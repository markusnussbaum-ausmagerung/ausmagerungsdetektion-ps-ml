# -*- coding: utf-8 -*-
"""
N02a — CVAT-Masken und Patch-Labels

Rastert CVAT-Polygone (Export-XML) zu Masken und vergibt je Patch ein
Soll-Label (strikt/locker, Purity-Regel). Eingang: ps_output.npz und
CVAT-Exports, Ausgang: Patch-Label-Tabellen.
"""


# -*- coding: utf-8 -*-
"""20260514_nb-02a_CVAT_Mask_Patches_v02.ipynb

Version v02 - Mehrklassen-CVAT-Annotation mit Exclusion-Logik
"""

# ============================================================
# NOTEBOOK 02a v02 — CVAT-XML → 3 BINÄRE MASKEN → PATCH-LABELS
# ============================================================
# Liest CVAT-Annotationen (XML-Format), rastert Polygone in
# drei separate binäre Masken (eine pro Label-Klasse), legt
# das Patch-Raster darüber und vergibt Labels nach Purity-Regel
# mit Exclusion-Mechanismus.
#
# CVAT-Labels:
#   - "ausgemagert"      → Ziel-Klasse für Purity-Regel
#   - "uebergang_unklar" → Exclusion (Patch nicht trainierbar)
#   - "andere_schaeden"  → Exclusion (Patch nicht trainierbar)
#
# Output:
#   - {STANDORT_ID}_mask_{label}.png   (drei Masken pro Standort)
#   - patch_labels_all.csv             (Labels strikt + locker + Anteile)
# ============================================================

import numpy as np
import pandas as pd
import os, glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# In Colab: Drive mounten

# ============================================================
# KONFIGURATION
# ============================================================
# ============================================================
# UMGEBUNG UND PFADE
# ============================================================
# In Colab wird Google Drive eingehaengt, lokal/im Repo die Projektwurzel
# verwendet. Die Unterstruktur (03_Hardware_Fotobox, 04_Daten_Felderhebung,
# 05_ML_Pipeline) entspricht der in der Arbeit dokumentierten Ablage.
# Datenpfad per Umgebungsvariable MA_BASIS ueberschreibbar.
import os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJEKT = "/content/drive/MyDrive/MA_Ausmagerungsdetektion_PS"
except ImportError:
    PROJEKT = os.environ.get("MA_BASIS", ".")

BASIS = os.path.join(PROJEKT, "04_Daten_Felderhebung")
VERARBEITET_DIR = os.path.join(BASIS, '02_Verarbeitet/02_Standorte_gemittelt')
ANNOTATION_DIR   = os.path.join(BASIS, '03_Annotation')
CVAT_EXPORTS     = os.path.join(ANNOTATION_DIR, '01_Cvat_Exports')
MASKEN_DIR       = os.path.join(ANNOTATION_DIR, '02_Cvat_Masken')
PATCH_LABELS_DIR = os.path.join(ANNOTATION_DIR, '03_Patch_Labels')

# CVAT-Label-Klassen
LABEL_ZIEL      = 'ausgemagert'
LABEL_EXCLUSION = ['uebergang_unklar', 'andere_schaeden']
ALLE_LABEL      = [LABEL_ZIEL] + LABEL_EXCLUSION

# Purity-Regeln (zwei Varianten parallel berechnet)
PURITY_AUSGEMAGERT_STRIKT = 0.80
PURITY_INTAKT_STRIKT      = 0.05
PURITY_AUSGEMAGERT_LOCKER = 0.50
PURITY_INTAKT_LOCKER      = 0.20

# Exclusion: Patches mit zu viel "uebergang_unklar" oder "andere_schaeden"
# werden vom Training komplett ausgeschlossen (Label = 'exkludiert')
EXCLUSION_MAX = 0.10

PATCH_SIZE_MM = 50

os.makedirs(MASKEN_DIR, exist_ok=True)
os.makedirs(PATCH_LABELS_DIR, exist_ok=True)


# ============================================================
# 1. CVAT-XML PARSEN
# ============================================================
def parse_cvat_xml(xml_pfad):
    """
    Liest CVAT-XML und extrahiert pro Bild die Polygone.

    Rückgabe: dict {image_name: {'width', 'height', 'polygons': [...]}}
    """
    tree = ET.parse(xml_pfad)
    root = tree.getroot()

    annotations = {}

    for image in root.findall('image'):
        img_name = image.get('name')
        img_width  = int(image.get('width'))
        img_height = int(image.get('height'))

        polygons = []
        for poly in image.findall('polygon'):
            label = poly.get('label')
            points_str = poly.get('points')
            points = [tuple(map(float, p.split(',')))
                      for p in points_str.split(';')]
            polygons.append({
                'label': label,
                'points': points,
            })

        annotations[img_name] = {
            'width': img_width,
            'height': img_height,
            'polygons': polygons,
        }

    return annotations


# ============================================================
# 2. POLYGONE → DREI BINÄRE MASKEN RASTERN
# ============================================================
def polygone_zu_masken(polygons, width, height):
    """
    Rastert die drei CVAT-Label in drei separate binäre Masken.

    Rückgabe: dict {label: np.ndarray (H,W) uint8} mit Werten {0, 255}
    """
    masken_pil = {lbl: Image.new('L', (width, height), 0) for lbl in ALLE_LABEL}
    draws      = {lbl: ImageDraw.Draw(img) for lbl, img in masken_pil.items()}

    polygone_pro_label = {lbl: 0 for lbl in ALLE_LABEL}
    unbekannte_label   = []

    for poly in polygons:
        if poly['label'] in draws:
            draws[poly['label']].polygon(poly['points'], fill=255)
            polygone_pro_label[poly['label']] += 1
        else:
            unbekannte_label.append(poly['label'])

    if unbekannte_label:
        print(f"  ⚠ Unbekannte Label ignoriert: {set(unbekannte_label)}")

    return ({lbl: np.array(img) for lbl, img in masken_pil.items()},
            polygone_pro_label)


# ============================================================
# 3. PATCH-LABEL MIT EXCLUSION-LOGIK
# ============================================================
def label_patch_beide(patch_masken):
    """
    Vergibt strikt/locker-Labels + Exclusion-Flag für einen Patch.

    patch_masken: dict mit Patch-Ausschnitten der drei Masken

    Rückgabe: (label_strikt, label_locker, anteil_aus, anteil_excl)
    """
    anteil_aus  = (patch_masken[LABEL_ZIEL]            > 127).mean()
    anteil_un   = (patch_masken['uebergang_unklar']    > 127).mean()
    anteil_an   = (patch_masken['andere_schaeden']     > 127).mean()
    anteil_excl = anteil_un + anteil_an

    # Exclusion hat Vorrang
    if anteil_excl > EXCLUSION_MAX:
        return 'exkludiert', 'exkludiert', anteil_aus, anteil_excl

    # Strikte Purity-Regel
    if anteil_aus >= PURITY_AUSGEMAGERT_STRIKT:
        label_strikt = 'ausgemagert'
    elif anteil_aus <= PURITY_INTAKT_STRIKT:
        label_strikt = 'intakt'
    else:
        label_strikt = 'unklar'

    # Lockere Purity-Regel
    if anteil_aus >= PURITY_AUSGEMAGERT_LOCKER:
        label_locker = 'ausgemagert'
    elif anteil_aus <= PURITY_INTAKT_LOCKER:
        label_locker = 'intakt'
    else:
        label_locker = 'unklar'

    return label_strikt, label_locker, anteil_aus, anteil_excl


# ============================================================
# 4. HAUPTSCHLEIFE — ALLE STANDORTE
# ============================================================
xml_files = sorted(glob.glob(os.path.join(CVAT_EXPORTS, '*.xml')))
print(f"Gefunden: {len(xml_files)} CVAT-XML-Dateien")
for f in xml_files:
    print(f"  - {os.path.basename(f)}")

if len(xml_files) == 0:
    raise RuntimeError("Keine CVAT-Exports gefunden. Bitte erst annotieren.")


alle_patch_labels = []

for xml_pfad in xml_files:
    fname = os.path.basename(xml_pfad)
    standort_id = fname.replace('_annotations.xml', '').replace('.xml', '')

    print(f"\n{'='*60}")
    print(f"Verarbeite: {standort_id}")
    print(f"{'='*60}")

    # PS-Output dieses Standorts laden
    npz_pfad = os.path.join(VERARBEITET_DIR, standort_id, 'ps_output.npz')
    if not os.path.exists(npz_pfad):
        print(f"  ⚠ Kein ps_output.npz gefunden: {npz_pfad}")
        continue

    data = np.load(npz_pfad)
    H, W = data['Nx'].shape
    gsd  = float(data['gsd_mm_per_px'])
    patch_px = int(round(PATCH_SIZE_MM / gsd))
    n_rows = H // patch_px
    n_cols = W // patch_px

    print(f"  PS-Bild:     {H}×{W} px  (GSD {gsd:.4f} mm/px)")
    print(f"  Patch-Grid:  {n_rows}×{n_cols} = {n_rows*n_cols} Patches "
          f"(je {patch_px} px = {patch_px*gsd:.1f} mm)")

    # XML einlesen
    anns = parse_cvat_xml(xml_pfad)
    if len(anns) == 0:
        print(f"  ⚠ XML enthält keine Bild-Annotationen.")
        continue

    # Polygone ÜBER ALLE Layer (Bilder) im XML sammeln.
    # In CVAT liegen v1a/v2/v3/v4 als einzelne Frames in EINER Aufgabe;
    # annotiert wird nur auf einem Layer. Welcher das ist, ist egal.
    # Binäre Masken-Vereinigung ist unkritisch (überlappende Polygone
    # füllen dieselben Pixel, der Flächenanteil bleibt korrekt).
    alle_polygone = []
    for name, d in anns.items():
        if d['width'] != W or d['height'] != H:
            print(f"  ⚠ Layer {name} ({d['width']}×{d['height']}) "
                  f"≠ PS-Bild ({W}×{H}) — übersprungen.")
            continue
        if len(d['polygons']) > 0:
            print(f"  Annotierter Layer: {name}  ({len(d['polygons'])} Polygone)")
        alle_polygone.extend(d['polygons'])

    print(f"  Polygone gesamt: {len(alle_polygone)}")
    for i, p in enumerate(alle_polygone):
        print(f"    [{i+1}] {p['label']}: {len(p['points'])} Punkte")

    # Drei Masken rastern
    masken, anzahl_polygone = polygone_zu_masken(alle_polygone, W, H)

    # Alle drei Masken speichern
    for lbl, m in masken.items():
        masken_pfad = os.path.join(MASKEN_DIR, f'{standort_id}_mask_{lbl}.png')
        Image.fromarray(m).save(masken_pfad)

    # Globale Flächenanteile
    print(f"\n  Globale Anteile im Bild:")
    for lbl, m in masken.items():
        anteil = (m > 127).mean() * 100
        print(f"    {lbl:20s}: {anteil:5.1f} %  ({anzahl_polygone[lbl]} Polygone)")

    # Patch-Labels berechnen — WICHTIG: Liste hier initialisieren!
    patches_dieses_standorts = []
    for iy in range(n_rows):
        for ix in range(n_cols):
            y0, y1 = iy * patch_px, (iy + 1) * patch_px
            x0, x1 = ix * patch_px, (ix + 1) * patch_px

            # Drei Patch-Ausschnitte als dict
            patch_masken = {lbl: m[y0:y1, x0:x1] for lbl, m in masken.items()}

            label_s, label_l, purity_aus, purity_excl = label_patch_beide(patch_masken)
            patches_dieses_standorts.append({
                'standort_id': standort_id,
                'patch_idx': iy * n_cols + ix,
                'patch_row': iy,
                'patch_col': ix,
                'patch_label_strikt': label_s,
                'patch_label_locker': label_l,
                'anteil_ausgemagert': purity_aus,
                'anteil_exclusion':   purity_excl,
            })

    # Zusammenfassung pro Standort
    df_s = pd.DataFrame(patches_dieses_standorts)
    print(f"\n  Patch-Verteilung (STRIKT ≥{PURITY_AUSGEMAGERT_STRIKT}/≤{PURITY_INTAKT_STRIKT}, Excl>{EXCLUSION_MAX}):")
    print(df_s['patch_label_strikt'].value_counts().to_string())
    print(f"\n  Patch-Verteilung (LOCKER ≥{PURITY_AUSGEMAGERT_LOCKER}/≤{PURITY_INTAKT_LOCKER}, Excl>{EXCLUSION_MAX}):")
    print(df_s['patch_label_locker'].value_counts().to_string())

    alle_patch_labels.extend(patches_dieses_standorts)


# ============================================================
# 5. GESAMT-CSV SCHREIBEN
# ============================================================
df = pd.DataFrame(alle_patch_labels)
out_csv = os.path.join(PATCH_LABELS_DIR, 'patch_labels_all.csv')
df.to_csv(out_csv, index=False)

print(f"\n{'='*60}")
print(f"GESAMT-ZUSAMMENFASSUNG")
print(f"{'='*60}")
print(f"  Gespeichert: {out_csv}")
print(f"  Standorte:   {df['standort_id'].nunique()}")
print(f"  Patches:     {len(df)}")
print(f"\n  Label-Verteilung gesamt — STRIKT:")
print(df['patch_label_strikt'].value_counts().to_string())
print(f"\n  Label-Verteilung gesamt — LOCKER:")
print(df['patch_label_locker'].value_counts().to_string())

n_strikt = (df['patch_label_strikt'] == 'unklar').sum()
n_locker = (df['patch_label_locker'] == 'unklar').sum()
n_excl   = (df['patch_label_strikt'] == 'exkludiert').sum()
print(f"\n  Anteil 'unklar'     STRIKT: {n_strikt/len(df)*100:5.1f} %")
print(f"  Anteil 'unklar'     LOCKER: {n_locker/len(df)*100:5.1f} %")
print(f"  Anteil 'exkludiert' (beide): {n_excl/len(df)*100:5.1f} %")


# ============================================================
# 6. VISUALISIERUNG — FÜR ALLE STANDORTE (je ein PNG)
# ============================================================
# Erzeugt pro Standort ein {standort_id}_visualisierung.png in
# 03_Patch_Labels/. ZEIGE_PLOTS=True zeigt sie zusätzlich inline
# in Colab; auf False nur speichern (kein Zuscrollen bei 7 Plots).
ZEIGE_PLOTS = True

label_farben = {
    'intakt':      'green',
    'ausgemagert': 'red',
    'unklar':      'gray',
    'exkludiert':  'purple',
}
farben_maske = {
    'ausgemagert':      ('Reds',    0.45),
    'uebergang_unklar': ('Blues',   0.45),
    'andere_schaeden':  ('Purples', 0.45),
}

for sid_v in sorted(df['standort_id'].unique()):
    # Drei Masken laden
    masken_v = {}
    for lbl in ALLE_LABEL:
        p = os.path.join(MASKEN_DIR, f'{sid_v}_mask_{lbl}.png')
        if os.path.exists(p):
            masken_v[lbl] = np.array(Image.open(p))

    if len(masken_v) == 0:
        print(f"  ⚠ {sid_v}: keine Masken gefunden — übersprungen.")
        continue

    df_v = df[df['standort_id'] == sid_v]

    # PS-Output für Hintergrund
    npz_v = np.load(os.path.join(VERARBEITET_DIR, sid_v, 'ps_output.npz'))
    Nz_v = npz_v['Nz']
    patch_px = int(round(PATCH_SIZE_MM / float(npz_v['gsd_mm_per_px'])))

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # 1: Nz-Karte als Hintergrund
    axes[0, 0].imshow(Nz_v, cmap='gray', vmin=0.7, vmax=1.0)
    axes[0, 0].set_title(f'{sid_v} — Nz-Karte', fontsize=13)

    # 2: Alle drei Masken überlagert in Farben
    axes[0, 1].imshow(Nz_v, cmap='gray', vmin=0.7, vmax=1.0)
    for lbl, (cmap, alpha) in farben_maske.items():
        if lbl in masken_v:
            axes[0, 1].imshow(masken_v[lbl], cmap=cmap, alpha=alpha)
    axes[0, 1].set_title('Masken — rot=aus, blau=unklar, lila=anderer Schaden',
                          fontsize=11)

    # 3: Patch-Labels STRIKT
    axes[1, 0].imshow(Nz_v, cmap='gray', vmin=0.7, vmax=1.0, alpha=0.6)
    for _, row in df_v.iterrows():
        r, c = int(row['patch_row']), int(row['patch_col'])
        farbe = label_farben.get(row['patch_label_strikt'], 'black')
        rect = Rectangle((c * patch_px, r * patch_px), patch_px, patch_px,
                         linewidth=2, edgecolor=farbe,
                         facecolor=farbe, alpha=0.25)
        axes[1, 0].add_patch(rect)
    axes[1, 0].set_title('STRIKT — grün=intakt, rot=aus, grau=unklar, lila=exkl.',
                          fontsize=11)

    # 4: Patch-Labels LOCKER
    axes[1, 1].imshow(Nz_v, cmap='gray', vmin=0.7, vmax=1.0, alpha=0.6)
    for _, row in df_v.iterrows():
        r, c = int(row['patch_row']), int(row['patch_col'])
        farbe = label_farben.get(row['patch_label_locker'], 'black')
        rect = Rectangle((c * patch_px, r * patch_px), patch_px, patch_px,
                         linewidth=2, edgecolor=farbe,
                         facecolor=farbe, alpha=0.25)
        axes[1, 1].add_patch(rect)
    axes[1, 1].set_title('LOCKER — grün=intakt, rot=aus, grau=unklar, lila=exkl.',
                          fontsize=11)

    for ax in axes.flatten():
        ax.axis('off')

    plt.suptitle(f'{sid_v}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PATCH_LABELS_DIR, f'{sid_v}_visualisierung.png'),
                dpi=120, bbox_inches='tight')
    if ZEIGE_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"  ✓ {sid_v}_visualisierung.png gespeichert")

print(f"\n  Alle Visualisierungen in: {PATCH_LABELS_DIR}")

import pandas as pd
df = pd.read_csv(os.path.join(PROJEKT,
                 '04_Daten_Felderhebung/03_Annotation/03_Patch_Labels/'
                 'patch_labels_all.csv'))
print(pd.crosstab(df['standort_id'], df['patch_label_strikt']))
print()
print(pd.crosstab(df['standort_id'], df['patch_label_locker']))