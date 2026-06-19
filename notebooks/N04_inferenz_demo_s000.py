# -*- coding: utf-8 -*-
"""
N04 — Inferenz-Demonstration S000 (qualitativ)

Wendet das in N03 gespeicherte RF-Modell auf den Pilotstandort S000 an
und vergleicht qualitativ mit der CVAT-Annotation. KEIN Leistungsmass
(S000 wirkte an der Feature-Auswahl mit, Feature-Selection-Leakage).
"""


# -*- coding: utf-8 -*-
"""
20260603_nb04_Inferenz_Demo_S000.ipynb

============================================================
NOTEBOOK 04 — INFERENZ-DEMONSTRATION (S000_Test)
============================================================
Wendet das in N3 trainierte, gespeicherte Random-Forest-Modell
auf den Pilotstandort S000_Test an und vergleicht qualitativ
die Modellvorhersage mit der CVAT-Annotation.

WICHTIG — qualitative Demonstration, KEIN Leistungsmaß:
  S000_Test wirkte an der Feature-Auswahl mit (Pilot-Diskriminanz)
  → eine Trefferquote wäre zirkulär (Feature-Selection-Leakage).
  Zudem fehlt in der S000-Annotation die Exclusion-Klasse; die
  Soll-Karte kennt daher nur ausgemagert/intakt/unklar.

Konsistenz-Garantien:
  - Feature-Extraktion bitgleich zu N2b (gleiche Funktionen,
    MIN_CONFIDENCE=6 wie im Trainingslauf, NICHT 8!).
  - Soll-Karte mit N2a-Purity-Logik (LOCKER, ≥0,50/≤0,20) aus der CVAT-XML;
    locker gewählt, da S000 nur mit der Zielklasse annotiert ist.
  - Beide Karten auf demselben 7×7-Raster → patchweise vergleichbar.

Eingaben:
  ps_output: .../02_Verarbeitet/Archiv/S000_Test_v02/ps_output.npz
  CVAT-XML:  .../03_Annotation/Archiv/S000_Test_annotations.xml
  Modell:    .../05_ML_Pipeline/04_Modelle/final_rf.joblib
Ausgabe:
  .../05_ML_Pipeline/06_Inferenz_Demo/
============================================================
"""

import numpy as np
import pandas as pd
import os
import joblib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from scipy.stats import entropy, skew, kurtosis
from scipy.ndimage import sobel



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

BASIS = PROJEKT
PS_NPZ  = os.path.join(BASIS, '04_Daten_Felderhebung/02_Verarbeitet/'
                       'Archiv/S000_Test_v02/ps_output.npz')
CVAT_XML = os.path.join(BASIS, '04_Daten_Felderhebung/03_Annotation/'
                        'Archiv/S000_Test_annotations.xml')
MODELL  = os.path.join(BASIS, '05_ML_Pipeline/04_Modelle/final_rf.joblib')
META    = os.path.join(BASIS, '05_ML_Pipeline/04_Modelle/modell_meta.joblib')
OUT_DIR = os.path.join(BASIS, '05_ML_Pipeline/06_Inferenz_Demo')
os.makedirs(OUT_DIR, exist_ok=True)

STANDORT = 'S000_Test'

# --- MUSS mit dem Trainingslauf (N2b) übereinstimmen ---
PATCH_SIZE_MM   = 50
MIN_CONFIDENCE  = 6     # <-- 6, NICHT 8 (Trainingsdatensatz wurde mit 6 erzeugt)
MIN_VALID_RATIO = 0.70

# CVAT-Label-Klassen (N2a-Logik); Exclusion bei S000 faktisch nicht belegt
LABEL_ZIEL      = 'ausgemagert'
LABEL_EXCLUSION = ['uebergang_unklar', 'andere_schaeden']
ALLE_LABEL      = [LABEL_ZIEL] + LABEL_EXCLUSION
# LOCKERE Purity-Variante für die Soll-Karte:
# S000 ist nur mit der Zielklasse annotiert (keine Exclusion); die strikte
# 0,80-Schwelle würde hier zu viele Patches als 'unklar' statt 'ausgemagert'
# führen. Lockere Schwellen (N2a) bilden die Annotation realistischer ab.
PURITY_AUS_LOCKER = 0.50
PURITY_INT_LOCKER = 0.20
EXCLUSION_MAX     = 0.10


# ============================================================
# 1. FEATURE-BERECHNUNG — WORTGLEICH AUS N2b
# ============================================================
def berechne_patch_features(Nx_p, Ny_p, Nz_p, alb_p, conf_p,
                            G_p, G_global_threshold):
    valid = conf_p >= MIN_CONFIDENCE
    valid_ratio = float(valid.mean())
    if valid_ratio < MIN_VALID_RATIO:
        return {'valid_ratio': valid_ratio,
                'roughness_index': np.nan, 'var_Nz': np.nan,
                'surface_entropy_Nz': np.nan, 'albedo_variance': np.nan,
                'skewness_Nz': np.nan, 'kurtosis_Nz': np.nan,
                'edge_mean': np.nan, 'edge_std': np.nan,
                'edge_p95': np.nan, 'edge_density_global': np.nan}

    nx, ny, nz = Nx_p[valid], Ny_p[valid], Nz_p[valid]
    al, g = alb_p[valid], G_p[valid]

    std_Nx_v = float(np.std(nx)); std_Ny_v = float(np.std(ny))
    roughness = float(np.sqrt(std_Nx_v**2 + std_Ny_v**2))
    var_Nz_v = float(np.var(nz))
    hist, _ = np.histogram(nz, bins=30, range=(0.5, 1.0))
    surf_ent = float(entropy(hist + 1))
    skew_Nz = float(skew(nz)); kurt_Nz = float(kurtosis(nz))
    alb_var = float(np.var(al))
    edge_mean = float(g.mean()); edge_std = float(g.std())
    edge_p95 = float(np.percentile(g, 95))
    edge_density_global = float((g > G_global_threshold).mean())

    return {'valid_ratio': valid_ratio,
            'roughness_index': roughness, 'var_Nz': var_Nz_v,
            'surface_entropy_Nz': surf_ent, 'albedo_variance': alb_var,
            'skewness_Nz': skew_Nz, 'kurtosis_Nz': kurt_Nz,
            'edge_mean': edge_mean, 'edge_std': edge_std,
            'edge_p95': edge_p95, 'edge_density_global': edge_density_global}


# ============================================================
# 2. CVAT-XML → MASKEN → PATCH-SOLL-LABEL (N2a-Logik, union)
# ============================================================
def parse_und_maskiere(xml_pfad, W, H):
    root = ET.parse(xml_pfad).getroot()
    masken_pil = {lbl: Image.new('L', (W, H), 0) for lbl in ALLE_LABEL}
    draws = {lbl: ImageDraw.Draw(img) for lbl, img in masken_pil.items()}
    n_poly = 0
    for image in root.findall('image'):
        iw, ih = int(image.get('width')), int(image.get('height'))
        if iw != W or ih != H:
            print(f"  ⚠ Layer {image.get('name')} {iw}x{ih} ≠ {W}x{H} — übersprungen")
            continue
        for poly in image.findall('polygon'):
            lbl = poly.get('label')
            if lbl in draws:
                pts = [tuple(map(float, p.split(',')))
                       for p in poly.get('points').split(';')]
                draws[lbl].polygon(pts, fill=255)
                n_poly += 1
    print(f"  Polygone gerastert: {n_poly}")
    return {lbl: np.array(img) for lbl, img in masken_pil.items()}


def soll_label(masken_p):
    anteil_aus = (masken_p[LABEL_ZIEL] > 127).mean()
    anteil_excl = sum((masken_p[l] > 127).mean() for l in LABEL_EXCLUSION)
    if anteil_excl > EXCLUSION_MAX:
        return 'exkludiert'
    if anteil_aus >= PURITY_AUS_LOCKER:
        return 'ausgemagert'
    if anteil_aus <= PURITY_INT_LOCKER:
        return 'intakt'
    return 'unklar'


# ============================================================
# 3. DATEN LADEN
# ============================================================
data = np.load(PS_NPZ)
Nx, Ny, Nz = data['Nx'], data['Ny'], data['Nz']
albedo, conf = data['albedo'], data['n_valid_leds']
gsd = float(data['gsd_mm_per_px'])
H, W = Nz.shape
patch_px = int(round(PATCH_SIZE_MM / gsd))
n_y, n_x = H // patch_px, W // patch_px
print(f"{STANDORT}: {n_x}×{n_y} = {n_x*n_y} Patches ({patch_px} px = {patch_px*gsd:.1f} mm)")
print(f"  conf n_valid_leds: median={int(np.median(conf))}, Schwelle={MIN_CONFIDENCE}")

rf = joblib.load(MODELL)
meta = joblib.load(META)
FEATURES = meta['feature_names']
idx_aus = int(np.where(rf.classes_ == 1)[0][0])   # Spalte für P(ausgemagert)

masken = parse_und_maskiere(CVAT_XML, W, H)


# ============================================================
# 4. SOBEL (ganzes Bild) + GLOBALER THRESHOLD — wie N2b
# ============================================================
Gx = sobel(Nz, axis=1, mode='reflect')
Gy = sobel(Nz, axis=0, mode='reflect')
G  = np.sqrt(Gx**2 + Gy**2)
valid_global = conf >= MIN_CONFIDENCE
G_global_p90 = float(np.percentile(G[valid_global], 90)) if valid_global.sum() else 0.0


# ============================================================
# 5. PRO PATCH: SOLL-LABEL, FEATURES, VORHERSAGE
# ============================================================
zeilen = []
for iy in range(n_y):
    for ix in range(n_x):
        y0, y1 = iy*patch_px, (iy+1)*patch_px
        x0, x1 = ix*patch_px, (ix+1)*patch_px

        feats = berechne_patch_features(
            Nx[y0:y1, x0:x1], Ny[y0:y1, x0:x1], Nz[y0:y1, x0:x1],
            albedo[y0:y1, x0:x1], conf[y0:y1, x0:x1],
            G[y0:y1, x0:x1], G_global_p90)

        masken_p = {l: masken[l][y0:y1, x0:x1] for l in ALLE_LABEL}
        soll = soll_label(masken_p)

        gueltig = not np.isnan(feats['roughness_index'])
        if gueltig:
            x_vec = pd.DataFrame([[feats[f] for f in FEATURES]], columns=FEATURES)
            pred = int(rf.predict(x_vec)[0])
            proba_aus = float(rf.predict_proba(x_vec)[0, idx_aus])
        else:
            pred, proba_aus = None, None

        zeilen.append({'iy': iy, 'ix': ix, 'soll': soll,
                       'gueltig': gueltig, 'pred': pred, 'proba_aus': proba_aus,
                       'valid_ratio': feats['valid_ratio']})

pdf = pd.DataFrame(zeilen)
n_gueltig = int(pdf['gueltig'].sum())
print(f"\n  Gültige Patches: {n_gueltig} von {len(pdf)}")
print(f"  Vorhergesagt (gültige): "
      f"ausgemagert={int((pdf['pred']==1).sum())}, intakt={int((pdf['pred']==0).sum())}")
print(f"  Mittlere Konfidenz P(ausgemagert): "
      f"{pdf.loc[pdf['gueltig'],'proba_aus'].mean():.2f}")


# ============================================================
# 6. DREI KARTEN: SOLL | KONFIDENZ | HARTE VORHERSAGE
# ============================================================
soll_farben = {'intakt': 'green', 'ausgemagert': 'red',
               'unklar': 'gray', 'exkludiert': 'purple'}
cmap = cm.get_cmap('RdYlGn_r')   # 0=grün(intakt) … 1=rot(ausgemagert)
norm = Normalize(vmin=0, vmax=1)

fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
for ax in axes:
    ax.imshow(Nz, cmap='gray', vmin=0.7, vmax=1.0)
    ax.axis('off')

# (1) Soll
for _, r in pdf.iterrows():
    ax = axes[0]
    rect = Rectangle((r['ix']*patch_px, r['iy']*patch_px), patch_px, patch_px,
                     linewidth=1.5, edgecolor=soll_farben.get(r['soll'], 'black'),
                     facecolor=soll_farben.get(r['soll'], 'black'), alpha=0.30)
    ax.add_patch(rect)
axes[0].set_title(f'{STANDORT} — SOLL (CVAT, locker)\n'
                  'grün=intakt, rot=ausgemagert, grau=unklar', fontsize=12)

# (2) Konfidenz-Heatmap
for _, r in pdf.iterrows():
    ax = axes[1]
    if r['gueltig']:
        col = cmap(norm(r['proba_aus']))
        rect = Rectangle((r['ix']*patch_px, r['iy']*patch_px), patch_px, patch_px,
                         linewidth=1.5, edgecolor='k', facecolor=col, alpha=0.60)
    else:
        rect = Rectangle((r['ix']*patch_px, r['iy']*patch_px), patch_px, patch_px,
                         linewidth=1.5, edgecolor='k', facecolor='lightgray',
                         alpha=0.45, hatch='//')
    ax.add_patch(rect)
axes[1].set_title('IST — RF-Konfidenz P(ausgemagert)\n'
                  'grün→rot = 0→1, schraffiert grau = ungültig', fontsize=12)
sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=axes[1], fraction=0.046, label='P(ausgemagert)')

# (3) Harte Vorhersage
for _, r in pdf.iterrows():
    ax = axes[2]
    if r['gueltig']:
        col = 'red' if r['pred'] == 1 else 'green'
        rect = Rectangle((r['ix']*patch_px, r['iy']*patch_px), patch_px, patch_px,
                         linewidth=1.5, edgecolor='k', facecolor=col, alpha=0.35)
    else:
        rect = Rectangle((r['ix']*patch_px, r['iy']*patch_px), patch_px, patch_px,
                         linewidth=1.5, edgecolor='k', facecolor='lightgray',
                         alpha=0.45, hatch='//')
    ax.add_patch(rect)
axes[2].set_title('IST — harte Vorhersage\n'
                  'grün=intakt, rot=ausgemagert, grau=ungültig', fontsize=12)

plt.suptitle(f'Inferenz-Demonstration {STANDORT} — qualitativ, kein Leistungsmaß',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{STANDORT}_inferenz_demo.png'),
            dpi=130, bbox_inches='tight')
plt.show()

pdf.to_csv(os.path.join(OUT_DIR, f'{STANDORT}_patch_vorhersagen.csv'), index=False)


# ============================================================
# 7. DESKRIPTIVE EINORDNUNG (KEIN Leistungsmaß)
# ============================================================
vgl = pdf[(pdf['gueltig']) & (pdf['soll'].isin(['intakt', 'ausgemagert']))].copy()
vgl['soll01'] = (vgl['soll'] == 'ausgemagert').astype(int)
uebereinstimmung = (vgl['soll01'] == vgl['pred']).mean() if len(vgl) else np.nan
print(f"\n{'='*60}")
print("DESKRIPTIVE EINORDNUNG (rein illustrativ, KEIN Leistungsmaß)")
print('='*60)
print(f"  Patches mit eindeutigem Soll-Label und gültig: {len(vgl)}")
print(f"  davon vom Modell gleich eingeordnet: {uebereinstimmung*100:.0f} %")
print(f"  Hinweis: S000_Test wirkte an der Feature-Auswahl mit; zudem fehlt")
print(f"  die Exclusion-Klasse in der Annotation. Diese Zahl ist NICHT als")
print(f"  Genauigkeit zu interpretieren, sondern nur als grobe Plausibilität.")
print(f"\n  Ausgaben in: {OUT_DIR}")