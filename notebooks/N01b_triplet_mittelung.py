# -*- coding: utf-8 -*-
"""
N01b — Triplet-Mittelung je Standort

Mittelt die heterogenen Triplet-Aufnahmen je Standort zu einem PS-Output.
Normalen komponentenweise summiert und renormiert, Albedo arithmetisch,
Confidence pixelweise als Minimum. Eingang: 02_Verarbeitet/01_Triplets,
Ausgang: 02_Verarbeitet/02_Standorte_gemittelt.
"""


# ============================================================
# NOTEBOOK 01b — TRIPLET-MITTELUNG
# ============================================================
# Mittelt pro Standort die Triplet-PS-Outputs (S0XX-1, S0XX-2,
# S0XX-3) zu einem gemittelten PS-Output (S0XX).
#
# Sonderfall S001: nur 1 Aufnahme vorhanden → keine Mittelung,
# Daten werden 1:1 in den gemittelten Output-Ordner kopiert.
#
# Erzeugt zusätzlich:
#   - Repeatability-Karten (std Nz, std Albedo, Winkel-Abweichung)
#     nur wenn mehr als 1 Triplet vorhanden
#   - Annotations-Bilder v2, v3, v4 aus gemittelten Daten
#   - v1a aus dem mittleren Triplet (Index 2) kopieren —
#     so bleibt die "unabhängige" Realaufnahme verfügbar
#   - Diagnose-Plot pro Standort (Triplets vs. gemittelt)
#
# Eingabe-Pfad:  02_Verarbeitet/01_Triplets/S0XX-N/ps_output.npz
# Ausgabe-Pfad:  02_Verarbeitet/02_Standorte_gemittelt/S0XX/
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import datetime
import shutil
from PIL import Image
from collections import defaultdict



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
TRIPLETS_DIR = os.path.join(BASIS, '02_Verarbeitet/01_Triplets')
OUTPUT_DIR   = os.path.join(BASIS, '02_Verarbeitet/02_Standorte_gemittelt')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. STANDORTE GRUPPIEREN (Triplets ODER Einzelaufnahmen)
# ============================================================
alle_npz = sorted(glob.glob(os.path.join(TRIPLETS_DIR, '*/ps_output.npz')))
print(f"Gefunden: {len(alle_npz)} PS-Output-Dateien")

# Gruppieren: S0XX-N → Basis-ID S0XX
# Wenn kein -N im Namen: Standort ist Einzelaufnahme (z.B. S001)
gruppen = defaultdict(list)
for npz_pfad in alle_npz:
    ordner = os.path.basename(os.path.dirname(npz_pfad))
    m = re.match(r'(S\d+)[-_](\d+)$', ordner)
    if m:
        basis_id, trip_idx = m.group(1), int(m.group(2))
        gruppen[basis_id].append((trip_idx, npz_pfad, ordner))
    else:
        # Einzelaufnahme — Ordner heißt direkt S001, S002, ...
        m_single = re.match(r'(S\d+)$', ordner)
        if m_single:
            basis_id = m_single.group(1)
            gruppen[basis_id].append((1, npz_pfad, ordner))
        else:
            print(f"  ⚠ Ungültiger Ordner-Name: {ordner}")

print(f"\nBasis-Standorte gefunden: {len(gruppen)}")
for sid in sorted(gruppen.keys()):
    n_trip = len(gruppen[sid])
    indices = sorted([t[0] for t in gruppen[sid]])
    if n_trip == 1:
        print(f"  {sid}: 1 Aufnahme (KEINE Mittelung)")
    elif n_trip in (2, 3):
        print(f"  {sid}: {n_trip} Aufnahmen {indices} ✓")
    else:
        print(f"  {sid}: {n_trip} Aufnahmen {indices} — ungewöhnlich, bitte prüfen")


# ============================================================
# 2. MITTELUNG ODER 1:1-ÜBERNAHME PRO STANDORT
# ============================================================
def verarbeite_standort(sid, triplet_liste):
    """
    Wenn ≥2 Triplets: mittele.
    Wenn 1 Aufnahme: kopiere durch.
    Rückgabe: dict mit den finalen Arrays + Diagnose-Daten.
    """
    n = len(triplet_liste)
    triplet_liste_sortiert = sorted(triplet_liste, key=lambda x: x[0])

    # Lade alle vorhandenen Triplets
    Nx_stack, Ny_stack, Nz_stack = [], [], []
    alb_stack, conf_stack = [], []
    quell_ordner = []

    meta = None
    for trip_idx, npz_pfad, ordner in triplet_liste_sortiert:
        data = np.load(npz_pfad)
        Nx_stack.append(data['Nx'])
        Ny_stack.append(data['Ny'])
        Nz_stack.append(data['Nz'])
        alb_stack.append(data['albedo'])
        conf_stack.append(data['n_valid_leds'])
        quell_ordner.append(ordner)

        if meta is None:
            meta = {
                'gsd_mm_per_px': float(data['gsd_mm_per_px']),
                'crop_size_px':  int(data['crop_size_px']),
                'image_shape':   data['image_shape'],
                'standort_label': str(data['standort_label']) if 'standort_label' in data.files else 'unbekannt',
            }

    Nx_s = np.stack(Nx_stack, axis=0)
    Ny_s = np.stack(Ny_stack, axis=0)
    Nz_s = np.stack(Nz_stack, axis=0)
    alb_s = np.stack(alb_stack, axis=0)
    conf_s = np.stack(conf_stack, axis=0)

    if n == 1:
        # === Einzelaufnahme: keine Mittelung ===
        Nx_final = Nx_s[0]
        Ny_final = Ny_s[0]
        Nz_final = Nz_s[0]
        alb_final = alb_s[0]
        conf_final = conf_s[0]
        repeatability = None  # kein Repeatability-Output
    else:
        # === Triplet-Mittelung ===
        # Normalen: komponentenweise summieren + renormieren
        Nx_sum = Nx_s.sum(axis=0)
        Ny_sum = Ny_s.sum(axis=0)
        Nz_sum = Nz_s.sum(axis=0)
        norm = np.sqrt(Nx_sum**2 + Ny_sum**2 + Nz_sum**2)
        norm_safe = np.maximum(norm, 1e-10)
        Nx_final = Nx_sum / norm_safe
        Ny_final = Ny_sum / norm_safe
        Nz_final = Nz_sum / norm_safe

        # Albedo: arithmetischer Mittelwert
        alb_final = alb_s.mean(axis=0)

        # Confidence: Minimum über alle Triplets (konservativ)
        conf_final = conf_s.min(axis=0)

        # Repeatability-Karten
        std_Nx = Nx_s.std(axis=0)
        std_Ny = Ny_s.std(axis=0)
        std_Nz = Nz_s.std(axis=0)
        std_albedo = alb_s.std(axis=0)

        # Winkel-Abweichung zwischen Triplets und Mittelung
        angles = []
        for i in range(n):
            dot = (Nx_final * Nx_s[i] + Ny_final * Ny_s[i] +
                   Nz_final * Nz_s[i])
            dot_clipped = np.clip(dot, -1.0, 1.0)
            angles.append(np.degrees(np.arccos(dot_clipped)))
        angle_mean = np.mean(angles, axis=0)

        repeatability = {
            'std_Nx': std_Nx.astype(np.float32),
            'std_Ny': std_Ny.astype(np.float32),
            'std_Nz': std_Nz.astype(np.float32),
            'std_albedo': std_albedo.astype(np.float32),
            'angle_dev_deg': angle_mean.astype(np.float32),
        }

    return {
        'Nx': Nx_final.astype(np.float32),
        'Ny': Ny_final.astype(np.float32),
        'Nz': Nz_final.astype(np.float32),
        'albedo': alb_final.astype(np.float32),
        'n_valid_leds': conf_final.astype(np.uint8),
        # Originalwerte für Visualisierung
        'Nx_single': Nx_s.astype(np.float32),
        'Ny_single': Ny_s.astype(np.float32),
        'Nz_single': Nz_s.astype(np.float32),
        'albedo_single': alb_s.astype(np.float32),
        'n_triplets': n,
        'quell_ordner': quell_ordner,
        'repeatability': repeatability,
        'meta': meta,
    }


# ============================================================
# 3. SHADING-EXPORT FÜR CVAT (auf gemittelten Daten)
# ============================================================
def export_annotations_bilder(out_dir, sid, Nx, Ny, Nz, albedo,
                               quell_ordner_v1a=None):
    """
    Erzeugt v2 (Lambert geometric), v3 (Lambert × Albedo),
    v4 (Nz-Karte) aus den gemittelten Daten.
    Kopiert v1a (Original LED07) aus dem Quell-Ordner —
    bei Triplets nimmt sie die mittlere Aufnahme.
    """
    annot_dir = os.path.join(out_dir, 'annotation')
    os.makedirs(annot_dir, exist_ok=True)

    def stretch(img, lo=2, hi=98):
        a, b = np.percentile(img, [lo, hi])
        return np.clip((img - a) / (b - a), 0, 1)

    def save_png(arr_float01, pfad):
        img = (np.clip(arr_float01, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(pfad)

    # v2: Lambert geometric
    light_dir = np.array([0.5, -0.5, 0.707])
    light_dir /= np.linalg.norm(light_dir)
    shading_geom = np.clip(
        Nx * light_dir[0] + Ny * light_dir[1] + Nz * light_dir[2], 0, 1)
    save_png(stretch(shading_geom),
             os.path.join(annot_dir, f'{sid}_v2_lambert_geometric.png'))

    # v3: Lambert × Albedo
    alb_norm = albedo / np.percentile(albedo, 98)
    alb_norm = np.clip(alb_norm, 0, 1)
    shading_alb = shading_geom * alb_norm
    save_png(stretch(shading_alb),
             os.path.join(annot_dir, f'{sid}_v3_lambert_albedo.png'))

    # v4: Nz-Karte
    NZ_MIN, NZ_MAX = 0.5, 1.0
    Nz_clipped = np.clip(Nz, NZ_MIN, NZ_MAX)
    Nz_8bit = ((Nz_clipped - NZ_MIN) / (NZ_MAX - NZ_MIN) * 255).astype(np.uint8)
    Image.fromarray(Nz_8bit, mode='L').save(
        os.path.join(annot_dir, f'{sid}_v4_nz_karte.png'))

# v1a + v1b: Original-LED-Aufnahmen aus Quell-Ordner kopieren
    if quell_ordner_v1a is not None:
        for tag, suffix in [('v1a', 'original_LED07_vorne'),
                            ('v1b', 'original_LED10_hinten')]:
            src = os.path.join(TRIPLETS_DIR, quell_ordner_v1a, 'annotation',
                               f'{quell_ordner_v1a}_{tag}_{suffix}.png')
            if os.path.exists(src):
                dst = os.path.join(annot_dir, f'{sid}_{tag}_{suffix}.png')
                shutil.copy2(src, dst)
            else:
                print(f"    ⚠ {tag} nicht gefunden: {os.path.basename(src)}")


# ============================================================
# 4. HAUPTSCHLEIFE
# ============================================================
for sid in sorted(gruppen.keys()):
    triplet_liste = gruppen[sid]
    n_triplets = len(triplet_liste)

    print(f"\n{'='*60}")
    print(f"Verarbeite: {sid}  ({n_triplets} Aufnahme(n))")
    print(f"{'='*60}")

    result = verarbeite_standort(sid, triplet_liste)

    # === Speichern ===
    out_standort_dir = os.path.join(OUTPUT_DIR, sid)
    os.makedirs(out_standort_dir, exist_ok=True)

    out_pfad = os.path.join(out_standort_dir, 'ps_output.npz')
    np.savez_compressed(
        out_pfad,
        Nx=result['Nx'], Ny=result['Ny'], Nz=result['Nz'],
        albedo=result['albedo'],
        n_valid_leds=result['n_valid_leds'],
        gsd_mm_per_px=np.float32(result['meta']['gsd_mm_per_px']),
        crop_size_px=np.int32(result['meta']['crop_size_px']),
        image_shape=result['meta']['image_shape'],
        standort_id=np.array(sid),
        standort_label=np.array(result['meta']['standort_label']),
        n_triplets_avg=np.int32(n_triplets),
        erzeugt_am=np.array(datetime.datetime.now().isoformat()),
        pipeline_version=np.array('1.1-triplet-avg'),
    )

    # Repeatability nur wenn ≥2 Triplets
    if result['repeatability'] is not None:
        rep_pfad = os.path.join(out_standort_dir, 'repeatability.npz')
        np.savez_compressed(rep_pfad, **result['repeatability'])

    # v1a aus mittlerem Triplet kopieren (Index 2 bei drei Triplets)
    if n_triplets == 3:
        v1a_quelle = result['quell_ordner'][1]  # Triplet 2 = Mitte
    else:
        v1a_quelle = result['quell_ordner'][0]  # bei 1 Aufnahme die einzige

    export_annotations_bilder(
        out_standort_dir, sid,
        result['Nx'], result['Ny'], result['Nz'], result['albedo'],
        quell_ordner_v1a=v1a_quelle,
    )

    # === Diagnose-Print ===
    if result['repeatability']:
        ang = result['repeatability']['angle_dev_deg']
        mean_angle = ang.mean()
        p95_angle = np.percentile(ang, 95)
        mean_std_Nz = result['repeatability']['std_Nz'].mean()
        print(f"  Repeatability:")
        print(f"    Mittlere Winkel-Abweichung: {mean_angle:.2f}°")
        print(f"    95. Perzentil Winkel:        {p95_angle:.2f}°")
        print(f"    Mittlere Nz-Std:             {mean_std_Nz:.4f}")
    else:
        print(f"  Einzelaufnahme — keine Repeatability berechnet")

    print(f"  Gespeichert: {out_pfad}")

    # === Visualisierung ===
    if n_triplets >= 2:
        # Diagnose-Plot für gemittelte Standorte (2 oder 3 Aufnahmen)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        rep = result['repeatability']
        col_mean = n_triplets            # Spalte der Mittelung: 2 bei n=2, 3 bei n=3

        # Reihe 1: Nz je Aufnahme + gemittelt
        for i in range(n_triplets):
            axes[0, i].imshow(result['Nz_single'][i], cmap='gray', vmin=0.7, vmax=1.0)
            axes[0, i].set_title(f'Nz Aufnahme {i+1}', fontsize=11)
            axes[0, i].axis('off')
        axes[0, col_mean].imshow(result['Nz'], cmap='gray', vmin=0.7, vmax=1.0)
        axes[0, col_mean].set_title('Nz gemittelt', fontsize=11, fontweight='bold')
        axes[0, col_mean].axis('off')
        for j in range(col_mean + 1, 4):
            axes[0, j].axis('off')

        # Reihe 2: Albedo je Aufnahme + gemittelt
        v_lo, v_hi = np.percentile(result['albedo'], [2, 98])
        for i in range(n_triplets):
            axes[1, i].imshow(result['albedo_single'][i], cmap='gray', vmin=v_lo, vmax=v_hi)
            axes[1, i].set_title(f'Albedo Aufnahme {i+1}', fontsize=11)
            axes[1, i].axis('off')
        axes[1, col_mean].imshow(result['albedo'], cmap='gray', vmin=v_lo, vmax=v_hi)
        axes[1, col_mean].set_title('Albedo gemittelt', fontsize=11, fontweight='bold')
        axes[1, col_mean].axis('off')
        for j in range(col_mean + 1, 4):
            axes[1, j].axis('off')

        # Reihe 3: Repeatability (4 feste Panels)
        im = axes[2, 0].imshow(rep['std_Nz'], cmap='hot', vmin=0, vmax=0.1)
        axes[2, 0].set_title('std(Nz)', fontsize=11); axes[2, 0].axis('off')
        plt.colorbar(im, ax=axes[2, 0], fraction=0.046)
        im = axes[2, 1].imshow(rep['std_albedo'], cmap='hot',
                               vmin=0, vmax=rep['std_albedo'].mean() * 3)
        axes[2, 1].set_title('std(Albedo)', fontsize=11); axes[2, 1].axis('off')
        plt.colorbar(im, ax=axes[2, 1], fraction=0.046)
        im = axes[2, 2].imshow(rep['angle_dev_deg'], cmap='hot', vmin=0, vmax=5)
        axes[2, 2].set_title('Winkel-Abweichung [°]', fontsize=11); axes[2, 2].axis('off')
        plt.colorbar(im, ax=axes[2, 2], fraction=0.046)
        axes[2, 3].hist(rep['angle_dev_deg'].ravel(), bins=100, range=(0, 10),
                        color='steelblue', alpha=0.7)
        axes[2, 3].axvline(rep['angle_dev_deg'].mean(), color='red', linestyle='--',
                           label=f"Mittel: {rep['angle_dev_deg'].mean():.2f}°")
        axes[2, 3].axvline(np.percentile(rep['angle_dev_deg'], 95), color='orange',
                           linestyle='--',
                           label=f"P95: {np.percentile(rep['angle_dev_deg'], 95):.2f}°")
        axes[2, 3].set_xlabel('Winkel-Abweichung [°]')
        axes[2, 3].set_ylabel('Anzahl Pixel (log)')
        axes[2, 3].set_yscale('log'); axes[2, 3].legend(); axes[2, 3].grid(alpha=0.3)
        axes[2, 3].set_title('Verteilung Winkel-Abweichung')

        # Reihe 4: Differenz je Aufnahme zur Mittelung
        for i in range(n_triplets):
            diff = result['Nz_single'][i] - result['Nz']
            im = axes[3, i].imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            axes[3, i].set_title(f'Nz_A{i+1} − Nz_mean', fontsize=11)
            axes[3, i].axis('off')
        for j in range(n_triplets, 4):
            axes[3, j].axis('off')

        plt.suptitle(f'{sid} — Mittelung aus {n_triplets} Aufnahmen und Repeatability',
                     fontsize=14)
    else:
        # n == 1: echte Einzelaufnahme (S001)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(result['Nz'], cmap='gray', vmin=0.7, vmax=1.0)
        axes[0].set_title(f'{sid} — Nz (Einzelaufnahme)', fontsize=12)
        axes[0].axis('off')
        v_lo, v_hi = np.percentile(result['albedo'], [2, 98])
        axes[1].imshow(result['albedo'], cmap='gray', vmin=v_lo, vmax=v_hi)
        axes[1].set_title(f'{sid} — Albedo (Einzelaufnahme)', fontsize=12)
        axes[1].axis('off')


    plt.tight_layout()
    plt.savefig(os.path.join(out_standort_dir,
                              f'{sid}_triplet_diagnose.png'),
                dpi=110, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. ZUSAMMENFASSUNG
# ============================================================
print(f"\n{'='*60}")
print(f"GESAMT-ZUSAMMENFASSUNG")
print(f"{'='*60}")
print(f"  Verarbeitete Standorte: {len(gruppen)}")
print(f"  Output-Verzeichnis:     {OUTPUT_DIR}")
print(f"\n  Pro Standort gespeichert:")
print(f"    ps_output.npz          (gemittelt oder Einzelaufnahme)")
print(f"    repeatability.npz      (nur bei ≥2 Triplets)")
print(f"    annotation/v1a-v4.png  (für CVAT)")
print(f"    triplet_diagnose.png   (visueller Vergleich)")