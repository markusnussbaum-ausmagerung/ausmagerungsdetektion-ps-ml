# -*- coding: utf-8 -*-
"""
N02b — Feature-Extraction

Berechnet je Patch 10 ML-Features (roughness, var_Nz, Entropie, Albedo-
Varianz, Schiefe, Kurtosis, vier Sobel-Kantenmasse) plus Diagnose-Features.
Filter: MIN_CONFIDENCE>=6/12 LEDs, MIN_VALID_RATIO>=0.70.
Ausgang: features_all_patches.csv.
"""


# ============================================================
# NOTEBOOK 02b — FEATURE EXTRACTION
# ============================================================
# Liest alle ps_output.npz Dateien und berechnet pro Patch:
#
#   ML-Features (gehen in die Modellierung):
#     - roughness_index, var_Nz, surface_entropy_Nz
#     - albedo_variance
#     - skewness_Nz, kurtosis_Nz
#     - edge_mean, edge_std, edge_p95, edge_density_global  ← NEU (Sobel)
#
#   Diagnose-Features (in CSV, aber nicht für ML):
#     - std_Nx, std_Ny (X/Y-Asymmetrie monitoren)
#     - albedo_mean (Bias-Diagnose)
#     - mean_deviation_deg
#     - valid_ratio
#
# Output: features_all_patches.csv
# ============================================================

import numpy as np
import pandas as pd
import os, glob
from scipy.stats import entropy, skew, kurtosis
from scipy.ndimage import sobel
import matplotlib.pyplot as plt

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
OUTPUT_CSV = os.path.join(PROJEKT,
              '05_ML_Pipeline/01_Features/features_all_patches.csv')

PATCH_SIZE_MM   = 50        # 5×5 cm
MIN_CONFIDENCE  = 6         # Pixel braucht ≥ 6 von 12 LEDs mit Signal
MIN_VALID_RATIO = 0.70      # Patch braucht ≥ 70% gültige Pixel

print(f"Suche PS-Outputs in: {VERARBEITET_DIR}")
ps_files = sorted(glob.glob(os.path.join(VERARBEITET_DIR, '*/ps_output.npz')))
print(f"Gefunden: {len(ps_files)} Standorte")
for f in ps_files:
    print(f"  - {os.path.basename(os.path.dirname(f))}")

if len(ps_files) == 0:
    raise RuntimeError("Keine ps_output.npz gefunden.")


# ============================================================
# FEATURE-BERECHNUNG PRO PATCH
# ============================================================
def berechne_patch_features(Nx_p, Ny_p, Nz_p, alb_p, conf_p,
                            G_p, G_global_threshold):
    """
    Berechnet alle Features für einen Patch.
    G_p: vorab berechneter Sobel-Gradient für diesen Patch.
    G_global_threshold: bildweites 90. Perzentil von G (für edge_density_global).
    """
    # Confidence-Maske
    valid = conf_p >= MIN_CONFIDENCE
    valid_ratio = float(valid.mean())

    # Wenn zu wenig gültige Pixel: alle Features NaN
    if valid_ratio < MIN_VALID_RATIO:
        return {
            'valid_ratio': valid_ratio,
            # ML-Features
            'roughness_index': np.nan, 'var_Nz': np.nan,
            'surface_entropy_Nz': np.nan, 'albedo_variance': np.nan,
            'skewness_Nz': np.nan, 'kurtosis_Nz': np.nan,
            'edge_mean': np.nan, 'edge_std': np.nan,
            'edge_p95': np.nan, 'edge_density_global': np.nan,
            # Diagnose
            'std_Nx': np.nan, 'std_Ny': np.nan, 'albedo_mean': np.nan,
            'mean_deviation_deg': np.nan,
        }

    nx = Nx_p[valid]
    ny = Ny_p[valid]
    nz = Nz_p[valid]
    al = alb_p[valid]
    g  = G_p[valid]

    # === Statistische Features (permutations-invariant) ===
    std_Nx_v = float(np.std(nx))
    std_Ny_v = float(np.std(ny))
    roughness = float(np.sqrt(std_Nx_v**2 + std_Ny_v**2))
    var_Nz_v = float(np.var(nz))

    # Surface-Entropy aus Histogramm der Nz-Werte
    hist, _ = np.histogram(nz, bins=30, range=(0.5, 1.0))
    surf_ent = float(entropy(hist + 1))  # +1 vermeidet log(0)

    # Höhere Momente
    skew_Nz = float(skew(nz))
    kurt_Nz = float(kurtosis(nz))  # excess kurtosis (Normal = 0)

    # Albedo
    alb_var = float(np.var(al))
    alb_mean = float(np.mean(al))

    # Winkel-Abweichung
    nz_clipped = np.clip(nz, -1.0, 1.0)
    mean_dev_deg = float(np.degrees(np.mean(np.arccos(nz_clipped))))

    # === Räumliche Features (Sobel-Gradient) ===
    edge_mean = float(g.mean())
    edge_std  = float(g.std())
    edge_p95  = float(np.percentile(g, 95))
    edge_density_global = float((g > G_global_threshold).mean())

    return {
        'valid_ratio': valid_ratio,
        # ML-Features
        'roughness_index': roughness,
        'var_Nz': var_Nz_v,
        'surface_entropy_Nz': surf_ent,
        'albedo_variance': alb_var,
        'skewness_Nz': skew_Nz,
        'kurtosis_Nz': kurt_Nz,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'edge_p95': edge_p95,
        'edge_density_global': edge_density_global,
        # Diagnose
        'std_Nx': std_Nx_v,
        'std_Ny': std_Ny_v,
        'albedo_mean': alb_mean,
        'mean_deviation_deg': mean_dev_deg,
    }


# ============================================================
# HAUPTSCHLEIFE
# ============================================================
alle_patches = []

for fpath in ps_files:
    data = np.load(fpath)

    Nx = data['Nx']
    Ny = data['Ny']
    Nz = data['Nz']
    albedo = data['albedo']
    conf = data['n_valid_leds']

    gsd = float(data['gsd_mm_per_px'])
    sid = str(data['standort_id'])

    H, W = Nx.shape
    patch_px = int(round(PATCH_SIZE_MM / gsd))
    n_y = H // patch_px
    n_x = W // patch_px

    # === Sobel-Gradient auf das gesamte Nz-Bild ===
    # mode='reflect' → spiegelt am Rand, vermeidet künstliche Kanten
    Gx = sobel(Nz, axis=1, mode='reflect')
    Gy = sobel(Nz, axis=0, mode='reflect')
    G  = np.sqrt(Gx**2 + Gy**2)

    # Bildweites 90. Perzentil als Referenz-Schwelle
    # Nur über vertrauenswürdige Pixel berechnen
    valid_global = conf >= MIN_CONFIDENCE
    if valid_global.sum() > 0:
        G_global_p90 = float(np.percentile(G[valid_global], 90))
    else:
        G_global_p90 = 0.0

    print(f"\n{sid}: {n_x}×{n_y} = {n_x*n_y} Patches "
          f"({patch_px} px = {patch_px*gsd:.1f} mm)")
    print(f"  Sobel-Gradient bildweit:  "
          f"mean={G[valid_global].mean():.3f}, p90={G_global_p90:.3f}")

    # === Patches durchlaufen ===
    for iy in range(n_y):
        for ix in range(n_x):
            y0, y1 = iy * patch_px, (iy + 1) * patch_px
            x0, x1 = ix * patch_px, (ix + 1) * patch_px

            features = berechne_patch_features(
                Nx[y0:y1, x0:x1],
                Ny[y0:y1, x0:x1],
                Nz[y0:y1, x0:x1],
                albedo[y0:y1, x0:x1],
                conf[y0:y1, x0:x1],
                G[y0:y1, x0:x1],
                G_global_p90,
            )

            features.update({
                'standort_id': sid,
                'patch_idx': iy * n_x + ix,
                'patch_row': iy,
                'patch_col': ix,
                'patch_x_mm': ix * PATCH_SIZE_MM,
                'patch_y_mm': iy * PATCH_SIZE_MM,
                'patch_size_px': patch_px,
            })
            alle_patches.append(features)


# ============================================================
# DATAFRAME ERSTELLEN UND SPEICHERN
# ============================================================
df = pd.DataFrame(alle_patches)

# Spaltenreihenfolge: Meta, ML-Features, Diagnose
meta_cols = ['standort_id', 'patch_idx', 'patch_row', 'patch_col',
             'patch_x_mm', 'patch_y_mm', 'patch_size_px',
             'valid_ratio']

ml_feature_cols = ['roughness_index', 'var_Nz', 'surface_entropy_Nz',
                   'albedo_variance', 'skewness_Nz', 'kurtosis_Nz',
                   'edge_mean', 'edge_std', 'edge_p95',
                   'edge_density_global']

diag_cols = ['std_Nx', 'std_Ny', 'albedo_mean', 'mean_deviation_deg']

df = df[meta_cols + ml_feature_cols + diag_cols]

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

n_valid = df['roughness_index'].notna().sum()

print(f"\n{'='*60}")
print(f"FERTIG")
print(f"{'='*60}")
print(f"  Gespeichert: {OUTPUT_CSV}")
print(f"  Standorte:   {df['standort_id'].nunique()}")
print(f"  Patches:     {len(df)} (davon gültig: {n_valid})")
print(f"\nML-Features:")
print(df[ml_feature_cols].describe().round(4))
print(f"\nDiagnose-Features:")
print(df[diag_cols].describe().round(4))

# ============================================================
# DIAGNOSE / EXPLORATION (nicht Teil der Pipeline-Ausgabe)
# Ad-hoc-Kontrolle der Gueltigkeitsquoten je Standort.
# ============================================================
import pandas as pd
df = pd.read_csv(os.path.join(PROJEKT,
                 '05_ML_Pipeline/01_Features/features_all_patches.csv'))
df['gueltig'] = df['roughness_index'].notna()

# 1) Gültige Patches pro Standort + mittleres valid_ratio
print(df.groupby('standort_id').apply(
    lambda g: pd.Series({
        'gueltig':       int(g['gueltig'].sum()),
        'von':           len(g),
        'gueltig_%':     round(100*g['gueltig'].mean(), 1),
        'valid_ratio_mean': round(g['valid_ratio'].mean(), 3),
    })))

# 2) Wie knapp sind die Ausfälle?
print('\n>=0.70 (gültig):      ', (df['valid_ratio'] >= 0.70).sum())
print('0.50–0.70 (knapp raus):', ((df['valid_ratio']>=0.5)&(df['valid_ratio']<0.7)).sum())
print('<0.50 (klar raus):     ', (df['valid_ratio'] < 0.50).sum())

lab = pd.read_csv(os.path.join(PROJEKT,
                  '04_Daten_Felderhebung/03_Annotation/03_Patch_Labels/'
                  'patch_labels_all.csv'))
m = df.merge(lab, on=['standort_id','patch_idx'])
train = m[m['gueltig'] & m['patch_label_strikt'].isin(['intakt','ausgemagert'])]
print(pd.crosstab(train['standort_id'], train['patch_label_strikt'], margins=True))

m = df.merge(lab, on=['standort_id','patch_idx'])
sub = m[m['patch_label_strikt'].isin(['intakt','ausgemagert'])]
print(sub.groupby('patch_label_strikt')['valid_ratio'].agg(['mean','median','min','max','count']))