# -*- coding: utf-8 -*-
"""
N02c — Diskriminanz-Test (Pilot, NICHT produktiv)

ACHTUNG: Diagnose-/Plausibilitaetstest auf dem Pilotstandort, KEIN
Bestandteil der produktiven Pipeline. Die belastbare Diskriminanz der
Vollerhebung steht in N02e. Hier nur zur methodischen Absicherung.
"""


# ============================================================
# NOTEBOOK 02c — DISKRIMINANZ-TEST MIT CVAT-LABELS
# ============================================================
# Joint features_all_patches.csv (N2b) mit patch_labels_all.csv (N2a)
# und prüft, ob die ML-Features zwischen intakt und ausgemagert
# diskriminieren — basierend auf den echten CVAT-Annotationen.
#
# Output:
#   - Statistik-Tabellen (Cohen's d, t-Test) für STRIKT und LOCKER
#   - Boxplots pro Feature
#   - Patch-Übersichts-Visualisierung pro Standort
# ============================================================

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

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

BASIS_DRIVE = PROJEKT

FEATURES_CSV = os.path.join(BASIS_DRIVE,
    '05_ML_Pipeline/01_Features/features_all_patches.csv')

# Falls die Features-CSV noch im alten Pfad liegt (ohne 01_Features/),
# automatischer Fallback:
if not os.path.exists(FEATURES_CSV):
    FEATURES_CSV_ALT = os.path.join(BASIS_DRIVE,
        '05_ML_Pipeline/features_all_patches.csv')
    if os.path.exists(FEATURES_CSV_ALT):
        FEATURES_CSV = FEATURES_CSV_ALT
        print(f"Features-CSV gefunden im alten Pfad: {FEATURES_CSV_ALT}")

LABELS_CSV = os.path.join(BASIS_DRIVE,
    '04_Daten_Felderhebung/03_Annotation/03_Patch_Labels/patch_labels_all.csv')

VERARBEITET_DIR = os.path.join(BASIS_DRIVE,
    '04_Daten_Felderhebung/02_Verarbeitet')

print(f"Features:  {FEATURES_CSV}")
print(f"  exists: {os.path.exists(FEATURES_CSV)}")
print(f"Labels:    {LABELS_CSV}")
print(f"  exists: {os.path.exists(LABELS_CSV)}")


# ============================================================
# 1. DATEN LADEN UND JOINEN
# ============================================================
features = pd.read_csv(FEATURES_CSV)
labels   = pd.read_csv(LABELS_CSV)

print(f"\nFeatures geladen: {len(features)} Patches, "
      f"{features['standort_id'].nunique()} Standorte")
print(f"Labels geladen:   {len(labels)} Patches, "
      f"{labels['standort_id'].nunique()} Standorte")

# Join über standort_id + patch_idx
df = features.merge(
    labels[['standort_id', 'patch_idx',
            'patch_label_strikt', 'patch_label_locker',
            'anteil_ausgemagert', 'anteil_exclusion']],
    on=['standort_id', 'patch_idx'],
    how='inner',
)

print(f"\nNach Join: {len(df)} Patches")
print(f"\nLabel-Verteilung STRIKT:")
print(df['patch_label_strikt'].value_counts().to_string())
print(f"\nLabel-Verteilung LOCKER:")
print(df['patch_label_locker'].value_counts().to_string())

# Exclusion-Statistik (aus nb02a übernommen)
n_excl_strikt = (df['patch_label_strikt'] == 'exkludiert').sum()
n_excl_locker = (df['patch_label_locker'] == 'exkludiert').sum()
print(f"\nExclusion-Statistik:")
print(f"  Patches mit Label 'exkludiert' (STRIKT): {n_excl_strikt}")
print(f"  Patches mit Label 'exkludiert' (LOCKER): {n_excl_locker}")
print(f"  → Diese Patches werden in der Diskriminanzanalyse "
      f"durch isin()-Filter automatisch ausgeschlossen.")

if n_excl_strikt > 0:
    print(f"\n  Mittlerer Exclusion-Anteil bei exkludierten Patches: "
          f"{df[df['patch_label_strikt']=='exkludiert']['anteil_exclusion'].mean()*100:.1f} %")


# ============================================================
# 2. FEATURES DEFINIEREN
# ============================================================
ml_features = [
    'roughness_index', 'var_Nz', 'surface_entropy_Nz',
    'albedo_variance', 'skewness_Nz', 'kurtosis_Nz',
    'edge_mean', 'edge_std', 'edge_p95', 'edge_density_global',
]

# Nur Patches mit gültigen Features (NaN raus)
df_valid = df.dropna(subset=ml_features).copy()
print(f"\nNach NaN-Filter: {len(df_valid)} Patches gültig (von {len(df)})")


# ============================================================
# 3. STATISTISCHER VERGLEICH — FUNKTION
# ============================================================
def vergleiche_features(df_test, label_col, label_intakt='intakt',
                        label_ausgemagert='ausgemagert'):
    """
    Macht t-Test + Cohen's d für jedes Feature.
    """
    intakt   = df_test[df_test[label_col] == label_intakt]
    ausg     = df_test[df_test[label_col] == label_ausgemagert]

    n_i = len(intakt)
    n_a = len(ausg)

    print(f"\n  n_intakt = {n_i}, n_ausgemagert = {n_a}")

    if n_i < 3 or n_a < 3:
        print(f"  ⚠ Zu wenig Patches in einer Klasse — keine Auswertung")
        return None

    ergebnisse = []
    print(f"\n  {'Feature':<24} {'Intakt (μ±σ)':<22} {'Ausgem. (μ±σ)':<22} "
          f"{'t':>7} {'p':>9} {'Cohen d':>9}")
    print("  " + "-" * 96)

    for feat in ml_features:
        iv = intakt[feat].values
        av = ausg[feat].values

        # Welch's t-Test (nicht annehmend gleiche Varianzen)
        t_stat, p_val = stats.ttest_ind(iv, av, equal_var=False)

        # Cohen's d (gepoolt)
        pooled = np.sqrt((iv.var(ddof=1) + av.var(ddof=1)) / 2)
        d = (av.mean() - iv.mean()) / pooled if pooled > 0 else np.nan

        print(f"  {feat:<24} "
              f"{iv.mean():>9.4f}±{iv.std():.4f}   "
              f"{av.mean():>9.4f}±{av.std():.4f}   "
              f"{t_stat:>7.2f} {p_val:>9.4f} {d:>9.2f}")

        ergebnisse.append({
            'feature': feat, 't': t_stat, 'p': p_val, 'd': d,
            'mean_intakt': iv.mean(), 'mean_ausgem': av.mean(),
        })

    return pd.DataFrame(ergebnisse)


# ============================================================
# 4. AUSWERTUNG — STRIKT
# ============================================================
print(f"\n{'='*60}")
print(f"STRIKT (≥0.80 / ≤0.05)")
print(f"{'='*60}")
df_strikt = df_valid[df_valid['patch_label_strikt'].isin(['intakt', 'ausgemagert'])]
res_strikt = vergleiche_features(df_strikt, 'patch_label_strikt')


# ============================================================
# 5. AUSWERTUNG — LOCKER
# ============================================================
print(f"\n{'='*60}")
print(f"LOCKER (≥0.50 / ≤0.20)")
print(f"{'='*60}")
df_locker = df_valid[df_valid['patch_label_locker'].isin(['intakt', 'ausgemagert'])]
res_locker = vergleiche_features(df_locker, 'patch_label_locker')


# ============================================================
# 6. BOXPLOTS — beide Schwellen nebeneinander
# ============================================================
n_features = len(ml_features)
n_cols = 5
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, feat in enumerate(ml_features):
    ax = axes[i]

    # Daten für STRIKT
    iv_s = df_strikt[df_strikt['patch_label_strikt'] == 'intakt'][feat].dropna()
    av_s = df_strikt[df_strikt['patch_label_strikt'] == 'ausgemagert'][feat].dropna()

    # Daten für LOCKER
    iv_l = df_locker[df_locker['patch_label_locker'] == 'intakt'][feat].dropna()
    av_l = df_locker[df_locker['patch_label_locker'] == 'ausgemagert'][feat].dropna()

    box_data = [iv_s, av_s, iv_l, av_l]
    labels_box = ['I\nstrikt', 'A\nstrikt', 'I\nlocker', 'A\nlocker']
    colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']

    bp = ax.boxplot(box_data, labels=labels_box, widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Trennlinie zwischen strikt und locker
    ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)

    # Einzelpunkte
    for j, vals in enumerate(box_data, start=1):
        ax.scatter([j]*len(vals), vals, alpha=0.5, s=15,
                   color='blue' if j%2==1 else 'red', zorder=3)

    ax.set_title(feat, fontsize=10)
    ax.grid(True, alpha=0.3)

# Leere Achsen ausblenden
for j in range(n_features, len(axes)):
    axes[j].axis('off')

plt.suptitle('Features pro Klasse — STRIKT vs LOCKER\n'
             'I = intakt, A = ausgemagert', fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig('diskriminanz_cvat_boxplots.png',
            dpi=110, bbox_inches='tight')
plt.show()


# ============================================================
# 7. VISUALISIERUNG: PATCH-LABELS AUF DEM BILD
# ============================================================
# Pro Standort: Nz-Map mit eingefärbten Patches nach Label.
# NaN-Patches (Bildqualitäts-Filter) werden grau schraffiert
# dargestellt — als visueller Verweis auf Abb. 4-9-3 (vor Filter).
for sid in df['standort_id'].unique():
    npz_pfad = os.path.join(VERARBEITET_DIR, sid, 'ps_output.npz')
    if not os.path.exists(npz_pfad):
        print(f"⚠ Kein ps_output.npz für {sid}")
        continue

    data = np.load(npz_pfad)
    Nz = data['Nz']
    df_s = df[df['standort_id'] == sid]
    patch_px = int(df_s['patch_size_px'].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    label_farben = {'intakt':      'green',
                    'ausgemagert': 'red',
                    'unklar':      'gray',
                    'exkludiert':  'purple'}

    # === STRIKT ===
    axes[0].imshow(Nz, cmap='gray', vmin=0.7, vmax=1.0, alpha=0.6)
    for _, row in df_s.iterrows():
        r, c = int(row['patch_row']), int(row['patch_col'])

        # NaN-Patches (Bildqualitäts-Filter) als schraffiert darstellen
        ist_nan = pd.isna(row[ml_features]).any()
        if ist_nan:
            rect = Rectangle((c * patch_px, r * patch_px),
                 patch_px, patch_px,
                 linewidth=1.8, edgecolor='black',
                 facecolor='black', alpha=0.75,
                 hatch='xxx', zorder=2)
            axes[0].add_patch(rect)
            continue

        # Normale Label-Färbung
        farbe = label_farben.get(row['patch_label_strikt'], 'black')
        rect = Rectangle((c * patch_px, r * patch_px),
                         patch_px, patch_px,
                         linewidth=2, edgecolor=farbe,
                         facecolor=farbe, alpha=0.3)
        axes[0].add_patch(rect)
    axes[0].set_title(f'{sid} — STRIKT (grün=intakt, rot=aus, '
                      f'grau=unklar, lila=exkl., schraffiert=NaN)',
                      fontsize=11)
    axes[0].axis('off')

    # === LOCKER ===
    axes[1].imshow(Nz, cmap='gray', vmin=0.7, vmax=1.0, alpha=0.6)
    for _, row in df_s.iterrows():
        r, c = int(row['patch_row']), int(row['patch_col'])

        ist_nan = pd.isna(row[ml_features]).any()
        if ist_nan:
            rect = Rectangle((c * patch_px, r * patch_px),
                 patch_px, patch_px,
                 linewidth=1.8, edgecolor='black',
                 facecolor='black', alpha=0.75,
                 hatch='xxx', zorder=2)
            axes[1].add_patch(rect)
            continue

        farbe = label_farben.get(row['patch_label_locker'], 'black')
        rect = Rectangle((c * patch_px, r * patch_px),
                         patch_px, patch_px,
                         linewidth=2, edgecolor=farbe,
                         facecolor=farbe, alpha=0.3)
        axes[1].add_patch(rect)
    axes[1].set_title(f'{sid} — LOCKER (grün=intakt, rot=aus, '
                      f'grau=unklar, lila=exkl., schraffiert=NaN)',
                      fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{sid}_patches_nach_qualitaetsfilter.png',
                dpi=140, bbox_inches='tight')
    plt.show()


# ============================================================
# 8. INTERPRETATION
# ============================================================
print(f"\n{'='*60}")
print(f"INTERPRETATION (Cohen's d Schwellen)")
print(f"{'='*60}")
print(f"  |d| > 0.8: großer Effekt   (diskriminativ)")
print(f"  |d| > 0.5: mittlerer Effekt")
print(f"  |d| < 0.2: kein Effekt")

for variante, res in [('STRIKT', res_strikt), ('LOCKER', res_locker)]:
    if res is None:
        continue
    print(f"\n--- {variante} ---")
    res_sorted = res.reindex(res['d'].abs().sort_values(ascending=False).index)
    stark = res_sorted[(res_sorted['d'].abs() > 0.8)]
    print(f"  Starke Diskriminatoren (|d|>0.8): {len(stark)}")
    for _, r in stark.iterrows():
        print(f"    {r['feature']:<24} d={r['d']:+.2f}  p={r['p']:.4f}")

    if len(stark) >= 3:
        print(f"  → Brauchbare Diskrimination, Pipeline ist trainings-tauglich.")
    elif len(stark) >= 1:
        print(f"  → Schwache Diskrimination — vor Vollerhebung kritisch prüfen.")
    else:
        print(f"  → Keine starken Diskriminatoren — Pipeline neu hinterfragen.")

# Korrelationsmatrix der ML-Features im Pilot
korr = df_valid[ml_features].corr()
print(korr.round(2))

# Visualisierung
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(korr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={'label': 'Pearson r'})
plt.tight_layout()
plt.show()

# ============================================================
# 9. EBENE-1-HISTOGRAMME (Pixel-Verteilung pro Patch)
# ============================================================
# Zeigt die Verteilung der Nz-Werte INNERHALB EINES typischen
# intakten und eines typischen ausgemagerten Patches mit den
# vier statistischen Verteilungsmomenten.
#
# Dient als Pendant zu den Boxplots (Ebene 2):
# - Boxplots zeigen die Streuung EINES Features ÜBER die Patches
# - Histogramme zeigen die Verteilung der Pixel INNERHALB EINES Patches
# ============================================================
from scipy import stats as sp_stats

# Variante für die Auswahl: locker (n=14 intakt + 13 ausgemagert)
df_locker_eval = df_locker.copy()

# Pilot-Standort
sid_pilot = df_locker_eval['standort_id'].iloc[0]

# Nz-Karte und Confidence laden
npz = np.load(os.path.join(VERARBEITET_DIR, sid_pilot, 'ps_output.npz'))
Nz_map   = npz['Nz']
n_valid_map = npz['n_valid'] if 'n_valid' in npz.files else None
patch_px = int(df_locker_eval[df_locker_eval['standort_id']==sid_pilot]
               ['patch_size_px'].iloc[0])

# Schwelle für vertrauenswürdige Pixel (konsistent mit nb02b)
MIN_LEDS = 8

def extrahiere_patch_pixel(row):
    """Holt vertrauenswürdige Nz-Pixel aus einem Patch."""
    r, c = int(row['patch_row']), int(row['patch_col'])
    y0, y1 = r * patch_px, (r + 1) * patch_px
    x0, x1 = c * patch_px, (c + 1) * patch_px

    nz_patch = Nz_map[y0:y1, x0:x1]
    if n_valid_map is not None:
        nv_patch = n_valid_map[y0:y1, x0:x1]
        maske = (nv_patch >= MIN_LEDS)
        return nz_patch[maske]
    else:
        return nz_patch.flatten()

# Median-Patch pro Klasse auswählen
def waehle_repraesentativen_patch(df_klasse, kriterium='var_Nz'):
    """Patch mit dem Median des Kriteriums — also kein Ausreißer."""
    df_sortiert = df_klasse.sort_values(kriterium)
    median_idx = len(df_sortiert) // 2
    return df_sortiert.iloc[median_idx]

intakt_patches = df_locker_eval[df_locker_eval['patch_label_locker']=='intakt']
ausg_patches   = df_locker_eval[df_locker_eval['patch_label_locker']=='ausgemagert']

patch_intakt = waehle_repraesentativen_patch(intakt_patches)
patch_ausg   = waehle_repraesentativen_patch(ausg_patches)

px_intakt = extrahiere_patch_pixel(patch_intakt)
px_ausg   = extrahiere_patch_pixel(patch_ausg)

# Vier Verteilungsmomente direkt aus den Pixel-Werten
def vier_momente(arr):
    return {
        'mu':   float(np.mean(arr)),
        'sigma':float(np.std(arr)),
        'skew': float(sp_stats.skew(arr)),
        'kurt': float(sp_stats.kurtosis(arr)),
        'n':    int(len(arr)),
    }

m_i = vier_momente(px_intakt)
m_a = vier_momente(px_ausg)

print(f"\nRepräsentativer intakter Patch (idx={int(patch_intakt['patch_idx'])}, "
      f"row={int(patch_intakt['patch_row'])}, col={int(patch_intakt['patch_col'])})")
print(f"  μ={m_i['mu']:.4f}, σ={m_i['sigma']:.4f}, "
      f"skew={m_i['skew']:+.2f}, kurt={m_i['kurt']:+.2f}, n_pixel={m_i['n']}")
print(f"\nRepräsentativer ausgemagerter Patch (idx={int(patch_ausg['patch_idx'])}, "
      f"row={int(patch_ausg['patch_row'])}, col={int(patch_ausg['patch_col'])})")
print(f"  μ={m_a['mu']:.4f}, σ={m_a['sigma']:.4f}, "
      f"skew={m_a['skew']:+.2f}, kurt={m_a['kurt']:+.2f}, n_pixel={m_a['n']}")


# === Visualisierung ===
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white')

# Gemeinsame Bins für direkten Vergleich
nz_min = min(px_intakt.min(), px_ausg.min())
nz_max = max(px_intakt.max(), px_ausg.max())
bins   = np.linspace(nz_min, nz_max, 50)

konfig = [
    (axes[0], px_intakt, m_i, 'intakt',
     '#A8C8E8', '#234A75', patch_intakt),
    (axes[1], px_ausg,   m_a, 'ausgemagert',
     '#F2B5B5', '#8B2E2E', patch_ausg),
]

for ax, pixel, m, klassenname, hg_farbe, text_farbe, p_info in konfig:
    ax.hist(pixel, bins=bins, color=hg_farbe,
            edgecolor=text_farbe, linewidth=0.4, alpha=0.85)
    ax.axvline(m['mu'], color=text_farbe, linewidth=1.4,
               linestyle='--', alpha=0.85)
    ax.set_xlabel(r'$N_z$-Wert pro Pixel', fontsize=10.5)
    ax.set_ylabel('Anzahl Pixel', fontsize=10.5)
    ax.set_title(
        f"Klasse: {klassenname}  —  Patch (row={int(p_info['patch_row'])}, "
        f"col={int(p_info['patch_col'])})",
        fontsize=11, fontweight='bold', color=text_farbe)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.5)

    # Statistik-Box
    text = (f"$\\mu = {m['mu']:.4f}$\n"
            f"$\\sigma = {m['sigma']:.4f}$\n"
            f"Skewness $= {m['skew']:+.2f}$\n"
            f"Kurtosis $= {m['kurt']:+.2f}$\n"
            f"$n_{{Pixel}} = {m['n']:,}$".replace(',', '.'))
    ax.text(0.04, 0.96, text,
            transform=ax.transAxes, fontsize=10,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.45',
                      facecolor='white', edgecolor=text_farbe,
                      linewidth=0.8, alpha=0.92),
            color=text_farbe)

# Gleiche x-Achsen-Skala
for ax in axes:
    ax.set_xlim(nz_min, nz_max)

plt.tight_layout()
plt.savefig(f'{sid_pilot}_ebene1_histogramme.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\nGespeichert: {sid_pilot}_ebene1_histogramme.png")