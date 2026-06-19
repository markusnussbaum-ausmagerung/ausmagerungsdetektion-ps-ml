# -*- coding: utf-8 -*-
"""
N02e — Diskriminanzanalyse Gesamtdatensatz

Primaeres quantitatives Ergebnis: Effektstaerken (Cohen's d), AUC,
Feature-Korrelationen und Bezug zur Laser-MPD (deskriptiv, n=7).
Eingang: dataset_for_training.csv.
"""


# -*- coding: utf-8 -*-
"""
20260603_nb02e_Diskriminanz_Gesamtdatensatz.ipynb

============================================================
NOTEBOOK 02e — DISKRIMINANZANALYSE ÜBER DEN GESAMTDATENSATZ
============================================================
Liest dataset_for_training.csv (aus N2d) und untersucht die
Trennschärfe der 10 ML-Features zwischen intakt und ausgemagert.

Methodische Leitlinien (siehe Statusprotokolle 03.06.2026):
  - HAUPTMASSE sind EFFEKTSTÄRKEN (Cohen's d) und AUC — beide
    weitgehend n-unabhängig und damit robust bei n=7 Standorten.
  - p-WERTE auf Patch-Ebene sind DESKRIPTIV (Pseudoreplikation:
    Patches eines Standorts sind nicht unabhängig).
  - Zusätzlich EHRLICHE STANDORT-EBENE: Features pro Standort
    gemittelt, dann 4 ausgemagerte vs. 3 intakte Standorte.
    ACHTUNG: Bei 4 vs. 3 ist der kleinste zweiseitige MWU-p-Wert
    ≈ 0,057 — Signifikanz ist dort konstruktionsbedingt unmöglich.
    Deshalb auch hier die Effektstärke, nicht der p-Wert.
  - STRIKT ist primär; LOCKER als Sensitivität.

Ausgabe (05_ML_Pipeline/03_Diskriminanz/):
  - diskriminanz_summary.csv        (Haupttabelle)
  - 01_effektstaerken_patch.png     (Ranking |Cohen's d|)
  - 02_patch_vs_standort_d.png      (überlebt der Effekt die Aggregation?)
  - 03_boxplots_features.png        (Verteilung je Feature/Klasse)
  - 04_korrelationsmatrix.png       (Feature-Redundanz)
  - 05_mtd_korrelation.png          (Bezug zur Laser-MPD, Standort-Ebene)
============================================================
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score



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
DATASET_CSV = os.path.join(
    BASIS, '05_ML_Pipeline/02_Joined_Dataset/dataset_for_training.csv')
OUT_DIR = os.path.join(BASIS, '05_ML_Pipeline/03_Diskriminanz')
os.makedirs(OUT_DIR, exist_ok=True)

# Die 10 ML-Features (NICHT die Diagnose-Features)
ML_FEATURES = [
    'roughness_index', 'var_Nz', 'surface_entropy_Nz', 'albedo_variance',
    'skewness_Nz', 'kurtosis_Nz', 'edge_mean', 'edge_std', 'edge_p95',
    'edge_density_global',
]

# Lesbare Namen für Plots
FEATURE_LABELS = {
    'roughness_index': 'roughness_index',
    'var_Nz': 'var_Nz',
    'surface_entropy_Nz': 'surface_entropy_Nz',
    'albedo_variance': 'albedo_variance',
    'skewness_Nz': 'skewness_Nz',
    'kurtosis_Nz': 'kurtosis_Nz',
    'edge_mean': 'edge_mean',
    'edge_std': 'edge_std',
    'edge_p95': 'edge_p95',
    'edge_density_global': 'edge_density_global',
}

LABEL_COL_PRIMARY = 'patch_label_strikt'   # primär
LABEL_COL_SENS    = 'patch_label_locker'   # Sensitivität


# ============================================================
# HILFSFUNKTIONEN
# ============================================================
def cohens_d(a, i):
    """Cohen's d, gepoolte SD. Positiv = höher bei 'ausgemagert'."""
    a = np.asarray(a, float)
    i = np.asarray(i, float)
    na, ni = len(a), len(i)
    if na < 2 or ni < 2:
        return np.nan
    va, vi = a.var(ddof=1), i.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (ni - 1) * vi) / (na + ni - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - i.mean()) / sp


def d_interpret(d):
    ad = abs(d)
    if np.isnan(d):
        return '—'
    if ad < 0.2:
        return 'vernachlässigbar'
    if ad < 0.5:
        return 'klein'
    if ad < 0.8:
        return 'mittel'
    return 'groß'


def feature_auc(values, y_aus):
    """AUC eines Einzelfeatures als Klassifikator. 0.5 = keine Trennung."""
    try:
        return roc_auc_score(y_aus, values)
    except Exception:
        return np.nan


# ============================================================
# 1. DATEN LADEN UND FILTERN
# ============================================================
df = pd.read_csv(DATASET_CSV)
print(f"Geladen: {len(df)} Patches, {df['standort_id'].nunique()} Standorte")

fehlende = [f for f in ML_FEATURES if f not in df.columns]
if fehlende:
    raise RuntimeError(f"Fehlende Feature-Spalten: {fehlende}")

# valid = alle 10 ML-Features vorhanden; binäres Label
valid_mask = df[ML_FEATURES].notna().all(axis=1)


def make_binary(df_in, label_col):
    sub = df_in[valid_mask & df_in[label_col].isin(['intakt', 'ausgemagert'])].copy()
    sub['y_aus'] = (sub[label_col] == 'ausgemagert').astype(int)
    return sub


df_s = make_binary(df, LABEL_COL_PRIMARY)
print(f"\nTrainierbar (STRIKT, valid + binär): {len(df_s)} Patches")
print(df_s[LABEL_COL_PRIMARY].value_counts().to_string())
print(f"\nVerteilung über Standorte (STRIKT):")
print(pd.crosstab(df_s['standort_id'], df_s[LABEL_COL_PRIMARY]).to_string())


# ============================================================
# 2. STANDORT-KLASSEN (für die ehrliche Standort-Ebene)
# ============================================================
# Klasse je Standort = Mehrheitslabel seiner trainierbaren Patches
standort_klasse = (df_s.groupby('standort_id')[LABEL_COL_PRIMARY]
                   .agg(lambda s: s.value_counts().idxmax()))
print(f"\nStandort-Klassen (Mehrheit):")
print(standort_klasse.to_string())
n_aus_sites = (standort_klasse == 'ausgemagert').sum()
n_int_sites = (standort_klasse == 'intakt').sum()
print(f"  → {n_aus_sites} ausgemagerte vs. {n_int_sites} intakte Standorte")

# Standort-Mittelwerte je Feature
standort_means = df_s.groupby('standort_id')[ML_FEATURES].mean()
standort_means['klasse'] = standort_klasse


# ============================================================
# 3. HAUPTTABELLE: PATCH- UND STANDORT-EBENE
# ============================================================
rows = []
y_aus = df_s['y_aus'].values

for f in ML_FEATURES:
    a = df_s.loc[df_s['y_aus'] == 1, f].values   # ausgemagert
    i = df_s.loc[df_s['y_aus'] == 0, f].values   # intakt

    # Patch-Ebene
    d_patch = cohens_d(a, i)
    auc = feature_auc(df_s[f].values, y_aus)
    try:
        _, p_patch = mannwhitneyu(a, i, alternative='two-sided')
    except ValueError:
        p_patch = np.nan

    # Standort-Ebene
    a_s = standort_means.loc[standort_means['klasse'] == 'ausgemagert', f].values
    i_s = standort_means.loc[standort_means['klasse'] == 'intakt', f].values
    d_standort = cohens_d(a_s, i_s)
    try:
        _, p_standort = mannwhitneyu(a_s, i_s, alternative='two-sided')
    except ValueError:
        p_standort = np.nan

    rows.append({
        'feature': f,
        'mean_intakt': i.mean(),
        'mean_ausgemagert': a.mean(),
        'cohens_d_patch': d_patch,
        'abs_d_patch': abs(d_patch),
        'effekt_patch': d_interpret(d_patch),
        'auc_patch': auc,
        'p_mwu_patch_deskriptiv': p_patch,
        'cohens_d_standort': d_standort,
        'effekt_standort': d_interpret(d_standort),
        'p_mwu_standort_illustrativ': p_standort,
    })

summary = pd.DataFrame(rows).sort_values('abs_d_patch', ascending=False)
summary_out = summary.drop(columns='abs_d_patch')

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print(f"\n{'='*70}")
print("DISKRIMINANZ — sortiert nach |Cohen's d| (Patch-Ebene)")
print('='*70)
print(summary_out.round(3).to_string(index=False))

summary_out.round(5).to_csv(os.path.join(OUT_DIR, 'diskriminanz_summary.csv'),
                            index=False)

# Hinweis zur Standort-Ebene
from math import comb
min_p = 2.0 / comb(n_aus_sites + n_int_sites, n_int_sites)
print(f"\n  Hinweis: Bei {n_aus_sites} vs. {n_int_sites} Standorten ist der kleinste")
print(f"  mögliche zweiseitige MWU-p-Wert ≈ {min_p:.3f}. Standort-p-Werte daher")
print(f"  nur illustrativ; maßgeblich ist Cohen's d auf Standort-Ebene.")


# ============================================================
# 4. PLOT 1 — EFFEKTSTÄRKEN-RANKING (PATCH-EBENE)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
s = summary.sort_values('cohens_d_patch')
farben = ['firebrick' if d > 0 else 'steelblue' for d in s['cohens_d_patch']]
ax.barh(range(len(s)), s['cohens_d_patch'], color=farben, alpha=0.8)
ax.set_yticks(range(len(s)))
ax.set_yticklabels([FEATURE_LABELS[f] for f in s['feature']])
for x in [0.2, 0.5, 0.8]:
    ax.axvline(x, color='gray', ls=':', lw=0.8)
    ax.axvline(-x, color='gray', ls=':', lw=0.8)
ax.axvline(0, color='black', lw=1)
ax.set_xlabel("Cohen's d  (rot: höher bei ausgemagert, blau: höher bei intakt)")
ax.set_title("Effektstärken der ML-Features — Patch-Ebene\n"
             "Referenzlinien: 0,2 klein · 0,5 mittel · 0,8 groß")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_effektstaerken_patch.png'),
            dpi=130, bbox_inches='tight')
plt.show()


# ============================================================
# 5. PLOT 2 — PATCH- vs. STANDORT-EBENE (überlebt der Effekt?)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
order = summary.sort_values('abs_d_patch')['feature'].tolist()
yp = [summary.set_index('feature').loc[f, 'cohens_d_patch'] for f in order]
ys = [summary.set_index('feature').loc[f, 'cohens_d_standort'] for f in order]
y = np.arange(len(order))
ax.plot(yp, y, 'o', color='firebrick', label='Patch-Ebene (n groß, pseudorepliziert)')
ax.plot(ys, y, 's', color='darkgreen', label='Standort-Ebene (n=7, ehrlich)')
for k in range(len(order)):
    ax.plot([yp[k], ys[k]], [y[k], y[k]], color='gray', lw=0.8, alpha=0.6)
ax.set_yticks(y)
ax.set_yticklabels([FEATURE_LABELS[f] for f in order])
ax.axvline(0, color='black', lw=1)
for xv in [0.5, 0.8, -0.5, -0.8]:
    ax.axvline(xv, color='gray', ls=':', lw=0.7)
ax.set_xlabel("Cohen's d")
ax.set_title("Effektstärke patch- vs. standortweise\n"
             "Bleibt der Effekt bei standortweiser Aggregation erhalten?")
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_patch_vs_standort_d.png'),
            dpi=130, bbox_inches='tight')
plt.show()


# ============================================================
# 6. PLOT 3 — BOXPLOTS JE FEATURE NACH KLASSE
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, f in zip(axes.flatten(), ML_FEATURES):
    data = [df_s.loc[df_s['y_aus'] == 0, f].values,
            df_s.loc[df_s['y_aus'] == 1, f].values]
    bp = ax.boxplot(data, labels=['intakt', 'ausgemagert'],
                    patch_artist=True, widths=0.6)
    for patch, col in zip(bp['boxes'], ['steelblue', 'firebrick']):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    d = summary.set_index('feature').loc[f, 'cohens_d_patch']
    auc = summary.set_index('feature').loc[f, 'auc_patch']
    ax.set_title(f"{FEATURE_LABELS[f]}\nd={d:.2f} · AUC={auc:.2f}", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle("Feature-Verteilung nach Klasse (STRIKT, valid)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_boxplots_features.png'),
            dpi=120, bbox_inches='tight')
plt.show()


# ============================================================
# 7. PLOT 4 — KORRELATIONSMATRIX (FEATURE-REDUNDANZ)
# ============================================================
corr = df_s[ML_FEATURES].corr(method='spearman')
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(ML_FEATURES)))
ax.set_yticks(range(len(ML_FEATURES)))
ax.set_xticklabels([FEATURE_LABELS[f] for f in ML_FEATURES],
                   rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([FEATURE_LABELS[f] for f in ML_FEATURES], fontsize=9)
for r in range(len(ML_FEATURES)):
    for c in range(len(ML_FEATURES)):
        ax.text(c, r, f"{corr.values[r, c]:.2f}", ha='center', va='center',
                fontsize=7,
                color='white' if abs(corr.values[r, c]) > 0.6 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046, label='Spearman-ρ')
ax.set_title("Feature-Korrelation (Spearman) — Redundanz-Check")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_korrelationsmatrix.png'),
            dpi=130, bbox_inches='tight')
plt.show()

# stark korrelierte Paare melden
print(f"\nStark korrelierte Feature-Paare (|ρ| ≥ 0.8):")
gemeldet = False
for r in range(len(ML_FEATURES)):
    for c in range(r + 1, len(ML_FEATURES)):
        rho = corr.values[r, c]
        if abs(rho) >= 0.8:
            print(f"  {ML_FEATURES[r]} ↔ {ML_FEATURES[c]}: ρ={rho:.2f}")
            gemeldet = True
if not gemeldet:
    print("  keine")


# ============================================================
# 8. PLOT 5 — BEZUG ZUR LASER-MPD (STANDORT-EBENE, deskriptiv)
# ============================================================
# Verknüpft die Feature-Stärke mit der unabhängigen Lasermessung.
if 'mtd_mittel_mm' in df.columns:
    mtd_pro_standort = (df_s.groupby('standort_id')['mtd_mittel_mm'].first()
                        if 'mtd_mittel_mm' in df_s.columns else None)
    if mtd_pro_standort is not None and mtd_pro_standort.notna().sum() >= 3:
        # Top-Feature nach |d| gegen MTD plotten
        top_feat = summary.iloc[0]['feature']
        x = mtd_pro_standort.reindex(standort_means.index).values
        yv = standort_means[top_feat].values
        rho, p = spearmanr(x, yv)
        fig, ax = plt.subplots(figsize=(8, 6))
        cols = ['firebrick' if k == 'ausgemagert' else 'steelblue'
                for k in standort_means['klasse']]
        ax.scatter(x, yv, c=cols, s=90, edgecolor='k', zorder=3)
        for sid, xi, yi in zip(standort_means.index, x, yv):
            ax.annotate(sid, (xi, yi), fontsize=8,
                        xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel('MTD (Laser, mm) — Standort-Mittel')
        ax.set_ylabel(f'{top_feat} — Standort-Mittel')
        ax.set_title(f'Stärkstes Feature vs. Laser-MTD (Standort-Ebene, n=7)\n'
                     f'Spearman ρ={rho:.2f} (p={p:.3f}, deskriptiv)')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, '05_mtd_korrelation.png'),
                    dpi=130, bbox_inches='tight')
        plt.show()

        # Alle Features gegen MTD (Standort-Ebene), Spearman
        print(f"\nKorrelation Feature-Standortmittel ↔ MTD (Spearman, n=7, deskriptiv):")
        for f in ML_FEATURES:
            rho, p = spearmanr(x, standort_means[f].values)
            print(f"  {f:22s}: ρ={rho:+.2f}  (p={p:.3f})")
    else:
        print("\n  MTD-Spalte vorhanden, aber zu wenig Werte für Korrelation.")
else:
    print("\n  Keine MTD-Spalte im Datensatz — MTD-Bezug übersprungen.")


# ============================================================
# 9. SENSITIVITÄT: STRIKT vs. LOCKER (nur Cohen's d patchweise)
# ============================================================
df_l = make_binary(df, LABEL_COL_SENS)
print(f"\n{'='*70}")
print(f"SENSITIVITÄT — Cohen's d (Patch) STRIKT vs. LOCKER")
print('='*70)
print(f"  (LOCKER: {len(df_l)} Patches, "
      f"{(df_l['y_aus']==1).sum()} aus / {(df_l['y_aus']==0).sum()} int)")
sens_rows = []
for f in ML_FEATURES:
    a_s = df_s.loc[df_s['y_aus'] == 1, f].values
    i_s = df_s.loc[df_s['y_aus'] == 0, f].values
    a_l = df_l.loc[df_l['y_aus'] == 1, f].values
    i_l = df_l.loc[df_l['y_aus'] == 0, f].values
    sens_rows.append({
        'feature': f,
        'd_strikt': cohens_d(a_s, i_s),
        'd_locker': cohens_d(a_l, i_l),
    })
sens = pd.DataFrame(sens_rows)
sens['delta'] = (sens['d_strikt'] - sens['d_locker']).abs()
print(sens.round(3).to_string(index=False))
print(f"\n  Maximale |Δd| zwischen strikt und locker: {sens['delta'].max():.3f}")
print("  → kleine Werte bedeuten: Befund hängt nicht an der Purity-Schwelle.")

print(f"\n{'='*70}")
print(f"FERTIG — Ausgaben in: {OUT_DIR}")
print('='*70)