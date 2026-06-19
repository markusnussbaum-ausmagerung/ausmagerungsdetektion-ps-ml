# -*- coding: utf-8 -*-
"""
N03 — Binaere Klassifikation (LOGO, RF/SVM)

Trainiert Random Forest und SVM auf den 10 ML-Features, evaluiert ueber
Leave-One-Group-Out (Gruppe=Standort) mit gepoolter Out-of-Fold-Matrix.
Hauptmasse Balanced Accuracy und MCC. Demonstriert die Lernbarkeit
(Hauptergebnis bleibt die Diskriminanz, N02e).
"""


# -*- coding: utf-8 -*-
"""
20260603_nb03_Klassifikation_LOGO.ipynb

============================================================
NOTEBOOK 03 — BINÄRE KLASSIFIKATION (intakt / ausgemagert)
============================================================
Trainiert Random Forest und SVM auf den 10 ML-Features und
evaluiert sie ehrlich über Leave-One-Group-Out (Gruppe =
Standort) mit GEPOOLTER Out-of-Fold-Auswertung.

Methodische Festlegungen (siehe Statusprotokolle 03.06.2026):
  - Split: LeaveOneGroupOut, Gruppe = standort_id.
    Grund: Patches eines Standorts sind nicht unabhängig;
    sie dürfen nie gleichzeitig in Training und Test liegen
    (sonst lernt das Modell die Oberfläche auswendig → Leakage).
  - Auswertung: ALLE Out-of-Fold-Vorhersagen in EINEN Topf,
    daraus EINE Konfusionsmatrix. KEIN Mitteln von Fold-Werten,
    da 5 von 7 Test-Folds einklassig sind.
  - Hauptkennzahl: Balanced Accuracy (nicht rohe Accuracy;
    Trivialmodell 'immer ausgemagert' erreicht sonst ~69 %).
  - class_weight='balanced' gegen die 2,25:1-Schieflage.
  - KEIN Tuning auf den Testdaten; feste, dokumentierte Defaults.
  - STRIKT primär, LOCKER als Sensitivitätsanalyse.
  - Das Hauptergebnis der Arbeit bleibt die Diskriminanz (N2e);
    der Klassifikator demonstriert die Lernbarkeit.

Ausgabe (05_ML_Pipeline/):
  04_Modelle/   final_rf.joblib, final_svm.joblib, modell_meta.joblib
  05_Klassifikation/  ergebnisse_summary.csv, Abbildungen
============================================================
"""

import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                             recall_score, precision_score, matthews_corrcoef)



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
MODELL_DIR = os.path.join(BASIS, '05_ML_Pipeline/04_Modelle')
OUT_DIR    = os.path.join(BASIS, '05_ML_Pipeline/05_Klassifikation')
os.makedirs(MODELL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ML_FEATURES = [
    'roughness_index', 'var_Nz', 'surface_entropy_Nz', 'albedo_variance',
    'skewness_Nz', 'kurtosis_Nz', 'edge_mean', 'edge_std', 'edge_p95',
    'edge_density_global',
]

RANDOM_STATE = 42  # Reproduzierbarkeit

# Feste, dokumentierte Modell-Defaults (KEIN Tuning auf Test)
def make_rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)

def make_svm():
    # StandardScaler IM Pipeline → wird je Fold neu gefittet (kein Leakage)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale',
                    class_weight='balanced', random_state=RANDOM_STATE)),
    ])


# ============================================================
# 1. DATEN LADEN UND FILTERN
# ============================================================
df = pd.read_csv(DATASET_CSV)
print(f"Geladen: {len(df)} Patches, {df['standort_id'].nunique()} Standorte")

valid_mask = df[ML_FEATURES].notna().all(axis=1)


def baue_datensatz(label_col):
    sub = df[valid_mask & df[label_col].isin(['intakt', 'ausgemagert'])].copy()
    sub = sub.reset_index(drop=True)
    X = sub[ML_FEATURES]
    y = (sub[label_col] == 'ausgemagert').astype(int)   # 1 = ausgemagert
    groups = sub['standort_id']
    return X, y, groups, sub


# ============================================================
# 2. LOGO-EVALUATION MIT GEPOOLTER OOF-AUSWERTUNG
# ============================================================
def logo_evaluieren(model_factory, X, y, groups, modellname):
    """
    Trainiert je Fold neu (6 Standorte), sagt den ausgelassenen
    Standort vorher, sammelt ALLE Vorhersagen in einen Topf.
    Rückgabe: DataFrame mit standort_id, y_true, y_pred je Patch.
    """
    logo = LeaveOneGroupOut()
    oof_true, oof_pred, oof_sid = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        test_sid = groups.iloc[test_idx].unique()
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])

        oof_true.extend(y.iloc[test_idx].tolist())
        oof_pred.extend(pred.tolist())
        oof_sid.extend(groups.iloc[test_idx].tolist())

    return pd.DataFrame({'standort_id': oof_sid,
                         'y_true': oof_true,
                         'y_pred': oof_pred})


def kennzahlen(oof, modellname, label_variant):
    yt, yp = oof['y_true'].values, oof['y_pred'].values
    bacc = balanced_accuracy_score(yt, yp)
    rec_aus = recall_score(yt, yp, pos_label=1)
    rec_int = recall_score(yt, yp, pos_label=0)
    prec_aus = precision_score(yt, yp, pos_label=1, zero_division=0)
    prec_int = precision_score(yt, yp, pos_label=0, zero_division=0)
    mcc = matthews_corrcoef(yt, yp)
    acc = (yt == yp).mean()
    return {
        'modell': modellname, 'label': label_variant, 'n': len(yt),
        'balanced_accuracy': bacc,
        'recall_ausgemagert': rec_aus, 'recall_intakt': rec_int,
        'precision_ausgemagert': prec_aus, 'precision_intakt': prec_int,
        'mcc': mcc, 'accuracy_roh': acc,
    }


# ============================================================
# 3. DURCHLAUF — STRIKT (primär) UND LOCKER (Sensitivität)
# ============================================================
ergebnisse = []
oof_speicher = {}

for label_variant, label_col in [('strikt', 'patch_label_strikt'),
                                 ('locker', 'patch_label_locker')]:
    X, y, groups, sub = baue_datensatz(label_col)
    print(f"\n{'='*60}")
    print(f"LABEL-VARIANTE: {label_variant}")
    print(f"{'='*60}")
    print(f"  Patches: {len(X)}  |  ausgemagert: {(y==1).sum()}  intakt: {(y==0).sum()}")
    print(f"  Standorte (Gruppen): {groups.nunique()}")

    for name, factory in [('RandomForest', make_rf), ('SVM', make_svm)]:
        oof = logo_evaluieren(factory, X, y, groups, name)
        kz = kennzahlen(oof, name, label_variant)
        ergebnisse.append(kz)
        oof_speicher[(label_variant, name)] = oof

        print(f"\n  {name}:")
        print(f"    Balanced Accuracy: {kz['balanced_accuracy']:.3f}")
        print(f"    Recall ausgemagert: {kz['recall_ausgemagert']:.3f} | "
              f"Recall intakt: {kz['recall_intakt']:.3f}")
        print(f"    MCC: {kz['mcc']:.3f}  (rohe Accuracy: {kz['accuracy_roh']:.3f})")

erg_df = pd.DataFrame(ergebnisse)
erg_df.round(4).to_csv(os.path.join(OUT_DIR, 'ergebnisse_summary.csv'),
                       index=False)
print(f"\n{'='*60}")
print("ERGEBNIS-ÜBERSICHT")
print('='*60)
print(erg_df.round(3).to_string(index=False))


# ============================================================
# 4. KONFUSIONSMATRIZEN (STRIKT, beide Modelle)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name in zip(axes, ['RandomForest', 'SVM']):
    oof = oof_speicher[('strikt', name)]
    cm = confusion_matrix(oof['y_true'], oof['y_pred'], labels=[0, 1])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['intakt', 'ausgemagert'])
    ax.set_yticklabels(['intakt', 'ausgemagert'])
    ax.set_xlabel('Vorhergesagt'); ax.set_ylabel('Wahr (Label)')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                    fontsize=18,
                    color='white' if cm[r, c] > cm.max()/2 else 'black')
    bacc = balanced_accuracy_score(oof['y_true'], oof['y_pred'])
    ax.set_title(f'{name} (STRIKT, gepoolte OOF)\nBalanced Acc = {bacc:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_konfusionsmatrizen_strikt.png'),
            dpi=130, bbox_inches='tight')
plt.show()


# ============================================================
# 5. KORREKTHEIT PRO STANDORT (RF, STRIKT)
# ============================================================
# Zeigt, ob ein ausgelassener Standort als Ganzes erkannt wird.
oof_rf = oof_speicher[('strikt', 'RandomForest')]
pro_standort = (oof_rf.assign(korrekt=oof_rf['y_true'] == oof_rf['y_pred'])
                .groupby('standort_id')
                .agg(n=('korrekt', 'size'),
                     anteil_korrekt=('korrekt', 'mean'),
                     wahre_klasse=('y_true', 'mean')))  # 0=intakt,1=aus
pro_standort['klasse'] = pro_standort['wahre_klasse'].apply(
    lambda m: 'ausgemagert' if m > 0.5 else 'intakt')
print(f"\nKorrektheit je ausgelassenem Standort (RF, STRIKT):")
print(pro_standort.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
cols = ['firebrick' if k == 'ausgemagert' else 'steelblue'
        for k in pro_standort['klasse']]
ax.bar(pro_standort.index, pro_standort['anteil_korrekt'], color=cols, alpha=0.8)
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Anteil korrekt klassifizierter Patches')
ax.set_title('Korrektheit je ausgelassenem Standort (RF, LOGO, STRIKT)\n'
             'rot = ausgemagerter Standort, blau = intakter Standort')
for sid, v in zip(pro_standort.index, pro_standort['anteil_korrekt']):
    ax.text(sid, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_korrektheit_pro_standort.png'),
            dpi=130, bbox_inches='tight')
plt.show()


# ============================================================
# 6. FEATURE-IMPORTANCES (RF, auf allen STRIKT-Daten refit)
# ============================================================
X_s, y_s, groups_s, _ = baue_datensatz('patch_label_strikt')
rf_final = make_rf()
rf_final.fit(X_s, y_s)
imp = pd.Series(rf_final.feature_importances_, index=ML_FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(imp.index, imp.values, color='seagreen', alpha=0.8)
ax.set_xlabel('Gini-Importance (Random Forest)')
ax.set_title('Feature-Wichtigkeit (RF, auf allen STRIKT-Daten)\n'
             'Hinweis: stark korrelierte Relief-Features teilen sich die Wichtigkeit')
for f, v in imp.items():
    ax.text(v + 0.002, f, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_feature_importance_rf.png'),
            dpi=130, bbox_inches='tight')
plt.show()
print(f"\nFeature-Importances (RF, STRIKT):")
print(imp.sort_values(ascending=False).round(4).to_string())


# ============================================================
# 7. FINALE MODELLE FÜR N4 SPEICHERN (auf allen STRIKT-Daten)
# ============================================================
# N4 wendet das finale Modell auf S000_Test an (Inferenz-Demo).
svm_final = make_svm()
svm_final.fit(X_s, y_s)

joblib.dump(rf_final,  os.path.join(MODELL_DIR, 'final_rf.joblib'))
joblib.dump(svm_final, os.path.join(MODELL_DIR, 'final_svm.joblib'))
joblib.dump({
    'feature_names': ML_FEATURES,
    'label_map': {0: 'intakt', 1: 'ausgemagert'},
    'trainiert_auf': 'STRIKT, valid, S001–S007 (172 Patches)',
    'random_state': RANDOM_STATE,
    'hinweis': 'Inferenz-Demo (N4) ist qualitativ; S000_Test wirkte an '
               'der Feature-Auswahl mit → keine unabhängige Leistungsbewertung.',
}, os.path.join(MODELL_DIR, 'modell_meta.joblib'))

print(f"\n{'='*60}")
print(f"FERTIG")
print('='*60)
print(f"  Ergebnisse: {OUT_DIR}")
print(f"  Modelle für N4: {MODELL_DIR}")
print(f"\n  Einordnung: Hohe Werte sind bei den großen Effektstärken (N2e)")
print(f"  erwartbar. Sie betreffen eindeutig gelabelte Flächen; wegen der")
print(f"  Klasse-Standort-Konfundierung ist ein Teil der Leistung ggf.")
print(f"  Standort-Wiedererkennung. Hauptergebnis bleibt die Diskriminanz.")

