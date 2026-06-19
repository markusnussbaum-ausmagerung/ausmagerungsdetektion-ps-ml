# -*- coding: utf-8 -*-
"""
N02d — Drei-Wege-Join zum Trainingsdatensatz

Fuehrt Features (N02b), Patch-Labels (N02a) und Standort-Metadaten
(master.csv) zusammen. Ausgang: dataset_for_training.csv, Basis aller
weiteren ML-Schritte.
"""


# ============================================================
# NOTEBOOK 02d — DREI-WEGE-JOIN ZUM TRAININGSDATENSATZ
# ============================================================
# Führt drei Quellen zusammen:
#   1. features_all_patches.csv   (aus N2b)  → Features pro Patch
#   2. patch_labels_all.csv       (aus N2a)  → Labels pro Patch
#   3. master.csv                 (manuell)  → Metadaten pro Standort
#
# Output: dataset_for_training.csv
# Diese Datei ist die Basis für alle weiteren ML-Schritte.
# ============================================================

import numpy as np
import pandas as pd
import os

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

# Fallback für altes Pfad-Schema
if not os.path.exists(FEATURES_CSV):
    FEATURES_CSV_ALT = os.path.join(BASIS_DRIVE,
        '05_ML_Pipeline/features_all_patches.csv')
    if os.path.exists(FEATURES_CSV_ALT):
        FEATURES_CSV = FEATURES_CSV_ALT

LABELS_CSV = os.path.join(BASIS_DRIVE,
    '04_Daten_Felderhebung/03_Annotation/03_Patch_Labels/patch_labels_all.csv')

MASTER_CSV = os.path.join(BASIS_DRIVE,
    '04_Daten_Felderhebung/00_Master/master.csv')

OUTPUT_CSV = os.path.join(BASIS_DRIVE,
    '05_ML_Pipeline/02_Joined_Dataset/dataset_for_training.csv')

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print(f"Features:  {FEATURES_CSV}")
print(f"  exists:  {os.path.exists(FEATURES_CSV)}")
print(f"Labels:    {LABELS_CSV}")
print(f"  exists:  {os.path.exists(LABELS_CSV)}")
print(f"Master:    {MASTER_CSV}")
print(f"  exists:  {os.path.exists(MASTER_CSV)}")


# ============================================================
# 1. LADEN
# ============================================================
features = pd.read_csv(FEATURES_CSV)
labels   = pd.read_csv(LABELS_CSV)

print(f"\nFeatures: {len(features)} Patches, {features['standort_id'].nunique()} Standorte")
print(f"Labels:   {len(labels)} Patches, {labels['standort_id'].nunique()} Standorte")

# Master-CSV ist optional — wenn fehlend, weiter ohne master-Spalten
master = None
if os.path.exists(MASTER_CSV):
    master = pd.read_csv(MASTER_CSV)
    print(f"Master:   {len(master)} Standorte mit Metadaten")
else:
    print(f"\n⚠ master.csv fehlt — Join erfolgt ohne Standort-Metadaten.")
    print(f"   Für die Vollerhebung muss master.csv gepflegt werden.")


# ============================================================
# 2. CONSISTENCY-CHECKS
# ============================================================
# Welche standort_ids in den drei Datenquellen?
sids_features = set(features['standort_id'].unique())
sids_labels   = set(labels['standort_id'].unique())
sids_master   = set(master['standort_id'].unique()) if master is not None else set()

print(f"\n{'='*60}")
print(f"CONSISTENCY-CHECKS")
print(f"{'='*60}")

# In Features, aber nicht in Labels
nur_in_features = sids_features - sids_labels
if nur_in_features:
    print(f"\n⚠ {len(nur_in_features)} Standorte in Features, aber NICHT in Labels:")
    for sid in sorted(nur_in_features):
        print(f"    - {sid}  (CVAT-Annotation fehlt)")

# In Labels, aber nicht in Features
nur_in_labels = sids_labels - sids_features
if nur_in_labels:
    print(f"\n⚠ {len(nur_in_labels)} Standorte in Labels, aber NICHT in Features:")
    for sid in sorted(nur_in_labels):
        print(f"    - {sid}  (PS-Pipeline fehlt)")

# Falls Master vorhanden:
if master is not None:
    nur_in_master = sids_master - sids_features
    if nur_in_master:
        print(f"\n⚠ {len(nur_in_master)} Standorte in Master, aber NICHT in Features:")
        for sid in sorted(nur_in_master):
            print(f"    - {sid}  (PS-Pipeline noch offen)")

    nur_features_ohne_master = sids_features - sids_master
    if nur_features_ohne_master:
        print(f"\n⚠ {len(nur_features_ohne_master)} Standorte in Features, aber NICHT in Master:")
        for sid in sorted(nur_features_ohne_master):
            print(f"    - {sid}  (master.csv aktualisieren!)")

if not (nur_in_features or nur_in_labels or
        (master is not None and nur_in_master)):
    print(f"  ✓ Alle Datenquellen konsistent.")


# ============================================================
# 3. JOIN
# ============================================================
print(f"\n{'='*60}")
print(f"JOIN")
print(f"{'='*60}")

# Inner Join: nur Patches, die in beiden Quellen vorkommen
dataset = features.merge(
    labels[['standort_id', 'patch_idx',
            'patch_label_strikt', 'patch_label_locker',
            'anteil_ausgemagert']],
    on=['standort_id', 'patch_idx'],
    how='inner',
)
print(f"Nach Features+Labels Join: {len(dataset)} Patches")

# Master-Daten dazu (falls verfügbar)
if master is not None:
    # Welche Spalten aus master übernehmen?
    # standort_id ist schon drin, also auswählen welche Metadaten relevant sind
    master_cols = ['standort_id', 'cluster_id', 'datum', 'mischgut_typ_geschaetzt',
                   'groesstkorn_mm', 'flickstelle', 'mtd_mittel_mm',
                   'label_visuell_autor', 'label_visuell_betreuer',
                   'belichtung_s']

    # Nur die Spalten nehmen, die tatsächlich existieren
    verfuegbar = [c for c in master_cols if c in master.columns]
    fehlend    = [c for c in master_cols if c not in master.columns]

    if fehlend:
        print(f"  Master-Spalten fehlend (übersprungen): {fehlend}")

    dataset = dataset.merge(master[verfuegbar],
                            on='standort_id', how='left')
    print(f"Nach Master-Join: {len(dataset)} Patches, "
          f"{dataset['cluster_id'].nunique() if 'cluster_id' in dataset.columns else 0} Cluster")
else:
    print(f"Master-Join übersprungen.")


# ============================================================
# 4. ZUSAMMENFASSUNG
# ============================================================
print(f"\n{'='*60}")
print(f"FINALE ÜBERSICHT")
print(f"{'='*60}")
print(f"  Standorte:  {dataset['standort_id'].nunique()}")
if 'cluster_id' in dataset.columns:
    print(f"  Cluster:    {dataset['cluster_id'].nunique()}")
print(f"  Patches:    {len(dataset)}")

# Gültige Patches (Bildqualität)
ml_features_keys = ['roughness_index', 'var_Nz', 'surface_entropy_Nz',
                    'albedo_variance', 'skewness_Nz', 'kurtosis_Nz',
                    'edge_mean', 'edge_std', 'edge_p95',
                    'edge_density_global']

n_features_valid = dataset[ml_features_keys].notna().all(axis=1).sum()
print(f"  davon bildtechn. gültig: {n_features_valid}")

# Klassen-Verteilungen
print(f"\nLabel-Verteilung STRIKT:")
print(dataset['patch_label_strikt'].value_counts().to_string())

print(f"\nLabel-Verteilung LOCKER:")
print(dataset['patch_label_locker'].value_counts().to_string())

# Patches, die fürs Training verwendbar sind (gültig UND klar gelabelt)
trainings_strikt = dataset[
    dataset[ml_features_keys].notna().all(axis=1)
    & dataset['patch_label_strikt'].isin(['intakt', 'ausgemagert'])
]
trainings_locker = dataset[
    dataset[ml_features_keys].notna().all(axis=1)
    & dataset['patch_label_locker'].isin(['intakt', 'ausgemagert'])
]

print(f"\nTrainings-tauglich (STRIKT, valid+klares Label):  {len(trainings_strikt)} Patches")
if len(trainings_strikt) > 0:
    print(trainings_strikt['patch_label_strikt'].value_counts().to_string())

print(f"\nTrainings-tauglich (LOCKER, valid+klares Label):  {len(trainings_locker)} Patches")
if len(trainings_locker) > 0:
    print(trainings_locker['patch_label_locker'].value_counts().to_string())

# Wenn Cluster vorhanden: Klassen-Balance pro Cluster
if 'cluster_id' in dataset.columns and master is not None:
    print(f"\nPatches pro Cluster (LOCKER):")
    cluster_summary = trainings_locker.groupby(
        ['cluster_id', 'patch_label_locker']
    ).size().unstack(fill_value=0)
    print(cluster_summary)


# ============================================================
# 5. SPEICHERN
# ============================================================
dataset.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*60}")
print(f"GESPEICHERT")
print(f"{'='*60}")
print(f"  {OUTPUT_CSV}")
print(f"  Spalten: {list(dataset.columns)}")