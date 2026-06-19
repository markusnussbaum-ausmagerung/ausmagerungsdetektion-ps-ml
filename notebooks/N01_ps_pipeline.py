# -*- coding: utf-8 -*-
"""
N01 — Photometric-Stereo-Pipeline (Queau-Modell, ORIGINAL-undist-1:1)

Rechnet aus 12 DNG-Aufnahmen je Standort Normalenkarte und Albedo.
Modell: cos^4-Vignettierung, Psi-Korrektur, 1/r^2-Nahfeld, Anisotropie,
Lichtvektoren aus entzerrten Pixelkoordinaten.

Hinweis: Dieses Notebook benoetigt die DNG-Rohdaten, die nicht Teil des
oeffentlichen Repositoriums sind (Groesse, Ortsbezug). Es ist hier als
dokumentierter Verfahrensschritt enthalten. Die Notebooks ab N02 arbeiten
mit dem Zwischenprodukt ps_output.npz und laufen am Beispieldatensatz.
"""

# !pip install rawpy scipy   # Pakete siehe requirements.txt

# ============================================================
# PHOTOMETRIC STEREO — ORIGINAL PIPELINE (Quéau-Modell)
# MIT undistortPoints + EXIF-Verifikation
#
# Anwendung:
#   1. Ordner und Präfix eintragen
#   2. Zelle ausführen
#   3. Ergebnis: Normal Map, Albedo, RGB-Visualisierung
#
# Modell: cos⁴ + Ψ + 1/r² + Anisotropie
# Lichtvektoren: aus entzerrten Pixelkoordinaten berechnet
# Intensitäten: aus verzerrtem Bild (so wie der Sensor sie sieht)
# ============================================================

import rawpy
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2

# ============================================================
# UMGEBUNG UND PFADE
# ============================================================
# In Colab Google Drive einhaengen, lokal/im Repo Projektwurzel verwenden.
# Unterstruktur (03_Hardware_Fotobox, 04_Daten_Felderhebung) wie in der Arbeit.
# Datenpfad per Umgebungsvariable MA_BASIS ueberschreibbar.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJEKT = "/content/drive/MyDrive/MA_Ausmagerungsdetektion_PS"
except ImportError:
    PROJEKT = os.environ.get("MA_BASIS", ".")

KAL_PFAD     = os.path.join(PROJEKT, "03_Hardware_Fotobox/05_Kalibrierung/001_Kamerakalibrierung/kalibrierung_pixel9a_v2.npz")
LED_KAL_PFAD = os.path.join(PROJEKT, "03_Hardware_Fotobox/05_Kalibrierung/009_LED_Kalibrierung/led_kalibrierung_v2.npz")
GSD_PFAD     = os.path.join(PROJEKT, "03_Hardware_Fotobox/05_Kalibrierung/007_Validerung_Plakat/gsd_final.npz")
DATEN_DIR    = os.path.join(PROJEKT, "04_Daten_Felderhebung")

# ============================================================
# HIER NUR ORDNER UND DATEI-PRÄFIX EINTRAGEN
# ============================================================
# Rohdaten-Ordner EINES Standorts (12 DNG je Aufnahme), pro Lauf anpassen.
DNG_ORDNER = os.path.join(DATEN_DIR, "01_Rohdaten/standort_beispiel")
DATEI_PRAEFIX = 'Standort002'   # → xx_LED01_Links_20.dng usw.

LED_SUFFIX = {
    1: 'Links_20',   2: 'Links_35',   3: 'Links_60',
    4: 'Rechts_20',  5: 'Rechts_35',  6: 'Rechts_60',
    7: 'Vorne_20',   8: 'Vorne_35',   9: 'Vorne_60',
    10: 'Hinten_20', 11: 'Hinten_35', 12: 'Hinten_60',
}

DNG_PFADE = []
for led_nr in range(1, 13):
    dateiname = f'{DATEI_PRAEFIX}_LED{led_nr:02d}_{LED_SUFFIX[led_nr]}.dng'
    DNG_PFADE.append(os.path.join(DNG_ORDNER, dateiname))

print(f"Ordner:  {DNG_ORDNER}")
print(f"Präfix:  {DATEI_PRAEFIX}")
print(f"Beispiel: {os.path.basename(DNG_PFADE[0])}")

# ============================================================
# KALIBRIERUNGSDATEN LADEN
# ============================================================
# Kalibrierpfade siehe PROJEKT-Block oben

# Kamera
kal = np.load(KAL_PFAD)
K = kal['K']
D = kal['D']
fx, fy = K[0, 0], K[1, 1]
cx_full, cy_full = K[0, 2], K[1, 2]
f_full = (fx + fy) / 2

# LED-Kalibrierung
led_kal = np.load(LED_KAL_PFAD)
LED_POSITIONS = led_kal['led_positions']
LED_HAUPTACHSEN = led_kal['led_hauptachsen']
PSI = led_kal['psi']
MU = float(led_kal['mu'])

# GSD
GSD = float(np.load(GSD_PFAD)['gsd_final'])

print(f"\n{'='*55}")
print(f"KAMERA-KALIBRIERUNG (OpenCV)")
print(f"{'='*55}")
print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx_full:.1f}  cy={cy_full:.1f}")
print(f"  D = {D.ravel()[:5]}")
print(f"  GSD (voll) = {GSD:.5f} mm/px")
print(f"  f_mean = {f_full:.1f} px")

# ============================================================
# EXIF-VERIFIKATION: Intrinsics aus DNG-Metadaten lesen
# ============================================================
print(f"\n{'='*55}")
print(f"EXIF-VERIFIKATION")
print(f"{'='*55}")

erstdatei = DNG_PFADE[0]
if os.path.exists(erstdatei):
    with rawpy.imread(erstdatei) as raw:
        # Sensor-Dimensionen
        raw_h, raw_w = raw.sizes.raw_height, raw.sizes.raw_width
        img_h, img_w = raw.sizes.height, raw.sizes.width
        flip = raw.sizes.flip
        black = raw.black_level_per_channel
        white = raw.white_level
        color_desc = raw.color_desc.decode()
        bayer_pattern = raw.raw_pattern.tolist()

    print(f"  RAW-Größe:     {raw_w} × {raw_h}")
    print(f"  Bild-Größe:    {img_w} × {img_h}")
    print(f"  Flip:          {flip}")
    print(f"  Bayer-Pattern: {color_desc}")
    print(f"  Bayer-Matrix:  {bayer_pattern}")
    print(f"  Black-Level:   {black}")
    print(f"  White-Level:   {white}")

    # EXIF über subprocess (falls exiftool installiert)
    try:
        import subprocess
        result = subprocess.run(
            ['exiftool', '-FocalLength', '-FocalLengthIn35mmFormat',
             '-ImageWidth', '-ImageHeight', '-Model',
             '-ExifImageWidth', '-ExifImageHeight',
             erstdatei],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\n  EXIF (exiftool):")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"\n  exiftool nicht verfügbar — EXIF-Vergleich übersprungen")
        print(f"  Tipp: !apt install libimage-exiftool-perl")
else:
    print(f"  ⚠ Erste Datei nicht gefunden: {erstdatei}")


# ============================================================
# GRÜNKANAL-PARAMETER (halbe Auflösung)
# ============================================================
# Hauptpunkt exakt für gemittelten Grünkanal:
# green[i,j] = Mittel von Vollbild (2i, 2j) und (2i+1, 2j+1)
# Effektives Zentrum in Vollbild: (2i+0.5, 2j+0.5)
# → cx_g = (cx_full - 0.5) / 2
cx_g = (cx_full - 0.5) / 2
cy_g = (cy_full - 0.5) / 2
f_g = f_full / 2
gsd_g = GSD * 2

# K-Matrix für halbe Auflösung (für undistortPoints)
K_half = np.array([
    [f_full / 2,  0,           cx_g],
    [0,           f_full / 2,  cy_g],
    [0,           0,           1   ]
], dtype=np.float64)

print(f"\n{'='*55}")
print(f"GRÜNKANAL-PARAMETER")
print(f"{'='*55}")
print(f"  Hauptpunkt (voll):  cx={cx_full:.1f}  cy={cy_full:.1f}")
print(f"  Hauptpunkt (grün):  cx={cx_g:.2f}  cy={cy_g:.2f}")
print(f"  (Naiv cx/2={cx_full/2:.2f} — Differenz: "
      f"{abs(cx_g - cx_full/2):.2f} px = "
      f"{abs(cx_g - cx_full/2) * gsd_g:.3f} mm)")
print(f"  f_g = {f_g:.1f} px")
print(f"  GSD (grün) = {gsd_g:.4f} mm/px")


# ============================================================
# SCHRITT 1: Grünkanal laden
# ============================================================
def lade_gruenkanal(dng_pfad):
    """
    GRBG-Pattern (Google Pixel 9a):
      Zeile 0:  G  R  G  R ...   → g1 = bayer[0::2, 0::2]
      Zeile 1:  B  G  B  G ...   → g2 = bayer[1::2, 1::2]
    """
    with rawpy.imread(dng_pfad) as raw:
        bayer = raw.raw_image.copy().astype(np.float32)
        flip = raw.sizes.flip
    bayer = np.maximum(bayer - 64, 0)
    g1 = bayer[0::2, 0::2]
    g2 = bayer[1::2, 1::2]
    green = (g1 + g2) / 2.0
    if flip == 6:
        green = np.rot90(green, k=-1)
    return green


# ============================================================
# SCHRITT 2: Bilder laden
# ============================================================
print(f"\nLade 12 DNG-Aufnahmen ...")
bilder = []
for i, pfad in enumerate(DNG_PFADE):
    if not os.path.exists(pfad):
        raise FileNotFoundError(f"LED{i+1:02d}: {pfad}")
    green = lade_gruenkanal(pfad)
    bilder.append(green)
    print(f"  LED{i+1:02d} {LED_SUFFIX[i+1]:<12}: "
          f"max={green.max():.0f}  mean={green.mean():.0f}")

I_stack_raw = np.stack(bilder, axis=0)
N_LEDS, H_g, W_g = I_stack_raw.shape
print(f"\nBild-Stack: {N_LEDS} LEDs × {H_g} × {W_g} Pixel")


# ============================================================
# SCHRITT 2.5: Quadratischer Crop (1:1) auf Hauptpunkt
# ============================================================
# Entfernt die schlecht beleuchteten Bildecken.
# Crop-Größe = kürzere Seite (W_g), zentriert auf Hauptpunkt cy.
CROP_SIZE = min(H_g, W_g)  # 1500 bei 2000×1500

# Vertikaler Crop (die lange Seite wird beschnitten)
v_center = int(round(cy_g))
v_start = max(0, v_center - CROP_SIZE // 2)
v_end = v_start + CROP_SIZE
if v_end > H_g:
    v_end = H_g
    v_start = v_end - CROP_SIZE

# Horizontaler Crop (falls Bild breiter als CROP_SIZE)
u_center = int(round(cx_g))
u_start = max(0, u_center - CROP_SIZE // 2)
u_end = u_start + CROP_SIZE
if u_end > W_g:
    u_end = W_g
    u_start = u_end - CROP_SIZE

# Crop anwenden
I_stack_raw = I_stack_raw[:, v_start:v_end, u_start:u_end]
N_LEDS, H_g, W_g = I_stack_raw.shape

# Hauptpunkt im neuen Koordinatensystem
cx_g = cx_g - u_start
cy_g = cy_g - v_start

# K_half aktualisieren
K_half[0, 2] = cx_g
K_half[1, 2] = cy_g

print(f"\n1:1 Crop: {CROP_SIZE}×{CROP_SIZE} px")
print(f"  v_start={v_start}  v_end={v_end}  (oben {v_start} px, "
      f"unten {2000-v_end} px abgeschnitten)")
print(f"  u_start={u_start}  u_end={u_end}")
print(f"  Hauptpunkt im Crop: cx={cx_g:.1f}  cy={cy_g:.1f}")
print(f"  Aufnahmefläche: {CROP_SIZE * gsd_g:.0f} × "
      f"{CROP_SIZE * gsd_g:.0f} mm = "
      f"{CROP_SIZE * gsd_g / 10:.1f} × "
      f"{CROP_SIZE * gsd_g / 10:.1f} cm")


# ============================================================
# SCHRITT 3: Pixelkoordinaten entzerren (undistortPoints)
# ============================================================
print(f"\nEntzerren der Pixelkoordinaten ...")

# Alle Pixel-Positionen im Grünkanal
u_grid_raw, v_grid_raw = np.meshgrid(
    np.arange(W_g, dtype=np.float32),
    np.arange(H_g, dtype=np.float32))

# Entzerren: verzerrte Pixel → entzerrte Pixel
pts_raw = np.stack([u_grid_raw.ravel(),
                    v_grid_raw.ravel()], axis=1).astype(np.float32)
pts_undist = cv2.undistortPoints(
    pts_raw.reshape(-1, 1, 2), K_half, D, P=K_half)
u_undist = pts_undist[:, 0, 0].reshape(H_g, W_g)
v_undist = pts_undist[:, 0, 1].reshape(H_g, W_g)

# Maximale Verschiebung
delta = np.sqrt((u_undist - u_grid_raw)**2 + (v_undist - v_grid_raw)**2)
print(f"  Max. Verschiebung: {delta.max():.2f} px "
      f"= {delta.max() * gsd_g:.2f} mm")
print(f"  Mitte:             {delta[H_g//2, W_g//2]:.4f} px")

# Welt-Koordinaten aus ENTZERRTEN Positionen
# (Das ist der einzige Ort wo undistortPoints eingeht)
x_world = (u_undist - cx_g) * gsd_g
y_world = -(v_undist - cy_g) * gsd_g
z_world = np.zeros_like(x_world)

print(f"  Welt-Bereich X: {x_world.min():.1f} bis {x_world.max():.1f} mm")
print(f"  Welt-Bereich Y: {y_world.min():.1f} bis {y_world.max():.1f} mm")


# ============================================================
# SCHRITT 4: cos⁴(α)-Korrektur
# ============================================================
print(f"\ncos⁴(α)-Korrektur ...")

# cos⁴ ebenfalls mit entzerrten Koordinaten berechnen
u_rel = u_undist - cx_g
v_rel = v_undist - cy_g
cos4_alpha = (f_g / np.sqrt(u_rel**2 + v_rel**2 + f_g**2)) ** 4

print(f"  Mitte: {cos4_alpha[H_g//2, W_g//2]:.4f}")
print(f"  Ecke:  {cos4_alpha[0, 0]:.4f}")
print(f"  Abfall: {(1 - cos4_alpha.min())*100:.1f}%")

# Intensitäten korrigieren (Bild bleibt verzerrt — nur Division)
I_stack = I_stack_raw / cos4_alpha[np.newaxis, :, :]


# ============================================================
# SCHRITT 5: Lichtvektor-Stack T(x) mit Ψ, 1/r², cos^μ(θ)
# ============================================================
print(f"Berechne Lichtvektor-Stack T(x) ...")

T_stack = np.zeros((N_LEDS, H_g, W_g, 3), dtype=np.float32)

for i in range(N_LEDS):
    led_pos = LED_POSITIONS[i]
    led_axis = LED_HAUPTACHSEN[i]
    psi_i = PSI[i]

    # Vektor von Pixel zur LED (aus ENTZERRTEN Welt-Koordinaten)
    dx = led_pos[0] - x_world
    dy = led_pos[1] - y_world
    dz = led_pos[2] - z_world
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Einheits-Lichtrichtung
    sx, sy, sz = dx/r, dy/r, dz/r

    # Anisotropie
    cos_theta = -(led_axis[0]*sx + led_axis[1]*sy + led_axis[2]*sz)
    cos_theta = np.maximum(cos_theta, 0)

    # Gesamtfaktor
    skalar = psi_i * (cos_theta ** MU) / (r**2)

    T_stack[i, :, :, 0] = skalar * sx
    T_stack[i, :, :, 1] = skalar * sy
    T_stack[i, :, :, 2] = skalar * sz

print(f"  T-Stack: {T_stack.shape}")


# ============================================================
# SCHRITT 6: Shadow-Trimming + Least Squares
# ============================================================
# Trimming entfernt pro Pixel die Ausreißer-Intensitäten:
#   - Die 2 dunkelsten (= Schlagschatten, Selbstschatten)
#   - Die 1 hellste (= Spekularitäten, Glanzpunkte)
# Übrig bleiben 9 von 12 Messungen, die dem Lambert-Modell folgen.
#
# Warum? Eine LED deren Licht von einem Objekt blockiert wird
# (Schlagschatten) erzeugt Intensität ~0 an diesem Pixel.
# Der Solver würde das als "Normale steht senkrecht zur LED"
# interpretieren — was falsch ist. Durch Entfernen der 2
# dunkelsten Werte wird der Schatten ignoriert.
# ============================================================

N_TRIM_LOW = 2    # 2 dunkelste entfernen (Schatten)
N_TRIM_HIGH = 1   # 1 hellste entfernen (Spekularität)
N_KEEP = N_LEDS - N_TRIM_LOW - N_TRIM_HIGH  # = 9

print(f"Shadow-Trimming: {N_TRIM_LOW} dunkelste + "
      f"{N_TRIM_HIGH} hellste entfernen → {N_KEEP} von {N_LEDS} behalten")

# Pro Pixel: Intensitäten sortieren, mittlere 9 behalten
sort_idx = np.argsort(I_stack, axis=0)           # (12, H, W)
keep_idx = sort_idx[N_TRIM_LOW:N_LEDS-N_TRIM_HIGH]  # (9, H, W)

# Getrimmte Stacks aufbauen
I_trim = np.zeros((N_KEEP, H_g, W_g), dtype=np.float32)
T_trim = np.zeros((N_KEEP, H_g, W_g, 3), dtype=np.float32)

for k in range(N_KEEP):
    idx_map = keep_idx[k]     # (H, W) — welche LED pro Pixel an Position k
    for led in range(N_LEDS):
        maske = (idx_map == led)
        I_trim[k][maske] = I_stack[led][maske]
        T_trim[k][maske] = T_stack[led][maske]

# Least Squares mit den getrimmten Daten
print(f"Löse Gleichungssystem ({N_KEEP} LEDs pro Pixel) ...")

A = np.transpose(T_trim, (1, 2, 0, 3)).reshape(-1, N_KEEP, 3)
I_flat = np.transpose(I_trim, (1, 2, 0)).reshape(-1, N_KEEP, 1)

AtA = np.einsum('pij,pik->pjk', A, A)
AtI = np.einsum('pij,pik->pjk', A, I_flat).squeeze(-1)
AtA += np.eye(3)[np.newaxis, :, :] * 1e-10

m_flat = np.linalg.solve(AtA, AtI[:, :, np.newaxis]).squeeze(-1)
m = m_flat.reshape(H_g, W_g, 3)
print(f"  Fertig.")

# ============================================================
# SCHRITT 7: Normale und Albedo
# ============================================================
albedo = np.linalg.norm(m, axis=2)
albedo_safe = np.maximum(albedo, 1e-6)
normalen = m / albedo_safe[:, :, np.newaxis]

Nx = normalen[:, :, 0]
Ny = normalen[:, :, 1]
Nz = normalen[:, :, 2]

print(f"\nErgebnis:")
print(f"  Albedo:  min={albedo.min():.2f}  max={albedo.max():.2f}  "
      f"mean={albedo.mean():.2f}")
print(f"  Nz:     min={Nz.min():.3f}  max={Nz.max():.3f}  "
      f"mean={Nz.mean():.3f}")
print(f"  Nx:     min={Nx.min():.3f}  max={Nx.max():.3f}  "
      f"mean={Nx.mean():.3f}")
print(f"  Ny:     min={Ny.min():.3f}  max={Ny.max():.3f}  "
      f"mean={Ny.mean():.3f}")


# ============================================================
# SCHRITT 7.5: PS-Output als .npz auf Google Drive speichern
# ============================================================
import datetime

# === HIER PRO STANDORT ANPASSEN ===
STANDORT_ID = 'S007-2'           # globale Standort-ID (manuell vergeben)
STANDORT_LABEL = 'ausgemagert' # 'intakt' / 'leicht' / 'mittel' / 'schwer'
PIPELINE_VERSION = '1.1'       # nach jeder Pipeline-Änderung erhöhen

OUTPUT_BASE = os.path.join(DATEN_DIR, "02_Verarbeitet/01_Triplets")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, STANDORT_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PFAD = os.path.join(OUTPUT_DIR, 'ps_output.npz')

# Confidence-Map: wie viele LEDs liefern pro Pixel "echtes" Signal?
# Schwelle 100 ADU = deutlich über Black-Level (64) + Sensor-Rauschen
SNR_SCHWELLE = 100
n_valid_per_pixel = (I_stack_raw > SNR_SCHWELLE).sum(axis=0).astype(np.uint8)

# Speichern (komprimiert spart ~70% Plattenplatz)
np.savez_compressed(
    OUTPUT_PFAD,
    # Haupt-Output (das was die Feature-Extraction braucht)
    Nx=Nx.astype(np.float32),
    Ny=Ny.astype(np.float32),
    Nz=Nz.astype(np.float32),
    albedo=albedo.astype(np.float32),
    # Confidence pro Pixel
    n_valid_leds=n_valid_per_pixel,
    # Metadaten
    standort_id=np.array(STANDORT_ID),
    standort_label=np.array(STANDORT_LABEL),
    quell_ordner=np.array(os.path.basename(DNG_ORDNER)),
    datei_praefix=np.array(DATEI_PRAEFIX),
    gsd_mm_per_px=np.float32(gsd_g),
    crop_size_px=np.int32(CROP_SIZE),
    image_shape=np.array([H_g, W_g], dtype=np.int32),
    n_trim_low=np.int32(N_TRIM_LOW),
    n_trim_high=np.int32(N_TRIM_HIGH),
    pipeline_version=np.array(PIPELINE_VERSION),
    erzeugt_am=np.array(datetime.datetime.now().isoformat()),
)

# Diagnose-Ausgabe
dateigroesse_mb = os.path.getsize(OUTPUT_PFAD) / 1024**2
n_gesamt = n_valid_per_pixel.size
n_schwach = (n_valid_per_pixel < 6).sum()

print(f"\n{'='*55}")
print(f"PS-OUTPUT GESPEICHERT")
print(f"{'='*55}")
print(f"  Pfad:           {OUTPUT_PFAD}")
print(f"  Dateigröße:     {dateigroesse_mb:.1f} MB")
print(f"  Bild-Shape:     {H_g} × {W_g} px ({CROP_SIZE * gsd_g / 10:.1f} cm)")
print(f"  Standort-ID:    {STANDORT_ID}")
print(f"  Standort-Label: {STANDORT_LABEL}")
print(f"\n  Confidence (LEDs mit Signal > {SNR_SCHWELLE} ADU pro Pixel):")
print(f"    Median:       {int(np.median(n_valid_per_pixel))} von 12 LEDs")
print(f"    Pixel mit <6: {n_schwach / n_gesamt * 100:.1f}%")
if n_schwach / n_gesamt > 0.05:
    print(f"  ⚠ Hoher Anteil schwacher Pixel — Belichtung evaluieren!")



# ============================================================
# SHADING-EXPORT FÜR CVAT-ANNOTATION
# ============================================================
# Erzeugt ein 2-Panel-PNG zur visuellen Annotation in CVAT:
#   - Links:  Original-Aufnahme (LED07 Vorne_20, eine zentrale LED)
#   - Rechts: Lambert-Shading mit virtueller Beleuchtung
#
# Speichert das Bild im selben Ordner wie ps_output.npz.
# Diese Datei wird in CVAT als Annotations-Grundlage hochgeladen.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1) Lambert-Shading mit virtueller Beleuchtung ---
# Virtuelles Licht aus 45° [0.5, -0.5, 0.707] schräg oben rechts vorne.
# Virtuelles Licht aus 30° [0.612, -0.612, 0.5] schräg oben rechts vorne.
# Diese Lichtrichtung gibt gut sichtbares Relief ohne extreme Schatten.
light_dir = np.array([0.612, -0.612, 0.5])
light_dir /= np.linalg.norm(light_dir)

# Lambert: I = max(0, n · L) * albedo
# Mit "synthetischer" konstanter Albedo wirkt das Relief ohne
# Materialvariation — gut für Geometrie-Annotation.
# Mit echter Albedo wirkt's natürlicher — gut um Körner zu sehen.

shading_geometric = np.clip(
    Nx * light_dir[0] + Ny * light_dir[1] + Nz * light_dir[2],
    0, 1
)

# Albedo-gewichtete Variante (zeigt Material-Variation)
albedo_norm = albedo / np.percentile(albedo, 98)  # auf 98%-Perzentil normieren
albedo_norm = np.clip(albedo_norm, 0, 1)
shading_albedo = shading_geometric * albedo_norm

# Kontrast-Anpassung für bessere Sichtbarkeit
def stretch(img, p_lo=2, p_hi=98):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img - lo) / (hi - lo), 0, 1)

shading_geometric_s = stretch(shading_geometric)
shading_albedo_s = stretch(shading_albedo)


# --- 2) Original-LED-Aufnahme als Referenz ---
# LED07 = Vorne_20, eine der zentralen oberen LEDs.
# LED07 = Vorne_20 (Index 6), LED10 = Hinten_20 (Index 9)
original_led07 = I_stack_raw[6]
original_led10 = I_stack_raw[9]
original_led07_s = stretch(original_led07)
original_led10_s = stretch(original_led10)


# --- 3) Alle Varianten als hochauflösende PNGs für CVAT ---
# Reine Pixel-Daten, keine Matplotlib-Achsen/Titel/Padding.
# Damit bleibt die Auflösung exakt 1500×1500 und die Pixel-Koordinaten
# in CVAT entsprechen 1:1 den Pipeline-Koordinaten.
# OUTPUT_DIR liegt bereits aus dem Speicher-Block vor.
ANNOT_DIR = os.path.join(OUTPUT_DIR, 'annotation')
os.makedirs(ANNOT_DIR, exist_ok=True)

from PIL import Image

def save_as_png(arr_float01, pfad):
    """Konvertiert float[0,1]-Array zu uint8 und speichert als PNG."""
    img_uint8 = (np.clip(arr_float01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(pfad)

# Variante 1: Original LED07 (rohe Aufnahme, eine LED)
# Variante 1a: Original LED07 (von vorne)
pfad_v1a = os.path.join(ANNOT_DIR, f'{STANDORT_ID}_v1a_original_LED07_vorne.png')
save_as_png(original_led07_s, pfad_v1a)

# Variante 1b: Original LED10 (von hinten — Gegenlicht)
pfad_v1b = os.path.join(ANNOT_DIR, f'{STANDORT_ID}_v1b_original_LED10_hinten.png')
save_as_png(original_led10_s, pfad_v1b)

# Variante 2: Lambert-Shading rein (Geometrie ohne Material)
pfad_v2 = os.path.join(ANNOT_DIR, f'{STANDORT_ID}_v2_lambert_geometric.png')
save_as_png(shading_geometric_s, pfad_v2)

# Variante 3: Lambert × Albedo (Geometrie + Material)
pfad_v3 = os.path.join(ANNOT_DIR, f'{STANDORT_ID}_v3_lambert_albedo.png')
save_as_png(shading_albedo_s, pfad_v3)


# --- 4) Zusätzlich: Vergleichs-Plot für die Auswahl ---
# Dieses Bild ist NUR für deine eigene Begutachtung, nicht für CVAT.
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

axes[0].imshow(original_led07_s, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('v1: Original LED07', fontsize=14)
axes[0].axis('off')

axes[1].imshow(shading_geometric_s, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('v2: Lambert Geometric', fontsize=14)
axes[1].axis('off')

axes[2].imshow(shading_albedo_s, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('v3: Lambert × Albedo', fontsize=14)
axes[2].axis('off')

plt.suptitle(f'{STANDORT_ID} — Annotations-Varianten im Vergleich',
             fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(ANNOT_DIR, f'{STANDORT_ID}_vergleich.png'),
            dpi=120, bbox_inches='tight', facecolor='white')
plt.show()


# --- 5) Ausgabe ---
print(f"\n{'='*55}")
print(f"SHADING-BILDER GESPEICHERT (alle 1500×1500 px)")
print(f"{'='*55}")
print(f"  Verzeichnis: {ANNOT_DIR}")
print(f"\n  Für CVAT-Upload (auswählen):")
print(f"    v1 — Original LED07:        {os.path.basename(pfad_v1a)}")
print(f"    v2 — Lambert Geometric:     {os.path.basename(pfad_v2)}")
print(f"    v3 — Lambert × Albedo:      {os.path.basename(pfad_v3)}")
print(f"\n  Zur Auswahl (Vergleichs-Plot): {STANDORT_ID}_vergleich.png")
print(f"  GSD: {gsd_g:.4f} mm/px")

# ============================================================
# EXPORT NZ-KARTE UND NORMAL MAP ALS HOCHAUFGELÖSTE PNGs
# Für CVAT-Annotation und Methodik-Abbildungen in Kapitel 4.8
# ============================================================
from PIL import Image as PILImage

# --- Nz-Karte als Graustufenbild (1500 × 1500 px) ---
# Wertebereich [0.5, 1.0] auf [0, 255] strecken;
# typische Asphalt-Nz-Werte liegen zwischen 0.7 und 1.0
NZ_MIN, NZ_MAX = 0.5, 1.0
Nz_clipped = np.clip(Nz, NZ_MIN, NZ_MAX)
Nz_8bit = ((Nz_clipped - NZ_MIN) / (NZ_MAX - NZ_MIN) * 255).astype(np.uint8)

pfad_nz = os.path.join(ANNOT_DIR, f'{STANDORT_ID}_v4_nz_karte.png')
PILImage.fromarray(Nz_8bit, mode='L').save(pfad_nz)
print(f"  Nz-Karte (Annotations-Grundlage): {pfad_nz}")

# --- Normal Map als RGB-Bild (Konvention: Nx→R, Ny→G, Nz→B) ---
# Mapping [-1, +1] → [0, 255] linear
normal_rgb = np.stack([Nx, Ny, Nz], axis=-1)              # (H, W, 3)
normal_rgb_8bit = ((normal_rgb + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

pfad_normal = os.path.join(ANNOT_DIR, f'{STANDORT_ID}_v5_normal_map.png')
PILImage.fromarray(normal_rgb_8bit, mode='RGB').save(pfad_normal)
print(f"  Normal Map (RGB):                 {pfad_normal}")


# ============================================================
# SCHRITT 8: Visualisierung
# ============================================================
print(f"\nErstelle Visualisierung ...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

im0 = axes[0, 0].imshow(Nx, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0, 0].set_title('Nx (links ↔ rechts)\n'
                      'Rot = nach rechts geneigt\n'
                      'Blau = nach links geneigt')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(Ny, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0, 1].set_title('Ny (vorne ↔ hinten)\n'
                      'Rot = nach vorne geneigt\n'
                      'Blau = nach hinten geneigt')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

im2 = axes[0, 2].imshow(Nz, cmap='gray', vmin=0.5, vmax=1.0)
axes[0, 2].set_title('Nz (Steilheit)\n'
                      'Hell = flach | Dunkel = steil')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

v_lo, v_hi = np.percentile(albedo, [2, 98])
im3 = axes[1, 0].imshow(albedo, cmap='gray', vmin=v_lo, vmax=v_hi)
axes[1, 0].set_title('Albedo')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

normal_rgb = np.zeros((H_g, W_g, 3))
normal_rgb[:, :, 0] = (Nx + 1) / 2
normal_rgb[:, :, 1] = (Ny + 1) / 2
normal_rgb[:, :, 2] = np.clip(Nz, 0, 1)
normal_rgb = np.clip(normal_rgb, 0, 1)
axes[1, 1].imshow(normal_rgb)
axes[1, 1].set_title('RGB Normal Map')

v99 = np.percentile(I_stack_raw[0], 99)
axes[1, 2].imshow(I_stack_raw[0] / v99, cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title(f'Eingangsbild LED01 (roh)')

for ax in axes.flatten():
    ax.set_xlabel('Pixel u')
    ax.set_ylabel('Pixel v')

fig.suptitle(f'PS ORIGINAL (cos⁴+Ψ+1/r²+Aniso) + undistortPoints\n'
             f'{os.path.basename(DNG_ORDNER)} | {DATEI_PRAEFIX}',
             fontsize=13)
plt.tight_layout()
plt.savefig('ps_ergebnis_ORIGINAL_undist.png',
            dpi=120, bbox_inches='tight')
plt.show()

print(f"\n✓ Pipeline abgeschlossen.")

# Test: Datei wieder einlesen und Werte prüfen
test_data = np.load(OUTPUT_PFAD)

print("Verfügbare Arrays:")
for key in test_data.files:
    arr = test_data[key]
    if arr.ndim == 0:  # Skalar
        print(f"  {key:<20} = {arr}")
    else:
        print(f"  {key:<20} shape={arr.shape}, dtype={arr.dtype}, "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")

# Sanity-Check: Nx² + Ny² + Nz² ≈ 1 (Einheitsvektor)
n_norm = np.sqrt(test_data['Nx']**2 + test_data['Ny']**2 + test_data['Nz']**2)
print(f"\nNormalen-Magnitude: mean={n_norm.mean():.4f}, "
      f"std={n_norm.std():.4f}")
print("→ Sollte sehr nah an 1.0 mit kleiner std liegen")

# ============================================================
# INTERAKTIVE NORMAL MAPS — Hover für exakte Werte
# Voraussetzung: normalen, albedo, gsd_g aus PS-Pipeline
# ============================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Nx = normalen[:, :, 0]
Ny = normalen[:, :, 1]
Nz = normalen[:, :, 2]
H, W = Nx.shape

print(f"Normal Map: {H}×{W} px")
print(f"Hover über die Bilder um Werte abzulesen.")
print(f"Nx=0.707 → 45° Neigung, Nx=0.5 → 30°, Nx=0.866 → 60°\n")

# ── Nx interaktiv ───────────────────────────────────────────
fig_nx = go.Figure(data=go.Heatmap(
    z=Nx,
    colorscale='RdBu_r',
    zmin=-0.8, zmax=0.8,
    colorbar=dict(title='Nx'),
    hovertemplate=(
        'u=%{x}<br>v=%{y}<br>'
        'Nx=%{z:.4f}<br>'
        'Winkel=%{customdata:.1f}°'
        '<extra></extra>'
    ),
    customdata=np.degrees(np.arcsin(np.clip(Nx, -1, 1))),
))
fig_nx.update_layout(
    title='Nx (links ↔ rechts) — Hover für Werte<br>'
          '<sub>Rot=rechts geneigt, Blau=links geneigt | '
          'Winkel = arcsin(Nx)</sub>',
    xaxis_title='Pixel u',
    yaxis_title='Pixel v',
    width=800, height=900,
    yaxis=dict(autorange='reversed'),
)
fig_nx.show()

# ── Ny interaktiv ───────────────────────────────────────────
fig_ny = go.Figure(data=go.Heatmap(
    z=Ny,
    colorscale='RdBu_r',
    zmin=-0.8, zmax=0.8,
    colorbar=dict(title='Ny'),
    hovertemplate=(
        'u=%{x}<br>v=%{y}<br>'
        'Ny=%{z:.4f}<br>'
        'Winkel=%{customdata:.1f}°'
        '<extra></extra>'
    ),
    customdata=np.degrees(np.arcsin(np.clip(Ny, -1, 1))),
))
fig_ny.update_layout(
    title='Ny (vorne ↔ hinten) — Hover für Werte<br>'
          '<sub>Rot=vorne geneigt, Blau=hinten geneigt</sub>',
    xaxis_title='Pixel u',
    yaxis_title='Pixel v',
    width=800, height=900,
    yaxis=dict(autorange='reversed'),
)
fig_ny.show()

# ── Nz interaktiv ───────────────────────────────────────────
fig_nz = go.Figure(data=go.Heatmap(
    z=Nz,
    colorscale='Gray',
    zmin=0.3, zmax=1.0,
    colorbar=dict(title='Nz'),
    reversescale=True,
    hovertemplate=(
        'u=%{x}<br>v=%{y}<br>'
        'Nz=%{z:.4f}<br>'
        'Neigung=%{customdata:.1f}° von Senkrechte'
        '<extra></extra>'
    ),
    customdata=np.degrees(np.arccos(np.clip(Nz, -1, 1))),
))
fig_nz.update_layout(
    title='Nz (Steilheit) — Hover für Werte<br>'
          '<sub>Hell=flach (Nz≈1), Dunkel=steil (Nz<1) | '
          'Neigung = arccos(Nz)</sub>',
    xaxis_title='Pixel u',
    yaxis_title='Pixel v',
    width=800, height=900,
    yaxis=dict(autorange='reversed'),
)
fig_nz.show()

# ── Albedo interaktiv ───────────────────────────────────────
fig_alb = go.Figure(data=go.Heatmap(
    z=albedo,
    colorscale='Gray',
    colorbar=dict(title='ρ'),
    hovertemplate=(
        'u=%{x}<br>v=%{y}<br>'
        'Albedo=%{z:.4f}'
        '<extra></extra>'
    ),
))
fig_alb.update_layout(
    title='Albedo — Hover für Werte',
    xaxis_title='Pixel u',
    yaxis_title='Pixel v',
    width=800, height=900,
    yaxis=dict(autorange='reversed'),
)
fig_alb.show()

print("4 interaktive Plots erstellt.")
print("Tipp: Hover zeigt Nx/Ny/Nz + den Winkel in Grad.")