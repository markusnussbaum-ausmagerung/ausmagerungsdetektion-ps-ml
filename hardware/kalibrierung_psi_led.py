# ============================================================
# PIPELINE-SCHRITT 2 KORRIGIERT: Ψⁱ-Kalibrierung
# Bug-Fix: y_world Vorzeichen invertiert (v wächst → y schrumpft)
# ============================================================

import rawpy
import numpy as np
import os

# ============================================================
# UMGEBUNG UND PFADE
# ============================================================
# In Colab Google Drive einhaengen, lokal/im Repo Projektwurzel verwenden.
# Unterstruktur (03_Hardware_Fotobox/05_Kalibrierung) wie in der Arbeit.
import os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJEKT = "/content/drive/MyDrive/MA_Ausmagerungsdetektion_PS"
except ImportError:
    PROJEKT = os.environ.get("MA_BASIS", ".")
import glob
import matplotlib.pyplot as plt

# ── Pfade ────────────────────────────────────────────────────
BASIS = os.path.join(PROJEKT,
         '03_Hardware_Fotobox/05_Kalibrierung/'
         '003_Belichtungstests/001_Weissreferenzen-NEU/'
         '20260504_1_60s')
KAL_PFAD = os.path.join(PROJEKT,
            '03_Hardware_Fotobox/05_Kalibrierung/'
            '001_Kamerakalibrierung/kalibrierung_pixel9a_v2.npz')
OFFSET_PFAD = os.path.join(PROJEKT,
               '03_Hardware_Fotobox/05_Kalibrierung/'
               '007_Validerung_Plakat/'
               'linsenoffset_und_led_korrektur.npz')
GSD_PFAD = os.path.join(PROJEKT,
            '03_Hardware_Fotobox/05_Kalibrierung/'
            '007_Validerung_Plakat/gsd_final.npz')
SAVE_DIR = os.path.join(PROJEKT,
            '03_Hardware_Fotobox/05_Kalibrierung/'
            '009_LED_Kalibrierung')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Kalibrierungs-Daten laden ────────────────────────────────
kal = np.load(KAL_PFAD)
K = kal['K']
fx, fy = K[0, 0], K[1, 1]
cx_full, cy_full = K[0, 2], K[1, 2]
f_full = (fx + fy) / 2

offset = np.load(OFFSET_PFAD)
LED_POSITIONS = offset['led_positionen_korr']
gsd_data = np.load(GSD_PFAD)
GSD = float(gsd_data['gsd_final'])

LED_NAMES = {1:'Links_20', 2:'Links_35', 3:'Links_60',
             4:'Rechts_20', 5:'Rechts_35', 6:'Rechts_60',
             7:'Vorne_20', 8:'Vorne_35', 9:'Vorne_60',
             10:'Hinten_20', 11:'Hinten_35', 12:'Hinten_60'}

# ── LED-Hauptachsen aus konstruktiver Annahme ───────────────
LED_HAUPTACHSEN = np.zeros((12, 3))
for i in range(12):
    richtung = -LED_POSITIONS[i]
    LED_HAUPTACHSEN[i] = richtung / np.linalg.norm(richtung)

MU = 1.0

# ── Hilfsfunktion ───────────────────────────────────────────
def lade_gruenkanal(dng_pfad, black_level=64):
    with rawpy.imread(dng_pfad) as raw:
        bayer = raw.raw_image.copy().astype(np.float32)
        flip = raw.sizes.flip
    bayer = np.maximum(bayer - black_level, 0)
    g1 = bayer[0::2, 0::2]
    g2 = bayer[1::2, 1::2]
    green = (g1 + g2) / 2.0
    if flip == 6:
        green = np.rot90(green, k=-1)
    return green

# ── Geometrie-Faktor G^i(p) – KORRIGIERT ───────────────────
def berechne_G(led_pos, led_axis, mu, H, W, gsd, f_full,
               cx_full, cy_full):
    """Geometriefaktor pro Pixel auf Lambert-Platte (Z=0).

       WICHTIG: Achsen-Konvention nach empirischer Diagnose:
         u (Bild-Spalte) → Welt-x  (positiv u = positiv x)
         v (Bild-Zeile)  → -Welt-y (positiv v = NEGATIV y)
                            (oben im Bild = Vorne = positiv y)
    """
    v_grid, u_grid = np.indices((H, W))
    cx_g = cx_full / 2
    cy_g = cy_full / 2
    gsd_g = gsd * 2

    # Welt-Koordinaten (KORRIGIERT)
    x_world = (u_grid - cx_g) * gsd_g
    y_world = -(v_grid - cy_g) * gsd_g   # Vorzeichen-Invertierung
    z_world = np.zeros_like(x_world)

    # Vektor von Pixel zur LED
    dx = led_pos[0] - x_world
    dy = led_pos[1] - y_world
    dz = led_pos[2] - z_world
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Lichtrichtung (Pixel → LED, normiert)
    sx, sy, sz = dx/r, dy/r, dz/r

    # Anisotropie: Winkel zwischen Lichtrichtung und LED-Hauptachse
    # n_s zeigt von LED zur Boxmitte → entgegengesetzt zu s
    cos_theta = -(led_axis[0]*sx + led_axis[1]*sy + led_axis[2]*sz)
    cos_theta = np.maximum(cos_theta, 0)

    # Lambert: Plattennormale = (0, 0, 1)
    cos_alpha = sz
    cos_alpha = np.maximum(cos_alpha, 0)

    G = (cos_theta ** mu) * cos_alpha / (r**2)
    return G

# ── cos⁴(α)-Korrekturmatrix in Grünauflösung ───────────────
H_g, W_g = 2000, 1500
cx_g_half = cx_full / 2
cy_g_half = cy_full / 2
f_g = f_full / 2
v_grid, u_grid = np.indices((H_g, W_g))
u_rel = u_grid - cx_g_half
v_rel = v_grid - cy_g_half
cos4_alpha = (f_g / np.sqrt(u_rel**2 + v_rel**2 + f_g**2)) ** 4

SATURATION = (1023 - 64) - 5

# ── Hauptschleife: Ψⁱ pro LED ──────────────────────────────
psi_values = np.zeros(12)
fit_qualitaet = np.zeros(12)

print(f"{'='*70}")
print("Ψⁱ-Kalibrierung pro LED (korrigiert)")
print(f"{'='*70}")
print(f"{'LED':<25s} {'Ψⁱ':>15s} {'gültig':>10s} {'Fit %':>10s}")
print('-' * 70)

for i in range(12):
    led_nr = i + 1
    muster = os.path.join(BASIS, f'wr_1_60s_LED{led_nr:02d}_*.dng')
    treffer = glob.glob(muster)
    if not treffer:
        print(f"  LED{led_nr:02d} fehlt!")
        continue

    green = lade_gruenkanal(treffer[0])
    I = green / cos4_alpha

    G = berechne_G(LED_POSITIONS[i], LED_HAUPTACHSEN[i], MU,
                   H_g, W_g, GSD, f_full, cx_full, cy_full)

    mask = (green < SATURATION) & (G > 1e-12) & (I > 0)
    n_valid = int(mask.sum())

    I_v = I[mask]
    G_v = G[mask]
    psi = float(np.sum(I_v * G_v) / np.sum(G_v**2))

    pred = psi * G_v
    rms = float(np.sqrt(np.mean((I_v - pred)**2)) / np.mean(I_v))

    psi_values[i] = psi
    fit_qualitaet[i] = rms

    status = '✓' if rms < 0.15 else '!' if rms < 0.30 else '✗'
    print(f"  LED{led_nr:02d} {LED_NAMES[led_nr]:<11s} "
          f"{psi:>15.2e} {n_valid:>10d} {rms*100:>8.2f}% {status}")

# ── Statistik ───────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Ψⁱ-Statistik:")
print(f"  Mittelwert: {psi_values.mean():.2e}")
print(f"  Min:        {psi_values.min():.2e} (LED "
      f"{int(np.argmin(psi_values))+1})")
print(f"  Max:        {psi_values.max():.2e} (LED "
      f"{int(np.argmax(psi_values))+1})")
print(f"  Streuung:   {psi_values.std()/psi_values.mean()*100:.1f}%")

print(f"\nFit-Qualität:")
print(f"  Mittel:     {fit_qualitaet.mean()*100:.2f}%")
print(f"  Worst:      {fit_qualitaet.max()*100:.2f}%")

# ── Visualisierung ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = [f"{i+1}\n{LED_NAMES[i+1]}" for i in range(12)]

axes[0].bar(range(12), psi_values, color='steelblue')
axes[0].set_xticks(range(12))
axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('Ψ (LED-Intensität, relative Einheit)')
axes[0].set_title('LED-Intensitäten Ψⁱ (korrigiert)')
axes[0].axhline(psi_values.mean(), color='red', ls='--',
                label=f'Mittel = {psi_values.mean():.2e}')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(range(12), fit_qualitaet * 100,
            color=['green' if q < 0.10 else 'orange' if q < 0.20 else 'red'
                   for q in fit_qualitaet])
axes[1].set_xticks(range(12))
axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Relativer RMS-Fehler [%]')
axes[1].set_title('Fit-Qualität (kleiner = besser)')
axes[1].axhline(10, color='black', ls='--', alpha=0.5,
                label='10% (gut)')
axes[1].axhline(20, color='red', ls='--', alpha=0.5,
                label='20% (Grenze)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('psi_kalibrierung_v2.png', dpi=100)
plt.show()

# ── Speichern ──────────────────────────────────────────────
SAVE_PFAD = os.path.join(SAVE_DIR, 'led_kalibrierung_v2.npz')
np.savez(SAVE_PFAD,
         led_positions=LED_POSITIONS,
         led_hauptachsen=LED_HAUPTACHSEN,
         psi=psi_values,
         mu=MU,
         fit_qualitaet=fit_qualitaet,
         GSD=GSD,
         achsen_konvention='u→x (positiv), v→-y (invertiert)')
print(f"\nGespeichert: {SAVE_PFAD}")