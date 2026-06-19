# ============================================================
# PIPELINE-SCHRITT 1: cos⁴(α)-Korrektur (Quéau Gl. 2.10-2.12)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

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

# ── Kamerakalibrierung laden (v2) ────────────────────────────
KAL_PFAD = os.path.join(PROJEKT,
            '03_Hardware_Fotobox/05_Kalibrierung/'
            '001_Kamerakalibrierung/kalibrierung_pixel9a_v2.npz')
kal = np.load(KAL_PFAD)
K = kal['K']
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Mittelwert der Brennweiten (fx, fy fast identisch)
f = (fx + fy) / 2
print(f"Brennweite f = {f:.2f} px (Mittel aus fx, fy)")
print(f"Hauptpunkt (cx, cy) = ({cx:.2f}, {cy:.2f})")

# ── Bildgröße nach Postprocess (Hochkant) ───────────────────
H, W = 4000, 3000

# ── Pixelraster relativ zum Hauptpunkt ──────────────────────
v_grid, u_grid = np.indices((H, W))
u_rel = u_grid - cx
v_rel = v_grid - cy

# ── cos(α) für jeden Pixel ──────────────────────────────────
distance_squared = u_rel**2 + v_rel**2 + f**2
cos_alpha = f / np.sqrt(distance_squared)

# ── cos⁴(α) als Korrekturmatrix ─────────────────────────────
cos4_alpha = cos_alpha ** 4

print(f"\ncos⁴(α) Statistik:")
print(f"  Bildmitte:         {cos4_alpha[int(cy), int(cx)]:.4f}")
print(f"  Minimum (Ecke):    {cos4_alpha.min():.4f}")
print(f"  Maximum:           {cos4_alpha.max():.4f}")
print(f"  Helligkeitsabfall: "
      f"{(1 - cos4_alpha.min())*100:.1f} %")

# ── Visualisierung ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap der cos⁴-Werte
im0 = axes[0].imshow(cos4_alpha, cmap='viridis', vmin=0.5, vmax=1.0)
axes[0].plot(cx, cy, 'r+', markersize=15, mew=2,
             label=f'Hauptpunkt ({cx:.0f}, {cy:.0f})')
axes[0].set_title('cos⁴(α) – Helligkeit relativ zur Bildmitte')
axes[0].set_xlabel('Pixel x')
axes[0].set_ylabel('Pixel y')
axes[0].legend(loc='upper right')
plt.colorbar(im0, ax=axes[0], label='cos⁴(α)')

# Korrekturfaktor (= 1 / cos⁴) als Multiplikator zum Wiederherstellen
korrektur_faktor = 1.0 / cos4_alpha
im1 = axes[1].imshow(korrektur_faktor, cmap='hot',
                      vmin=1.0, vmax=korrektur_faktor.max())
axes[1].plot(cx, cy, 'b+', markersize=15, mew=2,
             label=f'Hauptpunkt')
axes[1].set_title(f'1/cos⁴(α) – Korrekturfaktor '
                   f'(max {korrektur_faktor.max():.3f})')
axes[1].set_xlabel('Pixel x')
axes[1].set_ylabel('Pixel y')
axes[1].legend(loc='upper right')
plt.colorbar(im1, ax=axes[1], label='Korrekturfaktor')

plt.tight_layout()
plt.savefig('cos4_korrektur.png', dpi=100)
plt.show()

# ── Speichern für spätere Pipeline-Schritte ────────────────
SAVE_PFAD = os.path.join(PROJEKT,
             '03_Hardware_Fotobox/05_Kalibrierung/'
             '008_cos4_korrektur/cos4_korrektur.npz')
import os
os.makedirs(os.path.dirname(SAVE_PFAD), exist_ok=True)
np.savez(SAVE_PFAD,
         cos4_alpha=cos4_alpha,
         korrektur_faktor=korrektur_faktor,
         cx=cx, cy=cy, f=f, H=H, W=W)
print(f"\nGespeichert: {SAVE_PFAD}")