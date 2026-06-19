#!/usr/bin/env python3
"""
Fotobox Photometric Stereo – Automatisiertes Aufnahmescript
Läuft auf Raspberry Pi 4
Steuert 12 LEDs per GPIO → Relaismodul → LDD-700LW → LED
Kamera: Google Pixel 9a per ADB

Steuerungskonzept (active-LOW Relaismodul mit Optokoppler, NC-Klemme):
───────────────────────────────────────────────────────────────────────
Verkabelung Relais-Ausgang:
  COM → DIM-Eingang LDD (weißes Kabel)
  NC  → Minus-Schiene Stromverteiler (GND)
  NO  → nicht belegt

GPIO HIGH (3.3V) → Relais AUS → COM+NC verbunden → DIM auf GND → LED AUS
GPIO LOW  (0V)   → Relais AN  → COM+NC getrennt  → DIM offen   → LED AN
"""

import RPi.GPIO as GPIO
import subprocess
import time
import os
from datetime import datetime

# ── GPIO Konfiguration ──────────────────────────────────────
LED_PINS = {
    1:  17, 2:  18, 3:  27,
    4:  22, 5:  23, 6:  24,
    7:  25, 8:   8, 9:   7,
    10: 12, 11: 16, 12: 20,
}

LED_NAMES = {
    1:  "Links_20",  2:  "Links_35",  3:  "Links_60",
    4:  "Rechts_20", 5:  "Rechts_35", 6:  "Rechts_60",
    7:  "Vorne_20",  8:  "Vorne_35",  9:  "Vorne_60",
    10: "Hinten_20", 11: "Hinten_35", 12: "Hinten_60",
}

DELAY_LED_ON    = 0.5   # Sek. warten nach LED an (Kamera stabilisieren)
DELAY_AFTER_PIC = 0.3   # Sek. warten nach Foto


# ── GPIO Setup ───────────────────────────────────────────────
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in LED_PINS.values():
        # OPTIMIERT: initial=GPIO.HIGH verhindert das Relais-Zucken beim Setup
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
    print("GPIO initialisiert – alle Relais aus, alle LEDs aus")

def all_leds_off():
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.HIGH)

def led_on(led_num):
    all_leds_off()
    GPIO.output(LED_PINS[led_num], GPIO.LOW)

def led_off(led_num):
    GPIO.output(LED_PINS[led_num], GPIO.HIGH)


# ── ADB Kamera ───────────────────────────────────────────────
def adb_check():
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    devices = [l for l in lines[1:] if 'device' in l]
    if not devices:
        raise RuntimeError("Kein Android-Geraet via ADB! USB-Kabel pruefen.")
    print(f"ADB Geraet gefunden: {devices[0]}")

def prepare_camera():
    """Weckt das Display auf und startet gezielt die Open Camera App (Nur 1x pro Standort nötig)"""
    # Handy aufwecken (falls Bildschirm schwarz)
    subprocess.run(['adb', 'shell', 'input', 'keyevent', 'KEYCODE_WAKEUP'], capture_output=True)
    time.sleep(0.5)
    
    # OPTIMIERT: Startet explizit "ProShot" statt der Standard-Kamera
    subprocess.run(['adb', 'shell', 'monkey', '-p',
    'com.riseupgames.proshot2', '-c',
    'android.intent.category.LAUNCHER', '1'],
    capture_output=True)
    time.sleep(2.0) # Der App kurz Zeit zum Starten geben

def take_raw_photo(filename):
    """
    Drückt virtuell den Auslöser, wartet intelligent auf das NEUE Bild in ProShot
    und kopiert es. Das Original bleibt als Backup auf dem Smartphone!
    """
    # DEIN NEUER PFAD:
    cam_path = '/storage/emulated/0/DCIM/Masterarbeit/*.dng'
    
    # 1. STATUS QUO CHECK: Welches ist VOR dem Klick das neueste Bild?
    res_before = subprocess.run(['adb', 'shell', 'ls', '-t', cam_path], capture_output=True, text=True)
    latest_before = res_before.stdout.strip().split('\n')[0] if res_before.stdout else ""

    # 2. AUSLÖSEN: Drückt die Leiser-Taste -> Macht das Foto
    subprocess.run(['adb', 'shell', 'input', 'keyevent', 'KEYCODE_VOLUME_DOWN'], capture_output=True)
    
    # 3. WARTEN AUF DIE NEUE DATEI (Polling)
    max_retries = 10  # Maximal 10 Sekunden warten
    new_latest = ""
    
    for _ in range(max_retries):
        time.sleep(1.0) # Jede Sekunde einmal nachschauen
        res_after = subprocess.run(['adb', 'shell', 'ls', '-t', cam_path], capture_output=True, text=True)
        current_latest = res_after.stdout.strip().split('\n')[0] if res_after.stdout else ""
        
        # Prüfen, ob eine Datei gefunden wurde UND ob sie einen ANDEREN Namen hat als vorher
        if current_latest and ".dng" in current_latest and current_latest != latest_before:
            new_latest = current_latest
            break # Das neue Bild ist da! Schleife abbrechen.

    # 4. KOPIEREN (Ohne Löschen = 100% Backup auf dem Handy)
    if new_latest:
        # Foto auf den Raspberry Pi kopieren
        subprocess.run(['adb', 'pull', new_latest.strip(), filename], capture_output=True)
        print(f"  Foto geladen: {os.path.basename(filename)}")
        return True
        
    print("  FEHLER: OpenCamera hat innerhalb von 10 Sekunden kein neues DNG gespeichert!")
    return False


# ── Messung ──────────────────────────────────────────────────
def capture_white_reference(output_dir):
    print("\n=== WEISSREFERENZ ===")
    print("Weisses Blatt Papier flach in die Box legen.")
    input("Bereit? Enter druecken...")
    
    ref_dir = os.path.join(output_dir, 'weissreferenz')
    os.makedirs(ref_dir, exist_ok=True)
    
    prepare_camera() # Kamera EINMAL vorbereiten
    
    for led_num in range(1, 13):
        print(f"Ref LED {led_num:2d}/12 ({LED_NAMES[led_num]})...", end=' ')
        led_on(led_num)
        time.sleep(DELAY_LED_ON)
        
        fname = os.path.join(ref_dir, f'Weissreferenz_LED{led_num:02d}_{LED_NAMES[led_num]}.dng')
        take_raw_photo(fname)
        
        all_leds_off()
        time.sleep(DELAY_AFTER_PIC)
        
    all_leds_off()
    print("\nWeissreferenz abgeschlossen.")
    input("Papier entfernen und Enter druecken...")


def capture_measurement(output_dir, standort_nr, label_text):
    """
    12 RAW-Fotos aufnehmen und in einem gelabelten Ordner speichern.
    """
    # NEU: Der Ordnername enthält jetzt das Label!
    meas_dir = os.path.join(output_dir, f'standort_{standort_nr:03d}_{label_text}')
    os.makedirs(meas_dir, exist_ok=True)
    print(f"\n=== Standort {standort_nr:03d} [{label_text.upper()}] ===")
    
    prepare_camera()
    
    success = 0
    for led_num in range(1, 13):
        print(f"  LED {led_num:2d}/12 ({LED_NAMES[led_num]})...", end=' ')
        led_on(led_num)
        time.sleep(DELAY_LED_ON)
        
        # Der Dateiname bleibt sauber, der Ordnername verrät ja schon das Label
        fname = os.path.join(meas_dir, f'Standort{standort_nr:03d}_LED{led_num:02d}_{LED_NAMES[led_num]}.dng')
        if take_raw_photo(fname):
            success += 1
            
        all_leds_off()
        time.sleep(DELAY_AFTER_PIC)
        
    all_leds_off()
    print(f"Fertig: {success}/12 Fotos OK")
    return success == 12


# ── Hauptprogramm ────────────────────────────────────────────
def main():
    print("=" * 50)
    print("FOTOBOX PHOTOMETRIC STEREO")
    print("=" * 50)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    # Ausgabeordner auf dem Raspberry Pi (Standard-Benutzer pi)
    output_dir = f'/home/pi/messungen/messung_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ordner: {output_dir}")
    
    setup_gpio()
    adb_check()
    
    do_ref = input("\nWeissreferenz aufnehmen? (j/n): ")
    if do_ref.lower() == 'j':
        capture_white_reference(output_dir)
        
    standort_nr = 1
    
    # NEU: CSV-Logbuch erstellen und Kopfzeile schreiben
    csv_pfad = os.path.join(output_dir, 'dataset_labels.csv')
    with open(csv_pfad, 'w') as f:
        f.write("Standort_Nr,Ordnername,Label_Klasse,Timestamp\n")
        
    print("\nMessablauf gestartet. Ctrl+C zum Beenden.")
    
    # Definition der Label-Kategorien
    label_map = {
        "0": "intakt",
        "1": "leicht",
        "2": "mittel",
        "3": "schwer",
        "x": "sonstiges" # Falls mal ein Riss oder Flickstelle da ist
    }
    
    try:
        while True:
            print(f"\n--- Vorbereitung Standort {standort_nr:03d} ---")
            print("Zustand bewerten: [0] Intakt | [1] Leicht | [2] Mittel | [3] Schwer | [x] Sonstiges")
            
            # 1. Label abfragen (mit Fehlerabfangung auf der Straße)
            user_input = ""
            while user_input not in label_map:
                user_input = input("Bitte Zahl eingeben: ").strip().lower()
                
            label_text = label_map[user_input]
            
            # 2. Warten bis du die Box richtig hingestellt hast
            input(f"\nBox für Standort {standort_nr:03d} positionieren und Enter drücken...")
            
            # 3. Messung starten (Label übergeben)
            ok = capture_measurement(output_dir, standort_nr, label_text)
            
            if ok:
                # NEU: Wenn erfolgreich, in die CSV-Datei schreiben
                with open(csv_pfad, 'a') as f:
                    zeit = datetime.now().strftime('%H:%M:%S')
                    ordner = f'standort_{standort_nr:03d}_{label_text}'
                    f.write(f"{standort_nr},{ordner},{label_text},{zeit}\n")
                    
                standort_nr += 1
            else:
                retry = input("Fehler – nochmal versuchen? (j/n): ")
                if retry.lower() == 'j':
                    print("Wiederhole Standort...")
                else:
                    standort_nr += 1
                    
    except KeyboardInterrupt:
        print(f"\n\nMessung beendet. Standorte: {standort_nr - 1}")
        print(f"Daten: {output_dir}")
    finally:
        all_leds_off()
        GPIO.cleanup()
        print("GPIO aufgeraeumt. Alle Relais aus, alle LEDs aus.")

if __name__ == '__main__':
    main()