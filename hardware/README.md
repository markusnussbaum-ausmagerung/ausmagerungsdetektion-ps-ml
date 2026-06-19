# Hardware der Fotobox

Dieser Ordner dokumentiert die selbstgebaute, lichtdichte Fotobox, mit der die
Aufnahmen fuer das Photometric Stereo entstehen, sowie die Skripte zur
Steuerung und Kalibrierung.

## Aufbau in Kurzform

Die Box besitzt eine Aufnahmeflaeche von 56 mal 42 cm. Der fuer die Auswertung
verwendete Ausschnitt betraegt 39 mal 39 cm (1500 mal 1500 Pixel). Zwoelf LEDs
sitzen in einer Halbkugelgeometrie ueber der Probe und werden einzeln
geschaltet, sodass je Aufnahme genau eine Lichtrichtung aktiv ist.

## Stueckliste

Hersteller-Datenblaetter sind aus urheberrechtlichen Gruenden nicht Teil dieses
Repositoriums. Die folgende Liste nennt die verbauten Komponenten mit
Modellbezeichnung, sodass sich die zugehoerigen Datenblaetter beim jeweiligen
Hersteller auffinden lassen.

| Komponente | Typ / Modell | Anzahl | Funktion |
|---|---|---|---|
| LED | 3 W, 4000 K, Stern-PCB 20 mm | 12 | Beleuchtung je Lichtrichtung |
| LED-Treiber | Mean Well LDD-700LW (700 mA Konstantstrom) | 12 | je eine LED konstant bestromt |
| Steuerrechner | Raspberry Pi 4 Model B | 1 | GPIO-Steuerung und Ablauf |
| Relaismodul | Yizhet 8-Kanal (Active-LOW, NC-Klemmen) | 2 | schaltet die LEDs einzeln |
| Kamera | Google Pixel 9a | 1 | RAW/DNG-Aufnahme |
| Aufnahme-App | ProShot | 1 | manuelle RAW-Aufnahme, ADB-gesteuert |
| Laststromquelle | TalentCell LiFePO4 12 V / 6,6 Ah | 1 | Versorgung der LED-Treiber (galvanisch getrennt) |
| Steuerstromquelle | 5-V-Powerbank | 1 | Versorgung des Raspberry Pi |

## Steuerungskonzept

Die Relais arbeiten mit Active-LOW-Logik an den NC-Klemmen (normally closed).
Beim Booten des Raspberry Pi sind dadurch alle LEDs aus (Fail-Safe). Ein
GPIO-Pegel HIGH schaltet das Relais aus und damit die LED aus, ein Pegel LOW
schaltet die LED ein. Die Kamera wird per ADB ueber USB-C ausgeloest.

## Skripte

`fotobox_steuerung.py` laeuft auf dem Raspberry Pi. Es schaltet die zwoelf LEDs
nacheinander und loest je Lichtrichtung eine RAW-Aufnahme aus. Das Skript
benoetigt `RPi.GPIO` sowie eine ADB-Verbindung zur Kamera und ist nur auf der
realen Hardware lauffaehig.

`kalibrierung_cos4_weissreferenz.py` berechnet aus einer Weissreferenz die
cos^4-Vignettierungskorrektur und speichert sie fuer die Pipeline.

`kalibrierung_psi_led.py` bestimmt die LED-Geometrie (Psi-Kalibrierung) aus
Weissreferenzaufnahmen und erzeugt die LED-Kalibrierdatei, die N01 verwendet.

Die beiden Kalibrierskripte folgen demselben Pfad-Schema wie die Notebooks
(siehe Haupt-README). Sie erzeugen die Kalibrierdateien, die unter
`03_Hardware_Fotobox/05_Kalibrierung/` abgelegt werden.

## Ergaenzende eigene Dokumente

Weitere selbst erstellte Unterlagen zur Box (Schaltungskonzept, Boxgeometrie,
Einkaufsliste) koennen hier als PDF beigelegt werden. Sie sind eigenes Werk und
unterliegen keiner fremden Lizenz.
