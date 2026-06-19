# Ausmagerungsdetektion auf Asphalt mit Photometric Stereo und Machine Learning

Dieses Repositorium enthaelt den Quellcode der Verarbeitungspipeline zur
Masterarbeit von Markus Nussbaum (Bauingenieurwesen, Hochschule Campus Wien, 2026).

Die Arbeit untersucht, wie zuverlaessig sich Ausmagerungen und beginnende
Kornausbrueche auf Asphaltoberflaechen mit bildbasierten, KI-gestuetzten
Verfahren erkennen und klassifizieren lassen. Der Ansatz kombiniert Photometric
Stereo (Rekonstruktion der Oberflaechennormalen aus Mehrfachbeleuchtung) mit
einer merkmalsbasierten Klassifikation.

Der Charakter der Arbeit ist ein Machbarkeitsnachweis der gesamten
Verfahrenskette. Es besteht kein Anspruch auf Generalisierung oder
statistische Repraesentativitaet (sieben Standorte, eine Deckschicht AC 8).

## Wichtiger Hinweis zu den Daten

Die DNG-Rohaufnahmen sind nicht Teil dieses Repositoriums. Sie sind zu gross
und tragen einen Ortsbezug. Enthalten ist stattdessen ein kleiner
Beispieldatensatz (ein Standort als Zwischenprodukt `ps_output.npz`), mit dem
sich die Schritte ab N02 nachvollziehen lassen. Die beiden ersten Schritte
N01 und N01b benoetigen die Rohbilder und sind hier als dokumentierter
Verfahrensschritt enthalten, laufen am Beispiel aber nicht mit.

## Aufbau des Repositoriums

```
.
├── README.md
├── LICENSE                       MIT-Lizenz
├── requirements.txt              benoetigte Python-Pakete
├── .gitignore
│
├── notebooks/                    die Verarbeitungspipeline N01 bis N04
│
├── 03_Hardware_Fotobox/
│   └── 05_Kalibrierung/          Kamera-, LED- und GSD-Kalibrierdateien (.npz)
│
├── 04_Daten_Felderhebung/        Beispieldatensatz (ein Standort)
│   ├── 00_Master/master.csv      Standort-Metadaten (ortsbezogene Spalten entfernt)
│   ├── 02_Verarbeitet/           ps_output.npz des Beispielstandorts
│   └── 03_Annotation/            zugehoerige CVAT-Annotation (XML)
│
└── hardware/                     Steuerungs- und Kalibrierskripte der Fotobox
```

Die Ordnernamen unter `03_` und `04_` entsprechen der in der Masterarbeit
beschriebenen Projektablage. Die vollstaendige Datenmenge der sieben Standorte
liegt ausserhalb des Repositoriums.

## Installation

Die Pipeline ist fuer Python 3 ausgelegt. Die benoetigten Pakete lassen sich
mit folgendem Befehl installieren:

```
pip install -r requirements.txt
```

Die Notebooks sind in Google Colab entwickelt worden und laufen dort
unveraendert. Lokal funktionieren sie ebenso, sobald die Pakete installiert
sind und die Daten am erwarteten Ort liegen.

## Ausfuehrung und Datenpfade

Jedes Notebook entscheidet zu Beginn selbst, wo es laeuft. In Colab haengt es
Google Drive ein und zeigt auf die dortige Projektablage. Lokal verwendet es
das aktuelle Verzeichnis als Wurzel. Der Wurzelpfad laesst sich ueber die
Umgebungsvariable `MA_BASIS` ueberschreiben.

```python
# Beispiel lokal: Daten liegen im aktuellen Ordner
# (Standard, keine Einstellung noetig)

# Beispiel: Daten liegen woanders
import os
os.environ["MA_BASIS"] = "/pfad/zu/den/daten"
```

Wer den Beispieldatensatz nutzt, legt ihn in die oben gezeigte Ordnerstruktur
und startet ab N02a.

## Die Pipeline N01 bis N04

Die Notebooks bauen aufeinander auf und werden in dieser Reihenfolge
ausgefuehrt.

1. **N01** rechnet aus zwoelf DNG-Aufnahmen je Standort die Normalenkarte und
   die Albedo (Near-Field Photometric Stereo nach Queau).
2. **N01b** mittelt die mehrfachen Aufnahmen je Standort (Triplets) zu einem
   einzigen PS-Ergebnis.
3. **N02a** rastert die CVAT-Annotation zu Masken und vergibt je Bildausschnitt
   (Patch) ein Soll-Label.
4. **N02b** berechnet je Patch die zehn Merkmale fuer die Modellierung plus
   Diagnose-Groessen.
5. **N02c** ist ein nicht produktiver Plausibilitaetstest auf dem Pilotstandort
   und gehoert nicht zur eigentlichen Pipeline.
6. **N02d** fuehrt Merkmale, Labels und Standort-Metadaten zum
   Trainingsdatensatz zusammen.
7. **N02e** enthaelt die Diskriminanzanalyse des Gesamtdatensatzes
   (Effektstaerken, das primaere quantitative Ergebnis der Arbeit).
8. **N03** trainiert und evaluiert die binaere Klassifikation (Random Forest
   und SVM, Leave-One-Group-Out je Standort).
9. **N04** wendet das Modell zur Veranschaulichung auf den Pilotstandort an
   (qualitative Demonstration, kein Leistungsmass).

## Datenschutz

Die Standort-Metadaten (`master.csv`) sind vor der Veroeffentlichung um die
ortsbezogenen Spalten bereinigt (Koordinaten, Strassenname). Die raeumliche
Naehe der Standorte bleibt ueber die gemeinsame Cluster-Kennung erkennbar, die
genaue Lage ist nicht enthalten.

## Methodik und Hilfsmittel

Die Photometric-Stereo-Methodik dieser Arbeit beruht auf dem Near-Field-Modell
von Queau et al. (2017/2018), das in eigenem Code umgesetzt ist. Die
vollstaendige Quellenangabe findet sich in der Masterarbeit.

Bei der Erstellung des Programmcodes sind KI-gestuetzte Werkzeuge eingesetzt
worden. Konzeption der Verfahrenskette, Wahl der Methodik und Parameter, Tests
sowie die Integration in Hardware und Datenpipeline liegen beim Autor.

## Lizenz

Der Code steht unter der MIT-Lizenz (siehe `LICENSE`).

## Zitation

Der dauerhaft zitierbare Versions-DOI wird beim Einfrieren des Abgabestands
ueber Zenodo erzeugt und an dieser Stelle ergaenzt.

```
DOI: https://doi.org/10.xxxx/zenodo.xxxxxxx
```

## Autor

Markus Nussbaum, Hochschule Campus Wien, 2026.
