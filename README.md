# SAMSON Tree Tracking

Dieses Projekt entstand im Rahmen des Grundprojektes an der HAW. 
Die Aufgabe bestand darin, aus einer aufeinanderfolgenden Reihe von Bildern ein Bild für jeden individuellen Baum auszuschneiden. Dies erforderte die Erkennung und Verfolgung des Baumes auf den Bildern.

## Table of Contents

- [Bemerkung](#Bemerkung)
- [Funktionsweise](#Funktionsweise)
- [YOLOV8](#YOLOV8)
- [Installation](#Installation)
- [Verwendung](#Verwendung)




## Bemerkung

Das trainierte YOLOv8-Modell erweist sich als unzureichend, robust und präzise, um es weiterhin einzusetzen. 
Bis zur Messfahrt am 7. Juli 2023 (Aufzeichnungsnr. 37, siehe Aufnahmen_Übersicht.xlsx) hat es funktioniert. Jedoch treten bei dieser und anderen Messfahrten im Sommer wiederholt Probleme auf: Es erfolgt entweder die falsche Identifikation von Bäumen oder es werden in aufeinanderfolgenden Bildern keine Bäume erkannt, obwohl sie deutlich sichtbar sind.
Im Winter ist dies aber durchaus einsetzbar.

## Funktionsweise
Das Konzept basiert darauf, in jedem Frame YOLOv8 anzuwenden, um die Bäume zu erkennen. Zwischen zwei aufeinanderfolgenden Frames wird die Kamerabewegung auf Bildebene geschätzt, um die Position der bisher verfolgten Bäume im aktuellen Bild zu aktualisieren. Die neu erkannten Bäume werden dann anhand der IoU (Intersection over Union) oder der Distanz mit den bereits verfolgten Bäumen verglichen, um festzustellen, ob es sich um denselben Baum handelt. Auf diese Weise erfolgt das Tracking der Bäume.
## YOLOV8
Es wurde ein [YOLOv8](https://github.com/ultralytics/ultralytics)-Modell der Version small trainiert, mit Bilddaten aus allen Berschauer aufnahmen bis zur Nr 40 (siehe Aufnahmen_Übersicht.xlsx). Das Modell wurde speziell darauf trainiert, die Baumstämme zu erkennen. Das bedeutet, dass bei der Annotation die Bounding Box um den Baumstamm gelegt wurde.

## Installation
Zur Verwendung des Skriptes müssen einige Pakete installiert sein im Zusammenhang mit Python (3.8):
- [Pytorch](https://pytorch.org/get-started/locally/)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- pip install argparse
- pip install math
- ip install os
- pip install cv2

Für das Verwenden des navsat Skriptes sollte noch ROS installiert sein.


## Verwendung
Falls noch nicht gemacht, sollte aus den Messfahrten Videos extrahiert werden. Mithilfe des Skriptes von David Berschauer (navsat_to_kml.py) können die GPS Daten aus den Rosbags geladen werden, womit dann der Videoaufzeichnung die entsprechende Reihe, der Ort und der Bereich zugeordnet werden kann.
Die Videos sollten so zugeschnitten sein, dass sie kurz vor der Reihe anfangen und kurz vor der Reihe aufhören, damit die ausgeschnittenen Bilder den Reihen zuzuordnen sind.
Die Videos könnten dann so heißen:
- 2023-03-01_14-59-46_Horizontal 1.bagReihe9Rechts.avi

Wichtig sind aber nur die Zeitangaben, die in diesem Format sein müssen und wie das Video dann mindestens heißen müsste.

Danach kann dann das Tracking Skript aufgerufen werden mit
- python3 track_trees.py "Pfad zum Video" True/False

Dem Skript übergibt man dann den Pfad zum Video und ob ein Ergebnis Video gemacht werden soll, in dem man sich das Tracking im Nachhinein nochmal angucken kann.

Die ausgeschnittenen Bilder sind dann im Output Ordner zu finden, mit dem Namen des Videos.

Falls noch nicht gemacht, kann mit "make_folder.py" die Ordner erstellt werden für die jeweilige Reihe.

- python3 make_folder.py 174 "Reihe 9" Berschauer A

Dem Skript übergibt man dann, wie viele Bäume in dieser Reihe sind, die Reihe, den Ort und den Bereich.

Mit dem Skript "insert_cut_images.py" können dann die ausgeschnittenen Bilder dort eingefügt werden.

- python3 insert_cut_images.py output/2023-04-26_17-49-24_Elstar_1_laenge_left.svo.avi/ False "Reihe 9" Right Berschauer A

Dem Skript wird den Pfad zum Ordner übergeben, ob die Messfahrt von rechts nach links (False) oder von links nach rechts (True) aufgenommen wurde (siehe qgis), die entsprechende Reihe, ob die Reihe von Rechts (oben) oder Links (unten) aufgenommen wurde (siehe qgis) den Ort und entsprechenden Bereich.