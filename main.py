import numpy
import shutil
import pandas
import cv2
import csv
import sys
import os
import mediapipeMesh
import networkx
import importlib
import igraph as ig
import ricciData
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
from os import listdir

nomi = []
count = 1
csv_file = open('WebMorph.csv', mode='w')
nomicolonne = ['Nome immagine', 'Etichetta', 'Manhattan', 'Curvature']
writer = csv.DictWriter(csv_file, fieldnames=nomicolonne)
writer.writeheader()
csv_file.close
folder_dir = "/home/sasa/Scrivania/Tir o cigno/FRLL-Morphs/facelab_london/morph_webmorph"
for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        #print(images)
        csv_file = open('WebMorph.csv', mode='a')
        image = cv2.imread(folder_dir+"/"+images)
        G = mediapipeMesh.buildGraphNorm(image, "manhattan")
        if (G is None):
            nomi.append(images)
            print(nomi)
        else:
            orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
            orc.compute_ricci_curvature()
            G_orc = orc.G.copy() 
            x = (networkx.get_edge_attributes(G, "weight").values())
            if("morphed" in images):
                t = 1
            else:
                t = 0
            writer.writerow({'Nome immagine': images, 'Etichetta': 1, 'Manhattan': list(x), 'Curvature': ricciData.r_file(G_orc)})
            count += 1
            csv_file.close
            print("Immagini ", count-1)
