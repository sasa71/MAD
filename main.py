import cv2
import csv
import sys
import os
import mediapipeMesh
import dlibMesh
import networkx
import importlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import igraph as ig
import delaunay
import ricciData
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np
from os import listdir
count=1
csv_file=open('Workout.csv', mode='w')
nomicolonne = ['Indice','Nome immagine', 'Etichetta', 'Manhattan','Curvature']
writer = csv.DictWriter(csv_file, fieldnames=nomicolonne)
writer.writeheader()
folder_dir ="/home/sasa/Scrivania/Test"
for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        image = cv2.imread(folder_dir+"/"+images)
        G=mediapipeMesh.buildGraph(image, "manhattan") 
        orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy() 
        print ("\n\n",images,"\n")
        x=(networkx.get_edge_attributes(G_orc, "weight").values())
        print(list(x))
        ricciData.show_results(G_orc)
        if("morphed" in images):
            t="Morphed"
        else:
            t="Bona Fide"          
        writer.writerow({'Indice': count,'Nome immagine': images,'Etichetta': t, 'Manhattan': list(x),'Curvature':ricciData.r_file(G_orc)})
        count+=1
print("Immagini ",count-1)
csv_file.close
#image = cv2.imread("/home/sasa/Scrivania/Test")
#mediapipeMesh.showGraph(image, "euclidean")
#dlibMesh.showGraph(image, "euclidean")
#g=mediapipeMesh.buildFormanRicciGraph(image, "euclidean")
#print(networkx.info(G))
#orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
#orc.compute_ricci_curvature()
#G_orc = orc.G.copy()  # save an intermediate result


#ricciData.show_results(G_orc)
#print("/---------/")


#print(networkx.info(G_orc))
#for edge in G.size:
#x=networkx.get_edge_attributes(G_orc, "weight")
#print(networkx.get_edge_attributes(G, "weight"))
#(362, 381): 11.180339887498949, (107, 374): 109.59014554237986, (107, 381): 89.1852005660132, (374, 381): 22.090722034374522} euclidean
#(362, 381): 11.180339887498949, (107, 374): 109.59014554237986, (107, 381): 89.1852005660132, (374, 381): 22.090722034374522} Ollivier-euclidean
#(362, 381): 5.5280884372566064e-05, (107, 374): 0.0006409809759132523, (107, 381): 7.671928445551757e-05, (374, 381): 0.0002742192151178324} coseno
#(362, 381): 5.5280884372566064e-05, (107, 374): 0.0006409809759132523, (107, 381): 7.671928445551757e-05, (374, 381): 0.0002742192151178324} Ollivier-cosine

