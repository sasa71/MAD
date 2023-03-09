import time
import gc
import numpy
import shutil
import pandas
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
from memory_profiler import profile

#start_time = time.time()
count = 1
csv_file = open('Workout.csv', mode='w')
nomicolonne = ['Nome immagine', 'Etichetta', 'Manhattan', 'Curvature']
writer = csv.DictWriter(csv_file, fieldnames=nomicolonne)
writer.writeheader()
csv_file.close
del nomicolonne
gc.collect()
folder_dir = "/home/sasa/Scrivania/Tir o cigno/SMDD_release_train/m15k_t"
for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        csv_file = open('Workout.csv', mode='a')
        image = cv2.imread(folder_dir+"/"+images)
        G = mediapipeMesh.buildGraph(image, "manhattan")
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy() 
        #print ("\n\n",images,"\n")
        x = (networkx.get_edge_attributes(G, "weight").values())
        #print(list(x))
        #ricciData.show_results(G_orc)
        if("morphed" in images):
            t = "Morphed"
        else:
            t = "Bona Fide"
        writer.writerow({'Nome immagine': images, 'Etichetta': t, 'Manhattan': list(x), 'Curvature': ricciData.r_file(G_orc)})
        count += 1
        csv_file.close
        del G
        gc.collect()
        del orc
        gc.collect()
        del G_orc
        gc.collect()
        del x
        gc.collect()
        del image
        gc.collect()
print("Immagini ", count-1)
#print("--- %s seconds ---" % (time.time() - start_time))

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

"""
df = pandas.DataFrame({
    "Nome Immagine":[],
            "Etichetta":[],
            "Manhattan":[],
            "Curvature":[]
            
})
folder_dir = "/home/sasa/Scrivania/Test"
for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        image = cv2.imread(folder_dir+"/"+images)
        graph = mediapipeMesh.buildGraph(image, "manhattan")
        x = (networkx.get_edge_attributes(graph, "weight").values())
        orc = OllivierRicci(graph, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy() 
        if("morphed" in images):
            t = "Morphed"
        else:
            t = "Bona Fide"  
        df = df.append({
            "Nome Immagine":images,
            "Etichetta":t,
            "Manhattan":list(x),
            "Curvature":ricciData.r_file(G_orc)
            }, ignore_index=True)
    
#df.to_csv("Workout.csv", mode='a')
"""