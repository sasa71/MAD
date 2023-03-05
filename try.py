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

image = cv2.imread("/home/sasa/Scrivania/Test/img000064.png")
mediapipeMesh.showGraph(image, "manhattan") 
 