import math
import networkx
import cv2
from scipy.spatial import distance
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import dlib


def buildGraph(image, distType):
    detector = dlib.get_frontal_face_detector()
    lm_predictor = dlib.shape_predictor("/home/luigi/Scaricati/pythonProject/shape_predictor_68_face_landmarks.dat")

    faces = detector(image)

    if (len(faces) < 1):
        return

    graph = networkx.Graph()

    face_landmarks = lm_predictor(image, faces[0])
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        pos = (x, y)
        graph.add_node(n, pos=pos)

    nodesPosition = networkx.get_node_attributes(graph, "pos")

    for faceEdge in FACE_EDGES:

        match distType:

            case 'euclidean':
                weight = distance.euclidean(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])
            case 'cosine':
                weight = distance.cosine(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])
            case _:
                weight = distance.euclidean(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])

        if (weight != 0):
            graph.add_edge(faceEdge[0], faceEdge[1], weight=weight)
        else:
            graph.add_edge(faceEdge[0], faceEdge[1], weight=0.001)

    return graph


def showGraph(image,a):
    graph = buildGraph(image, a)

    nodesPositions = networkx.get_node_attributes(graph, "pos")

    for faceEdge in FACE_EDGES:
        cv2.line(image, nodesPositions[faceEdge[0]], nodesPositions[faceEdge[1]], (0, 0, 255), 1)

    for faceLandmark in FACE_LANDMARKS:
        cv2.circle(image, nodesPositions[faceLandmark], 2, (0, 0, 0))
        cv2.putText(image, str(faceLandmark), nodesPositions[faceLandmark], 0, 0.25, (255, 0, 0))

    cv2.imshow("image   SSIIUUUUU", image)
    cv2.waitKey(0)


def buildFormanRicciGraph(image, distType):
    if (image is None):
        return

    graph = buildGraph(image, distType)

    if (graph is None):
        return
    # Computes Forman-Ricci curv
    ricciCurvGraph = FormanRicci(graph)
    ricciCurvGraph.compute_ricci_curvature()
    return ricciCurvGraph


def buildOllivierRicciGraph(image, distType):
    if (image is None):
        return

    graph = buildGraph(image, distType)

    if (graph is None):
        return
    # Computes Ollivier-Ricci curv
    ricciCurvGraph = OllivierRicci(graph)
    ricciCurvGraph.compute_ricci_curvature()
    return ricciCurvGraph


FACE_LANDMARKS = frozenset([
     0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67
])

FACE_EDGES = frozenset([
     #Jaw
     (0,1),
     (1,2),
     (2,3),
     (3,4),
     (4,5),
     (5,6),
     (6,7),
     (7,8),
     (8,9),
     (9,10),
     (10,11),
     (11,12),
     (12,13),
     (13,14),
     (14,15),
     (15,16),

    # Left eyebrow
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    # Right eyebrow
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    # Nose
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 33),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
    # Left eye
    (36, 37),
    (37, 38),
    (38, 39),
    (39, 40),
    (40, 41),
    (41, 36),
    # Right eye
    (42, 43),
    (43, 44),
    (44, 45),
    (45, 46),
    (46, 47),
    (47, 42),
    # Right eye
    (42, 43),
    (43, 44),
    (44, 45),
    (45, 46),
    (46, 47),
    (47, 42),
    # Mouth
    (48, 49),
    (49, 50),
    (50, 51),
    (51, 52),
    (52, 53),
    (53, 54),
    (54, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    (58, 59),
    (59, 48),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
    (65, 66),
    (66, 67),
    (67, 60),
    # Eyebrows
    (21, 22)

])