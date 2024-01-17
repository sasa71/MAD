import pandas as pd
import shutil
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, det_curve, auc
from joblib import dump, load

def get_apcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FMR operating point
    Definition:
    ZeroFMR: is defined as the lowest FNMR at which no false matches occur.
    Others FMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest bpcer at which the probability of apcer == op
    @rtype: float
    """
    index = np.argmin(abs(apcer - op))
    return index, bpcer[index], threshold[index]

def get_bpcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FNMR operating point
    Definition:
    ZeroFNMR: is defined as the lowest FMR at which no non-false matches occur.
    Others FNMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest apcer at which the probability of bpcer == op
    @rtype: float
    """
    temp = abs(bpcer - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]

    return index, apcer[index], threshold[index]

def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    eer = fpr[index]

    return eer, index, threshold[index]

def performances_compute(prediction_scores, gt_labels, threshold_type, op_val, verbose):
    # fpr = apcer, 1-tpr = bpcer
    # op_val: 0 - 1
    # gt_labels: list of ints,  0 for attack, 1 for bonafide
    # prediction_scores: list of floats, higher value should be bonafide
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=1, drop_intermediate=False)
    bpcer = 1 - tpr
    val_eer, _, eer_threshold = get_eer_threhold(fpr, tpr, threshold)
    val_auc = auc(fpr, tpr)

    if threshold_type=='eer':
        threshold = eer_threshold
    elif threshold_type=='apcer':
        _, _, threshold = get_apcer_op(fpr, bpcer, threshold, op_val)
    elif threshold_type=='bpcer':
        _, _, threshold = get_bpcer_op(fpr, bpcer, threshold, op_val)
    else:
        threshold = 0.5

    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    threshold_APCER = type2 / num_fake
    threshold_BPCER = type1 / num_real
    threshold_ACER = (threshold_APCER + threshold_BPCER) / 2.0

    if verbose is True:
        print(f'AUC@ROC: {val_auc}, threshold:{threshold}, EER: {val_eer}, APCER:{threshold_APCER}, BPCER:{threshold_BPCER}, ACER:{threshold_ACER}')

    return val_auc, val_eer, [threshold, threshold_APCER, threshold_BPCER, threshold_ACER]


def compute_eer(label, pred):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, drop_intermediate=False)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

"""

train = pd.read_csv("C:\\Users\\salva\\Desktop\\DatasetArchiColonne\\Train\\Workout.csv")

x = train.drop(['Nome immagine', "Etichetta"],axis=1)
y = train['Etichetta']
mappa_valori = {0: 1, 1: 0} # Sostituzione dei valori 
y = y.replace(mappa_valori)

array1=[('',0,0)]
array2=[('',0,0)]
array3=[('',0,0)]
array4=[('',0,0)]

thresholds=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001,
            0.2, 0.02, 0.002, 0.0002, 0.00002, 0.000002,
            0.3, 0.03, 0.003, 0.0003, 0.00003, 0.000003,
            0.4, 0.04, 0.004, 0.0004, 0.00004, 0.000004,
            0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005,
            0.6, 0.06, 0.006, 0.0006, 0.00006, 0.000006,
            0.7, 0.07, 0.007, 0.0007, 0.00007, 0.000007,
            0.8, 0.08, 0.008, 0.0008, 0.00008, 0.000008,
            0.9, 0.09, 0.009, 0.0009, 0.00009, 0.000009]

threshold_max=[0.009, 0.008, 0.006]

directory ="C:\\Users\\salva\\Desktop\\DatasetArchiColonneLMA e MorGan"

for threshold in threshold_max:
    
    try:
        # Applica la selezione delle feature
        sel = VarianceThreshold(threshold=threshold)
        X_train_sel = sel.fit_transform(x)
        
        # Addestra il modello
        model = SVC()
        model.fit(X_train_sel, y)

        # Itera attraverso ogni file nella directory
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):

                # Carica il file di test
                test = pd.read_csv(directory+'\\'+filename)

                # Seleziona le feature
                x_test = test.drop(['Nome immagine', "Etichetta"],axis=1)

                # Seleziona la variabile target
                y_test = test["Etichetta"]

                # Sostituisci i valori della variabile target
                mappa_valori = {0: 1, 1: 0}
                y_test = y_test.replace(mappa_valori)

                # Applica la selezione delle feature sul dataset di test
                X_test_sel = sel.transform(x_test)

                # Esegui le predizioni sul dataset di test
                prediction = model.predict(X_test_sel)

                # Calcola le metriche di valutazione
                print("Threshold: ", threshold)
                print(filename)

                print('The accuracy is: ', accuracy_score(prediction, y_test))
                acc = accuracy_score(prediction, y_test)

                print("Confusion Matrix: ")
                print(confusion_matrix(y_test, prediction))
            
                #z=confusion_matrix(y_test, prediction)
                #apcer = z[0][1] / (z[0][0] + z[0][1])
                #bpcer = z[1][0] / (z[1][0] + z[1][1])
                
                #print("APCER: ", '{:.2%}'.format(apcer))
                #print("BPCER: ", '{:.2%}'.format(bpcer))
                #print("ACER: ", compute_eer(y_test, prediction))

                if(filename=="LMA64_test.csv" and acc > array1[0][1]):
                    array1=[(filename,acc,threshold)]
                elif(filename=="MorGAN_test.csv" and acc > array2[0][1]):
                    array2=[(filename,acc,threshold)]
                elif(filename=="test_digital_imgs_attack.csv" and acc > array3[0][1]):
                    array3=[(filename,acc,threshold)]
                elif(filename=="test_ps_imgs_attack.csv" and acc > array4[0][1]):
                    array4=[(filename,acc,threshold)]

    except ValueError:
        print("la threshold", threshold, "non è presente nel CSV")
        continue


print(array1, array2, array3, array4)"""


train = pd.read_csv("C:\\Users\\salva\\Desktop\\DatasetArchiColonne\\Train\\Workout.csv")

x = train.drop(['Nome immagine', "Etichetta"],axis=1)
y = train['Etichetta']
mappa_valori = {0: 1, 1: 0} # Sostituzione dei valori 
y = y.replace(mappa_valori)

threshold=0.017

sel = VarianceThreshold(threshold)
X_train_sel = sel.fit_transform(x)
#print(X_train_sel)
#print(x.shape)

model = GaussianNB()
# select the svm algorithm
model.fit(X_train_sel, y)

#model = load('Modello_train_treshold.sav')

test = pd.read_csv("C:\\Users\\salva\\Desktop\\WebMorph.csv")
x_test = test.drop(['Nome immagine', "Etichetta"],axis=1)
y_test = test["Etichetta"]
z_test = test["Nome immagine"]

X_test_sel = sel.transform(x_test)
#print(x_test.shape)
prediction = model.predict(X_test_sel)
#print(X_test_sel)  
print('The accuracy is: ', accuracy_score(prediction, y_test))

print("Confusion Matrix: ")
print(confusion_matrix(y_test, prediction))

        
"""
directory ="C:\\Users\\salva\\Desktop\\CSV colonne giuste"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Carica il file di test
        test = pd.read_csv(directory+'\\'+filename)
        x_test = test.drop(['Nome immagine', "Etichetta"],axis=1)
        y_test = test["Etichetta"]
        mappa_valori = {0: 1, 1: 0} # Sostituzione dei valori 
        y_test = y_test.replace(mappa_valori)
        z_test = test["Nome immagine"]

        X_test_sel = sel.transform(x_test)
        #print(x_test.shape)
        prediction = model.predict(X_test_sel)
        print(filename)
        #print(X_test_sel)  
        print('The accuracy is: ', accuracy_score(prediction, y_test))

        print("Confusion Matrix: ")
        print(confusion_matrix(y_test, prediction))
        
        # Calcolare l'APCER e il BPCER per ogni punto di lavoro
        #decision_scores = model.predict_proba(X_test_sel)
        #print(decision_scores)
        #y_pred_morphed = decision_scores[:, 1]
        #print(y_pred_morphed)
        #performances_compute(decision_scores, y_test, "eer", 0.001, True)
        
        fpr, tpr, threshold = roc_curve(y_test.values, y_pred_morphed, pos_label=1, drop_intermediate=False)
        val_eer, _, eer_threshold = get_eer_threhold(fpr, tpr, threshold)
        print("EER: ", val_eer)
        val_auc = auc(fpr, tpr)
        print("auc@roc: ", val_auc)
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % val_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()
        
#print("finito")
#src_folder = 'C:\\Users\\salva\\Desktop\\Tirocinio\\FRLL-Morphs\\facelab_london\\morph_opencv e bonafide'
#dst_folder_good = 'C:\\Users\\salva\\Desktop\\Prediction GaussianNB() immagini\\FRLL-OpenCV giuste'
#dst_folder_bad = 'C:\\Users\\salva\\Desktop\\Prediction GaussianNB() immagini\\FRLL-OpenCV sbagliate'

#test = pd.read_csv("C:\\Users\\salva\\Desktop\\DatasetArchiColonne\\TestNonBilanciati\\OpenCVNonBilanciato.csv")

for i in range(len(prediction)):
    if(prediction[i]==y_test[i]):
        shutil.copy(src_folder + "\\" + z_test[i], dst_folder_good + "\\" + z_test[i])
    elif(prediction[i]!=y_test[i]):
        shutil.copy(src_folder + "\\" + z_test[i], dst_folder_bad + "\\" + z_test[i])

        #print("Classifiction Report: ")
        #print(classification_report(y_test, prediction))

        #print("ACER: ", compute_eer(y_test, prediction))
        #z=confusion_matrix(y_test, prediction)

        #apcer = z[0][1] / (z[0][0] + z[0][1])
        #bpcer = z[1][0] / (z[1][0] + z[1][1])
        #eer = (apcer + bpcer) / 2
           
        #print("APCER: ", '{:.2%}'.format(apcer))
        #print("BPCER: ", '{:.2%}'.format(bpcer))
        #print("ACER: ", '{:.2%}'.format(eer))


"""
