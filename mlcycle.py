import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os


classifiers = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    MLPClassifier(random_state=42),
    BaggingClassifier(random_state=42),
    SGDClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    SVC(),
    LinearSVC(),
    AdaBoostClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(),
]


train = pd.read_csv("C:\\Users\\salva\\Desktop\\DatasetArchiColonne\\Train\\Workout.csv")
x = train.drop(['Nome immagine', "Etichetta"],axis=1)
y = train['Etichetta']

directory ='C:\\Users\\salva\\Desktop\\Giggin'

for clf in classifiers:
    clf.fit(x, y)

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            
            test = pd.read_csv(directory+'\\'+filename)
            x_test = test.drop(['Nome immagine', "Etichetta"], axis=1)
            y_test = test["Etichetta"]

            prediction = clf.predict(x_test)

            print(clf)
            print(filename)
            print('The accuracy is: ', accuracy_score(prediction, y_test))

            print("Classification Report: ")
            print(classification_report(y_test, prediction))

            z=confusion_matrix(y_test, prediction)
            print("Confusion Matrix: ")
            print(z)

            apcer = z[1][0] / (z[1][0] + z[1][1])
            bpcer =z[0][1] / (z[0][0] + z[0][1])
            eer = (apcer + bpcer) / 2
           
            print("APCER: ", '{:.2%}'.format(apcer))
            print("BPCER: ", '{:.2%}'.format(bpcer))
            print("ACER: ", '{:.2%}'.format(eer))
