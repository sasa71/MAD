"""archi_set=set()
image = cv2.imread("C:\\Users\\salva\\Desktop\\Test\\morphed_img000184_img282038.png")
G=mediapipeMesh.buildGraphNorm(image, "manhattan")
orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
orc.compute_ricci_curvature()
G_orc = orc.G.copy() 
x=[]
x=list(G_orc.edges())
train = pd.read_csv("C:\\Users\\salva\\Desktop\\Dataset\\Train\\Workout.csv")
x_train = train.drop(['Nome immagine', "Etichetta"],axis=1)
y = train['Etichetta']
a=[]
selector = VarianceThreshold(threshold=0.017)
x_train_variance = selector.fit_transform(x_train)
features_discarded = (selector.get_support())
print("x_train: ",x_train.shape)
print("x_train_variance: ",x_train_variance.shape)
for i in range (2644):
    if(features_discarded[i]==True):
        a.append(i)
        print(features_discarded[i],i)

for j in range(10):
    a[j]=a[j]-1322
print("indici: ",a)

for q in a:
    archi_set.add(x[q])
    print("archi da tenere: ",x[q])

print(archi_set)

print(x)

ricciData.show_results(G_orc)
"""

"""
train = pd.read_csv("C:/Users/luigi/OneDrive/Desktop/Workout/Workout.csv")
x_train = train.drop(['Nome immagine', "Etichetta"],axis=1)
y = train['Etichetta']
selector = VarianceThreshold(threshold=0.017)
x_train_variance = selector.fit_transform(x_train)
features_discarded = (selector.get_support())
print("x_train: ",x_train.shape)
print("x_train_variance: ",x_train_variance.shape)
for i in range (2644):
    if(features_discarded[i]==True):
        print(features_discarded[i],i)
        print (x_train.columns[i])

"""
