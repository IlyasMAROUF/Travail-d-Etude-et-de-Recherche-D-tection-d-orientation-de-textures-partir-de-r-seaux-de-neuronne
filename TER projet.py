## Importation des packages

import math
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

## Importation des données

import numpy as np
loaded_I = np.loadtxt("Images1.txt")
Images = loaded_I.reshape(40000, 30, 30,1)
Images = (Images-np.mean(Images))/np.std(Images)
Lphi = np.loadtxt("Phi1.txt")

loaded_I = np.loadtxt("Images_test.txt")
Images_test = loaded_I.reshape(1000, 30, 30,1)
Images_test = (Images_test-np.mean(Images_test))/np.std(Images_test)
Lphi_test = np.loadtxt("Phi_test.txt")

loaded_I = np.loadtxt("Images_val.txt")
Images_val = loaded_I.reshape(1000, 30, 30,1)
Images_val = (Images_val-np.mean(Images_val))/np.std(Images_val)
Lphi_val = np.loadtxt("Phi_val.txt")

## Classification

k=2
h = np.pi/k
Cls = np.arange(0,k+1)
Cls = -np.pi/2 + Cls * h

Cphi = np.array([],dtype = int)

(n,)=np.shape(Lphi)

for i in range (0,n):
    for j in range (0,k):
        if Lphi[i] >= Cls[j] and Lphi[i]<= Cls[j+1]:
            Cphi = np.append(Cphi,[j])



Cphi_test = np.array([],dtype = int)

(n,)=np.shape(Lphi_test)

for i in range (0,n):
    for j in range (0,k):
        if Lphi_test[i] >= Cls[j] and Lphi_test[i]<= Cls[j+1]:
            Cphi_test = np.append(Cphi_test,[j])

Cphi_val = np.array([],dtype = int)

(n,)=np.shape(Lphi_val)

for i in range (0,n):
    for j in range (0,k):
        if Lphi_val[i] >= Cls[j] and Lphi_val[i]<= Cls[j+1]:
            Cphi_val = np.append(Cphi_val,[j])

## Création du modèle
model = keras.models.Sequential()

model.add(keras.layers.Input((30,30,1)))

model.add(keras.layers.Conv2D(16,(3,3),activation="relu"))
model.add(keras.layers.MaxPool2D((2,2)))

model.add(keras.layers.Conv2D(32,(3,3),activation="relu"))
model.add(keras.layers.MaxPool2D((2,2)))

model.add(keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(keras.layers.MaxPool2D((2,2)))



model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dense(k,activation='softmax'))


model.summary()


## Configuration et optimisation

opt = tf.keras.optimizers.SGD(learning_rate=1e-5)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

model.fit(  Images, Cphi,
            batch_size=100,
            epochs=20,
            validation_data=(Images_test,Cphi_test))

## Prédiction

Cphi_sigmoid = model.predict(Images_val)
Cphi_pred = np.argmax(Cphi_sigmoid,axis=-1)
Cphi_pred

## Matrice de confusion

conf = np.zeros((k,k))
for l in range(1000):
    i = Cphi_pred[l]
    j = Cphi_val[l]
    conf[i,j] = conf[i,j] + 1

fig, ax = plt.subplots(1,1)
data = conf
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,loc="center")

plt.show()

## Graphe Accuracy

Acc2 = [0.6105, 0.8174, 0.8510, 0.8702, 0.8804, 0.8846, 0.8905, 0.8972, 0.8962, 0.9018, 0.9024, 0.9061, 0.9113, 0.9130, 0.9155, 0.9153, 0.9196, 0.9204, 0.9193, 0.9252]

Acc5 = [0.3568, 0.6640, 0.7408, 0.7972, 0.8277, 0.8445, 0.8532, 0.8562, 0.8716, 0.8753, 0.8851, 0.8837, 0.8909, 0.8934, 0.8974, 0.9007, 0.9067, 0.9070, 0.9108, 0.9130]

Acc10 = [0.1015, 0.2201, 0.3368, 0.4383, 0.5523, 0.6420, 0.6878, 0.7125, 0.7305, 0.7437, 0.7595, 0.7636, 0.7793, 0.7850, 0.7988, 0.8012, 0.8113, 0.8141, 0.8224, 0.8301]

Acc20 = [0.0571, 0.2071, 0.3291, 0.3981, 0.4536, 0.4931, 0.5222, 0.5508, 0.5751, 0.5885, 0.6096, 0.6237, 0.6346, 0.6392, 0.6492, 0.6623, 0.6669, 0.6694, 0.6856, 0.6888]

Acc30 = [0.0369, 0.1011, 0.1337, 0.1801, 0.2138, 0.2446, 0.3058, 0.3640, 0.3947, 0.4222, 0.4425, 0.4615, 0.4780, 0.4895, 0.5001, 0.5036, 0.5205, 0.5255, 0.5308, 0.5360]


X = np.arange(1,21)

plt.axis([1,20,0,1])
plt.plot(X,Acc2, label = "2 classes")
plt.plot(X,Acc5, label = "5 classes")
plt.plot(X,Acc10, label = "10 classes")
plt.plot(X,Acc20, label = "20 classes")
plt.plot(X,Acc30, label = "30 classes")
plt.legend(loc="lower right")
plt.xlabel("Epoch", size = 16,)
plt.ylabel("Accurancy", size = 16)

plt.show()



