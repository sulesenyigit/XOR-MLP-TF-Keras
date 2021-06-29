# Learning XOR mapping with a MLP using TF (Keras)
En başta bu kodda kullanacağımız **fonksiyonları** çağıracağımız **kütüphaneleri** dahil ediyoruz.</br>
First we import the **libraries** to use the **functions** that we are going to invoke in this code.</br>
```
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np # numerical python kütüphanesi
import pandas as pd
import sys
from numpy import random
from google.colab import files
```
Bildiğimiz gibi, *XOR* problemi *Elektrik-Elektronik Mühendisliğinde* mantıksal devreler (**logic circuits**) alanında karşımıza çıkan bir problem. İki girişli ve tek çıkışlı (**two input one output**) bir sistemin girişleri *TRUE-FALSE* (veya *1-0* veya *HIGH-LOW*) olarak isimlendirilen *Boolean* veri tipinde değer alıyorlar. Benzer şekilde çıkış da yine *1-0* Boolean değişkenlerinden birisi oluyor.

<img src="figure/xor mapping.jpg" alt="xor problem mapping" height="200"/></br>
*Figure 1:* The mapping in XOR problem.

Aşağıda **numpy** kütüphanesi kullanarak giriş (**input**) ve hedef (**target**) verilerini oluşturduk. Dikkat edin, burada hedef verisine İngilizce'de **annonation** deniyor. Yani bizler hedef değişkenini oluşturduğumuzda giriş verisini **annotate** etmiş (etiket - **label**) oluyoruz.
```
trainingData = np.array([[0,0], [0,1], [1,0], [1,1]], "float32")
targetData = np.array([[0], [1], [1], [0]], "float32")
```
Yukarıda ürettiğimiz giriş-çıkış verisini aşağıda konsola yazdıralım.</br>
Let's print out the **input-output data** generated above in the console.
```
print(trainingData)
print(targetData)
```
**TF** kütüphanesinin içerisinde yer alan **Keras** kütüphanesiyle bir tane çok katmanlı sinir ağı (**multi-layer perceptron** - **MLP**) oluşturalım. Giriş sayısının iki, çıkış sayısının bir olduğunu, gizli katman (**hidden layer**) sayısını ve bu katmanlardaki nöron sayısını (**number of neurons**) ve de hangi aktivasyon fonksiyonunu (**activation function**) kullandığımızı adım adım seçeceğiz.
```
model = tf.keras.models.Sequential(layers=None, name='xor_MLP')
model.add(tf.keras.layers.Dense(16, input_dim = 2, activation = 'relu', name='HL1'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid', name='outputLayer'))
```
Yukarıdaki kodda yapay sinir ağımızın yapısını (**structure**) yani katman sayısı, katmanlardaki nöron miktarı, nöron çıkışlarında kullanılan aktivasyon fonksiyon tiplerini belirledik. Bir sonraki adım olarak öğrenme (**learning**) işlemini başlatacağız. Şimdi öğrenme işleminin nasıl olacağını belirleyelim.
```
model.compile(optimizer='adam', loss='mean_squared_error', metrics='binary_accuracy')
model.summary()
```
Son olarak konfigürasyonunu tamamladığımız çok katmanlı yapay sinir ağının eğitimini başlatıyoruz.</br>
Lastly, we begin the training of the configured multi-layer perceptron neural network.
```
history=model.fit(trainingData, targetData, epochs=150, verbose=1)
```
Eğitim süreci tamamlandıktan sonra ağın son halini test edelim.</br>
Let's test the network after the training process.
```
model.predict(trainingData).round()
```
Eğitim boyunca kayıp fonksiyonumuzun almış olduğu değerleri *Şekil 2*'de çizdirelim.</br>
Let's plot the values that the loss function has taken during the training in *Fig. 2*.
```
lossCurve = history.history["loss"]
plt.plot(lossCurve, label="Train")
# plt.legend(loc='upper right')
plt.title("Loss function")
plt.xlabel("epoch number")
```

<img src="figure/loss function (XOR problem).png" alt="loss function" height="240"/></br>
*Figure 2:* Loss function in XOR problem.

<img src="figure/accuracy XOR problem.png" alt="accuracy during training" height="240"/></br>
*Figure 3:* Accuracy change during the training in XOR problem.
