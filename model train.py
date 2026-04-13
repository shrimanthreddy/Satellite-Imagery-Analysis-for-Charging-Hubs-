import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,Flatten,Dense,Dropout,concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

from PIL import Image
import matplotlib.pyplot as plt

import zipfile

zip_path = "/content/EuroSAT.zip"
extract_path = "/content/EuroSAT"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)



IMG_SIZE = 64

data = []
labels = []

classes = ["Industrial","Residential","Highway"]

dataset_path = "EuroSAT/2750"

for cls in classes:

    path = os.path.join(dataset_path,cls)

    for img in os.listdir(path)[:700]:

        img_path = os.path.join(path,img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

        data.append(image)

        if cls in ["Industrial","Highway"]:
            labels.append(1)
        else:
            labels.append(0)

X = np.array(data)/255.0
y = np.array(labels)

print("Dataset:",X.shape)


X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# Data augmentation
datagen = ImageDataGenerator(

    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True

)

datagen.fit(X_train)



# CNN architecture
image_input = Input(shape=(64,64,3))

x = Conv2D(32,(3,3),activation='relu')(image_input)
x = BatchNormalization()(x)
x = MaxPool2D()(x)

x = Conv2D(64,(3,3),activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)

x = Conv2D(128,(3,3),activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)

x = Flatten()(x)

x = Dense(128,activation='relu')(x)
x = Dropout(0.5)(x)

feature_layer = Dense(32,activation='relu',name="feature_layer")(x)

output = Dense(1,activation='sigmoid')(feature_layer)

model = Model(inputs=image_input,outputs=output)


model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)


history = model.fit(

    datagen.flow(X_train,y_train,batch_size=32),

    validation_data=(X_test,y_test),

    epochs=10

)

loss,accuracy = model.evaluate(X_test,y_test)

print("Accuracy:",accuracy)

import pickle

with open("cnn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as cnn_model.pkl")