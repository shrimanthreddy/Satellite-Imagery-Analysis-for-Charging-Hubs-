import numpy as np
import cv2
from tensorflow.keras.models import load_model,Model

IMG_SIZE = 64

model = load_model("charging_hub_model.h5")

feature_model = Model(
inputs=model.input,
outputs=model.get_layer("feature_layer").output
)

def extract_features(image):

    img = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = img.reshape(1,64,64,3)

    features = feature_model.predict(img)[0]

    commercial = np.mean(features[:10])
    traffic = np.mean(features[10:20])
    powerline = np.mean(features[20:32])

    return commercial,traffic,powerline


def predict_location(image):

    img = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = img.reshape(1,64,64,3)

    score = model.predict(img)[0][0]

    commercial,traffic,powerline = extract_features(image)

    if score>0.5:
        label="Suitable"
    else:
        label="Not Suitable"

    return label,score,commercial,traffic,powerline