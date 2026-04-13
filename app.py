#streamlit code
%%writefile app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from feature_extractor import predict_location

st.set_page_config(page_title="EV Charging Hub Predictor")

st.title("⚡ EV Charging Hub Location Predictor")

uploaded_file = st.file_uploader(
"Upload Satellite Image",
type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image,use_column_width=True)

    image_np = np.array(image)

    if st.button("Analyze Location"):

        label,score,commercial,traffic,powerline = predict_location(image_np)

        st.subheader("Prediction")

        st.write("Suitability Score:",round(score,2))

        st.success(label)

        st.subheader("Extracted Infrastructure Parameters")

        st.write("Commercial Activity:",round(commercial,2))
        st.progress(int(commercial*100))

        st.write("Traffic Density:",round(traffic,2))
        st.progress(int(traffic*100))

        st.write("Powerline Proximity:",round(powerline,2))
        st.progress(int(powerline*100))


import streamlit as st

st.title("EV Charging Hub Predictor")

st.write("Streamlit is working successfully!")        