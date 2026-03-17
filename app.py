import streamlit as st
import cv2
import numpy as np
from detect import process_image

st.title(" Number Plate Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # process
    plate_img, plate_text, name, phone = process_image(image)

    # layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Car image", channels="BGR")

    with col2:
        if plate_img is not None:
            st.image(plate_img, caption="Number Plate", channels="BGR")

    # results
    if plate_text:
        st.success(f"Plate: {plate_text}")

        if name:
            st.info(f"Owner: {name}")
            st.info(f"Phone: {phone}")
        else:
            st.error("Owner not found")