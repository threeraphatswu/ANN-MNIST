import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import keras
import os

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

model_path = '67102010516_mnist_model.keras'

# ใช้ Cache เพื่อให้โหลดโมเดลครั้งเดียว
@st.cache_resource
def load_mnist_model(path):
    return keras.models.load_model(path)

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the model is saved correctly.")
else:
    model = load_mnist_model(model_path)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # โหลดรูปภาพ
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # แปลงเป็นภาพขาวดำ (Grayscale)
        img = img.convert('L')

        # ปรับขนาดเป็น 28x28
        img = img.resize((28, 28))

        # (ตัวเลือกเพิ่มเติม) MNIST ปกติใช้ตัวเลขสีขาวบนพื้นหลังสีดำ
        # หากผู้ใช้อัปโหลดรูปตัวเลขสีดำบนพื้นหลังสีขาว สามารถกลับสีภาพได้ด้วยบรรทัดล่างนี้:
        # img = ImageOps.invert(img)

        # แปลงเป็น NumPy Array
        img_array = np.array(img)

        # Normalize ค่าพิกเซลจาก [0, 255] เป็น [0, 1]
        img_array = img_array.astype("float32") / 255.0

        # เพิ่มมิติ Batch dimension เป็น (1, 28, 28)
        img_array = img_array.reshape(1, 28, 28)

        # ทำการทำนายผล
        prediction = model.predict(img_array)

        # หาค่า Class ที่มีความน่าจะเป็นสูงสุด
        predicted_digit = np.argmax(prediction)

        st.success(f"The model predicts the digit is: **{predicted_digit}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}. Please ensure the uploaded image is valid and the model is correctly loaded.")
