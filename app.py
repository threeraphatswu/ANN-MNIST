import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import keras

# ตั้งค่าหน้า Streamlit
st.set_page_config(page_title="MNIST Digit Predictor", page_icon="🔢")

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

model_path = '67102010516_mnist_model.keras'

# ใช้ @st.cache_resource เพื่อโหลดโมเดลเพียงครั้งเดียว (ช่วยให้แอปทำงานเร็วขึ้นมาก)
@st.cache_resource
def load_mnist_model(path):
    return keras.models.load_model(path)

# ตรวจสอบไฟล์โมเดล
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the model is saved correctly in the same folder.")
else:
    model = load_mnist_model(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # 1. โหลดรูปภาพ
            img = Image.open(uploaded_file)
            
            # แสดงรูปภาพต้นฉบับ
            st.image(img, caption='Uploaded Image', width=200)
            st.write("Classifying...")

            # 2. แปลงเป็นขาวดำ (Grayscale)
            img_gray = img.convert('L')

            # 3. ตรวจสอบและสลับสี (Invert Color)
            # โมเดล MNIST เทรนด้วยตัวเลขสีขาวบนพื้นดำ 
            # หากผู้ใช้อัปโหลดรูปตัวเลขสีดำบนพื้นขาว ระบบจะสลับสีให้อัตโนมัติ
            if np.mean(img_gray) > 127:
                img_gray = ImageOps.invert(img_gray)

            # 4. ย่อขนาดเป็น 28x28 พิกเซล
            img_resized = img_gray.resize((28, 28))

            # 5. แปลงเป็น NumPy Array และปรับค่าพิกเซลให้อยู่ในช่วง [0, 1]
            img_array = np.array(img_resized, dtype="float32") / 255.0

            # 6. ปรับทรงมิติข้อมูล (Shape) ให้ตรงตามที่โมเดลต้องการ
            # รองรับทั้งแบบ (1, 28, 28) และแบบ CNN (1, 28, 28, 1)
            if len(model.input_shape) == 4:
                img_array = np.expand_dims(img_array, axis=(0, -1))
            else:
                img_array = np.expand_dims(img_array, axis=0)

            # 7. ทำนายผล
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100

            # แสดงผลลัพธ์
            st.success(f"The model predicts the digit is: **{predicted_digit}** (Confidence: {confidence:.2f}%)")
            
            # แสดง กราฟความน่าจะเป็นของแต่ละตัวเลข (0-9)
            st.bar_chart(prediction[0])

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please ensure the uploaded image is valid.")
