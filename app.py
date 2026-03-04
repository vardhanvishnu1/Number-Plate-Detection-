import streamlit as st
from plate_rec import process_license_plate
import PIL.Image

st.title("License Plate Scanner")
st.write("Upload a photo to extract numbers instantly.")

uploaded_file = st.file_uploader("Choose an image : ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    result_text, processed_img = process_license_plate("temp.jpg")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image")
    with col2:
        st.image(processed_img, caption="Detected Area")
        
    st.success(f"Extracted Number: {result_text}")