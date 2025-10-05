import streamlit as st
from model import BLIPCaptioner
from PIL import Image
import torch

st.title("Image Captioning")

# --- Load BLIP model ---
@st.cache_resource
def load_model():
    return BLIPCaptioner()

captioner = load_model()

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

decoding = st.selectbox("Decoding Strategy", ["greedy", "beam", "nucleus"])
max_length = st.slider("Max caption length", 5, 50, 20)

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate caption button
    if st.button(" Generate Caption"):
        with st.spinner("Generating caption..."):
            # Pass uploaded_file directly to BLIP
            caption = captioner.generate_caption(
                uploaded_file,
                prompt=None,  # no prompt
                max_length=max_length,
                decoding=decoding
            )
        st.success("Caption generated!")
        st.markdown(f"**Generated Caption:** {caption}")
