import streamlit as st
import cv2
import numpy as np
import torch
import gdown
import os
from PIL import Image

# --- SOURCE [1]: Preprocessing Logic ---
def preprocess_mri(image_input):
    """Preprocess a corrupted MRI image for Restormer inference."""
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        else:
            file_bytes = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("input must be file path or bytes")
    
    if img is None:
        raise ValueError("Failed to load image — check file format")
    
    img = cv2.resize(img, (256, 256))
    img_np = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return tensor, img_np

# --- SOURCE [1]: Postprocessing Logic ---
def postprocess_output(output_tensor):
    """Convert model output tensor back to displayable numpy image."""
    img_float = output_tensor.cpu().squeeze().detach().numpy()
    img_float = np.clip(img_float, 0.0, 1.0)
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_float, img_uint8

# --- MODEL DOWNLOAD (Using gdown) ---
@st.cache_resource
def load_neurorefine_model():
    file_id = '1VW-F-SLnxhFg1Pvtv5sx44xQYo3c2v5B'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'neurorefine_model.pth'
    
    if not os.path.exists(output):
        with st.spinner("Downloading NeuroRefine model from Google Drive..."):
            gdown.download(url, output, quiet=False)
    
    # NOTE: You must define your model architecture class (e.g., Restormer) 
    # before loading the state_dict. This is external to the sources [1, 2].
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Placeholder: Assuming the model is a scripted torch model or 
    # you have the class defined here.
    try:
        model = torch.load(output, map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure the architecture is defined.")
        return None, device

# --- STREAMLIT UI ---
st.set_page_config(page_title="NeuroRefine AI", layout="wide")

st.title("🧠 NeuroRefine: AI-Powered MRI Artifact Reconstruction")
st.write("**NeuroRefine** is an AI platform that improves MRI scans affected by motion artifacts, noise, and scanning blur [3].")

# Sidebar - Project Context [4, 5]
st.sidebar.header("About NeuroRefine")
st.sidebar.info("Motion artifacts occur in up to **59% of MRI scans** [4].")
st.sidebar.write("### Model Performance [5]")
st.sidebar.write("- **PSNR:** 31.80 dB")
st.sidebar.write("- **SSIM:** 0.889")
st.sidebar.write("- **NRMSE:** 0.027")

# File Uploader [2, 3]
uploaded_file = st.file_uploader("Upload a corrupted brain MRI scan (.jpg, .png, .bmp)", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    model, device = load_neurorefine_model()
    
    if model:
        # Preprocessing [1]
        input_bytes = uploaded_file.read()
        input_tensor, original_np = preprocess_mri(input_bytes)
        input_tensor = input_tensor.to(device)

        # Inference Logic: Residual Learning Framework [1, 2]
        with torch.no_grad():
            # The model predicts the artifact (residual) [2]
            residual = model(input_tensor)
            # Reconstruct by adding residual and clamping [1]
            output_tensor = torch.clamp(input_tensor + residual, 0.0, 1.0)
        
        # Postprocessing [1]
        _, enhanced_np = postprocess_output(output_tensor)

        # Side-by-Side Comparison [6]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Motion-Corrupted Scan")
            st.image(original_np, use_container_width=True, caption="Original Input")
        
        with col2:
            st.subheader("Enhanced Reconstruction")
            st.image(enhanced_np, use_container_width=True, caption="NeuroRefine Output")
        
        # Download Result [2]
        result_img = Image.fromarray(enhanced_np)
        st.download_button("Download Enhanced Image", data=input_bytes, file_name="enhanced_mri.png", mime="image/png")
