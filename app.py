# app.py
import streamlit as st
import torch
import torchaudio
import torchvision
from torchvision import transforms
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from plot_audio import plot_specgram
from PIL import Image

# Set page config first
st.set_page_config(
    page_title="Heart Sound Classifier",
    page_icon="🎵",
    layout="wide"
)

# Define the model architecture (must match your training code)
class ImageMulticlassClassificationNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(25*25*16, 128)  # FIXED: Correct input size after conv layers
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)  # (BS, 1, 100, 100) → (BS, 6, 100, 100)
        x = self.relu(x)
        x = self.pool(x)   # (BS, 6, 100, 100) → (BS, 6, 50, 50)
        x = self.conv2(x)  # (BS, 6, 50, 50) → (BS, 16, 50, 50)
        x = self.relu(x)
        x = self.pool(x)   # (BS, 16, 50, 50) → (BS, 16, 25, 25)
        x = self.flatten(x) # (BS, 16, 25, 25) → (BS, 16*25*25 = 10000)
        x = self.fc1(x)    # (BS, 10000) → (BS, 128)
        x = self.relu(x)
        x = self.fc2(x)    # (BS, 128) → (BS, 64)
        x = self.relu(x)
        x = self.fc3(x)    # (BS, 64) → (BS, 4)
        x = self.softmax(x)
        return x

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = ImageMulticlassClassificationNet(num_classes=4)
        model.load_state_dict(torch.load('heart_sound_model.pth', map_location='cpu', weights_only=True))
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please make sure 'heart_sound_model.pth' exists in the same folder")
        return None

# Load class names
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        return ['artifact', 'extrahls', 'murmur', 'normal']

# Initialize
CLASSES = load_class_names()
model = load_model()

# Define transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# UI
st.title("🎵 Heart Sound Classification")
st.markdown("Upload a heart sound audio file to classify it into one of four categories")

# Sidebar for info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Classes:**
    - 🟢 **Normal**: Regular heart sounds
    - 🟡 **Murmur**: Abnormal heart sounds
    - 🔵 **Extrahls**: Extra heart sounds  
    - 🔴 **Artifact**: Noise/interference
    
    **Supported formats:** WAV, MP3, M4A
    """)

# Main content
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'm4a'],
    help="Upload heart sound recording"
)

if uploaded_file is None:
    st.info("👆 Please upload an audio file to begin analysis")

if uploaded_file is not None and model is not None:
    # Initialize spec_path variable
    spec_path = None
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and process audio
        waveform, sample_rate = torchaudio.load(tmp_path)
        
        # Resample if necessary (handle different sample rates)
        if sample_rate != 22050:  # Common standard rate
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=22050)
            waveform = resampler(waveform)
            sample_rate = 22050
        
        st.success(f"✅ Audio loaded successfully! Duration: {waveform.shape[1]/sample_rate:.2f} seconds")
        
        # Display audio player
        st.audio(uploaded_file)
        
        # Create visualizations
        st.subheader("📊 Audio Analysis")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Waveform plot
            st.markdown("**Waveform**")
            fig, ax = plt.subplots(figsize=(10, 3))
            time_axis = torch.arange(waveform.shape[1]) / sample_rate
            ax.plot(time_axis, waveform[0].numpy(), color='blue', alpha=0.7)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_title("Audio Waveform")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)  # Prevent memory leaks
        
        with viz_col2:
            # Spectrogram
            st.markdown("**Spectrogram**")
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as spec_file:
                plot_specgram(waveform, sample_rate, spec_file.name)
                st.image(spec_file.name, use_container_width=True)
                spec_path = spec_file.name
        
        # Prediction section
        st.subheader("🔍 Making Prediction...")
        
        with st.spinner("Processing audio and generating prediction..."):
            # Load and transform spectrogram
            image = Image.open(spec_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                predicted_class = CLASSES[predicted_idx]
                confidence = probabilities[predicted_idx].item() * 100
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        # Confidence bars
        st.markdown("**Confidence Scores:**")
        for i, class_name in enumerate(CLASSES):
            conf = probabilities[i].item() * 100
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(f"**{class_name}**")
            with col2:
                st.progress(conf/100)
            with col3:
                st.write(f"{conf:.1f}%")
        
        # Final prediction with emoji and color
        st.markdown("---")
        if confidence > 75:
            st.success(f"## ✅ **Prediction: {predicted_class.upper()}**")
        elif confidence > 50:
            st.warning(f"## ⚠️ **Prediction: {predicted_class.upper()}**")
        else:
            st.error(f"## ❓ **Prediction: {predicted_class.upper()}**")
        
        st.markdown(f"**Confidence: {confidence:.2f}%**")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure you're uploading a valid audio file.")
        
    finally:
        # Cleanup - check if variables exist before trying to delete
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        if 'spec_path' in locals() and spec_path and os.path.exists(spec_path):
            try:
                os.unlink(spec_path)
            except:
                pass

elif uploaded_file and model is None:
    st.error("❌ Model not loaded. Please check if 'heart_sound_model.pth' exists.")

# Add sample files section
with st.expander("🎧 Need a sample file to test?"):
    st.markdown("""
    You can use sample heart sound files from these sources:
    - [PhysioNet Heart Sound Database](https://physionet.org/content/challenge-2016/1.0.0/)
    - [CirCor DigiScope Phonocardiogram Dataset](https://physionet.org/content/circor-heart-sound/1.0.3/)
    
    Download some WAV files and upload them above to test the classifier!
    """)