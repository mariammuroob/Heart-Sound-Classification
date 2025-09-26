🎵 Heart Sound Audio Classifier

A content-based audio classification system built with PyTorch and Streamlit.
It classifies heart sound recordings into:
normal, murmur, extrahls, or artifact.
<img width="1341" height="561" alt="image" src="https://github.com/user-attachments/assets/859c8500-617b-4d5d-9641-0f6de9f308d1" />



The app is live here 👉 https://heart-sound-classification-auxxibxqngnq2uhkwxyemc.streamlit.app/


📝 Features
🎧 Upload a .wav heart sound file.
📈 Visualize waveform and spectrogram of the audio.
🤖 Get real-time prediction with confidence scores.


🛠️ Tech Stack

Python 3.13+
PyTorch / Torchaudio / Torchvision
Streamlit
Matplotlib / Seaborn / Librosa
Scikit-learn

🚦 Getting Started
1. Clone the repo
git clone https://github.com/your-username/audio-classifier.git
cd audio-classifier

2. Install dependencies
C:/Users/maria/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install -r requirements.txt

3. Run locally
C:/Users/maria/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m streamlit run streamlit_app.py

📂 Dataset
Expects .wav audio files.

Preprocessing converts them to spectrogram images for model input.



