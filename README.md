# Anti-Spoofing Liveness Detection

🛡️ **AI-powered system to detect real vs. spoofed faces in real time using YOLOv8 and Flask.**

---

## ✨ Features

### ✅ Real-Time Face Anti-Spoofing
- Detects whether a face in front of the camera is **real** or a **spoof**.
- Uses a **custom-trained YOLOv8 model**.

### ✅ Web Dashboard (Flask)
- Live video stream from webcam.
- Detection status with **color-coded labels**:
  - 🟢 **REAL**
  - 🔴 **FAKE**
  - ⚪ **WAITING**
- Confidence score and FPS monitoring.

### ✅ Model Training & Dataset Utilities
- Scripts included for:
  - Dataset collection via webcam.
  - Splitting dataset into **train/val/test** sets.
  - Training YOLOv8 with a custom dataset.

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nithish-Bathila/Anti-Spoofing-Liveness-Detection.git
cd Anti-Spoofing-Liveness-Detection
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt
```

⚠️ **Note**: Install **PyTorch** separately depending on your CUDA/CPU setup:  
👉 [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

## 📌 Notes
- The `models/` folder is included but empty in the repo (trained weights not pushed).
- Download trained model from [here](https://drive.google.com/file/d/1n7HjLH8E64jAXRiNQSIAHVl-Fllc3bhE/view?usp=sharing)
- Place the downloaded model in the `models/` folder

---

## 🎮 Usage

### Run the Web Dashboard
```bash
python main.py
```
- Opens a Flask server at PC IP address. (not the local host) Ex-http://192.xxx.x.1xx:5000
- Streams live video with detection results.


---

## 📂 Project Structure
```
main.py                 # Flask web dashboard (real-time detection)
train.py                # YOLOv8 training script
dataCollector.py        # Webcam dataset collection script
splitData.py            # Dataset splitting utility
models/                 # Store trained YOLO models (.pt files)
requirements.txt        # Python dependencies
README.md               # Project documentation
```

---
