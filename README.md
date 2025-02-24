
# 🛡️ SkinGuardian - AI-Powered Skin Cancer Detection

**SkinGuardian** is an AI-driven desktop application that enables **early skin cancer detection** using a fine-tuned **Beit model** from Qualcomm AI Hub. It runs entirely **on-device** with **ONNX Runtime**, ensuring **privacy, speed, and efficiency**. No internet connection is required once installed.

---

## 🚀 Features

✔ **AI-powered early detection**  
✔ **On-device inference** (No cloud dependency)  
✔ **Fast and private analysis**  
✔ **User-friendly interface** with drag-and-drop upload  
✔ **Simple one-click execution** (Windows EXE available)  
✔ **Qualcomm AI Hub Model - BEIT** fine-tuned for skin lesion classification  

---

## 🛠️ Installation

### **Option 1: Run from Source (Requires Python 3.9+)**
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure your model file is in the correct directory**:
   - Place `beit_finetuned_model.onnx` in the same folder as `app.py`.

3. **Run the application**:
   ```bash
   python app.py
   ```
4. **Access the web interface**:
   - Open your browser and go to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

### **Option 2: Run as a Standalone EXE (No Python Needed)**
1. **Download** the pre-built EXE from `dist/app.exe` (or build it yourself using PyInstaller).
2. **Double-click** `app.exe` to start the application.
3. **On your browser open the app** at:
   ```
   http://127.0.0.1:5000
   ```
4. **Upload an image** and receive an AI-based analysis.

---

## 🏗️ How to Build an Executable (`.exe`) Using PyInstaller

If you want to package the app into a Windows executable:

1. **Install PyInstaller**:
   ```bash
   pip install pyinstaller
   ```
2. **Run the following command** to create a standalone EXE:
   ```bash
   pyinstaller --onefile --noconsole app.py
   ```
3. **Find the EXE in the `dist/` folder**:
   - Run `dist/app.exe` to launch the application.

---

## 🧠 AI Model Information

- **Model**: Fine-tuned **Beit** from **Qualcomm AI Hub**  
- **Framework**: ONNX Runtime  
- **Input Size**: `224x224` pixels (RGB)  
- **Inference Output**: Binary classification (Cancerous vs. Non-Cancerous)  
- **Probability Scores**: AI model provides confidence levels for each prediction  

---

## 🖥️ Project Structure

```
/SkinGuardian
│── app.py                # Main Flask application
│── beit_finetuned_model.onnx  # AI model file 
│── requirements.txt       # Required dependencies
│── README.md              # Documentation
│── /templates             # HTML templates for UI
│   ├── index.html
│   ├── result.html
│   ├── about.html
│   └── contact.html
│── /dist                  # (Generated) Executable file after PyInstaller build
│── app.spec               # PyInstaller configuration
```

---

## 📜 License

This project is for educational and research purposes. Not intended for medical diagnosis or treatment.

---

### 🔹 **Git Large File Storage (LFS) for Model File**
Since the ONNX model file is large, it should be tracked using **Git LFS** to prevent repository size issues.

#### **How to Use Git LFS**
1. **Install Git LFS** (if not already installed):
   ```bash
   git lfs install
   ```
2. **Track the ONNX model file**:
   ```bash
   git lfs track "*.onnx"
   ```
3. **Commit and push** your changes:
   ```bash
   git add .gitattributes
   git add beit_finetuned_model.onnx
   git commit -m "Added ONNX model with LFS"
   git push origin main
   ```

---

### 🏆 **Why Choose SkinGuardian?**
🔹 **Runs Completely On-Device** – No internet dependency, ensuring privacy  
🔹 **Optimized for Qualcomm Hardware** – Fast, efficient execution on Snapdragon-powered devices  
🔹 **User-Friendly** – Drag-and-drop image upload, intuitive web-based UI  
🔹 **AI-Powered Precision** – Fine-tuned BEIT model ensures high accuracy in classification  

---

### 📢 **Upcoming Features**
🚀 **Mobile App Version** (Android/iOS)  
🖼️ **Heatmap Visualization** (Grad-CAM for interpretability)  
📊 **Detailed Probability Report** (Confidence scores for multiple lesion types)  
📩 **Exportable Reports** (PDF format for doctors/patients)  

**Developed for the Qualcomm AI Hackathon** 🏆 🚀
