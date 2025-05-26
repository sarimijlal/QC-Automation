# 🧵 Fabric Defect Detection – YOLOv8 Demo

A real-time web-based Quality Control (QC) system using YOLOv8 for detecting fabric defects. Built with **FastAPI** and a clean, interactive **HTML/JS frontend**.

## 🚀 Features

- 📸 Real-time defect detection from webcam
- 📁 Upload & analyze images or videos
- 🔍 Supports multiple defect classes
- 🖥 FastAPI backend + YOLOv8 inference
- 💡 Modular and ready for deployment

## 🛠 Tech Stack

- YOLOv8 (via Ultralytics)
- FastAPI
- OpenCV
- HTML, JavaScript, CSS
- WebSockets

## 🧪 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/yolo-defect-detection-demo.git
   cd yolo-defect-detection-demo

2. **Install dependencies**:

pip install -r requirements.txt

3. **Run the server**:

uvicorn main:app --reload
