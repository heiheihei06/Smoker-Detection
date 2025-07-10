# 🚬 Real-Time Smoking Detection with Flask and Computer Vision

This project is a **final semester project** for the **Computer Vision course**. The application is designed to **detect smoking behavior in real-time** using a combination of **YOLOv8** for object detection and **TensorFlow** for smoking classification.

## 🎯 Main Features
- Real-time person detection using **YOLOv8-nano**
- Smoking vs Not_Smoking classification using a custom-trained **TensorFlow CNN model**
- Live webcam video streaming via **Flask Web Interface**
- Colored bounding boxes: 🔴 Red for Smoking, 🟢 Green for Not Smoking
- Live **FPS display** for performance monitoring

## 🗂️ Project Structure
- app.py # Main Flask application
- index.html # Web interface (template)
- smoking_classifier_deployment_model.keras # Smoking classifier model
- yolov8n.pt # YOLOv8-nano model for person detection
- requirements.txt # Project dependencies

## 🧠 Technologies Used
- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/) (Smoking classifier)
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) (Object detection)
- [OpenCV](https://opencv.org/) (Video and image processing)
- [Flask](https://flask.palletsprojects.com/) (Web interface)

## ⚙️ How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/heiheihei06/Smoker-Detection.git
   cd Smoker-Detection
   
2. **Install all dependencies:**
   pip install -r requirements.txt
   
3. **Run the application:**
   python app.py
   
4. **Open your browser and visit:**
   http://127.0.0.1:5000

**📝 Notes**
- If yolov8n.pt has not been used before, it will automatically be downloaded by the ultralytics package.
- Ensure the smoking_classifier_deployment_model.keras file is placed in the root directory.
- The system uses the default webcam (cv2.VideoCapture(0)). Replace this with a file path if you want to use a recorded video.

**🙋‍♀️ About the Creator**
- 👩‍💻 Name: Permata Rezki Yulanda
- 🏫 Institution: Informatics Engineering, Politeknik Caltex Riau
- 📚 Project Type: Final Project for the Computer Vision Course
- 💬 Contact: 
    - GitHub : https://github.com/heiheihei06
    - LinkedIn : Permata Rezki Yulanda
    - Gmail : pry061101@gmail.com
