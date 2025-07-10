import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO  # Import YOLO for object detection
import time
from flask import Flask, render_template, Response  # For web interface

# --- 1. Load Smoking/Not_Smoking Classifier Model ---
# Ensure the filename matches the one downloaded from Colab
MODEL_CLASSIFIER_PATH = 'smoking_classifier_deployment_model.keras'
try:
    classifier_model = tf.keras.models.load_model(MODEL_CLASSIFIER_PATH)
    print(f"✅ Smoking/Not_Smoking classifier model successfully loaded from '{MODEL_CLASSIFIER_PATH}'.")
except Exception as e:
    print(f"❌ Error loading classifier model: {e}")
    print(f"Make sure '{MODEL_CLASSIFIER_PATH}' is in the same directory.")
    exit()

# --- 2. Load Object Detection Model (YOLOv8 for Person Detection) ---
# YOLOv8n (nano) is a lightweight and fast version
# The model will be downloaded automatically if not present
try:
    person_detector = YOLO('yolov8n.pt')  # Using pre-trained COCO model for various objects
    print("✅ YOLOv8 object detection model (yolov8n.pt) loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLOv8 model: {e}")
    print("Make sure you have an internet connection to download YOLOv8n.pt for the first time, or download manually.")
    exit()

# --- 3. Global Parameter Definitions ---
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Input size for the classifier model
CONFIDENCE_THRESHOLD_SMOKING = 0.6  # Confidence threshold for 'Smoking' classification
CONFIDENCE_THRESHOLD_PERSON = 0.5  # Confidence threshold for YOLO person detection

# 'person' class in COCO dataset is class ID 0
PERSON_CLASS_ID = 0

# --- 4. Smoking Classification Prediction Function ---
def predict_smoking_status(image_patch):
    if image_patch is None or image_patch.size == 0 or image_patch.shape[0] == 0 or image_patch.shape[1] == 0:
        return "Not_Smoking", 0.0

    # 1. Get the original aspect ratio of the patch
    h_orig, w_orig, _ = image_patch.shape
    aspect_ratio_orig = w_orig / h_orig

    # 2. Calculate new size while maintaining aspect ratio, then apply padding
    target_w, target_h = IMG_WIDTH, IMG_HEIGHT

    if aspect_ratio_orig > (target_w / target_h):  # If patch is wider than target
        new_w = target_w
        new_h = int(target_w / aspect_ratio_orig)
    else:  # If patch is taller than target
        new_h = target_h
        new_w = int(target_h * aspect_ratio_orig)

    resized_patch = cv2.resize(image_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create empty (black) canvas for padding
    padded_patch = np.full((target_h, target_w, 3), 0, dtype=np.uint8)  # Black padding

    # Calculate position to center the resized image on the canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste the resized image onto the canvas
    padded_patch[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_patch

    # Normalize the image
    processed_patch = np.expand_dims(padded_patch, axis=0) / 255.0

    prediction = classifier_model.predict(processed_patch, verbose=0)[0][0]
    label = "Smoking" if prediction >= CONFIDENCE_THRESHOLD_SMOKING else "Not_Smoking"
    confidence = prediction if prediction >= CONFIDENCE_THRESHOLD_SMOKING else (1 - prediction)

    return label, confidence

# --- 5. Frame Generator Function for Video Stream ---
def gen_frames():
    # Use camera (0) as the video source
    # Replace '0' with video file path (e.g., 'media/video_test.mp4') to use from file
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open video source. Make sure the camera is connected or the video path is correct.")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to read frame. May be end of video or camera issue.")
            break
        else:
            frame_count += 1
            # Flip frame horizontally for selfie view (optional, can be removed)
            frame = cv2.flip(frame, 1)

            # --- Person Detection Using YOLOv8 ---
            # 'verbose=False' disables YOLO logs
            # 'conf' sets the object detection confidence threshold
            # 'classes=[PERSON_CLASS_ID]' restricts detection to 'person' class only
            results = person_detector(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_PERSON, classes=[PERSON_CLASS_ID])

            # Iterate over each detected person
            for r in results:
                boxes = r.boxes  # Get bounding boxes from detection results
                for box in boxes:
                    # box.xyxy[0] returns [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Ensure bounding box coordinates are within frame boundaries
                    y1 = max(0, y1)
                    y2 = min(frame.shape[0], y2)
                    x1 = max(0, x1)
                    x2 = min(frame.shape[1], x2)

                    # Ensure cropped area is valid
                    if x2 > x1 and y2 > y1:
                        # Crop the person area from the original frame
                        person_patch = frame[y1:y2, x1:x2]

                        # --- Run Smoking/Not_Smoking classification on detected person area ---
                        label, confidence = predict_smoking_status(person_patch)

                        # Set color for bounding box and text
                        # Red for 'Smoking', Green for 'Not_Smoking'
                        color = (0, 0, 255) if label == "Smoking" else (0, 255, 0)
                        
                        # Draw bounding box around person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Prepare text to display
                        display_text = f"Person ({label}): {confidence:.2f}"
                        
                        # Draw label text above the bounding box
                        # Calculate position so it doesn’t go off-frame
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20  # Move down if too close to top
                        cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # Calculate and display FPS (Frames Per Second)
            current_time = time.time()
            if (current_time - start_time) > 1:  # Update FPS every 1 second
                fps = frame_count / (current_time - start_time)
                start_time = current_time
                frame_count = 0
            
            # Show FPS at top-left corner
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame into JPEG format for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\n'
                   b'Content-Type: image/jpeg\n\n' + frame_bytes + b'\n')

    # After loop ends, release video source and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream ended.")

# --- 6. Flask App Section ---
# Initialize Flask app and set template folder to current directory ('.')
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    """Main page of the web application."""
    # Flask will now look for 'index.html' in the same directory as 'app.py'
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint for video streaming."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n🚀 Running Flask application...")
    print("Access the app at http://127.0.0.1:5000 in your browser.")
    # 'debug=True' for development, 'use_reloader=False' to avoid loading model twice
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
