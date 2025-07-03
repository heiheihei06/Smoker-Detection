import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO # Mengimpor YOLO untuk deteksi objek
import time
from flask import Flask, render_template, Response # Untuk antarmuka web

# --- 1. Muat Model Klasifikasi Smoking/Not_Smoking ---
# Pastikan nama file cocok dengan yang diunduh dari Colab
MODEL_CLASSIFIER_PATH = 'smoking_classifier_deployment_model.keras'
try:
    classifier_model = tf.keras.models.load_model(MODEL_CLASSIFIER_PATH)
    print(f"✅ Model klasifikasi Smoking/Not_Smoking berhasil dimuat dari '{MODEL_CLASSIFIER_PATH}'.")
except Exception as e:
    print(f"❌ Error saat memuat model klasifikasi: {e}")
    print(f"Pastikan '{MODEL_CLASSIFIER_PATH}' ada di direktori yang sama.")
    exit()

# --- 2. Muat Model Deteksi Objek (YOLOv8 untuk Deteksi Orang) ---
# YOLOv8n (nano) adalah versi yang ringan dan cukup cepat
# Model akan otomatis diunduh jika belum ada
try:
    person_detector = YOLO('yolov8n.pt') # Menggunakan model pre-trained COCO untuk berbagai objek
    print("✅ Model deteksi objek YOLOv8 (yolov8n.pt) berhasil dimuat.")
except Exception as e:
    print(f"❌ Error saat memuat model YOLOv8: {e}")
    print("Pastikan Anda memiliki koneksi internet untuk mengunduh model YOLOv8n.pt pertama kali, atau unduh secara manual.")
    exit()

# --- 3. Definisi Parameter Global ---
IMG_WIDTH, IMG_HEIGHT = 150, 150 # Ukuran input untuk model klasifikasi
CONFIDENCE_THRESHOLD_SMOKING = 0.6 # Ambang batas kepercayaan untuk klasifikasi 'Smoking' (bisa disesuaikan)
CONFIDENCE_THRESHOLD_PERSON = 0.5 # Ambang batas kepercayaan untuk deteksi orang oleh YOLO

# Kelas 'person' di COCO dataset adalah kelas 0
PERSON_CLASS_ID = 0

# --- 4. Fungsi Prediksi Klasifikasi Merokok ---
def predict_smoking_status(image_patch):
    if image_patch is None or image_patch.size == 0 or image_patch.shape[0] == 0 or image_patch.shape[1] == 0:
        return "Not_Smoking", 0.0

    # 1. Dapatkan rasio aspek patch asli
    h_orig, w_orig, _ = image_patch.shape
    aspect_ratio_orig = w_orig / h_orig

    # 2. Hitung ukuran baru dengan mempertahankan rasio aspek, lalu tambahkan padding
    target_w, target_h = IMG_WIDTH, IMG_HEIGHT

    if aspect_ratio_orig > (target_w / target_h): # Jika patch lebih lebar dari target
        new_w = target_w
        new_h = int(target_w / aspect_ratio_orig)
    else: # Jika patch lebih tinggi dari target
        new_h = target_h
        new_w = int(target_h * aspect_ratio_orig)

    resized_patch = cv2.resize(image_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Buat kanvas kosong (hitam) untuk padding
    padded_patch = np.full((target_h, target_w, 3), 0, dtype=np.uint8) # Hitam sebagai padding

    # Hitung posisi untuk menempatkan gambar yang di-resize di tengah kanvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Tempel gambar yang di-resize ke kanvas
    padded_patch[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_patch

    # Normalisasi
    processed_patch = np.expand_dims(padded_patch, axis=0) / 255.0

    prediction = classifier_model.predict(processed_patch, verbose=0)[0][0]
    label = "Smoking" if prediction >= CONFIDENCE_THRESHOLD_SMOKING else "Not_Smoking"
    confidence = prediction if prediction >= CONFIDENCE_THRESHOLD_SMOKING else (1 - prediction)

    return label, confidence

# --- 5. Fungsi Generator Frame untuk Video Stream ---
def gen_frames():
    # Menggunakan kamera (0) sebagai sumber video
    # Ganti '0' dengan path file video (misal: 'media/video_test.mp4') jika ingin dari file
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Tidak dapat membuka sumber video. Pastikan kamera terhubung atau path video benar.")
        return

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Peringatan: Gagal membaca frame. Mungkin akhir video atau masalah kamera.")
            break
        else:
            frame_count += 1
            # Balik frame secara horizontal untuk tampilan selfie (opsional, bisa dihapus)
            frame = cv2.flip(frame, 1)

            # --- Deteksi Orang Menggunakan YOLOv8 ---
            # 'verbose=False' untuk menonaktifkan output log YOLO
            # 'conf' untuk ambang batas kepercayaan deteksi objek YOLO
            # 'classes=[PERSON_CLASS_ID]' untuk hanya mendeteksi kelas 'person'
            results = person_detector(frame, verbose=False, conf=CONFIDENCE_THRESHOLD_PERSON, classes=[PERSON_CLASS_ID])

            # Iterasi melalui setiap deteksi orang
            for r in results:
                boxes = r.boxes # Mendapatkan bounding boxes dari hasil deteksi
                for box in boxes:
                    # box.xyxy[0] memberikan [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Pastikan koordinat bounding box tidak di luar batas frame
                    y1 = max(0, y1)
                    y2 = min(frame.shape[0], y2)
                    x1 = max(0, x1)
                    x2 = min(frame.shape[1], x2)

                    # Pastikan area yang dipotong valid
                    if x2 > x1 and y2 > y1:
                        # Potong area orang dari frame asli
                        person_patch = frame[y1:y2, x1:x2]

                        # --- Klasifikasi Smoking/Not_Smoking pada area orang yang terdeteksi ---
                        label, confidence = predict_smoking_status(person_patch)

                        # Tentukan warna untuk bounding box dan teks
                        # Merah untuk 'Smoking', Hijau untuk 'Not_Smoking'
                        color = (0, 0, 255) if label == "Smoking" else (0, 255, 0)
                        
                        # Gambar bounding box di sekitar orang
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Siapkan teks untuk ditampilkan
                        display_text = f"Orang ({label}): {confidence:.2f}"
                        
                        # Tambahkan teks label di atas bounding box
                        # Hitung posisi teks agar tidak keluar dari frame
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 # Pindahkan ke bawah jika terlalu dekat atas
                        cv2.putText(frame, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # Hitung dan tampilkan FPS (Frames Per Second)
            current_time = time.time()
            if (current_time - start_time) > 1: # Update FPS setiap 1 detik
                fps = frame_count / (current_time - start_time)
                start_time = current_time
                frame_count = 0
            
            # Tampilkan FPS di sudut kiri atas frame
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame ke format JPEG untuk streaming web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\n'
                    b'Content-Type: image/jpeg\n\n' + frame_bytes + b'\n')

    # Setelah loop selesai, bebaskan sumber video dan tutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()
    print("Stream video selesai.")

# --- 6. Bagian Aplikasi Flask ---
# Menginisialisasi aplikasi Flask dan menentukan folder template sebagai direktori saat ini ('.')
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    """Halaman utama aplikasi web."""
    # Flask sekarang akan mencari 'index.html' di direktori yang sama dengan 'app.py'
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint untuk streaming video."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n🚀 Menjalankan aplikasi Flask...")
    print("Akses aplikasi di http://127.0.0.1:5000 di browser Anda.")
    # 'debug=True' untuk pengembangan, 'use_reloader=False' agar tidak memuat ulang model dua kali
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
