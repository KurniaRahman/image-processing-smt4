import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# Inisialisasi Text-to-Speech (TTS) engine
engine = pyttsx3.init()

# Memuat model
with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Variabel untuk logika baru
is_recording = False
sentence = []
last_added_char = ''
hand_detected_time = None
STABILITY_DELAY = 2  # Tunggu 1.5 detik hingga gestur stabil

print("Tekan 'R' untuk Mulai/Stop. Tekan 'Q' untuk Keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Logika deteksi
    if results.multi_hand_landmarks and is_recording:
        # Jika tangan terdeteksi dan sedang dalam mode rekam
        if hand_detected_time is None:
            # Mulai timer saat tangan pertama kali terdeteksi
            hand_detected_time = time.time()
        
        # Cek apakah tangan sudah stabil cukup lama
        elif time.time() - hand_detected_time > STABILITY_DELAY:
            data_aux, x_, y_ = [], [], []
            hand_landmarks = results.multi_hand_landmarks[0]
            
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            current_char = prediction[0]

            # Tambahkan huruf jika berbeda dari yang terakhir ditambahkan
            if current_char != last_added_char:
                sentence.append(current_char)
                last_added_char = current_char
                print(f"Huruf terdeteksi: {current_char}")

            # Reset timer agar program menunggu gestur stabil berikutnya
            hand_detected_time = None
        
        # Menggambar landmarks
        mp_drawing.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    else:
        # Jika tidak ada tangan terdeteksi, reset timer
        hand_detected_time = None
        last_added_char = ''

    # Tampilkan UI
    status_text = "MEREKAM" if is_recording else "TEKAN 'R' UNTUK MULAI"
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
    if sentence:
        cv2.putText(frame, ''.join(sentence), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('r'):
        is_recording = not is_recording
        if not is_recording:
            # Saat berhenti merekam
            if sentence:
                final_word = ''.join(sentence)
                print(f"Hasil akhir: {final_word}")
                engine.say(final_word)
                engine.runAndWait()
        else:
            # Saat mulai merekam
            print("Mulai merekam...")
            sentence = []
            hand_detected_time = None
            last_added_char = ''

cap.release()
cv2.destroyAllWindows()