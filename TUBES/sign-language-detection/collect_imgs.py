import os
import cv2
import string

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = list(string.ascii_uppercase)
dataset_size = 120

cap = cv2.VideoCapture(0)
exit_program = False

for char_class in classes:
    class_dir = os.path.join(DATA_DIR, char_class)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Mengumpulkan data untuk kelas: {char_class}. Tekan ESC untuk keluar.')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Siap? Tekan "Q" untuk kelas {char_class}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 , 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == 27: # Tombol ESC
            exit_program = True
            break
    
    if exit_program:
        break

    counter = 0
    cancelled = False
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        
        if key == ord('c'):
            print(f'Proses untuk kelas {char_class} dibatalkan.')
            cancelled = True
            break
        elif key == 27: # Tombol ESC
            exit_program = True
            break
            
        counter += 1

    if cancelled:
        try:
            for i in range(counter + 1):
                os.remove(os.path.join(class_dir, f'{i}.jpg'))
            if not os.listdir(class_dir):
                os.rmdir(class_dir)
        except Exception as e:
            print(f"Error: {e}")

    if exit_program:
        break

print("Keluar dari program.")
cap.release()
cv2.destroyAllWindows()