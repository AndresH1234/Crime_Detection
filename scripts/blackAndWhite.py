import cv2
import numpy as np

def es_blanco_y_negro(video_path, sample_rate=10, threshold=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analyzed_frames = 0
    bw_frames = 0

    for i in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convertir a escala de grises y comparar con el original
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame, cv2.merge([gray, gray, gray]))  # Comparar cada canal con la escala de grises
        diff_mean = np.mean(diff)  # Promedio de diferencias

        if diff_mean < threshold:
            bw_frames += 1  # Se considera blanco y negro

        analyzed_frames += 1

    cap.release()
    
    # Si la mayoría de los fotogramas son blanco y negro, el video se considera blanco y negro
    return bw_frames / analyzed_frames > 0.9

# Uso del programa
video_path = "video.mp4"
if es_blanco_y_negro(video_path):
    print("El video es en blanco y negro.")
else:
    print("El video es a color.")
