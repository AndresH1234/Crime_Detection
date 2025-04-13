import os
import cv2
import numpy as np
import time
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import KerasI3D

i3d = KerasI3D.KerasI3D
model_1 = i3d.build_model(num_classes=2, initial_learning_rate=1e-4)

# Cargar los mejores pesos entrenados
checkpoint_dir = "/mnt/DGX0Raid/aherrerag/Crime_Detection/notebooks/checkpoints/i3d_model"
model_1.load_weights(os.path.join(checkpoint_dir, "best_model"))

# Clase que indica crimen (ajústalo a tu caso)
CRIME_CLASS = 1  # Por ejemplo, clase 1 = actividad sospechosa

def preprocess(frame):
    # Redimensionar al tamaño que el modelo espera, ej: (224, 224)
    resized = cv2.resize(frame, (224, 224))
    # Normalizar (según cómo entrenaste)
    normalized = resized / 255.0
    return normalized

def trigger_alert():
    print("¡Actividad sospechosa detectada!")

def main():
    num_frames = 16  # Número de frames que espera el modelo
    video = '/mnt/DGX0Raid/aherrerag/Crime_Detection/data/test/Assault007_x264.mp4'
    import os
    print(os.path.exists(video))  
    video_stream = cv2.VideoCapture(video)  # Cambiar si usas un archivo local

    if not video_stream.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    try:
        while True:
            frames = []
            for _ in range(num_frames):
                ret, frame = video_stream.read()
                if not ret:
                    print("No se pudo leer el frame.")
                    time.sleep(0.1)
                    continue
                processed = preprocess(frame)  # Asegúrate que devuelve (224, 224, 3)
                frames.append(processed)

            if len(frames) < num_frames:
                continue

            # Convertir a formato esperado por el modelo
            input_tensor = np.expand_dims(frames, axis=0)  # (1, 16, 224, 224, 3)
            input_tensor = np.array(input_tensor, dtype=np.float32)

            # Hacer predicción
            prediction = model_1.predict(input_tensor)
            predicted_class = np.argmax(prediction, axis=-1)  # Devuelve array tipo [clase]

            if predicted_class[0] == CRIME_CLASS:
                print("Actividad sospechosa detectada!")
                trigger_alert()

            # Agrega salida con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    input("Desea iniciar la detección de crimen? (Presione Enter para continuar)")
    main()