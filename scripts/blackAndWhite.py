import cv2
import numpy as np
import os
import pandas as pd

class BlackAndWhite:
    def __init__(self, data_folder: str, df_videos: pd.DataFrame):
        self.data_folder = data_folder
        self.df_videos = df_videos

    def es_blanco_y_negro(self, video_path, sample_rate=10, threshold=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyzed_frames = 0
        bw_frames = 0

        for i in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calcular la diferencia entre los canales de color y la versión en grises
            diff_b = np.abs(frame[:, :, 0] - gray)
            diff_g = np.abs(frame[:, :, 1] - gray)
            diff_r = np.abs(frame[:, :, 2] - gray)

            # Usar desviación estándar como métrica de variabilidad de color
            diff_std = np.std([diff_b, diff_g, diff_r])

            if diff_std < threshold:
                bw_frames += 1

            analyzed_frames += 1

        cap.release()
        return (bw_frames / analyzed_frames) > 0.9 if analyzed_frames > 0 else False

    def encontrar_bw(self):
        resultados = []
        
        for _, row in self.df_videos.iterrows():
            video_path = os.path.join(self.data_folder, row['Evento'], row['Video'])
            if os.path.exists(video_path) and self.es_blanco_y_negro(video_path):
                print(f"[Blanco y Negro] Video: {row['Video']}")
                resultados.append(row)
        
        return pd.DataFrame(resultados)
