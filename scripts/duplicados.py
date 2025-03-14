import os
import cv2
import pandas as pd
import imagehash
from PIL import Image

class Duplicados:
    def __init__(self, data_folder, df_videos, time_sec=10):
        self.data_folder = data_folder
        self.df_videos = df_videos
        self.time_sec = time_sec

    def extract_frame(self, video_path):
        """ Extrae un fotograma en el segundo especificado. """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, self.time_sec * 1000)
        success, frame = cap.read()
        cap.release()
        return frame if success else None

    def get_video_hash(self, video_path):
        """ Calcula el hash perceptual del fotograma extraído. """
        frame = self.extract_frame(video_path)
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return imagehash.phash(img)
        return None

    def encontrar_duplicados(self):
        """ Encuentra videos duplicados según su hash perceptual. """
        video_hashes = {}
        duplicates = []

        for _, row in self.df_videos.iterrows():
            evento = row["Evento"]
            video = row["Video"]
            video_path = os.path.join(self.data_folder, evento, video)

            if not os.path.exists(video_path):
                print(f"[ERROR] No se encontró el archivo: {video_path}")
                continue
            
            if not video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  
                continue  # Ignorar archivos que no sean videos
            
            video_hash = self.get_video_hash(video_path)

            if video_hash is not None:
                if video_hash in video_hashes:
                    print(f"[DUPLICADO] {video} es similar a {video_hashes[video_hash]}")
                    duplicates.append(row)
                else:
                    video_hashes[video_hash] = video

        # Convertir los duplicados en un DataFrame
        df_duplicados = pd.DataFrame(duplicates)

        return df_duplicados
