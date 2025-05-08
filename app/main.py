
"""Comando para ejecutar el script:
PYTHONPATH=/mnt/DGX0Raid/aherrerag/Crime_Detection /home/aherrerag/anaconda3/envs/proyecto_i3d/bin/python /mnt/DGX0Raid/aherrerag/Crime_Detection/app/main.py --video /mnt/DGX0Raid/aherrerag/Crime_Detection/data/test/Robbery034_x264.mp4
"""

import tensorflow as tf
from kerastuner.tuners import RandomSearch
from tensorflow.keras import Model
from tensorflow.keras.layers import (ConvLSTM2D, Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
import numpy as np
import cv2
import tempfile
import os
import matplotlib.pyplot as plt
import argparse
from models.i3d import InceptionI3d

# --- CONFIG
CHECKPOINT_DIR = "/mnt/DGX0Raid/aherrerag/Crime_Detection/notebooks/checkpoints/i3d_convlstm_1/best_model"
NUM_FRAMES = 32
FRAME_SIZE = (224, 224)

# --- LOAD MODEL
def load_model():
    def build_model(hp):
        class Tuned_I3D_ConvLSTM(Model):
            def __init__(self, num_classes):
                super().__init__()
                self.i3d = InceptionI3d(num_classes=num_classes, is_training=True, final_endpoint="Mixed_5c")

                # Convertimos las opciones de kernel_size en strings y las evaluamos después
                kernel_size_choice = hp.Choice('kernel_size', ['3x3', '5x5'])
                if kernel_size_choice == '3x3':
                    kernel_size = (3, 3)
                else:
                    kernel_size = (5, 5)

                self.conv_lstm = ConvLSTM2D(
                    filters=hp.Choice('filters', [32, 64, 128]),
                    kernel_size=kernel_size,
                    padding='same',
                    return_sequences=False,
                    dropout=hp.Choice('dropout', [0.2, 0.3, 0.5]),
                    recurrent_dropout=hp.Choice('recurrent_dropout', [0.2, 0.3, 0.5]),
                    activation='tanh'
                )

                self.batch_norm = BatchNormalization()
                self.dense = Dense(hp.Choice('dense_units', [64, 128, 256]), activation='relu')
                self.dropout = Dropout(hp.Choice('dense_dropout', [0.2, 0.3, 0.5]))
                self.fc = Dense(num_classes, activation='sigmoid')

            def call(self, inputs, training=False):
                features, _ = self.i3d(inputs)
                x = self.conv_lstm(features)
                x = self.batch_norm(x, training=training)
                x = GlobalAveragePooling2D()(x)
                x = self.dense(x)
                x = self.dropout(x, training=training)
                return self.fc(x)

        num_classes = 2
        model = Tuned_I3D_ConvLSTM(num_classes=num_classes)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
        return model
    
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='i3d_convlstm_tuning'
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    tuner.search(None, 
                epochs=10, 
                validation_data=None, 
                callbacks=[early_stopping], 
                verbose=1)
    
    # Obtener el mejor conjunto de hiperparámetros
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Reconstruir el modelo usando los mejores hiperparámetros
    model = build_model(best_hp)

    # Compilar el modelo (esto ya se hace dentro de build_model, pero puedes asegurarte aquí)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy", AUC(name="auc")])
    model.load_weights(CHECKPOINT_DIR)

    return model


def preprocess_frame(frame, frame_size=FRAME_SIZE):
    frame = cv2.resize(frame, frame_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame / 255.0

def predict_crime(model, frames_window):
    input_tensor = np.expand_dims(np.array(frames_window, dtype=np.float32), axis=0)
    prediction = model.predict(input_tensor)[0]  # Devuelve [prob_no_crime, prob_crime]
    return prediction

def simulate_realtime_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    buffer = []

    frame_count = 0
    print(f"\nAnalizando {video_path}...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed = preprocess_frame(frame)
        buffer.append(preprocessed)

        if len(buffer) == NUM_FRAMES:
            prediction = predict_crime(model, buffer)
            prob_crime = prediction[1] * 100  # Suponemos clase 1 = crimen

            print(f"Frame {frame_count}: Probabilidad de crimen: {prob_crime:.2f}%")

            buffer.pop(0)  # Mantener ventana deslizante

        frame_count += 1

    cap.release()

def main():
    print("Cargando modelo...")
    model = load_model()

    video_dir = "./videos"
    video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi"))]

    if len(video_files) < 2:
        print("Se necesitan al menos dos videos en la carpeta /videos.")
        return

    for video_file in video_files[:2]:
        video_path = os.path.join(video_dir, video_file)
        simulate_realtime_from_video(video_path, model)

if __name__ == "__main__":
    main()