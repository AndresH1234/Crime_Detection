import os
import sys
# Agrega el path al directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.i3d import InceptionI3d
import tensorflow as tf

# Clase adaptadora para usar InceptionI3D como modelo de Keras
class KerasI3D(tf.keras.Model):
    def __init__(self, num_classes, endpoint = "Logits"):
        super(KerasI3D, self).__init__()
        self.i3d = InceptionI3d(num_classes=num_classes, is_training=True, final_endpoint=endpoint)

    def call(self, inputs, training=False):
        logits, _ = self.i3d(inputs)  # El modelo I3D devuelve logits y endpoints
        return logits
    
    def build_model(num_classes=2, initial_learning_rate=1e-4):
        model = KerasI3D(num_classes=num_classes)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate
        )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
        return model