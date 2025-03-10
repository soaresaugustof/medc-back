import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
       'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
       'Pleural Thickening', 'Pneumonia', 'Pneumothorax', 'Pneumoperitoneum',
       'Pneumomediastinum', 'Subcutaneous Emphysema', 'Tortuous Aorta',
       'Calcification of the Aorta']

def preprocess_image_with_generator(img_path, target_size=(320, 320)):
    # Cria um ImageDataGenerator com apenas a normalização usada no treino
    image_generator = ImageDataGenerator(
        samplewise_center=True,  # Centraliza por imagem individualmente
        samplewise_std_normalization=True  # Normaliza cada imagem
    )
    
    # Carrega a imagem
    img = load_img(img_path, target_size=target_size)
    
    # Converte a imagem em array numpy
    img_array = img_to_array(img)
    
    # Adiciona uma dimensão extra para representar o batch (necessário para a entrada do modelo)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Aplica o ImageDataGenerator ao array de imagem
    img_array = image_generator.standardize(img_array)
    
    return img_array

def weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    def loss(y_true, y_pred):
        # Converta y_true e y_pred para float32 para garantir a compatibilidade
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Initialize loss to zero
        loss = 0.0

        # Certifique-se de que os pesos sejam tensores
        pos_weights_tensor = tf.convert_to_tensor(pos_weights, dtype=tf.float32)
        neg_weights_tensor = tf.convert_to_tensor(neg_weights, dtype=tf.float32)

        for i in range(len(pos_weights)):
            # Para cada classe, adicione a média da perda ponderada para essa classe 
            class_loss = -(
                pos_weights_tensor[i] * y_true[:, i] * tf.math.log(y_pred[:, i] + epsilon) +
                neg_weights_tensor[i] * (1 - y_true[:, i]) * tf.math.log(1 - y_pred[:, i] + epsilon)
            )
            loss += tf.reduce_mean(class_loss)
        
        return loss

    return loss


img = preprocess_image_with_generator('cardiom3.png', target_size=(320, 320))

model = load_model('2409_2004i.h5', custom_objects = {'loss': weighted_loss})

predictions = model.predict(img)

for label, pred in zip(labels, predictions[0]):
    print(f'{label}: {pred * 100:.2f}%')