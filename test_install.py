import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les warnings et infos de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Désactive le message oneDNN spécifié dans votre terminal

import tensorflow as tf
import cv2
import numpy as np

print("TensorFlow :", tf.__version__)
print("OpenCV :", cv2.__version__)

# Test de la webcam
cap = cv2.VideoCapture(0)  # 0 = webcam par défaut
if not cap.isOpened():
    print("Webcam INTROUVABLE — essayez cv2.VideoCapture(1)")
else:
    ok, frame = cap.read()
    print("Webcam OK — résolution :", frame.shape)
cap.release()
