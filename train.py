import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les warnings et infos de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Désactive le message oneDNN

import datetime
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers

#parametres
TAILLE = 224
BATCH  = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSSIER  = os.path.join(BASE_DIR, "dataset")

#chargement des données
train_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER, 
    validation_split=0.2, 
    subset="training", 
    seed=123,
    image_size=(TAILLE, TAILLE), 
    batch_size=BATCH
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER, 
    validation_split=0.2, 
    subset="validation", 
    seed=123,
    image_size=(TAILLE, TAILLE), 
    batch_size=BATCH
)

noms_classes = train_ds.class_names
print("Classes :", noms_classes)

#Organisation des logs dans des sous-dossiers séparé
horodatage = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(BASE_DIR, "logs", f"run_{horodatage}")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#Récupération du CNN
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(TAILLE, TAILLE, 3), include_top=False, weights="imagenet"
)
print(base_model.output_shape) #taille couche de sortie
print(base_model.count_params()) #taille du reseau complet (une fois dégélée)

# Phase A : Feature Extraction
base_model.trainable = False   # On gèle le modèle de base

model = tf.keras.Sequential([
    layers.Input(shape=(TAILLE, TAILLE, 3)),
    layers.Rescaling(1./127.5, offset=-1), # MobileNetV2 attend des pixels entre -1 et 1
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(noms_classes), activation="softmax")
])

model.compile(
    loss = losses.SparseCategoricalCrossentropy(),
    optimizer = optimizers.Adam(learning_rate=1e-3),
    metrics = ["accuracy"]
)

print("Phase A : Feature Extraction")
model.fit(train_ds, validation_data=test_ds, epochs=10,
          callbacks=[tensorboard], verbose=2)


#Phase B : fine tuning
base_model.trainable = True   # On dégèle toute la base

model.compile(
    loss = losses.SparseCategoricalCrossentropy(),
    optimizer = optimizers.Adam(learning_rate=1e-4),
    metrics = ["accuracy"]
)

# organisation des logs pour la phase B 
log_dir_B = os.path.join(BASE_DIR, "logs", f"phaseB_{horodatage}")
tensorboard_B = tf.keras.callbacks.TensorBoard(log_dir=log_dir_B)

print("Phase B : Fine Tuning")
model.fit(train_ds, validation_data=test_ds, epochs=10,
          callbacks=[tensorboard_B], verbose=2)

model.evaluate(test_ds, verbose=2)

model.save(os.path.join(BASE_DIR, "models", "modele.keras"))
print("Modèle sauvegardé dans models/modele.keras")