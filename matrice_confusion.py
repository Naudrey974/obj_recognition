import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Masque les warnings et infos de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Désactive le message oneDNN

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# mêmes réglages que dans train.py
TAILLE = 224
BATCH  = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSSIER  = os.path.join(BASE_DIR, "dataset")

# rechargement du jeu de validation comme dans train.py
val_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER,
    validation_split=0.2,
    subset="validation",
    seed=123, # identique à train.py
    image_size=(TAILLE, TAILLE),
    batch_size=BATCH
)

#on récupère le nom des classes dans l'ordre
noms_classes = val_ds.class_names

# rechargemet du modèle déja entrainé
chemin_modele = os.path.join(BASE_DIR, "models", "modele.keras")
model = tf.keras.models.load_model(chemin_modele)

# prédictions du modèle
vraies_etiquettes = []   # vérité
predictions       = []   # réponses du modèle

#on parcourt le jeu de validation par lots 
# et pour chaque lot on récupère la bonne réponse et la reponse du modèle
for images, etiquettes in val_ds:
    #le modèle renvoie pour chaque image, 5 proba (1 par classe)
    proba = model.predict(images, verbose=0)
    classes_predites = np.argmax(proba, axis=1) 
    
    #on stocke les résultats dans ces deux listes
    predictions.extend(classes_predites)
    vraies_etiquettes.extend(etiquettes.numpy())

#on tranforme les listes en tableau
vraies_etiquettes = np.array(vraies_etiquettes)
predictions       = np.array(predictions)

# construction de la matrice
matrice = tf.math.confusion_matrix(
    labels=vraies_etiquettes,
    predictions=predictions,
    num_classes=len(noms_classes)
).numpy()

# affichage de la matrice avec Matplotlib
plt.figure(figsize=(7, 6))

plt.imshow(matrice, cmap="Blues")
plt.colorbar() # légende barre de couleur

# noms des classes 
plt.xticks(range(len(noms_classes)), noms_classes, rotation=45)
plt.yticks(range(len(noms_classes)), noms_classes)
plt.xlabel("Prédiction du modèle")
plt.ylabel("Vérité (classe réelle)")
plt.title("Matrice de confusion")

# nombres dans les cases
seuil = matrice.max() / 2
for i in range(len(noms_classes)):
    for j in range(len(noms_classes)):
        couleur = "white" if matrice[i, j] > seuil else "black"
        plt.text(j, i, matrice[i, j], ha="center", va="center", color=couleur)

plt.tight_layout()

# sauvegarde de l'image de la matrice
chemin_image = os.path.join(BASE_DIR, "matrice_confusion.png")
plt.savefig(chemin_image, dpi=150)

plt.show()