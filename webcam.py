import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf

#paramètre
TAILLE = 224   # taille d'image pour le modèle
SEUIL  = 0.80  # seuil de confiance pour valider la prediction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Les 5 classes dans le même ordre qu'à l'entrainement
noms_classes = ["autres", "camion", "fond", "licorne", "mosaique"]

# Message affiché par classe
MESSAGES = {
    "licorne": "Ah! la jolie licorne!",
    "camion": "Il est beau ton camion!",
    "mosaique": "Tu t'es mis au loisir créatif !",
    "autres": "Je ne reconnais pas cet objet...",
}

# ============================================================
# 1. CHARGER LE MODÈLE ENTRAÎNÉ
# ============================================================
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "modele.keras"))

# ============================================================
# 2. OUVRIR LA WEBCAM
#    0 = webcam par défaut (essayer 1 si ça ne marche pas)
# ============================================================
cap = cv2.VideoCapture(0)

# ============================================================
# 3. BOUCLE PRINCIPALE : lire une image, prédire, afficher
# ============================================================
while True:
    # Lire une image de la webcam
    ok, frame = cap.read()
    if not ok:
        break

    #préparation de l'image pour le modèle (convertit RGB/taille/nb dimension)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (TAILLE, TAILLE))
    image = np.expand_dims(image, axis=0)

    #prediction du modèle (1 par classe)
    proba = model.predict(image, verbose=0)[0]
    indice = np.argmax(proba)             
    classe = noms_classes[indice]
    confiance = proba[indice] * 100

    # affichage du msg sur la webcam
    if classe != "fond":
       #utilisation du seuil pour éviter les fausses prédictions
        if proba[indice] < SEUIL:
            texte = f"Incertain ({confiance:.0f}%)"
            couleur = (0, 165, 255)
        else:
            texte = f"{MESSAGES[classe]}"
            couleur = (0, 255, 0)

        cv2.putText(frame, texte, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, couleur, 2)

    cv2.imshow("Reconnaissance d'objets", frame)

    # quitter en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#fermeture
cap.release()
cv2.destroyAllWindows()