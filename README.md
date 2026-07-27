# obj_recognition

Projet de soutenance deep learning (ESGI) : reconnaissance d'objets en temps réel via webcam.

Un modèle (transfer learning sur MobileNetV2) distingue 5 classes :
- **licorne**, **camion**, **mosaique** — les 3 objets cibles
- **fond** — rien de particulier dans le cadre (décor vide, main/visage seuls)
- **autres** — un objet distinct présent, mais différent des 3 cibles

`fond` et `autres` sont nécessaires car la couche de sortie (softmax) force la somme des probabilités à 100 % : sans elles, le modèle serait obligé d'attribuer une des 3 classes cibles même en leur absence.

## Installation

```powershell
# 1. Créer un environnement isolé
conda create --name obj_recognition python=3.10 -y

# 2. L'activer (à refaire à chaque ouverture de terminal)
conda activate obj_recognition

# 3. Installer les bibliothèques
pip install tensorflow opencv-python matplotlib numpy
```

Paquets réellement utilisés par le projet :
- **tensorflow** — chargement du dataset, modèle (MobileNetV2), entraînement, inférence
- **opencv-python** (`cv2`) — capture webcam, lecture/écriture d'images
- **matplotlib** — visualisation du dataset (`explorer_dataset.py`) et matrice de confusion (`matrice_confusion.py`)
- **numpy** — manipulation des tableaux de prédictions (`webcam.py`, `matrice_confusion.py`)

`tensorboard` (utilisé pour la commande `tensorboard --logdir logs` plus bas) n'a pas besoin d'être installé séparément : c'est une dépendance de `tensorflow`, déjà présente après l'installation ci-dessus.

## Utilisation

Lancer les scripts dans cet ordre, depuis le dossier du projet (environnement `obj_recognition` activé) :

1. **`test_install.py`** — vérifie que tout fonctionne (TensorFlow, OpenCV, webcam) :
   ```powershell
   python test_install.py
   ```

2. **`extraire_frames.py`** — capture les images du dataset directement depuis la webcam.
   Ouvrir le fichier, changer `NOM_CLASSE` (`licorne`, `mosaique`, `camion`, `fond` ou `autres`), puis lancer :
   ```powershell
   python extraire_frames.py
   ```
   Relancer une fois par classe. `ESPACE` = pause/reprise (pour changer de fond/éclairage sans gaspiller de frames), `q` = quitter avant la fin. Le script reprend le comptage là où il s'était arrêté si la classe est déjà partiellement capturée.

3. **`explorer_dataset.py`** — vérifie l'équilibre des classes et affiche un échantillon aléatoire d'images par classe :
   ```powershell
   python explorer_dataset.py
   ```

4. **`train.py`** — entraîne le modèle (transfer learning MobileNetV2, phase A feature extraction puis phase B fine-tuning) :
   ```powershell
   python train.py
   ```
   Sauvegarde le modèle entraîné dans `models/modele.keras`. Suivi TensorBoard :
   ```powershell
   tensorboard --logdir logs
   ```

5. **`matrice_confusion.py`** — évalue le modèle sur le jeu de validation et affiche/sauvegarde sa matrice de confusion :
   ```powershell
   python matrice_confusion.py
   ```
   Génère `matrice_confusion.png`.

6. **`webcam.py`** — démo d'inférence en temps réel :
   ```powershell
   python webcam.py
   ```
   Affiche un message personnalisé par classe reconnue + le niveau de confiance. En dessous du seuil de confiance (`SEUIL` dans le fichier), affiche "Incertain" plutôt qu'une classe cible, pour éviter de classer à tort un objet inconnu. `q` = quitter.

## Structure du projet

```
obj_recognition/
├── dataset/               # images capturées, une sous-classe par dossier
│   ├── licorne/
│   ├── mosaique/
│   ├── camion/
│   ├── fond/
│   └── autres/
├── models/                # modèle entraîné (modele.keras)
├── logs/                  # logs TensorBoard, un sous-dossier horodaté par run
├── test_install.py        # vérification TF / OpenCV / webcam
├── extraire_frames.py     # capture d'images par classe depuis la webcam
├── explorer_dataset.py    # visualisation d'échantillons + équilibre des classes
├── train.py               # entraînement (transfer learning MobileNetV2)
├── matrice_confusion.py   # évaluation : matrice de confusion sur le jeu de validation
└── webcam.py              # démo d'inférence en temps réel
```
