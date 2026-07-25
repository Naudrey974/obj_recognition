# Setup de l'environnement — obj_recognition

Reconnaissance de 3 objets (+ classe "background"/"autre") en deep learning.

## Commandes (adaptées de votre tuto)

```powershell
# 1. Créer un environnement isolé (évite de casser vos autres projets Python)
conda create --name obj_recognition python=3.10 -y

# 2. L'activer (à refaire à CHAQUE ouverture de terminal !)
conda activate obj_recognition

# 3. Installer les bibliothèques
pip install tensorflow
pip install opencv-python   # capture webcam et traitement d'image
pip install matplotlib
pip install numpy
pip install scikit-learn    # matrice de confusion / rapport de classification
pip install tensorboard
```

Ou en une fois, une fois l'environnement activé :

```powershell
pip install -r requirements.txt
```

## Structure du projet

```
obj_recognition/
├── data/
│   ├── raw/          # images brutes capturées (webcam) par classe
│   └── processed/    # images prétraitées / splits train-val-test
├── models/            # modèles entraînés sauvegardés
├── notebooks/          # exploration / expérimentation
├── src/                # code source (capture, prétraitement, entraînement, inférence)
├── logs/                # logs TensorBoard
├── requirements.txt
└── SETUP.md
```

## Classes

4 classes au total : 3 objets à reconnaître + 1 classe "background" (ou "autre") pour différencier l'absence des 3 objets cibles.
