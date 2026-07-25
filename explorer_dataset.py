import os
import matplotlib.pyplot as plt
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSSIER = os.path.join(BASE_DIR, "dataset")
classes = sorted(os.listdir(DOSSIER))

# Vérifier équilibre des classes
comptes = [len(os.listdir(os.path.join(DOSSIER, c))) for c in classes]

plt.figure(figsize=(8, 4))
plt.bar(classes, comptes)
plt.title("Répartition des images par classe")
plt.ylabel("Nombre d'images")
plt.show()

# Afficher la grille d'échantillons
fig, axes = plt.subplots(len(classes), 5, figsize=(12, 3 * len(classes)), squeeze=False)

for ligne, classe in enumerate(classes):
    fichiers = sorted(os.listdir(os.path.join(DOSSIER, classe)))[:5]
    for col, fichier in enumerate(fichiers):
        image = cv2.imread(os.path.join(DOSSIER, classe, fichier))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV lit en BGR !
        axes[ligne, col].imshow(image)
        axes[ligne, col].axis("off")
    axes[ligne, 0].set_title(classe, loc="left", fontsize=12, fontweight="bold", pad=10)

plt.subplots_adjust(hspace=0.6)
plt.show()
