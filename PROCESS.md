# Journal de bord — obj_recognition

Projet de soutenance deep learning : reconnaissance de 3 objets + 1 classe "fond" (aucun des 3 objets).

## Objectif

Classifieur 5 classes (softmax) capable de distinguer en temps réel via webcam :
1. `deodorant`
2. `mosaique`
3. `balle`
4. `fond` — aucun objet distinct dans le cadre (décor vide, main/visage seuls)
5. `autres` — un objet distinct présent mais différent des 3 cibles (stylo, bouteille, clés, tasse…)

Les classes `fond` et `autres` sont nécessaires car softmax force la somme des probabilités à 100 % : sans elles, le modèle serait obligé d'attribuer une des 3 classes cibles même en leur absence.

**Décision (2026-07-25)** : séparer `fond` et `autres` en deux classes distinctes plutôt qu'un unique catch-all, pour une distinction plus fine (absence d'objet vs présence d'un objet non-cible). Frontière retenue : "y a-t-il un objet net et reconnaissable dans le cadre ?" → oui = `autres`, non = `fond`. Voir `videos/README.md` pour le détail et les cas limites à éviter.

## Étapes réalisées

### 1. Environnement (2026-07-25)
- Anaconda déjà installé (`C:\Users\audre\anaconda3`) mais absent du PATH → utilisation directe du hook conda PowerShell.
- Environnement `obj_recognition` créé (Python 3.10).
- Bibliothèques installées : `tensorflow`, `opencv-python`, `matplotlib`, `numpy`, `scikit-learn`, `tensorboard`.
- `pandas` volontairement absent : nos données sont des images organisées en dossiers, pas du tabulaire — on utilisera `image_dataset_from_directory` de Keras.
- Vérification (`test_install.py`) : TensorFlow 2.21.0, OpenCV 5.0.0, webcam détectée (résolution 640×480).

### 2. Structure du projet
```
obj_recognition/
├── videos/            # vidéos sources filmées par l'utilisateur (une par classe)
│   └── README.md      # protocole de tournage
├── dataset/            # images extraites, une sous-classe par dossier
│   ├── deodorant/
│   ├── mosaique/
│   ├── balle/
│   ├── fond/
│   └── autres/
├── models/              # modèles entraînés sauvegardés
├── logs/                  # logs TensorBoard
├── requirements.txt
├── SETUP.md              # procédure d'installation de l'environnement
├── PROCESS.md            # ce fichier — journal de bord
├── test_install.py       # vérification TF / OpenCV / webcam
└── extraire_frames.py    # extraction d'images depuis les vidéos
```

### 3. Choix des objets
- **déodorant** (cylindre allongé), **mosaïque carrée** (plat, carré), **balle bleue** (sphère) : formes et couleurs bien distinctes, objets rigides.
- Points de vigilance identifiés : taille de la mosaïque (filmer d'assez près), reflets sur surfaces brillantes (capuchon du déodorant, mosaïque vernie).

### 4. Script d'extraction de frames
- `extraire_frames.py` : lit chaque vidéo de `videos/*.mp4`, garde 1 frame sur 8 (~150 images pour une vidéo de 30-40s à 30 fps), sauvegarde dans `dataset/<classe>/`.
- Le nom du fichier vidéo (sans extension) = nom de la classe → aucune configuration supplémentaire nécessaire.

## Prochaines étapes

1. **Tournage** (utilisateur) : filmer `deodorant.mp4`, `mosaique.mp4`, `balle.mp4`, `fond.mp4`, `autres.mp4` selon le protocole dans `videos/README.md`, les placer dans `videos/`.
2. Lancer `python extraire_frames.py` pour peupler `dataset/`.
3. `explorer_dataset.py` : visualiser des échantillons, vérifier l'équilibre des classes.
4. `train.py` : construire et entraîner le modèle (Keras `image_dataset_from_directory`, suivi TensorBoard).
5. `finetune.py` : affiner le modèle.
6. `evaluer.py` : matrice de confusion / rapport de classification (scikit-learn).
7. `webcam.py` : démo d'inférence en temps réel.
