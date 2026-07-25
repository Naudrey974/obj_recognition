# Journal de bord — obj_recognition

Projet de soutenance deep learning : reconnaissance de 3 objets + 1 classe "fond" (aucun des 3 objets).

## Objectif

Classifieur 5 classes (softmax) capable de distinguer en temps réel via webcam :
1. `licorne`
2. `mosaique`
3. `camion`
4. `fond` — aucun objet distinct dans le cadre (décor vide, main/visage seuls)
5. `autres` — un objet distinct présent mais différent des 3 cibles (stylo, bouteille, clés, tasse…)

Les classes `fond` et `autres` sont nécessaires car softmax force la somme des probabilités à 100 % : sans elles, le modèle serait obligé d'attribuer une des 3 classes cibles même en leur absence.

**Décision (2026-07-25)** : séparer `fond` et `autres` en deux classes distinctes plutôt qu'un unique catch-all, pour une distinction plus fine (absence d'objet vs présence d'un objet non-cible). Frontière retenue : "y a-t-il un objet net et reconnaissable dans le cadre ?" → oui = `autres` (stylo, bouteille, clés, tasse, une autre balle de couleur différente…), non = `fond` (décor vide, main/visage seuls sans rien tenir). Cas limite à éviter : une main qui *tient* un objet est plus proche de `autres` que de `fond`.

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
├── dataset/            # images capturées, une sous-classe par dossier
│   ├── licorne/
│   ├── mosaique/
│   ├── camion/
│   ├── fond/
│   └── autres/
├── models/              # modèles entraînés sauvegardés
├── logs/                  # logs TensorBoard
├── requirements.txt
├── SETUP.md              # procédure d'installation de l'environnement
├── PROCESS.md            # ce fichier — journal de bord
├── test_install.py       # vérification TF / OpenCV / webcam
└── extraire_frames.py    # capture d'images par classe depuis la webcam
```

### 3. Choix des objets
- Choix initial : **déodorant** (cylindre allongé), **mosaïque carrée** (plat, carré), **balle bleue** (sphère) : formes et couleurs bien distinctes, objets rigides.
- **Révision (2026-07-25)** : `deodorant` → **licorne** (jouet, plus facile à travailler pour les images) ; `balle` → **camion** (jouet). `mosaique` conservée. Les 150 premières images capturées pour `deodorant` ont été écrasées par erreur par une capture de la licorne (le compteur de reprise n'avait pas encore la logique anti-écrasement actuelle) ; le dossier a simplement été renommé `licorne/` puisque le contenu correspondait déjà au nouvel objet choisi.
- Points de vigilance à surveiller pour la suite : taille de la mosaïque (filmer d'assez près), reflets sur surfaces brillantes (plastique verni des jouets licorne/camion).

### 4. Constitution du dataset — capture webcam directe

**Décision (2026-07-25)** : abandon de l'approche "filmer une vidéo puis extraire des frames" au profit d'une **capture directe à la webcam**, classe par classe. Raisons :
- Un pas de moins dans le pipeline (pas de fichier `.mp4` intermédiaire à filmer/déplacer).
- Feedback visuel immédiat (aperçu live + overlay du compteur pendant la capture).
- Les images d'entraînement viennent de la **même caméra** que celle utilisée en démo/inférence finale (`webcam.py`) → pas de décalage qualité/couleur/objectif entre train et prod, contrairement à un tournage fait avec un autre appareil (téléphone, etc.).

`extraire_frames.py` a été réécrit en conséquence :
- Ouvre `cv2.VideoCapture(0)`, sauvegarde 1 frame sur `INTERVALLE` (5) dans `dataset/<NOM_CLASSE>/`, jusqu'à atteindre `OBJECTIF` (150) images.
- `ESPACE` = pause/reprise (pour se repositionner, changer de fond, sans gaspiller de frames), `q` = quitter avant la fin.
- Reprend le comptage là où il s'était arrêté si on relance sur une classe déjà partiellement capturée (compte les `.jpg` déjà présents).
- **On relance le script une fois par classe**, en changeant `NOM_CLASSE` en haut du fichier avant chaque run : `licorne`, `mosaique`, `camion`, `fond`, `autres`. ⚠️ Bien vérifier `NOM_CLASSE` avant chaque lancement — c'est cet oubli qui a causé l'écrasement du dataset `deodorant` (voir section 3).

Le protocole de variabilité du tuto reste valable, juste appliqué en direct pendant la capture au lieu d'être filmé à l'avance : tourner l'objet à 360°, le rapprocher/éloigner, changer de fond (bureau/sol/mur) via une pause, varier l'éclairage (lumière du jour + artificielle), le cacher partiellement avec la main. Toujours attention au **biais de dataset** (shortcut learning) : ne pas garder le même arrière-plan pour toute une classe.

## Prochaines étapes

1. **Capture** (utilisateur) : lancer `python extraire_frames.py` une fois par classe (`licorne` ✅ fait, `mosaique`, `camion`, `fond`, `autres`), en variant angle/distance/fond/éclairage pendant la capture.
2. `explorer_dataset.py` : visualiser des échantillons, vérifier l'équilibre des classes.
3. `train.py` : construire et entraîner le modèle (Keras `image_dataset_from_directory`, suivi TensorBoard).
4. `finetune.py` : affiner le modèle.
5. `evaluer.py` : matrice de confusion / rapport de classification (scikit-learn).
6. `webcam.py` : démo d'inférence en temps réel.
