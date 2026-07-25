# Vidéos sources

Placez ici vos 5 vidéos, une par classe, nommées exactement ainsi (le nom du fichier = le nom de la classe) :

- `deodorant.mp4`
- `mosaique.mp4`
- `balle.mp4`
- `fond.mp4` — **aucun objet distinct** : bureau vide, mur, sol, main/visage seuls (sans rien tenir). L'idée : rien de particulier à regarder.
- `autres.mp4` — **un objet distinct mais différent des 3 cibles** : stylo, bouteille, livre, chargeur, clés, tasse, une autre balle de couleur différente… Filmez plusieurs objets différents à tour de rôle dans la vidéo, un maximum de variété.

⚠️ **Frontière fond / autres** : la règle est "y a-t-il un objet net et reconnaissable dans le cadre ?". Si oui (même si ce n'est pas un de vos 3 objets cibles) → `autres`. Si non (juste du décor, une main vide) → `fond`. Évitez les cas limites ambigus (ex. une main qui *tient* un stylo est plus proche de `autres`).

## Protocole de tournage (30 à 40 secondes par vidéo)

| Variation | Pourquoi |
|---|---|
| Tourner l'objet sur 360° | Le modèle doit le reconnaître sous tous les angles |
| Rapprocher / éloigner | Robustesse à l'échelle |
| 3 fonds différents (bureau, sol, mur) | Sinon le modèle apprend le fond, pas l'objet |
| Lumière du jour + lumière artificielle | Robustesse à l'éclairage |
| Objet partiellement caché par la main | Situation réelle en démo |

⚠️ **Piège n°1 du deep learning : le biais de dataset (shortcut learning).** Si toutes les photos de déodorant sont prises sur le même bureau, le modèle risque d'apprendre à reconnaître ce bureau plutôt que l'objet. Variez systématiquement les fonds pour chaque objet.

Une fois les 5 vidéos placées ici, lancez `python extraire_frames.py` depuis la racine du projet (environnement `obj_recognition` activé). Le script est générique : il traite automatiquement tous les `.mp4` présents dans ce dossier, quel que soit leur nombre.
