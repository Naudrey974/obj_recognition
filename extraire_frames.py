import os
import cv2

# ============================================================
# CAPTURE D'IMAGES DIRECTEMENT DEPUIS LA WEBCAM
# ------------------------------------------------------------
# On lance ce script UNE FOIS PAR CLASSE en changeant NOM_CLASSE.
# Il ouvre la webcam et sauvegarde une image toutes les
# INTERVALLE frames, tant qu'on n'a pas atteint OBJECTIF images.
#
# Commandes pendant la capture :
#   ESPACE = mettre en pause / reprendre (pour se repositionner)
#   q      = quitter avant la fin
# ============================================================

NOM_CLASSE = "deodorant"   # ← À CHANGER : deodorant, mosaique, balle, fond, autres
OBJECTIF   = 150           # nombre d'images visé pour cette classe
INTERVALLE = 5             # on sauvegarde 1 frame sur 5 (évite les quasi-doublons)

# 1. Créer le dossier de destination (le nom = le nom de la classe)
dossier = os.path.join("dataset", NOM_CLASSE)
os.makedirs(dossier, exist_ok=True)

# On repart du nombre d'images déjà présentes (pour compléter une classe)
deja_present = len([f for f in os.listdir(dossier) if f.endswith(".jpg")])
compteur_images = deja_present
print(f"{deja_present} image(s) déjà dans '{NOM_CLASSE}'. Objectif : {OBJECTIF}.")

# 2. Ouvrir la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam introuvable — essayez cv2.VideoCapture(1)")

numero_frame = 0
en_pause = False

print("Capture en cours. ESPACE = pause/reprise, q = quitter.")

while compteur_images < OBJECTIF:
    succes, frame = cap.read()
    if not succes:
        break

    # --- On ne sauvegarde que si on n'est PAS en pause ---
    if not en_pause and numero_frame % INTERVALLE == 0:
        chemin = os.path.join(dossier, f"{NOM_CLASSE}_{compteur_images:04d}.jpg")
        cv2.imwrite(chemin, frame)
        compteur_images += 1

    numero_frame += 1

    # --- Affichage avec un bandeau d'info ---
    affichage = frame.copy()
    etat = "PAUSE" if en_pause else "CAPTURE"
    couleur = (0, 165, 255) if en_pause else (0, 255, 0)
    texte = f"[{etat}] {NOM_CLASSE} : {compteur_images}/{OBJECTIF}"

    cv2.rectangle(affichage, (0, 0), (affichage.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(affichage, texte, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

    cv2.imshow("Capture dataset - ESGI", affichage)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    elif touche == ord(" "):      # barre espace
        en_pause = not en_pause

cap.release()
cv2.destroyAllWindows()
print(f"✅ Terminé : {compteur_images} images dans '{dossier}'")