"""
Extrait des images fixes depuis les vidéos de videos/ vers dataset/<classe>/.
Le nom du fichier vidéo (sans extension) devient le nom de la classe.
"""

import shutil
from pathlib import Path

import cv2

VIDEOS_DIR = Path("videos")
DATASET_DIR = Path("dataset")
FRAME_INTERVAL = 8  # on garde 1 frame sur 8 (~150 images pour 40s à 30fps)


def extraire_frames(video_path: Path, classe: str) -> int:
    sortie = DATASET_DIR / classe
    if sortie.exists():
        shutil.rmtree(sortie)
    sortie.mkdir(parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Impossible d'ouvrir {video_path}")
        return 0

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            nom = sortie / f"{classe}_{saved:04d}.jpg"
            cv2.imwrite(str(nom), frame)
            saved += 1
        frame_idx += 1
    cap.release()

    print(f"  {classe}: {saved} images extraites ({frame_idx} frames lues dans {video_path.name})")
    return saved


def main():
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        print(f"Aucune vidéo trouvée dans {VIDEOS_DIR}/ — voir {VIDEOS_DIR / 'README.md'}.")
        return

    print(f"{len(videos)} vidéo(s) trouvée(s) dans {VIDEOS_DIR}/\n")
    total = sum(extraire_frames(video_path, video_path.stem) for video_path in videos)
    print(f"\nTotal : {total} images extraites dans {DATASET_DIR}/")


if __name__ == "__main__":
    main()
