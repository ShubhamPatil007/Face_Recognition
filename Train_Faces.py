import numpy as np
import face_recognition as fr
import cv2
import os
import pickle

known_faces = []
known_names = []

TRAINING_DIR = "resources/FR_Dataset/train_set"

log = {}
for dir_name in os.listdir(TRAINING_DIR):
    processed_files = []
    for filename in os.listdir(f"{TRAINING_DIR}/{dir_name}"):
        training_image = fr.load_image_file(f"{TRAINING_DIR}/{dir_name}/{filename}")
        # half of the original size
        training_image = cv2.resize(training_image, (0, 0), fx=0.5, fy=0.5)

        # ignores corrupt photos
        if len(fr.face_encodings(training_image)) > 0:
            encoding = fr.face_encodings(training_image)[0]
            known_faces.append(encoding)
            known_names.append(dir_name)
            processed_files.append(str(filename))

    log[str(dir_name)] = processed_files

print(f"training successful...\nlog: {log}")
for key in log:
    print(f"{key}: {len(log[key])}")

pickle.dump((known_faces, known_names), open('saved_encodings', 'wb'))
