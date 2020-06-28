import face_recognition as fr
import cv2
import os
import pickle

known_faces, known_names = pickle.load(open('saved_encodings', 'rb'))
TESTING_DIR = "resources/FR_Dataset/test_set"


for filename in os.listdir(TESTING_DIR):
    identified_faces = []
    testing_image = fr.load_image_file(f"{TESTING_DIR}/{filename}")
    testing_image = cv2.resize(testing_image, (0, 0), fx=0.5, fy=0.5)
    output_image = cv2.cvtColor(testing_image, cv2.COLOR_RGB2BGR)
    face_locations = fr.face_locations(testing_image)
    encodings = fr.face_encodings(testing_image, face_locations)

    for encoding, face_location in zip(encodings, face_locations):
        # puts 'True' where face is matched.
        # len(predictions) == len(known_faces), i.e. checks if there is match for known face in the test image
        # if yes then put 'True' else 'False'
        predictions = fr.compare_faces(known_faces, encoding, tolerance=0.4)

        # # finds euclidean distance between two faces
        # distance = fr.face_distance(known_faces, encoding)
        # # small distance corresponds to best match
        # min_dist_index = np.argmin(distance)

        match = 'unknown'
        # even single face matches,
        if True in predictions:
            # take only the names whose face is matched in the test image
            match = known_names[predictions.index(True)]

        # identified faces in current test image
        identified_faces.append((match, face_location))

    # highlight identified faces in current test image
    for face in identified_faces:
        top, right, bottom, left = face[1]
        cv2.rectangle(output_image, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.rectangle(output_image, (left, top - 40), (right, top), (0, 0, 255), cv2.FILLED)
        cv2.putText(output_image, face[0], (left+5, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

    cv2.imshow(str(filename), output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()