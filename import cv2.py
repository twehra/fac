import cv2
import mediapipe as mp
import face_recognition
import os

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Initialisation de Mediapipe
mp_face_detection = mp.solutions.face_detection.FaceDetection()
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame")
        continue

    # Convertir l'image en RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détection des visages
    results = mp_face_detection.process(image_rgb)

    # Dessiner les résultats sur l'image
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

    
    cv2.imshow('Mediapipe Face Detection', image) # Afficher l'image avec les annotations

    
    if cv2.waitKey(5) & 0xFF == 27: # Sortie de la boucle avec la touche "Esc"
        break