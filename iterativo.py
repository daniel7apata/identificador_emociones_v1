import os
import cv2
import time

dataPath = 'C:/Users/Daniel/Desktop/itera_2'  # Asegúrate de definir tu ruta correcta
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model.xml")

cap = cv2.VideoCapture('dina_recopilacion_1.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

emotions = ['Alegria','Desagrado','Enojo','Miedo','Neutral','Sorpresa','Tristeza',]  # Ejemplo de emociones
emotion_folders = {emotion: os.path.join(dataPath, emotion) for emotion in emotions}

# Crear carpetas para cada emoción si no existen
for folder in emotion_folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        emotion = emotions[result[0]]  # Suponiendo que result[0] devuelve un índice de emoción
        cv2.putText(frame, '{}'.format(emotion), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # Guardar imagen en la carpeta correspondiente
        img_name = f"{emotion}_{time.time()}.jpg"
        cv2.imwrite(os.path.join(emotion_folders[emotion], img_name), rostro)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()