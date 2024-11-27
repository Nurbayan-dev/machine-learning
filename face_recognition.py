import cv2
import os
import numpy as np

# Инициализация face recognizer и классификатора Хаара
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Путь к папке с данными
dataset_path = r"dataset"

# Подготовка данных для обучения
def prepare_training_data(dataset_path):
    faces = []
    labels = []

    # Метки для Роналду и Месси
    label_mapping = {"ronaldo": 0, "messi": 1}

    for label_name, label in label_mapping.items():
        person_folder = os.path.join(dataset_path, label_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Обнаружение лица и добавление в данные
                detected_faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in detected_faces:
                    face = image[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, (100, 100))
                    faces.append(face_resized)
                    labels.append(label)

    return faces, labels

# Обучение модели
print("Подготовка данных для обучения...")
faces, labels = prepare_training_data(dataset_path)
if len(faces) > 0:
    print("Обучение модели...")
    face_recognizer.train(faces, np.array(labels))
    print("Модель обучена!")
else:
    print("Ошибка: не найдено данных для обучения.")

# Функция для распознавания лица
def predict(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face = gray_image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))
        label, confidence = face_recognizer.predict(face_resized)

        # Определяем, кто изображён
        person = "Ronaldo" if label == 0 else "Messi"
        print(f"Распознан: {person}, уверенность: {confidence}")

        # Отображение результата
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Показ изображения
    cv2.imshow("Распознавание лица", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Тестирование
test_image_path = os.path.join("../ronn.jpg")  # Укажите изображение для теста
predict(test_image_path)
