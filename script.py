import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Carregue o modelo pré-treinado FaceNet
facenet_model = load_model('facenet_keras.h5')

# Função para extrair embeddings de uma face
def get_face_embedding(face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = facenet_model.predict(samples)
    return yhat[0]

# Função para treinar o classificador SVM
def train_classifier(embeddings, labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(embeddings, labels_encoded)
    return classifier, label_encoder

# Dummy data para treinamento (substitua com seus dados)
# Importe ou gere suas imagens e rótulos reais para treino
dummy_embeddings = [np.random.rand(128) for _ in range(5)]  # Substitua pelos embeddings reais
dummy_labels = ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'] # Substitua pelos rótulos reais

# Treine o classificador
classifier, label_encoder = train_classifier(dummy_embeddings, dummy_labels)

# Inicialize a detecção de rostos
detector = MTCNN()

# Acesse a webcam
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar rostos
    faces = detector.detect_faces(frame)
    
    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)  # Certifique-se de que coordenadas não sejam negativas
        detected_face = frame[y:y+height, x:x+width]
        
        # Obter embedding do rosto detectado
        embedding = get_face_embedding(detected_face)
        
        # Classificar a face detectada
        prediction = classifier.predict([embedding])
        person = label_encoder.inverse_transform(prediction)
        
        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(frame, person[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Exibir a imagem com a detecção e reconhecimento de rostos
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
