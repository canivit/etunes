# Load the trained model
import cv2
import torch
from torchvision import transforms

from model import EmotionCNN

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
model_path = 'model/best_model.pth'


def predict_emotion_from_webcam():
    model = EmotionCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Capture the video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face using Haar Cascade
        haar_cascade_path = '../haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray_image[y:y + h, x:x + w]

            # Resize to 48x48
            resized_face = cv2.resize(face, (48, 48))

            # Normalize the image
            normalized_face = resized_face.astype('float32') / 255.0

            # Convert to a PyTorch tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            tensor_face = transform(normalized_face).unsqueeze(0).to(device)  # Add batch dimension and move to device

            # Predict the emotion
            with torch.no_grad():
                output = model(tensor_face)
                _, predicted = torch.max(output, 1)
                emotion = predicted.item()

                # Draw rectangle around the face and put the emotion text
                emotion_text = f'Emotion: {emotion_map[emotion]}'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


predict_emotion_from_webcam()
