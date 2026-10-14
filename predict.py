import os

# Suppress TensorFlow logging before importing Keras.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
from PIL import Image
import keras
from facenet_pytorch import MTCNN, InceptionResnetV1

# 1. Device Setup & Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN face detector & FaceNet feature extractor
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load Trained NN Classifier, Class Names, and Centroids
project_dir = os.path.dirname(os.path.abspath(__file__))
classifier = keras.models.load_model(os.path.join(project_dir, "football_player_classifier.keras"))
class_names = np.load(os.path.join(project_dir, "classes.npy"))
centroids = np.load(os.path.join(project_dir, "player_centroids.npy"), allow_pickle=True).item()

# Distance Threshold: > 0.55 means person is UNKNOWN
DISTANCE_THRESHOLD = 0.55

def cosine_distance(vector_a, vector_b):
    """Computes Cosine Distance between two 1D vectors."""
    a = vector_a / np.linalg.norm(vector_a)
    b = vector_b / np.linalg.norm(vector_b)
    return 1.0 - np.dot(a, b)

def predict_image(img):
    """Return the predicted player, or an explicit no-face/unknown result."""
    img = img.convert('RGB')
    face_tensor = mtcnn(img)
    
    if face_tensor is None:
        return {"status": "no_face"}

    # Standardize and extract 512-d FaceNet embedding
    face_tensor = (face_tensor - 127.5) / 128.0
    face_tensor = face_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = facenet(face_tensor).cpu().numpy().flatten()

    # --- STEP 1: DISTANCE THRESHOLD CHECK (GATEKEEPER) ---
    min_distance = min([cosine_distance(embedding, c) for c in centroids.values()])

    if min_distance > DISTANCE_THRESHOLD:
        return {"status": "unknown", "distance": float(min_distance)}

    # --- STEP 2: NEURAL NETWORK PREDICTION (IF KNOWN) ---
    # Reshape embedding for model input: (1, 512)
    embedding_input = np.expand_dims(embedding, axis=0)
    probabilities = classifier.predict(embedding_input, verbose=0)[0]
    
    best_class_idx = np.argmax(probabilities)
    top_indices = np.argsort(probabilities)[::-1]
    return {
        "status": "recognized",
        "player": str(class_names[best_class_idx]),
        "confidence": float(probabilities[best_class_idx]),
        "distance": float(min_distance),
        "top_predictions": [
            (str(class_names[idx]), float(probabilities[idx])) for idx in top_indices[:3]
        ],
    }


def predict_player(image_path):
    """Run prediction for a local image and print a readable CLI report."""
    with Image.open(image_path) as image:
        result = predict_image(image)

    print("\n" + "=" * 50)
    if result["status"] == "no_face":
        print(f"[Error] No face detected in: {image_path}")
    elif result["status"] == "unknown":
        print(" RESULT: UNKNOWN / UNRECOGNIZED PERSON")
        print(f" Minimum Cosine Distance: {result['distance']:.4f} (> {DISTANCE_THRESHOLD})")
        print(" Model prediction skipped because face is out-of-distribution.")
    else:
        print(f" RESULT: {result['player'].upper()}")
        print(f" NN Model Confidence: {result['confidence'] * 100:.2f}%")
        print(f" Closest Centroid Distance: {result['distance']:.4f} (Verified < {DISTANCE_THRESHOLD})")
        print("\nNeural Network Probability Distribution:")
        for player, probability in result["top_predictions"]:
            print(f"  • {player:<20} {probability * 100:6.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    
    test_directory = "test_images"
    for image_file in os.listdir(os.path.join(project_dir, test_directory)):
        image_path = os.path.join(project_dir, test_directory, image_file)
        print(f"\nProcessing: {image_path}")
        predict_player(image_path)