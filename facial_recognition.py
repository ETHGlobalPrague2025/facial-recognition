import cv2
from deepface import DeepFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Store face embeddings to track unique identities
known_face_embeddings = []
SIMILARITY_THRESHOLD = 0.7  # Adjust this value between 0 and 1 to control strictness

def get_face_embedding(frame):
    try:
        # DeepFace expects BGR image (as is from OpenCV)
        result = DeepFace.represent(frame, model_name='Facenet', enforce_detection=True, align=True)
        if result:
            return np.array(result[0]['embedding'])
    except Exception as e:
        print("No face or error:", e)
    return None

def is_known_face(embedding):
    if not known_face_embeddings:
        return False
    
    # Compare with all known embeddings
    similarities = [cosine_similarity(embedding.reshape(1, -1), 
                                    known_emb.reshape(1, -1))[0][0] 
                   for known_emb in known_face_embeddings]
    max_similarity = max(similarities)
    return max_similarity > SIMILARITY_THRESHOLD

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to open webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize for speed
    resized_frame = cv2.resize(frame, (320, 240))

    embedding = get_face_embedding(resized_frame)
    if embedding is not None:
        if not is_known_face(embedding):
            known_face_embeddings.append(embedding)
            print(f"🆕 New face detected! Total known faces: {len(known_face_embeddings)}")
        else:
            print(f"✅ Known face detected!")

    # Show webcam feed
    cv2.imshow("Face Recognition", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
