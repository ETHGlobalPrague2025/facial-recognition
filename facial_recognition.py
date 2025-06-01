import cv2
from deepface import DeepFace
import numpy as np
import time
import hashlib

# Store face embeddings and their hashes
known_faces = []  # Will store tuples of (embedding, hash_id)
SIMILARITY_THRESHOLD = 0.7

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_info(frame):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the first face detected
            (x, y, w, h) = faces[0]
            
            # Get face embedding
            result = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)
            if result:
                embedding = np.array(result[0]['embedding'])
                # Create a hash from the normalized embedding
                normalized_embedding = embedding / np.linalg.norm(embedding)
                hash_input = normalized_embedding.tobytes()
                face_hash = hashlib.sha256(hash_input).hexdigest()
                return (x, y, w, h), embedding, face_hash
                
    except Exception as e:
        print("No face or error:", e)
    return None, None, None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def find_best_matching_face(new_embedding):
    """Find the best matching face and its similarity score"""
    if not known_faces:
        return None, 0, -1
    
    similarities = [calculate_similarity(new_embedding, known_embedding) 
                   for known_embedding, _ in known_faces]
    
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    return known_faces[best_match_idx][1], best_similarity, best_match_idx

def print_known_faces():
    """Print all known face hashes with their index"""
    print("\n=== Known Faces ===")
    for idx, (_, hash_id) in enumerate(known_faces):
        print(f"Face {idx + 1}: {hash_id[:16]}")
    print("================")

# Start webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Unable to open webcam.")
    exit()

print("✅ Webcam started. Will run for 10 seconds...")

# Get start time
start_time = time.time()
last_hash_print = time.time()

while True:
    current_time = time.time()
    
    # Check if 10 seconds have passed
    if current_time - start_time > 10:
        print("✅ 10 seconds completed!")
        break

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Resize for speed
    resized_frame = cv2.resize(frame, (320, 240))
    display_frame = resized_frame.copy()

    face_coords, embedding, current_hash = get_face_info(resized_frame)
    
    if face_coords is not None and embedding is not None:
        x, y, w, h = face_coords
        
        best_match_hash, similarity_score, face_idx = find_best_matching_face(embedding)
        
        # Print similarity information
        if best_match_hash:
            print(f"\nSimilarity score with best match: {similarity_score:.3f}")
            print(f"Current hash: {current_hash[:16]}")
            print(f"Best match hash: {best_match_hash[:16]}")
        
        if similarity_score > SIMILARITY_THRESHOLD:
            # Green box for known faces
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Face {face_idx + 1} ({similarity_score:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"✅ Matched with Face {face_idx + 1}")
        else:
            # Blue box for new faces
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(display_frame, f"New: {current_hash[:8]}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            known_faces.append((embedding, current_hash))
            print(f"\n🆕 New face detected!")
            print(f"New Hash: {current_hash}")
            print_known_faces()

    # Show webcam feed
    cv2.imshow("Face Recognition", display_frame)
    
    # Print all known faces every 2 seconds
    if current_time - last_hash_print > 2:
        print_known_faces()
        last_hash_print = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print final known faces
print("\n=== Final Known Faces ===")
print_known_faces()

cap.release()
cv2.destroyAllWindows()
