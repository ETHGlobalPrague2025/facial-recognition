import cv2
from deepface import DeepFace
import numpy as np
import hashlib
import pickle # Added for database
import os # For checking file existence (though try-except is often preferred for open)

# --- Configuration ---
DB_FILE = "known_faces_db.pkl"
SIMILARITY_THRESHOLD = 0.7  # Threshold for considering a face "known"

# Initialize the face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    print("Please ensure OpenCV is installed correctly and the cascade file is accessible.")
    exit()

# --- Database Functions ---
def load_known_faces(filename=DB_FILE):
    """Load known faces from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                # It's good practice to ensure the file is not empty before loading
                if os.path.getsize(filename) > 0:
                    data = pickle.load(f)
                    # Basic validation: check if it's a list of tuples with 2 elements each (embedding, hash)
                    if isinstance(data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in data):
                        print(f"Successfully loaded {len(data)} known faces from '{filename}'.")
                        return data
                    else:
                        print(f"Data in '{filename}' is not in the expected format. Starting fresh.")
                        return []
                else:
                    print(f"Database file '{filename}' is empty. Starting fresh.")
                    return []
        except (pickle.UnpicklingError, EOFError) as e: # EOFError if file is empty and not checked by os.path.getsize
            print(f"Error unpickling data from '{filename}': {e}. Starting with an empty database.")
            return []
        except Exception as e: # Catch other potential errors like permission issues
            print(f"Could not load database from '{filename}': {e}. Starting with an empty database.")
            return []
    else:
        print(f"Database file '{filename}' not found. Starting with an empty database.")
        return []

def save_known_faces(data, filename=DB_FILE):
    """Save known faces to a pickle file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Database updated and saved to '{filename}'. Contains {len(data)} faces.")
    except Exception as e:
        print(f"Error saving database to '{filename}': {e}")

# --- Face Processing Functions ---
def get_face_info(frame):
    """
    Detects a face, extracts its embedding and hash.
    Returns face coordinates (x,y,w,h), embedding, and hash, or (None, None, None).
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Added minSize to filter out very small detections, adjust as needed
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80)) 
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Process the first detected face
            
            # DeepFace.represent is called on the whole frame.
            # Its internal detector will find faces. The embedding returned will be for the
            # most prominent face DeepFace finds in the 'frame'.
            # This might not perfectly align with faces[0] if multiple faces are present.
            representation_list = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)
            
            if representation_list and len(representation_list) > 0:
                # Take the first face representation found by DeepFace
                representation = representation_list[0]
                if 'embedding' in representation:
                    embedding = np.array(representation['embedding'])
                    
                    # Create a hash from the normalized embedding for a unique ID
                    norm = np.linalg.norm(embedding)
                    if norm == 0: # Should not happen with Facenet if embedding is valid
                        return None, None, None 
                    normalized_embedding = embedding / norm
                    hash_input = normalized_embedding.tobytes()
                    face_hash = hashlib.sha256(hash_input).hexdigest()
                    
                    return (x, y, w, h), embedding, face_hash
                else:
                    # print("DeepFace representation lacks 'embedding' key.")
                    pass # Silently pass
            else:
                # print("DeepFace could not generate representation for the detected face region.")
                pass # Silently pass
                
    except Exception as e:
        # Avoid excessive printing for common non-critical errors (e.g., no face in a frame)
        # print(f"Minor error in get_face_info: {e}") 
        pass
    return None, None, None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    embedding1 = np.asarray(embedding1).flatten()
    embedding2 = np.asarray(embedding2).flatten()
    
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0 # Avoid division by zero
        
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

def find_best_matching_face(new_embedding, known_faces_db):
    """Find the best matching face and its similarity score from the database."""
    if not known_faces_db:
        return None, 0.0, -1  # No known faces
    
    similarities = [calculate_similarity(new_embedding, known_embedding) 
                   for known_embedding, _ in known_faces_db]
    
    if not similarities:
        return None, 0.0, -1

    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    return known_faces_db[best_match_idx][1], best_similarity, best_match_idx # (hash_id, similarity, index)

def print_known_faces(known_faces_db):
    """Print all known face hashes (or part of them) with their index."""
    print("\n=== Known Faces in Database ===")
    if not known_faces_db:
        print("  No faces in the database.")
    else:
        for idx, (_, hash_id) in enumerate(known_faces_db):
            print(f"  Face {idx + 1}: {hash_id[:16]}...") # Print a shorter version of the hash
    print("==============================")

def face_recognized(face_idx):
    print(f"Face {face_idx + 1} recognized!", known_faces[face_idx][1])

# --- Main Application ---
if __name__ == "__main__":
    # Load known faces at startup
    known_faces = load_known_faces()
    print_known_faces(known_faces)

    # Start webcam (try index 0 if 1 doesn't work)
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("❌ Unable to open webcam. Please check camera index and permissions.")
        exit()

    print("\n✅ Webcam started.")
    print("Press 'a' when an unrecognized face is boxed to add it to the database.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from webcam.")
            break

        # Resize for faster processing, adjust dimensions as needed
        # Common sizes: (320, 240), (640, 480)
        resized_frame = cv2.resize(frame, (640, 480)) 
        display_frame = resized_frame.copy() # Always draw on a copy

        # Get key press (non-blocking)
        key = cv2.waitKey(1) & 0xFF

        # Get face info from the current frame
        face_coords, current_embedding, current_hash = get_face_info(resized_frame)
        
        if current_embedding is not None and current_hash is not None: # A face was detected and embedding generated
            x, y, w, h = face_coords # Coordinates from Haar Cascade
            
            best_match_hash, similarity_score, face_idx = find_best_matching_face(current_embedding, known_faces)
            
            text_on_frame = f"New: {current_hash[:8]}?" # Default text for unrecognized face
            box_color = (255, 0, 0)  # Blue for new/unknown

            if best_match_hash and similarity_score > SIMILARITY_THRESHOLD:
                # Face is recognized
                text_on_frame = f"Face {face_idx + 1} ({similarity_score:.2f})"
                box_color = (0, 255, 0)  # Green for recognized

                face_recognized(face_idx)
            else:
                # Face is not recognized (or database is empty)
                # Check if 'a' key is pressed to add this new face
                if key == ord('a'):
                    # Check if this specific face (by its hash) is already in our known_faces list
                    is_already_in_db = any(khash == current_hash for _, khash in known_faces)
                    
                    if not is_already_in_db:
                        known_faces.append((current_embedding, current_hash))
                        save_known_faces(known_faces) # Save updated list to pickle file
                        
                        print(f"\n🆕 Face Added! Hash: {current_hash[:16]}. Total known faces: {len(known_faces)}")
                        # Optionally, print the full list again, or just confirmation
                        # print_known_faces(known_faces) 
                        
                        # Update display for this frame to show it was added
                        text_on_frame = f"Added: {current_hash[:8]}"
                        box_color = (0, 165, 255)  # Orange for newly added this session
                    else:
                        print(f"\nℹ️ Face {current_hash[:8]} (or its hash) is already in the database.")
                        # It might be displayed as "New" if its similarity to its stored version
                        # is low due to variance, or if another very similar face was added.
            
            # Draw rectangle and text on the display_frame
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(display_frame, text_on_frame, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Show webcam feed
        cv2.imshow("Face Recognition - Press 'a' to add, 'q' to quit", display_frame)

        # Exit loop if 'q' is pressed
        if key == ord('q'):
            print("Quitting application...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")