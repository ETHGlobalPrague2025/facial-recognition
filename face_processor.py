import cv2
from deepface import DeepFace
import numpy as np
import hashlib

class FaceProcessor:
    def __init__(self, similarity_threshold=0.7):
        self.known_faces = []  # Will store tuples of (embedding, hash_id)
        self.similarity_threshold = similarity_threshold
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def find_best_matching_face(self, new_embedding):
        """Find the best matching face and its similarity score"""
        if not self.known_faces:
            return None, 0, -1
        
        similarities = [self.calculate_similarity(new_embedding, known_embedding) 
                       for known_embedding, _ in self.known_faces]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        return (self.known_faces[best_match_idx][1], 
                best_similarity, 
                best_match_idx)

    def get_face_info(self, frame):
        """Extract face information from a frame"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
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

    def process_frame(self, frame):
        """Process a frame and return face information"""
        face_coords, embedding, current_hash = self.get_face_info(frame)
        
        if face_coords is not None and embedding is not None:
            best_match_hash, similarity_score, face_idx = self.find_best_matching_face(embedding)
            
            similarity_info = {
                "best_match_hash": best_match_hash if best_match_hash else None,
                "similarity_score": float(similarity_score),
                "face_index": face_idx
            }
            
            # If similarity is below threshold, add as new face
            if similarity_score <= self.similarity_threshold:
                self.known_faces.append((embedding, current_hash))
                similarity_info["is_new_face"] = True
            else:
                similarity_info["is_new_face"] = False
            
            return face_coords, embedding, current_hash, similarity_info
            
        return None

    def get_known_faces(self):
        """Return list of known face hashes and their indices"""
        return [{
            "index": idx,
            "hash": face_hash
        } for idx, (_, face_hash) in enumerate(self.known_faces)] 