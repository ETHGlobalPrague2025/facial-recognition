import cv2
from deepface import DeepFace
import numpy as np
import hashlib
import pickle
import os
import threading # For background frame processing
import time # For yielding frames
from flask import Flask, Response, render_template_string, jsonify # Added Flask and jsonify

# --- Configuration ---
DB_FILE = "known_faces_db.pkl"
SIMILARITY_THRESHOLD = 0.7
WEBCAM_INDEX = 0 # or 0, or other if you have multiple cameras
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
HOST = '0.0.0.0' # Listen on all available network interfaces
PORT = 8080      # Port for the web server

# --- Global variables for Flask and Threading ---
output_frame = None # This will store the latest frame to be streamed
frame_lock = threading.Lock() # To safely access/modify output_frame
stop_processing_event = threading.Event() # To signal the processing thread to stop
HASH = None  # Global variable to store the current face hash

# Initialize Flask app
app = Flask(__name__)

# Initialize the face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    print("Please ensure OpenCV is installed correctly and the cascade file is accessible.")
    exit()

# --- Database Functions (Your existing functions - unchanged) ---
def load_known_faces(filename=DB_FILE):
    """Load known faces from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                if os.path.getsize(filename) > 0:
                    data = pickle.load(f)
                    if isinstance(data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in data):
                        print(f"Successfully loaded {len(data)} known faces from '{filename}'.")
                        return data
                    else:
                        print(f"Data in '{filename}' is not in the expected format. Starting fresh.")
                        return []
                else:
                    print(f"Database file '{filename}' is empty. Starting fresh.")
                    return []
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error unpickling data from '{filename}': {e}. Starting with an empty database.")
            return []
        except Exception as e:
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

# --- Face Processing Functions (Your existing functions - largely unchanged) ---
def get_face_info(frame):
    """
    Detects a face, extracts its embedding and hash.
    Returns face coordinates (x,y,w,h), embedding, and hash, or (None, None, None).
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            representation_list = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False, detector_backend='opencv') # Using opencv detector as it's faster for this case

            if representation_list and len(representation_list) > 0:
                representation = representation_list[0]
                if 'embedding' in representation:
                    embedding = np.array(representation['embedding'])
                    norm = np.linalg.norm(embedding)
                    if norm == 0:
                        return None, None, None
                    normalized_embedding = embedding / norm
                    hash_input = normalized_embedding.tobytes()
                    face_hash = hashlib.sha256(hash_input).hexdigest()
                    return (x, y, w, h), embedding, face_hash
                else:
                    pass
            else:
                pass
    except Exception as e:
        # print(f"Minor error in get_face_info: {e}") # Can be noisy
        pass
    return None, None, None

def calculate_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    embedding1 = np.asarray(embedding1).flatten()
    embedding2 = np.asarray(embedding2).flatten()
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

def find_best_matching_face(new_embedding, known_faces_db):
    if not known_faces_db:
        return None, 0.0, -1
    similarities = [calculate_similarity(new_embedding, known_embedding)
                   for known_embedding, _ in known_faces_db]
    if not similarities:
        return None, 0.0, -1
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    return known_faces_db[best_match_idx][1], best_similarity, best_match_idx

def print_known_faces(known_faces_db):
    print("\n=== Known Faces in Database ===")
    if not known_faces_db:
        print("  No faces in the database.")
    else:
        for idx, (_, hash_id) in enumerate(known_faces_db):
            print(f"  Face {idx + 1}: {hash_id[:16]}...")
    print("==============================")

def face_recognized_action(face_idx, face_hash): # Modified to take hash
    global HASH
    HASH = face_hash
    print(f"Face {face_idx + 1} recognized! Hash: {face_hash[:16]}")


# --- Camera Processing Thread Function ---
def process_camera_feed():
    global output_frame, frame_lock, known_faces, HASH # Add HASH to globals

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"❌ Unable to open webcam (index {WEBCAM_INDEX}). Please check.")
        stop_processing_event.set() # Signal main thread if camera fails
        return

    print(f"\n✅ Webcam (index {WEBCAM_INDEX}) started for processing.")
    print("Press 'a' in the OpenCV window when an unrecognized face is boxed to add it.")
    print("Press 'q' in the OpenCV window to quit the application.")
    print(f"MJPEG stream available at http://{HOST}:{PORT}/streaming")
    print(f"View stream in browser at http://{HOST}:{PORT}/")


    while not stop_processing_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from webcam. Exiting processing thread.")
            break

        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        display_frame = resized_frame.copy()

        key = cv2.waitKey(1) & 0xFF

        face_coords, current_embedding, current_hash = get_face_info(resized_frame)

        if current_embedding is not None and current_hash is not None:            
            x, y, w, h = face_coords
            best_match_hash, similarity_score, face_idx = find_best_matching_face(current_embedding, known_faces)
            text_on_frame = f"New: {current_hash[:8]}?"
            box_color = (255, 0, 0)

            if best_match_hash and similarity_score > SIMILARITY_THRESHOLD:
                text_on_frame = f"Face {face_idx + 1} ({similarity_score:.2f})"
                box_color = (0, 255, 0)
                # Call the action function for recognized face
                face_recognized_action(face_idx, known_faces[face_idx][1])
            else:
                if key == ord('a'):
                    is_already_in_db = any(khash == current_hash for _, khash in known_faces)
                    if not is_already_in_db:
                        with frame_lock: # Protect known_faces modification if accessed elsewhere too
                            known_faces.append((current_embedding, current_hash))
                            save_known_faces(known_faces) # Save updated list
                        print(f"\n🆕 Face Added! Hash: {current_hash[:16]}. Total: {len(known_faces)}")
                        text_on_frame = f"Added: {current_hash[:8]}"
                        box_color = (0, 165, 255)
                    else:
                        print(f"\nℹ️ Face {current_hash[:8]} is already in the database.")

            cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(display_frame, text_on_frame, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Update the global output frame
        with frame_lock:
            output_frame = display_frame.copy()

        # Show local webcam feed (optional, can be removed if only web stream is needed)
        #cv2.imshow("Face Recognition - Press 'a' to add, 'q' to quit", display_frame)

        #if key == ord('q'):
        #    print("Quit key ('q') pressed in OpenCV window. Stopping...")
        #    stop_processing_event.set()
        #    break
        
        # Small delay to yield CPU if processing is very fast, 
        # though cap.read() and DeepFace will likely be the main bottlenecks
        time.sleep(0.01)


    cap.release()
    cv2.destroyAllWindows()
    print("Camera processing thread finished.")
    # If 'q' was pressed, Flask might still be running.
    # A more robust shutdown would involve signaling Flask to stop too,
    # but for Ctrl+C, this is usually fine.


@app.route('/hash')
def hash():
    """Return the current face hash in JSON format."""
    return jsonify({'hash': HASH}), 200

# --- Flask Routes ---
@app.route('/')
def index():
    """Video streaming home page."""
    # Simple HTML page to display the MJPEG stream
    return render_template_string("""
    <html>
        <head>
            <title>Live Camera Stream</title>
        </head>
        <body>
            <h1>Live Camera Feed with Face Recognition</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
            <p>Database has {{ num_known_faces }} known faces.</p>
            <p>OpenCV window allows adding faces ('a') and quitting ('q').</p>
        </body>
    </html>
    """, num_known_faces=len(known_faces)) # Pass current count

def generate_frames():
    """Generator function for MJPEG streaming."""
    global output_frame, frame_lock
    while not stop_processing_event.is_set():
        with frame_lock:
            if output_frame is None:
                # Create a placeholder frame if processing hasn't started or no frame yet
                placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...", (30, FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                current_frame_bytes = cv2.imencode('.jpg', placeholder)[1].tobytes()
            else:
                # Encode the current output_frame to JPEG
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue # if encoding failed, skip this frame
                current_frame_bytes = encoded_image.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + current_frame_bytes + b'\r\n')
        time.sleep(0.03) # Adjust frame rate for streaming if needed (e.g., ~30 FPS)
    print("Frame generation stopped.")


@app.route('/streaming')
def video_feed():
    """Video streaming route. Uses the generator function."""
    if stop_processing_event.is_set():
        return "Processing has stopped.", 503
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Main Application ---
if __name__ == "__main__":
    # Load known faces at startup
    known_faces = load_known_faces() # Make sure this is global or accessible
    print_known_faces(known_faces)

    # Start the camera processing thread
    print("Starting camera processing thread...")
    processing_thread = threading.Thread(target=process_camera_feed)
    processing_thread.daemon = True # Allows main program to exit even if thread is running
    processing_thread.start()

    # Start Flask web server
    # Use threaded=True for Flask to handle multiple requests, e.g., multiple viewers for the stream
    # Use debug=False for production or when using custom threads like this to avoid Werkzeug reloader issues.
    print(f"Starting Flask server on http://{HOST}:{PORT}")
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Flask server received KeyboardInterrupt. Shutting down...")
    finally:
        print("Setting stop event for processing thread...")
        stop_processing_event.set()
        if processing_thread.is_alive():
            print("Waiting for processing thread to finish...")
            processing_thread.join(timeout=5) # Wait for the thread to close gracefully
        print("Application closed.")