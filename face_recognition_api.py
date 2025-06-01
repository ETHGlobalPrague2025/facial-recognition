from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from face_processor import FaceProcessor
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize face processor
face_processor = FaceProcessor()

@app.post("/process-frame/")
async def process_frame(frame: UploadFile = File(...)):
    """
    Process a video frame and return face information
    
    The frame should be sent as a base64 encoded image or regular image file
    Returns face hash, embedding, and coordinates if a face is detected
    """
    try:
        # Read the image file
        contents = await frame.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame
        result = face_processor.process_frame(img)
        
        if result:
            face_coords, embedding, face_hash, similarity_info = result
            
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist()
            
            return {
                "status": "success",
                "face_detected": True,
                "face_hash": face_hash,
                "embedding": embedding_list,
                "face_coordinates": {
                    "x": int(face_coords[0]),
                    "y": int(face_coords[1]),
                    "width": int(face_coords[2]),
                    "height": int(face_coords[3])
                },
                "similarity_info": similarity_info
            }
        else:
            return {
                "status": "success",
                "face_detected": False
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/known-faces/")
async def get_known_faces():
    """Get all known face hashes and their indices"""
    return {
        "status": "success",
        "known_faces": face_processor.get_known_faces()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 