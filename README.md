
# 🔍 Real-Time Face Recognition with Flask & DeepFace

This project provides a real-time face recognition system built using **OpenCV**, **DeepFace**, and a **Flask** server to stream the webcam feed over HTTP. It includes persistent face recognition with the ability to save and match faces using cosine similarity of deep embeddings.
In this project, we detect facial landmarks and extract feature embeddings from the user's face. These embeddings are then hashed and mapped to a hashed version of the user's email, enabling both pseudonymity and account abstraction.

---

## 📦 Features

* 🎥 Real-time webcam feed with face detection
* 🧠 Face recognition using **Facenet** model via **DeepFace**
* 🔐 Unique face hashing for persistent identification
* 📝 Face database with `pickle` storage
* 🌐 MJPEG video streaming via Flask (`/streaming`)
* 📡 Face hash API (`/hash`) to retrieve the latest recognized face
* 🖱️ Add unknown faces on-the-fly by pressing `'a'`
* 🚀 Optional FastAPI endpoint for server-side face processing

---

## 🛠️ Requirements

* Python 3.8+
* OpenCV
* DeepFace
* Flask
* NumPy
* PIL (for FastAPI version)
* `haarcascade_frontalface_default.xml` (bundled with OpenCV)

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
opencv-python
deepface
flask
numpy
pillow
fastapi
uvicorn
```

---

## 🚀 Getting Started

### 🔁 Run the Live Flask App

```bash
python main_flask_app.py
```

Once running:

* Open [http://localhost:8080/](http://localhost:8080/) to view the live stream.
* Press `'a'` in the OpenCV window to add a new face.
* Press `'q'` to quit the OpenCV preview (stream remains live).
* The face database persists to `known_faces_db.pkl`.

---

## 🌐 API Endpoints

| Endpoint     | Method | Description                       |
| ------------ | ------ | --------------------------------- |
| `/`          | GET    | Web page showing live stream      |
| `/streaming` | GET    | MJPEG stream of camera feed       |
| `/hash`      | GET    | Returns current face hash as JSON |

---

## 🧪 FastAPI Support (Optional)

Use FastAPI for processing uploaded images via HTTP.

### Start FastAPI Server

```bash
uvicorn fastapi_face_app:app --host 0.0.0.0 --port 8000
```

### Endpoint

```http
POST /process-frame/
```

**Form field**: `frame` (image or base64-encoded file)

Returns:

```json
{
  "hash": "d98c1dd404...",
  "coords": [x, y, w, h],
  "embedding": [0.23, 0.45, ...]
}
```

---

## 💾 Face Database

Faces are stored as a list of tuples:

```python
(embedding: np.ndarray, hash: str)
```

Saved to `known_faces_db.pkl` using `pickle`.

---

## 📊 Similarity & Hashing

* Embeddings are normalized and hashed using `SHA-256`.
* Cosine similarity is used to compare embeddings.
* Match threshold: `0.70` (can be tuned).

---

## 🔐 Security Note

This system is a **proof-of-concept**. It does **not** use liveness detection or spoofing protection.

---

## 📁 Project Structure

```
├── main_flask_app.py         # Flask-based face recognition and streaming
├── fastapi_face_app.py       # Optional FastAPI app for processing images
├── face_processor.py         # (Optional) Modular face logic for FastAPI
├── known_faces_db.pkl        # Pickle database of face embeddings
├── README.md                 # This file
```

---

## ✅ Future Improvements

* Add face name labeling
* Add liveness detection
* Integrate database via SQLite or PostgreSQL
* Web dashboard for face management
* Deploy using Docker

---

## 📄 License

MIT License. See `LICENSE` file for details.

---

Let me know if you'd like a `Dockerfile`, FastAPI example payload, or integration with external storage like Firebase or PostgreSQL.
