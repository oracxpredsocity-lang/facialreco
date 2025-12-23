from flask import Flask, render_template, Response, request, redirect
import face_recognition
import cv2
import os
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

FACES_DIR = "faces"
ENCODINGS_FILE = "encodings.pkl"
LOG_FILE = "logs.csv"
THRESHOLD = 0.5

# Source caméra par défaut
camera_source = 0
camera = cv2.VideoCapture(camera_source)

# Charger les visages
def load_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

known_faces = load_faces()

# Enregistrer les visages
def register_faces():
    data = {}
    if not os.path.exists(FACES_DIR):
        return data

    for person in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, person)
        if not os.path.isdir(person_path):
            continue

        data[person] = []
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                data[person].append(encodings[0])

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    return data

def gen_frames():
    global camera, known_faces

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for encoding, loc in zip(encodings, locations):
            name = "Visage non enregistré"
            min_dist = 1.0

            for person, encs in known_faces.items():
                distances = face_recognition.face_distance(encs, encoding)
                if len(distances) > 0:
                    d = np.min(distances)
                    if d < min_dist:
                        min_dist = d
                        name = person

            if min_dist < THRESHOLD:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{name},{now}\n")

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register", methods=["POST"])
def register():
    global known_faces
    known_faces = register_faces()
    return redirect("/")

@app.route("/set_camera", methods=["POST"])
def set_camera():
    global camera
    url = request.form.get("camera_url")

    camera.release()

    if url.isdigit():
        camera = cv2.VideoCapture(int(url))
    else:
        camera = cv2.VideoCapture(url)

    return redirect("/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
