import streamlit as st
import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
import sqlite3
from datetime import datetime
import io
import time

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Face detection and recognition
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=60)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Transform for face images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Database
DB_PATH = 'face_recognition.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, user_id INTEGER, image BLOB, FOREIGN KEY(user_id) REFERENCES users(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS encodings (id INTEGER PRIMARY KEY, user_id INTEGER, encoding BLOB, FOREIGN KEY(user_id) REFERENCES users(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, name TEXT, date TEXT, time TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Anti-spoofing based on Laplacian variance
def is_real_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > 35

# Save new user image and encoding
def save_image_to_db(user_id, image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        encoding = model(image_tensor).cpu().numpy().flatten()
    encoding = encoding / np.linalg.norm(encoding)

    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_blob = output.getvalue()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO images (user_id, image) VALUES (?, ?)', (user_id, image_blob))
    cursor.execute('INSERT INTO encodings (user_id, encoding) VALUES (?, ?)', (user_id, encoding.tobytes()))
    conn.commit()
    conn.close()

# Fetch all encodings
def fetch_encodings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT users.name, encodings.encoding FROM users JOIN encodings ON users.id = encodings.user_id''')
    data = cursor.fetchall()
    conn.close()
    return {name: np.frombuffer(enc, dtype=np.float32) for name, enc in data}

# Preprocess a detected face
def preprocess_face(img_rgb, box):
    x1, y1, x2, y2 = map(int, box)
    face = img_rgb[y1:y2, x1:x2]
    face_pil = Image.fromarray(face).convert('RGB')
    return transform(face_pil).unsqueeze(0).to(device)

# Attendance logging
def mark_attendance(name):
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, date_str))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date_str, time_str))
        conn.commit()
    conn.close()

def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (today,))
    data = cursor.fetchall()
    conn.close()
    return data

def get_registered_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users")
    data = cursor.fetchall()
    conn.close()
    return data

# Streamlit App
st.set_page_config(page_title="Face Recognition System", layout="centered")

page = st.sidebar.selectbox("Select Page", ["Register User", "Recognize & Mark Attendance", "View Attendance", "View Registered Users"])

if page == "Register User":
    st.title("📸 Register New User")
    name = st.text_input("Enter user's name")
    if st.button("Register") and name:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        cap = cv2.VideoCapture(0)
        directions = ["Look Center", "Look Left", "Look Right", "Look Up", "Look Down"]
        frame_placeholder = st.empty()
        progress = st.progress(0)
        count = 0

        for direction in directions:
            st.info(f"Please {direction}")
            time.sleep(1)
            captured = 0
            while captured < 12:
                ret, frame = cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(img_rgb)
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).convert('RGB')
                            save_image_to_db(user_id, face_pil)
                            captured += 1
                            count += 1
                            progress.progress(count / (len(directions) * 12))
                frame_placeholder.image(frame, channels="BGR")
        cap.release()
        st.success(f"Successfully registered {name}")

elif page == "Recognize & Mark Attendance":
    st.title("🎥 Live Face Recognition")
    frame_placeholder = st.empty()
    start = st.button("Start Recognition")
    stop = st.button("Stop")

    if start:
        encodings = fetch_encodings()
        cap = cv2.VideoCapture(0)
        st.session_state['stop'] = False

        while cap.isOpened() and not st.session_state.get('stop'):
            if stop:
                st.session_state['stop'] = True
                break
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)
            if boxes is not None:
                for box in boxes:
                    face_tensor = preprocess_face(img_rgb, box)
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    real = is_real_face(face_crop)
                    if not real:
                        label = "Fake"
                        color = (0, 0, 255)
                    else:
                        with torch.no_grad():
                            encoding = model(face_tensor).cpu().numpy().flatten()
                            encoding = encoding / np.linalg.norm(encoding)
                        best_match = "Unknown"
                        best_score = 0.5
                        for name, enc in encodings.items():
                            score = np.dot(encoding, enc)
                            if score > best_score:
                                best_match = name
                                best_score = score
                        label = f"{best_match} (Real)" if best_match != "Unknown" else "Unknown (Real)"
                        if best_match != "Unknown":
                            mark_attendance(best_match)
                        color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            frame_placeholder.image(frame, channels="BGR")
        cap.release()

elif page == "View Attendance":
    st.title("📅 Today's Attendance")
    if st.button("Show Records"):
        data = get_today_attendance()
        if data:
            st.table([{"Name": name, "Time": t} for name, t in data])
        else:
            st.info("No records yet today.")

elif page == "View Registered Users":
    st.title("👥 Registered Users")
    data = get_registered_users()
    if data:
        for uid, name in data:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{uid}: {name}")
            with col2:
                if st.button("Delete", key=uid):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM users WHERE id = ?", (uid,))
                    cursor.execute("DELETE FROM images WHERE user_id = ?", (uid,))
                    cursor.execute("DELETE FROM encodings WHERE user_id = ?", (uid,))
                    conn.commit()
                    conn.close()
                    st.experimental_rerun()
