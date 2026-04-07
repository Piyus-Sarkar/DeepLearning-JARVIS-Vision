import streamlit as st
import cv2
from deepface import DeepFace
from ultralytics import YOLO
import pyttsx3
import threading
import time

# --- 1. DASHBOARD UI SETUP ---
st.set_page_config(page_title="J.A.R.V.I.S. Vision", layout="wide")
st.title("👁️ J.A.R.V.I.S. Vision & Analytics Dashboard")

st.sidebar.header("System Controls")
run_camera = st.sidebar.checkbox("Activate Vision System")
enable_voice = st.sidebar.checkbox("Enable J.A.R.V.I.S. Voice", value=True)

# --- 2. CACHING AI BRAIN (Voice is no longer cached to fix the thread bug!) ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

# --- 3. THE FIX: VOICE ENGINE RUNS SAFELY INSIDE ITS OWN THREAD ---
def speak(text):
    if enable_voice:
        def run_speech():
            try:
                # Initializing INSIDE the thread prevents the silent crashing bug
                local_engine = pyttsx3.init()
                voices = local_engine.getProperty('voices')
                local_engine.setProperty('voice', voices[0].id) 
                local_engine.setProperty('rate', 170)
                local_engine.say(text)
                local_engine.runAndWait()
            except Exception as e:
                print(f"Voice Engine Error: {e}")
                
        threading.Thread(target=run_speech, daemon=True).start()

col1, col2 = st.columns([2, 1])
with col1:
    stframe = st.empty() 
with col2:
    st.subheader("System Logs")
    log_display = st.empty() 

# --- 4. MAIN CAMERA LOGIC ---
if run_camera:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    process_every_n_frames = 5
    
    current_yolo_objects = []
    current_faces = []
    
    # --- UPGRADED SHORT-TERM MEMORY ---
    known_objects = set()     
    known_face_count = 0      
    known_emotions = set()    # The AI now remembers how you felt a second ago!
    
    last_spoken_time = 0
    anti_flicker_cooldown = 4 
    
    current_log = "System booting..."

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame_count += 1
        current_time = time.time()

        if frame_count % process_every_n_frames == 0:
            # 1. Update Objects
            yolo_results = yolo_model(frame, verbose=False)[0]
            current_yolo_objects = yolo_results.boxes

            # 2. Update Faces
            try:
                face_results = DeepFace.analyze(frame, actions=['emotion', 'gender'], enforce_detection=False)
                current_faces = [face_results] if isinstance(face_results, dict) else face_results
            except ValueError:
                current_faces = []

            # --- 3. EMOTION & EVENT-DRIVEN LOGIC ---
            current_object_names = set()
            for box in current_yolo_objects:
                class_id = int(box.cls[0])
                obj_name = yolo_model.names[class_id]
                if obj_name != "person":
                    current_object_names.add(obj_name)
                    
            valid_faces = [f for f in current_faces if 'region' in f]
            current_face_count = len(valid_faces)
            
            # Grab all emotions currently on the screen
            current_emotions = set([f['dominant_emotion'] for f in valid_faces])

            # Mathematically find what is NEW in this frame
            new_objects = current_object_names - known_objects 
            new_face_added = current_face_count > known_face_count
            new_emotions_detected = current_emotions - known_emotions

            # Update memory for the next frame
            known_objects = current_object_names
            known_face_count = current_face_count
            known_emotions = current_emotions

            # Speak if there is a new object, a new face, OR a new emotion!
            if (new_objects or new_face_added or new_emotions_detected) and (current_time - last_spoken_time > anti_flicker_cooldown):
                announcement = []

                if new_objects:
                    announcement.append(f"New object identified: {', '.join(new_objects)}.")

                if new_face_added:
                    announcement.append(f"Human detected.")
                    
                if new_emotions_detected:
                    announcement.append(f"Emotion registered: {', '.join(new_emotions_detected)}.")

                if announcement:
                    final_speech = " ".join(announcement)
                    current_log = final_speech
                    speak(final_speech)
                    last_spoken_time = current_time

        # --- DRAWING ON THE FRAME ---
        for box in current_yolo_objects:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   
            class_id = int(box.cls[0])               
            object_name = yolo_model.names[class_id] 
            if object_name != "person": 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for face in current_faces:
            if 'region' in face:
                dominant_emotion = face['dominant_emotion']
                dominant_gender = max(face['gender'], key=face['gender'].get) 
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{dominant_gender} | {dominant_emotion}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert and Display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")
        log_display.info(current_log)

    cap.release()