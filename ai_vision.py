import cv2
from deepface import DeepFace
from ultralytics import YOLO
import pyttsx3
import threading
import time

# --- INITIALIZE VOICE ENGINE ---
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) 
engine.setProperty('rate', 170) 

def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

print("Loading YOLO AI Brain...")
yolo_model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(0)
print("Vision and Voice activated. Press 'q' to quit.")

frame_count = 0
process_every_n_frames = 5 

current_yolo_objects = []
current_faces = []
last_spoken_time = 0
cooldown_seconds = 8  

while True:
    ret, frame = cap.read()
    if not ret:
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

        # --- NEW: THE JARVIS BATCHING LOGIC ---
        # Only build an announcement if the 8-second cooldown has passed
        if current_time - last_spoken_time > cooldown_seconds:
            announcement = []

            # Step A: Gather all unique objects
            found_objects = set()
            for box in current_yolo_objects:
                class_id = int(box.cls[0])
                obj_name = yolo_model.names[class_id]
                if obj_name != "person":
                    found_objects.add(obj_name)
            
            if found_objects:
                # This joins multiple items together (e.g., "cell phone, cup")
                announcement.append(f"Objects detected: {', '.join(found_objects)}.")

            # Step B: Gather all human data
            # First, make sure the faces have valid data
            valid_faces = [face for face in current_faces if 'region' in face]
            
            if len(valid_faces) == 1:
                emotion = valid_faces[0]['dominant_emotion']
                announcement.append(f"One human present. Emotion is {emotion}.")
            elif len(valid_faces) > 1:
                # Grab a unique list of all the emotions in the crowd
                emotions = set([face['dominant_emotion'] for face in valid_faces])
                announcement.append(f"{len(valid_faces)} humans present. Emotions include: {', '.join(emotions)}.")

            # Step C: Speak the final batched sentence
            if announcement:
                final_speech = " ".join(announcement)
                print(f"AI Says: {final_speech}") # Print it to the terminal too
                speak(final_speech)
                last_spoken_time = current_time

    # --- DRAWING OBJECTS ON SCREEN ---
    for box in current_yolo_objects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])   
        class_id = int(box.cls[0])               
        object_name = yolo_model.names[class_id] 
        
        if object_name != "person": 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- DRAWING FACES ON SCREEN ---
    for face in current_faces:
        if 'region' in face:
            dominant_emotion = face['dominant_emotion']
            dominant_gender = max(face['gender'], key=face['gender'].get) 

            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{dominant_gender} | {dominant_emotion}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("AI Master Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()