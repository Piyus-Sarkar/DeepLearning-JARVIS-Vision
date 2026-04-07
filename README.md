# 👁️ J.A.R.V.I.S. Deep Learning Vision System

A real-time, event-driven computer vision dashboard built with Python and Streamlit. This project integrates multiple Deep Learning models to perform simultaneous object detection, facial recognition, emotion analysis, and demographic prediction, all narrated by a threaded text-to-speech engine.

## 🧠 Deep Learning Architecture

This system utilizes two primary neural network pipelines running in parallel:

1. **Ultralytics YOLOv8 (You Only Look Once):**
   * A state-of-the-art Convolutional Neural Network (CNN) used for high-speed, real-time object detection.
   * Tracks and identifies over 80 distinct classes of objects in the environment.

2. **DeepFace Framework:**
   * Utilizes deep CNNs (like VGG-Face) for precise facial recognition.
   * Analyzes facial landmarks to predict dominant emotions (Happy, Sad, Angry, Neutral, etc.) and demographic presentation.

## ✨ Key Features

* **Real-Time Web Dashboard:** Built with Streamlit for a clean, interactive user interface.
* **Event-Driven Memory:** The AI utilizes short-term state tracking to only announce *new* events (new objects, new humans, or changing emotions), preventing repetitive spam.
* **Threaded Voice Engine:** Uses `pyttsx3` running on a separate daemon thread to allow J.A.R.V.I.S. to speak announcements without freezing the video feed.
* **Smart Batching Logic:** Groups multiple detections into a single, cohesive spoken sentence.

## 🚀 How to Run Locally

1. Clone this repository.
2. Install the required deep learning and vision dependencies:
   ```bash
   pip install -r requirements.txt
3.Launch the Streamlit dashboard: streamlit run app.py
