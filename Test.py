import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

model = tf.keras.models.load_model('rps_model.h5')

def classify_hand_landmarks(landmarks):
    landmarks = np.array(landmarks).flatten()
    landmarks = np.expand_dims(landmarks, axis=0)
    prediction = model.predict(landmarks)
    return np.argmax(prediction)

def predict():
    ret, frame = cap.read()
    if not ret:
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            gesture_id = classify_hand_landmarks(landmarks)
            
            if gesture_id == 0:
                result = "Rock"
            elif gesture_id == 1:
                result = "Paper"
            else:
                result = "Scissors"
            
            lbl_result.config(text="Prediction: " + result)

    img_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    lbl_img.imgtk = imgtk
    lbl_img.configure(image=imgtk)

    lbl_img.after(10, predict)

root = Tk()
root.title("Rock-Paper-Scissors Prediction")

cap = cv2.VideoCapture(0)

lbl_img = Label(root)
lbl_img.pack()

lbl_result = Label(root, text="Prediction: ", font=("Arial", 24))
lbl_result.pack()

predict()

root.mainloop()

cap.release()
