import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy import stats
import mediapipe as mp

# Load your model here
model = load_model("D2M1.h5")

# Add your utility functions here (e.g., mediapipe_detection, draw_styled_landmarks, extract_keypoints, etc.)
actions = np.array(['ಶುಭೋದಯ',
 'ದೊಡ್ಡದು',
 'ನೀವು',
 'ದೊಡ್ಡದು',
 'ಧನ್ಯವಾದ',
 'ಸಮಯ',
 'ನಮಸ್ಕಾರ',
 'ವಿಮಾನ',
 'ಸಂತೋಷ',
 'ಒಳ್ಳೆಯದು',
 'ಶುಭ ರಾತ್ರಿ',
 'ಚಿಕ್ಕದು',
 'ನಾನು',
 'ಬೈಸಿಕಲ್',
 'ಎತ್ತರದ',
'ಯುವ',
'ನಿಧಾನ',
'ಹೊಸ',
'ಅನಾರೋಗ್ಯ',
'ತಂಪಾದ'])

no_sequences = 20
sequence_length = 40
start_folder = 0

label_map = {label: num for num, label in enumerate(actions)}

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.5
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        image, results = mediapipe_detection(img, self.holistic)
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-40:]

        if len(self.sequence) == 40:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            self.predictions.append(np.argmax(res))

            if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > self.threshold:
                    if len(self.sentence) > 0:
                        if actions[np.argmax(res)] != self.sentence[-1]:
                            self.sentence.append(actions[np.argmax(res)])
                    else:
                        self.sentence.append(actions[np.argmax(res)])

            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(self.sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image

st.title("Sign Language Live Detector")
webrtc_streamer(key="sign-language-detector", video_transformer_factory=VideoTransformer)
