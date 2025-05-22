import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.callbacks import TensorBoard

class StrokeTrainer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence_length = 30  # Number of frames to consider for each prediction
        self.data = []
        self.labels = []
        
    def extract_pose_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Extract key points
        landmarks = results.pose_landmarks.landmark
        
        # Get relevant body positions
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
            
        return np.array(features)
        
    def process_video(self, video_path, label):
        cap = cv2.VideoCapture(video_path)
        frame_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            features = self.extract_pose_features(frame)
            if features is not None:
                frame_sequence.append(features)
                
                if len(frame_sequence) == self.sequence_length:
                    self.data.append(np.array(frame_sequence))
                    self.labels.append(label)
                    frame_sequence = frame_sequence[self.sequence_length//2:]  # Overlap sequences
                    
        cap.release()
        
    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.sequence_length, 99, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')  # 2 classes: forehand, backhand
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
                     
        return model
        
    def train(self, data_dir):
        # Process all videos in the directory
        for stroke_type in ['forehand', 'backhand']:
            stroke_dir = os.path.join(data_dir, stroke_type)
            if not os.path.exists(stroke_dir):
                continue
                
            print(f"Processing {stroke_type} videos...")
            for video_file in os.listdir(stroke_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(stroke_dir, video_file)
                    label = {'forehand': 0, 'backhand': 1}[stroke_type]
                    self.process_video(video_path, label)
                    
        if not self.data:
            print("No valid data found!")
            return
            
        # Convert to numpy arrays
        X = np.array(self.data)
        y = to_categorical(np.array(self.labels))
        
        # Reshape for CNN
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = self.create_model()
        print("Training model...")
        
        # Add TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
        
        # Save model and training history
        model.save('tennis_stroke_model.h5')
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
            
        print("Model trained and saved!")
        return model, history

if __name__ == "__main__":
    trainer = StrokeTrainer()
    data_dir = "tennis_clips"  # Directory containing your video clips
    trainer.train(data_dir) 