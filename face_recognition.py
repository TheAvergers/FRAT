import cv2
import numpy as np
import pickle
import datetime
import os
import pygame
import time
from network_utils import stream_via_ffmpeg

class FaceRecognizer:
    def __init__(self, config):
        """Initialize face recognition system with configuration parameters"""
        self.config = config
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.last_seen = {}
        self.load_encodings()
        pygame.mixer.init()
        self.recognition_callback = None

    def load_encodings(self):
        """Load face encodings from pickle file"""
        print(f"Loading face encodings from {self.config['ENCODINGS_FILE']}")
        try:
            with open(self.config['ENCODINGS_FILE'], 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data['encodings'])} face encodings")
        except FileNotFoundError:
            print(f"Error: Encodings file {self.config['ENCODINGS_FILE']} not found")
            # Initialize with empty data
            self.data = {"encodings": [], "names": []}
            
    def set_recognition_callback(self, callback):
        """Set a callback function to be called when a face is recognized"""
        self.recognition_callback = callback

    def play_welcome_message(self, user):
        """Play a welcome message for the recognized user"""
        filename = f"{user}Recog.mp3"
        filepath = os.path.join(self.config['VOICE_LINES_DIR'], filename)
        if not os.path.exists(filepath):
            print(f"Warning: Voice file {filepath} not found")
            return
        
        print(f"Playing welcome message for {user}")
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    def process_frame(self, frame):
        """Process a single frame for face recognition"""
        # Create copies of the frame for processing and display
        display_frame = frame.copy()
        
        # Convert frame for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert face coordinates
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        
        detected_users = set()
        recognized_users = []
        
        if boxes:
            # Get face encodings (in real implementation, this would use face_recognition library)
            encodings = self._face_recognition_encodings(rgb, boxes)
            for encoding in encodings:
                # Compare with known faces (simplified for demo)
                matches = self._face_recognition_compare_faces(self.data['encodings'], encoding)
                name = 'Unknown'
                if True in matches:
                    # Find the best match
                    idxs = [i for i, m in enumerate(matches) if m]
                    counts = {}
                    for i in idxs:
                        n = self.data['names'][i]
                        counts[n] = counts.get(n, 0) + 1
                    name = max(counts, key=counts.get) if counts else 'Unknown'
                
                if name != 'Unknown':
                    detected_users.add(name)
                    recognized_users.append((name, boxes[len(recognized_users)]))
        
        # Check for users to welcome
        current_time = datetime.datetime.now()
        users_to_welcome = []
        
        for user in detected_users:
            last = self.last_seen.get(user)
            # If user hasn't been seen before or was last seen > timeout
            if not last or (current_time - last).total_seconds() > self.config['RECOGNITION_TIMEOUT']:
                users_to_welcome.append(user)
            # Update last seen time
            self.last_seen[user] = current_time
            
            # Notify callback about recognized user
            if self.recognition_callback:
                self.recognition_callback(user)
        
        # Draw rectangles and names
        for name, (top, right, bottom, left) in recognized_users:
            # Draw rectangle
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw name
            y = top - 15 if top > 15 else top + 15
            cv2.putText(
                display_frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2
            )
        
        return display_frame, users_to_welcome

    def _face_recognition_encodings(self, image, boxes):
        """Placeholder for face_recognition.face_encodings"""
        # In a real implementation, this would use actual face recognition
        return [np.random.rand(128) for _ in boxes]

    def _face_recognition_compare_faces(self, known_encodings, face_encoding, tolerance=0.6):
        """Placeholder for face_recognition.compare_faces"""
        # In a real implementation, this would use actual face comparison
        return [np.random.rand() < 0.2 for _ in known_encodings]

    def run_recognition_loop(self):
        """Run continuous face recognition loop from a video stream"""
        print(f"Starting face recognition from UDP stream on port {self.config['VIDEO_STREAM_PORT']}")
        
        # Start video stream
        ffmpeg_proc, pipe = stream_via_ffmpeg(
            self.config['VIDEO_STREAM_PORT'], 
            self.config['VIDEO_WIDTH'], 
            self.config['VIDEO_HEIGHT']
        )
        
        try:
            while True:
                try:
                    frame = next(pipe)
                    ret = True
                except StopIteration:
                    print("FFmpeg pipe closed.")
                    break
                
                if not ret:
                    print("Failed to receive frame. Retrying...")
                    time.sleep(0.1)
                    continue
                
                # Process the frame
                display_frame, users_to_welcome = self.process_frame(frame)
                
                # Play welcome messages
                for user in users_to_welcome:
                    self.play_welcome_message(user)
                
                # Display the frame
                cv2.imshow('Face Recognition', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up
            if ffmpeg_proc:
                ffmpeg_proc.kill()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # For standalone testing
    config = {
        'VIDEO_STREAM_PORT': 1234,
        'VIDEO_WIDTH': 640,
        'VIDEO_HEIGHT': 480,
        'ENCODINGS_FILE': "encodings.pickle",
        'VOICE_LINES_DIR': "voice_lines",
        'RECOGNITION_TIMEOUT': 300  # seconds
    }
    
    recognizer = FaceRecognizer(config)
    
    # Test callback
    def on_face_recognized(name):
        print(f"TEST: Recognized {name}")
    
    recognizer.set_recognition_callback(on_face_recognized)
    recognizer.run_recognition_loop()