import cv2
import numpy as np
import pickle
import datetime
import os
import pygame
import time
from network_utils import stream_via_ffmpeg
import face_recognition

class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.last_seen = {}
        self.load_encodings()
        pygame.mixer.init()
        self.recognition_callback = None

    def load_encodings(self):
        print(f"Loading face encodings from {self.config['ENCODINGS_FILE']}")
        try:
            with open(self.config['ENCODINGS_FILE'], 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data['encodings'])} face encodings")
        except FileNotFoundError:
            print(f"Error: Encodings file {self.config['ENCODINGS_FILE']} not found")
            self.data = {"encodings": [], "names": []}

    def set_recognition_callback(self, callback):
        self.recognition_callback = callback

    def play_welcome_message(self, user):
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
        display_frame = frame.copy()

        # Resize for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Convert boxes to (top, right, bottom, left) for the small frame
        boxes_small = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

        detected_users = set()
        recognized_users = []

        if boxes_small:
            encodings = self._face_recognition_encodings(rgb, boxes_small)
            for i, encoding in enumerate(encodings):
                matches = self._face_recognition_compare_faces(self.data['encodings'], encoding)
                name = 'Unknown'
                if True in matches:
                    idxs = [j for j, m in enumerate(matches) if m]
                    counts = {}
                    for j in idxs:
                        n = self.data['names'][j]
                        counts[n] = counts.get(n, 0) + 1
                    name = max(counts, key=counts.get) if counts else 'Unknown'

                if name != 'Unknown':
                    detected_users.add(name)
                    # Scale the box back up to original frame size for drawing
                    (top, right, bottom, left) = boxes_small[i]
                    full_box = (
                        int(top / scale),
                        int(right / scale),
                        int(bottom / scale),
                        int(left / scale)
                    )
                    recognized_users.append((name, full_box))

        current_time = datetime.datetime.now()
        users_to_welcome = []

        for user in detected_users:
            last = self.last_seen.get(user)
            if not last or (current_time - last).total_seconds() > self.config['RECOGNITION_TIMEOUT']:
                users_to_welcome.append(user)
            self.last_seen[user] = current_time

            if self.recognition_callback:
                self.recognition_callback(user)

        for name, (top, right, bottom, left) in recognized_users:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top > 15 else top + 15
            cv2.putText(
                display_frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2
            )

        return display_frame, users_to_welcome

    def _face_recognition_encodings(self, image, boxes):
        return face_recognition.face_encodings(image, boxes)

    def _face_recognition_compare_faces(self, known_encodings, face_encoding, tolerance=0.6):
        return face_recognition.compare_faces(known_encodings, face_encoding, tolerance)

    def run_recognition_loop(self):
        print(f"Starting face recognition from UDP stream on port {self.config['VIDEO_STREAM_PORT']}")

        ffmpeg_proc, pipe = stream_via_ffmpeg(
            self.config['VIDEO_STREAM_PORT'],
            self.config['VIDEO_WIDTH'],
            self.config['VIDEO_HEIGHT']
        )

        frame_count = 0

        try:
            while True:
                try:
                    frame = next(pipe)
                    frame_count += 1

                    if frame_count % 10 != 0:
                        continue

                    display_frame, users_to_welcome = self.process_frame(frame)

                    for user in users_to_welcome:
                        self.play_welcome_message(user)

                    cv2.imshow('Face Recognition', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except StopIteration:
                    print("FFmpeg pipe closed.")
                    break

        finally:
            if ffmpeg_proc:
                ffmpeg_proc.kill()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    config = {
        'VIDEO_STREAM_PORT': 1234,
        'VIDEO_WIDTH': 640,
        'VIDEO_HEIGHT': 480,
        'ENCODINGS_FILE': "encodings.pickle",
        'VOICE_LINES_DIR': "voice_lines",
        'RECOGNITION_TIMEOUT': 300
    }

    recognizer = FaceRecognizer(config)

    def on_face_recognized(name):
        print(f"TEST: Recognized {name}")

    recognizer.set_recognition_callback(on_face_recognized)
    recognizer.run_recognition_loop()
