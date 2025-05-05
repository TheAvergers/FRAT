import cv2
import numpy as np
import pickle
import datetime
import os
import pygame
import time
import threading
import queue
from collections import defaultdict
from network_utils import stream_via_ffmpeg

class FaceRecognizer:
    def __init__(self, config):
        self.config = config

        proto_path = config.get('DNN_PROTO_PATH', 'models/deploy.prototxt')
        model_path = config.get('DNN_MODEL_PATH', 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

        if not os.path.isfile(proto_path) or not os.path.isfile(model_path):
            raise FileNotFoundError(f"DNN model files not found. Expected: {proto_path}, {model_path}")

        self.face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU backend for face detection to ensure consistent performance.")

        self.last_seen = {}
        self.detection_streak = defaultdict(int)
        self.load_encodings()
        pygame.mixer.init()
        self.recognition_callback = None

        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

    def load_encodings(self):
        print(f"Loading face encodings from {self.config['ENCODINGS_FILE']}")
        try:
            with open(self.config['ENCODINGS_FILE'], 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data['encodings'])} face encodings")
        except FileNotFoundError:
            print(f"Error: Encodings file {self.config['ENCODINGS_FILE']} not found")
            self.data = {"encodings": [], "names": []}

    def run_recognition_loop(self):
        print(f"Starting face recognition from UDP stream on port {self.config['VIDEO_STREAM_PORT']}")

        ffmpeg_proc, pipe = stream_via_ffmpeg(
            self.config['VIDEO_STREAM_PORT'],
            self.config['VIDEO_WIDTH'],
            self.config['VIDEO_HEIGHT']
        )

        worker_thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        worker_thread.start()

        try:
            while not self.stop_event.is_set():
                try:
                    frame = next(pipe)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                except StopIteration:
                    print("FFmpeg pipe closed.")
                    break
        finally:
            self.stop_event.set()
            worker_thread.join()
            if ffmpeg_proc:
                ffmpeg_proc.kill()
            cv2.destroyAllWindows()

    def _process_frames_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                display_frame, confirmed_users = self.process_frame(frame)

                for user in confirmed_users:
                    self.play_welcome_message(user)

                cv2.imshow('Face Recognition', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
            except queue.Empty:
                continue

    def process_frame(self, frame):
        display_frame = frame.copy()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                boxes.append((y1, x2, y2, x1))

        detected_users = set()
        confirmed_users = []

        if boxes:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = self._face_recognition_encodings(rgb, boxes)
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
                    (top, right, bottom, left) = boxes[i]
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    y = top - 15 if top > 15 else top + 15
                    cv2.putText(
                        display_frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2
                    )

        current_time = datetime.datetime.now()

        for user in detected_users:
            self.detection_streak[user] += 1
            if self.detection_streak[user] >= 3:
                last = self.last_seen.get(user)
                if not last or (current_time - last).total_seconds() > self.config['RECOGNITION_TIMEOUT']:
                    confirmed_users.append(user)
                self.last_seen[user] = current_time
                self.detection_streak[user] = 0

            if self.recognition_callback:
                self.recognition_callback(user)

        for user in list(self.detection_streak.keys()):
            if user not in detected_users:
                self.detection_streak[user] = 0

        return display_frame, confirmed_users

    def _face_recognition_encodings(self, image, boxes):
        import face_recognition
        return face_recognition.face_encodings(image, boxes)

    def _face_recognition_compare_faces(self, known_encodings, face_encoding, tolerance=0.6):
        import face_recognition
        return face_recognition.compare_faces(known_encodings, face_encoding, tolerance)

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