import cv2
import pickle
import time
import numpy as np
import subprocess
import os

# Configuration
VIDEO_STREAM_PORT = 1234
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ENCODINGS_FILE = "encodings.pickle"


def purge_port(port):
    print(f"Purging processes on UDP port {port}...")
    if os.name == 'nt':
        try:
            output = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True
            ).decode(errors='ignore')
            for line in output.splitlines():
                parts = line.split()
                if parts:
                    pid = parts[-1]
                    subprocess.call(f'taskkill /PID {pid} /F', shell=True)
                    print(f"Killed process {pid} on port {port}")
        except subprocess.CalledProcessError:
            pass
    else:
        try:
            output = subprocess.check_output(
                f'lsof -i udp:{port} -t', shell=True
            ).decode(errors='ignore')
            for pid in output.split():
                subprocess.call(['kill', '-9', pid])
                print(f"Killed PID {pid} on UDP port {port}")
        except subprocess.CalledProcessError:
            pass


def stream_via_ffmpeg(port, width, height):
    """
    Spawn system FFmpeg to read UDP and pipe raw frames to Python.
    Returns the subprocess and a generator yielding frames.
    """
    url = f"udp://0.0.0.0:{port}?fifo_size=10000000&overrun_nonfatal=1"
    cmd = [
        'ffmpeg',
        '-i', url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    print(f"Starting FFmpeg subprocess: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def reader():
        frame_size = width * height * 3
        while True:
            raw = p.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            # Copy to make writeable
            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3)).copy()
            yield frame
    return p, reader()


def test_face_recognition():
    print("Starting Face Recognition Test")
    purge_port(VIDEO_STREAM_PORT)

    print(f"Loading face encodings from {ENCODINGS_FILE}")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['encodings'])} face encodings")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print(f"Spawning FFmpeg stream on UDP port {VIDEO_STREAM_PORT}")
    ffmpeg_proc, pipe = stream_via_ffmpeg(
        VIDEO_STREAM_PORT, VIDEO_WIDTH, VIDEO_HEIGHT
    )

    print("Starting face recognition loop...")
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

            names = []
            if boxes:
                encodings = face_recognition_encodings(rgb, boxes)
                for encoding in encodings:
                    matches = face_recognition_compare_faces(
                        data['encodings'], encoding
                    )
                    name = 'Unknown'
                    if True in matches:
                        idxs = [i for i, m in enumerate(matches) if m]
                        counts = {}
                        for i in idxs:
                            n = data['names'][i]
                            counts[n] = counts.get(n, 0) + 1
                        name = max(counts, key=counts.get)
                    names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top > 15 else top + 15
                cv2.putText(
                    frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2
                )

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if ffmpeg_proc:
            ffmpeg_proc.kill()
        cv2.destroyAllWindows()


def face_recognition_encodings(image, boxes):
    return [np.random.rand(128) for _ in boxes]


def face_recognition_compare_faces(known_encodings, face_encoding, tolerance=0.6):
    return [np.random.rand() < 0.2 for _ in known_encodings]

if __name__ == '__main__':
    test_face_recognition()
