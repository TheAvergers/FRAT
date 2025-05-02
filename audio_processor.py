import os
import time
import subprocess
import tempfile
import wave
import threading
import queue
import openai
from network_utils import stream_via_ffmpeg_audio
import numpy as np

class AudioProcessor:
    def __init__(self, config):
        """Initialize audio processing system"""
        self.config = config
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        self.wake_word = config['WAKE_WORD'].lower()
        self.alt_wake_word = "assistant"
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self.stop_event = threading.Event()
        self.audio_buffer = np.array([], dtype=np.int16)
        self.activation_callback = None
        self.last_recognized_user = None
        self.silence_threshold = config.get('SILENCE_THRESHOLD', 200)
        self.max_buffer_size = self.config['AUDIO_SAMPLE_RATE'] * 3  # 3-second buffer
        self.last_wake_word_time = 0
        self.wake_word_cooldown = 3.0

    def set_activation_callback(self, callback):
        """Set callback to call when wake word is detected"""
        self.activation_callback = callback

    def set_last_recognized_user(self, username):
        """Set the last recognized user"""
        self.last_recognized_user = username

    def start_audio_processing(self):
        """Start processing audio in a separate thread"""
        self.is_processing = True
        self.stop_event.clear()
        threading.Thread(target=self._process_audio_stream, daemon=True).start()

    def stop_audio_processing(self):
        """Stop the audio processing thread"""
        self.is_processing = False
        self.stop_event.set()

    def _process_audio_stream(self):
        """Process audio stream from UDP"""
        ffmpeg_proc, audio_pipe = stream_via_ffmpeg_audio(
            self.config['AUDIO_STREAM_PORT'],
            self.config['AUDIO_SAMPLE_RATE'],
            self.config['AUDIO_CHANNELS']
        )
        last_check_time = time.time()

        try:
            while not self.stop_event.is_set():
                audio_chunk = next(audio_pipe)
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
                if len(self.audio_buffer) > self.max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]

                current_time = time.time()
                if current_time - last_check_time >= 0.5 and \
                   current_time - self.last_wake_word_time > self.wake_word_cooldown and \
                   not self._is_mostly_silence(self.audio_buffer):
                    last_check_time = current_time
                    threading.Thread(target=self._check_for_wake_word, daemon=True).start()
        finally:
            if ffmpeg_proc:
                ffmpeg_proc.kill()

    def _is_mostly_silence(self, audio_array, threshold=None):
        threshold = threshold or self.silence_threshold
        rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        return rms < threshold

    def _check_for_wake_word(self):
        """Check if wake word is in audio buffer and immediately process command"""
        # Copy and reset buffer to avoid re-detect
        buffer_copy = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_wake_word_time = time.time()

        # Save buffer to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            fname = tmp_file.name
        with wave.open(fname, 'wb') as wf:
            wf.setnchannels(self.config['AUDIO_CHANNELS'])
            wf.setsampwidth(2)
            wf.setframerate(self.config['AUDIO_SAMPLE_RATE'])
            wf.writeframes(buffer_copy.tobytes())

        try:
            # Transcribe entire utterance
            with open(fname, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    temperature=0.0
                )
            text = transcript.get("text", "").lower().strip()
            if not text:
                return

            # Detect wake word presence
            if any(w in text for w in [self.wake_word, self.alt_wake_word, "hey assistant"]):
                # Remove wake word from text
                cmd = text
                for w in [self.wake_word, self.alt_wake_word, "hey assistant"]:
                    cmd = cmd.replace(w, "")
                cmd = cmd.strip()

                if cmd and self.activation_callback:
                    # Directly process command without separate listening
                    self.activation_callback(cmd, False)
        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            if os.path.exists(fname):
                os.unlink(fname)
