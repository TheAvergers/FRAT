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
        self.activation_callback = callback

    def set_last_recognized_user(self, username):
        self.last_recognized_user = username

    def start_audio_processing(self):
        self.is_processing = True
        self.stop_event.clear()
        threading.Thread(target=self._process_audio_stream, daemon=True).start()

    def stop_audio_processing(self):
        self.is_processing = False
        self.stop_event.set()

    def _is_mostly_silence(self, audio_array, threshold=None):
        threshold = threshold or self.silence_threshold
        rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        return rms < threshold

    def _capture_command_audio(self, max_duration=5.0, silence_timeout=1.0):
        print("Capturing command audio...")
        buffer = np.array([], dtype=np.int16)
        start_time = time.time()
        last_speech_time = time.time()

        while time.time() - start_time < max_duration:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                buffer = np.append(buffer, audio_chunk)

                if not self._is_mostly_silence(audio_chunk):
                    last_speech_time = time.time()
            else:
                time.sleep(0.05)

            if time.time() - last_speech_time > silence_timeout:
                print("Detected end of speech (silence).")
                break

        print(f"Captured {len(buffer)} samples of command audio.")
        return buffer

    def _check_for_wake_word(self):
        buffer_copy = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_wake_word_time = time.time()

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            fname = tmp_file.name
        with wave.open(fname, 'wb') as wf:
            wf.setnchannels(self.config['AUDIO_CHANNELS'])
            wf.setsampwidth(2)
            wf.setframerate(self.config['AUDIO_SAMPLE_RATE'])
            wf.writeframes(buffer_copy.tobytes())

        try:
            with open(fname, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    temperature=0.0
                )
            text = transcript.get("text", "").lower().strip()
            print(f"Wake word check transcription: {text}")

            if any(w in text for w in [self.wake_word, self.alt_wake_word, "hey assistant"]):
                print("Wake word detected. Capturing full command...")
                command_buffer = self._capture_command_audio()

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as cmd_file:
                    cmd_fname = cmd_file.name
                with wave.open(cmd_fname, 'wb') as wf:
                    wf.setnchannels(self.config['AUDIO_CHANNELS'])
                    wf.setsampwidth(2)
                    wf.setframerate(self.config['AUDIO_SAMPLE_RATE'])
                    wf.writeframes(command_buffer.tobytes())

                try:
                    with open(cmd_fname, 'rb') as audio_file:
                        cmd_transcript = openai.Audio.transcribe(
                            model="whisper-1",
                            file=audio_file,
                            language="en",
                            temperature=0.0
                        )
                    full_text = cmd_transcript.get("text", "").lower().strip()
                    print(f"Full command transcription: {full_text}")

                    if full_text and self.activation_callback:
                        self.activation_callback(full_text, False)

                except Exception as e:
                    print(f"Error processing command audio: {e}")
                finally:
                    if os.path.exists(cmd_fname):
                        os.unlink(cmd_fname)

        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def _process_audio_stream(self):
        print("Starting audio processing loop with self-healing...")

        while not self.stop_event.is_set():
            try:
                ffmpeg_proc, audio_pipe = stream_via_ffmpeg_audio(
                    self.config['AUDIO_STREAM_PORT'],
                    self.config['AUDIO_SAMPLE_RATE'],
                    self.config['AUDIO_CHANNELS']
                )
                last_audio_chunk_time = time.time()
                last_wake_check_time = time.time()
                print("FFmpeg audio stream started successfully.")

                for audio_chunk in audio_pipe:
                    if self.stop_event.is_set():
                        print("Stop event detected, exiting audio loop.")
                        break

                    last_audio_chunk_time = time.time()
                    self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
                    if len(self.audio_buffer) > self.max_buffer_size:
                        self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]

                    # Feed into audio queue for command capture
                    self.audio_queue.put(audio_chunk)

                    current_time = time.time()
                    if (current_time - last_wake_check_time >= 0.5 and
                        current_time - self.last_wake_word_time > self.wake_word_cooldown and
                        not self._is_mostly_silence(self.audio_buffer)):

                        last_wake_check_time = current_time
                        threading.Thread(target=self._check_for_wake_word, daemon=True).start()

                    if time.time() - last_audio_chunk_time > 10:
                        print("No audio received for 10 seconds, restarting FFmpeg...")
                        break

            except StopIteration:
                print("FFmpeg audio pipe closed (StopIteration). Restarting FFmpeg...")
            except Exception as e:
                print(f"Audio processing error: {e}. Restarting FFmpeg...")

            if ffmpeg_proc:
                try:
                    ffmpeg_proc.kill()
                except Exception:
                    pass

            print("Restarting FFmpeg audio stream in 2 seconds...")
            time.sleep(2)
