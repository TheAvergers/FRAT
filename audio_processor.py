import os
import time
import subprocess
import tempfile
import wave
import threading
import queue
import openai
from network_utils import stream_via_ffmpeg_audio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np

class AudioProcessor:
    def __init__(self, config):
        """Initialize audio processing system"""
        self.config = config
        
        # Using OpenAI v0.28.0 syntax
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        self.wake_word = config['WAKE_WORD'].lower()
        # Add "assistant" as a secondary wake word
        self.alt_wake_word = "assistant"
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self.stop_event = threading.Event()
        self.audio_buffer = np.array([], dtype=np.int16)
        self.activation_callback = None
        self.last_recognized_user = None
        
        # Better detection parameters
        self.silence_threshold = config.get('SILENCE_THRESHOLD', 500)
        self.min_speech_duration = config.get('MIN_SPEECH_DURATION', 0.5)  # seconds
        self.prompt_silence = config.get('PROMPT_SILENCE', 0.5)  # seconds
        self.max_buffer_size = self.config['AUDIO_SAMPLE_RATE'] * 3  # 3-second buffer
        
        # Add wake word cooldown to prevent repeated detections
        self.last_wake_word_time = 0
        self.wake_word_cooldown = 3.0  # seconds
        
        # Track recent transcripts to avoid duplicates
        self.recent_transcripts = []
        self.max_recent_transcripts = 5
        
        # Wake word detection state
        self.processing_wake_word = False
        
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
        print(f"Starting audio processing from UDP port {self.config['AUDIO_STREAM_PORT']}")
        
        # Start audio stream
        ffmpeg_proc, audio_pipe = stream_via_ffmpeg_audio(
            self.config['AUDIO_STREAM_PORT'],
            self.config['AUDIO_SAMPLE_RATE'],
            self.config['AUDIO_CHANNELS']
        )
        
        try:
            # Process audio chunks
            last_check_time = time.time()
            
            while not self.stop_event.is_set():
                try:
                    audio_chunk = next(audio_pipe)
                    self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
                    
                    # Keep a rolling buffer of the most recent audio
                    if len(self.audio_buffer) > self.max_buffer_size:
                        self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
                    
                    # Check for wake word less frequently (every 0.5 seconds)
                    # and only if we're not already processing a wake word
                    current_time = time.time()
                    if (not self.processing_wake_word and 
                        current_time - last_check_time >= 0.5 and 
                        len(self.audio_buffer) >= self.config['AUDIO_SAMPLE_RATE'] and
                        current_time - self.last_wake_word_time > self.wake_word_cooldown):
                        
                        last_check_time = current_time
                        # Start wake word detection in a separate thread to avoid blocking audio collection
                        threading.Thread(target=self._check_for_wake_word, daemon=True).start()
                        
                except StopIteration:
                    print("Audio pipe closed.")
                    break
                
        finally:
            if ffmpeg_proc:
                ffmpeg_proc.kill()
                
    def _is_mostly_silence(self, audio_array, threshold=None):
        """Check if audio is mostly silence based on RMS amplitude"""
        if threshold is None:
            threshold = self.silence_threshold
            
        # Use RMS (root mean square) for better silence detection
        rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        return rms < threshold
                
    def _check_for_wake_word(self):
        """Check if wake word is in audio buffer"""
        # Set flag to indicate we're processing a wake word detection
        self.processing_wake_word = True
        
        try:
            # Skip processing if the buffer is mostly silence
            if self._is_mostly_silence(self.audio_buffer):
                self.audio_buffer = self.audio_buffer[-int(self.config['AUDIO_SAMPLE_RATE'] * 0.5):]
                self.processing_wake_word = False
                return
                
            # Make a copy of the buffer for processing
            buffer_copy = self.audio_buffer.copy()
            
            # Save audio buffer to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_filename = tmp_file.name
                
            # Save buffer to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.config['AUDIO_CHANNELS'])
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.config['AUDIO_SAMPLE_RATE'])
                wf.writeframes(buffer_copy.tobytes())
                
            try:
                # Use OpenAI Whisper for STT with v0.28.0 syntax
                with open(temp_filename, 'rb') as audio_file:
                    transcript = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file,
                        language="en",
                        temperature=0.0,
                        prompt=f"The wake word is '{self.wake_word}'. Listen carefully."
                    )
                    
                text = transcript["text"].lower().strip()
                if text:
                    print(f"Transcribed: {text}")
                
                # Check if this is a duplicate transcript
                is_duplicate = text in self.recent_transcripts
                
                # Add to recent transcripts list and maintain its size
                if not is_duplicate:
                    self.recent_transcripts.append(text)
                    if len(self.recent_transcripts) > self.max_recent_transcripts:
                        self.recent_transcripts.pop(0)
                
                # Check if wake word is in the transcript (including alternative wake word)
                if (self.wake_word in text or self.alt_wake_word in text) and not is_duplicate:
                    detected_word = self.wake_word if self.wake_word in text else self.alt_wake_word
                    print(f"Wake word '{detected_word}' detected!")
                    
                    # Update the last detection time
                    self.last_wake_word_time = time.time()
                    
                    # Reset buffer AFTER detection to avoid re-detecting the same audio
                    self.audio_buffer = np.array([], dtype=np.int16)
                    
                    # Create personalized greeting with proper format
                    if self.last_recognized_user:
                        greeting = f"What do you need {self.last_recognized_user}?"
                    else:
                        greeting = "What do you need?"
                    
                    # Call the activation callback if one is set
                    if self.activation_callback:
                        self.activation_callback(greeting, True)  # True indicates this is a greeting
                        
                        # Wait for TTS to finish
                        time.sleep(1.0)
                        
                        # Wait for user response
                        command = self.listen_for_command(6)  # 6 second timeout
                        
                        if command:
                            # Echo back the command through TTS
                            self.activation_callback(f"You said: {command}", False)
                
            except Exception as e:
                print(f"Error processing audio: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        finally:
            # Reset processing flag
            self.processing_wake_word = False
                
    def _enhance_audio(self, audio_data):
        """Apply basic audio enhancement to improve transcription quality"""
        # Normalize audio to use full dynamic range
        if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0:
            normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9)
            return normalized
        return audio_data
    
    def listen_for_command(self, timeout=6):
        """Listen for a command immediately after wake word detection"""
        print(f"Listening for command with {timeout} second timeout...")
        
        # Create a temporary file for the recording
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_filename = tmp_file.name
            
        # Record audio for command
        start_time = time.time()
        command_buffer = np.array([], dtype=np.int16)
        
        # Start audio stream
        ffmpeg_proc, audio_pipe = stream_via_ffmpeg_audio(
            self.config['AUDIO_STREAM_PORT'],
            self.config['AUDIO_SAMPLE_RATE'],
            self.config['AUDIO_CHANNELS']
        )
        
        try:
            # Wait briefly for user to start speaking
            initial_wait = time.time()
            while time.time() - initial_wait < self.prompt_silence:
                try:
                    next(audio_pipe)  # Discard initial silence
                except StopIteration:
                    print("Audio pipe closed during initial wait.")
                    break
            
            print("Ready for command, recording started...")
            
            # Collect audio for the command
            speech_started = False
            silence_start = None
            last_significant_audio = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    audio_chunk = next(audio_pipe)
                    
                    # Add to buffer
                    command_buffer = np.append(command_buffer, audio_chunk)
                    
                    # Check if speech has started
                    if not speech_started and not self._is_mostly_silence(audio_chunk, threshold=200):
                        speech_started = True
                        last_significant_audio = time.time()
                        print("Speech detected, recording...")
                    
                    # Check for end of speech (silence after speech)
                    if speech_started:
                        if self._is_mostly_silence(audio_chunk, threshold=200):
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > 1.0:  # 1 second of silence after speech
                                print("End of speech detected, processing command...")
                                break
                        else:
                            silence_start = None  # Reset if we hear something
                            last_significant_audio = time.time()
                            
                    # Emergency timeout
                    if speech_started and time.time() - last_significant_audio > 2.5:
                        print("No speech for 2.5 seconds, ending recording...")
                        break
                            
                except StopIteration:
                    print("Audio pipe closed during command recording.")
                    break
            
            # Skip processing if no speech detected
            if not speech_started or len(command_buffer) < self.config['AUDIO_SAMPLE_RATE'] * self.min_speech_duration:
                print("No clear speech detected in recording")
                return ""
            
            # Enhance audio quality
            enhanced_audio = self._enhance_audio(command_buffer)
                
            # Save the recording
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.config['AUDIO_CHANNELS'])
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.config['AUDIO_SAMPLE_RATE'])
                wf.writeframes(enhanced_audio.tobytes())
                
            # Transcribe the command
            with open(temp_filename, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    temperature=0.0,
                    prompt="This is a voice command. Transcribe exactly what the person said."
                )
                
            command = transcript["text"].strip()
            print(f"Command transcribed: {command}")
            return command
            
        except Exception as e:
            print(f"Error recording command: {e}")
            return ""
        finally:
            if ffmpeg_proc:
                ffmpeg_proc.kill()
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
                

if __name__ == "__main__":
    # For standalone testing
    config = {
        'AUDIO_STREAM_PORT': 1235,
        'AUDIO_SAMPLE_RATE': 16000,
        'AUDIO_CHANNELS': 1,
        'WAKE_WORD': 'hey assistant',
        'SILENCE_THRESHOLD': 500,
        'MIN_SPEECH_DURATION': 0.5,
        'PROMPT_SILENCE': 0.5
    }
    
    # Make sure OPENAI_API_KEY is set in environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)
        
    processor = AudioProcessor(config)
    
    def on_activation(message, is_greeting):
        print(f"{'Greeting' if is_greeting else 'Response'}: {message}")
        
    processor.set_activation_callback(on_activation)
    processor.set_last_recognized_user("John")  # For testing
    processor.start_audio_processing()
    
    try:
        print("Audio processor running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping audio processor...")
        processor.stop_audio_processing()