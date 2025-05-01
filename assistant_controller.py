import os
import time
import threading
import openai
from face_recognition import FaceRecognizer
from audio_processor import AudioProcessor
from text_to_speech import TextToSpeech
from network_utils import purge_port

class AssistantController:
    def __init__(self, config):
        """Initialize the assistant controller with all components"""
        self.config = config
        
        # Make sure OPENAI_API_KEY is set
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI with v0.28.0 syntax
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        
        # Clean up ports before starting
        purge_port(config['VIDEO_STREAM_PORT'])
        purge_port(config['AUDIO_STREAM_PORT'])
        
        # Initialize components
        self.face_recognizer = FaceRecognizer(config)
        self.audio_processor = AudioProcessor(config)
        self.tts = TextToSpeech(config)
        
        # Last recognized user for personalized responses
        self.last_recognized_user = None
        
        # Configure audio processor callback
        self.audio_processor.set_activation_callback(self.handle_wake_word)
        
        # Threads
        self.face_thread = None
        self.audio_thread = None
        self.running = False
        
        # Track if TTS is currently speaking
        self.is_speaking = False
        
        # Set up face recognition callback
        self.setup_face_recognition_callback()
    
    def setup_face_recognition_callback(self):
        """Set up callback for face recognition updates"""
        def on_face_recognized(name):
            """Called when a face is recognized"""
            print(f"Face recognition update: {name}")
            self.last_recognized_user = name
            # Update audio processor with latest recognized user
            self.audio_processor.set_last_recognized_user(name)
            
        # Add the callback to our face recognizer
        self.face_recognizer.set_recognition_callback(on_face_recognized)
    
    def handle_wake_word(self, message, is_greeting):
        """Handle wake word detection with greeting or response"""
        if is_greeting:
            # This is the personalized greeting after wake word
            print(f"Playing greeting: {message}")
            self.is_speaking = True
            self.tts.speak(message)
            self.is_speaking = False
        else:
            # Handle the user's command
            print(f"Processing command: {message}")
            
            # Extract just the command part from "You said: [command]"
            command = message
            if message.startswith("You said: "):
                command = message[len("You said: "):]
                
            # If it's an actual command (not just echoing back), process it
            if not command.strip():
                self.tts.speak("I didn't catch that. Could you try again?")
                return
                
            response = self.process_command(command)
            print(f"Speaking response: {response}")
            self.is_speaking = True
            self.tts.speak(response)
            self.is_speaking = False
    
    def process_command(self, command):
        """Process a command using OpenAI API"""
        print(f"Processing command: {command}")
        
        try:
            # Using OpenAI v0.28.0 syntax
            completion = openai.ChatCompletion.create(
                model="gpt-4",  # Using gpt-4 instead of gpt-4o which wasn't available in v0.28.0
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Keep responses brief and conversational as they will be spoken aloud."},
                    {"role": "user", "content": command}
                ]
            )
            
            response = completion.choices[0].message.content
            print(f"Assistant response: {response}")
            return response
            
        except Exception as e:
            print(f"Error processing command: {e}")
            return "Sorry, I had trouble processing your request."
    
    def start(self):
        """Start the assistant system"""
        if self.running:
            print("Assistant already running")
            return
            
        self.running = True
        
        # Start face recognition in a separate thread
        self.face_thread = threading.Thread(
            target=self.face_recognizer.run_recognition_loop,
            daemon=True
        )
        self.face_thread.start()
        
        # Start audio processing
        self.audio_processor.start_audio_processing()
        
        print("Assistant system started and running")
    
    def stop(self):
        """Stop the assistant system"""
        if not self.running:
            print("Assistant not running")
            return
            
        print("Stopping assistant system...")
        self.running = False
        
        # Stop audio processing
        self.audio_processor.stop_audio_processing()
        
        # Face recognition thread will be terminated as daemon
        print("Assistant system stopped")


if __name__ == "__main__":
    # Configuration with updated parameters
    config = {
        'VIDEO_STREAM_PORT': 1234,
        'VIDEO_WIDTH': 640,
        'VIDEO_HEIGHT': 480,
        'AUDIO_STREAM_PORT': 1235,
        'AUDIO_SAMPLE_RATE': 16000,
        'AUDIO_CHANNELS': 1,
        'ENCODINGS_FILE': "encodings.pickle",
        'VOICE_LINES_DIR': "voice_lines",
        'RECOGNITION_TIMEOUT': 300,  # seconds
        'WAKE_WORD': 'hey assistant',
        'TTS_VOICE': 'nova',  # Options: alloy, echo, fable, onyx, nova, shimmer
        'SILENCE_THRESHOLD': 500,  # Updated from 1200 to 500
        'MIN_SPEECH_DURATION': 0.5,
        'PROMPT_SILENCE': 0.5  # Updated from 1.0 to 0.5
    }
    
    # Start assistant
    try:
        assistant = AssistantController(config)
        assistant.start()
        
        print("Assistant running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down assistant...")
        if assistant:
            assistant.stop()