import os
import time
import threading
import openai
from face_recognition import FaceRecognizer
from command_handler import CommandHandler
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
        
        # Clean up ports before starting
        purge_port(config['VIDEO_STREAM_PORT'])
        purge_port(config['AUDIO_STREAM_PORT'])
        
        # Initialize components - important to initialize TTS first!
        self.tts = TextToSpeech(config)
        
        # Wait a moment for TTS to initialize properly
        time.sleep(0.5)
        
        # Initialize remaining components after TTS
        self.face_recognizer = FaceRecognizer(config)
        self.audio_processor = AudioProcessor(config)
        
        # Initialize command handler after TTS is initialized
        self.command_handler = CommandHandler(tts_engine=self.tts)

        # Initialize OpenAI with v0.28.0 syntax
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        
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
        
        # Flag to prevent TTS overlap
        self.tts_lock = threading.Lock()
        
        # Set up face recognition callback
        self.setup_face_recognition_callback()
        
        # Conversation history for context
        self.conversation_history = []
        self.max_history_length = 10
    
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
        with self.tts_lock:  # Prevent multiple TTS calls from overlapping
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
                    
                # Process the command with OpenAI
                response = self.process_command(command)
                    
                # Speak the response
                self.is_speaking = True
                self.tts.speak(response)
                self.is_speaking = False
                
                print("Command processed, returning to wake word detection")
    
    def process_command(self, command):
        """Process a command using OpenAI API and CommandHandler"""
        print(f"Processing command: {command}")
        
        try:
            # Parse the command to determine intent
            cmd_type, cmd_args, raw_text = self.command_handler.parse_command(command)
            
            # For system commands, use direct handling
            if cmd_type != 'general_query':
                response = self.command_handler.execute_command(cmd_type, cmd_args, raw_text)
                print(f"Command executed: {cmd_type}, Response: {response}")
                return response
            
            # Otherwise, use OpenAI for general queries
            # Personalize the system message based on the user if available
            system_message = "You are a helpful home assistant. Keep responses brief and conversational as they will be spoken aloud."
            
            if self.last_recognized_user:
                system_message += f" The user's name is {self.last_recognized_user}."
                
            # Add context from conversation history if available
            messages = [{"role": "system", "content": system_message}]
            
            # Include recent conversation history for context
            for msg in self.conversation_history:
                messages.append(msg)
                
            # Add the current command
            messages.append({"role": "user", "content": command})
            
            # Using OpenAI v0.28.0 syntax
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=150,  # Keep responses concise for voice
                temperature=0.7
            )
            
            response = completion.choices[0].message.content
            print(f"Assistant response: {response}")
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": command})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Limit conversation history size
            if len(self.conversation_history) > self.max_history_length * 2:  # *2 because each exchange has 2 messages
                self.conversation_history = self.conversation_history[-self.max_history_length*2:]
                
            return response
            
        except Exception as e:
            print(f"Error processing command: {e}")
            return "Sorry, I had trouble processing your request."
            
    def start(self):
        """Start all assistant components"""
        if self.running:
            print("Assistant is already running")
            return
            
        self.running = True
        
        # Start TTS first to ensure it's ready for other components
        print("Starting audio processing...")
        # Start audio processing first
        self.audio_processor.start_audio_processing()
        
        # Give audio processing a moment to initialize
        time.sleep(0.5)
        
        # Start face recognition in a separate thread
        print("Starting face recognition...")
        self.face_recognizer.run_recognition_loop()
        
        print("Assistant started")
        
    def stop(self):
        """Stop all assistant components"""
        if not self.running:
            print("Assistant is not running")
            return
            
        self.running = False
        
        # Stop audio processing
        if hasattr(self.audio_processor, 'stop_audio_processing'):
            self.audio_processor.stop_audio_processing()
            
        # Stop face recognition
        if hasattr(self.face_recognizer, 'stop_face_recognition'):
            self.face_recognizer.stop_face_recognition()
            
        print("Assistant stopped")


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
        'SILENCE_THRESHOLD': 500,
        'MIN_SPEECH_DURATION': 0.5,
        'PROMPT_SILENCE': 0.5
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