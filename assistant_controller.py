# assistant_controller.py
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

        # Mode: 'command' or 'conversation'
        self.mode = config.get('MODE', 'command')

        # Ensure API key is available
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Purge any existing streams on configured ports
        purge_port(config['VIDEO_STREAM_PORT'])
        purge_port(config['AUDIO_STREAM_PORT'])

        # Initialize Text-to-Speech first
        self.tts = TextToSpeech(config)
        # Give TTS a moment to initialize
        time.sleep(0.5)

        # Initialize face recognizer and audio processor
        self.face_recognizer = FaceRecognizer(config)
        self.audio_processor = AudioProcessor(config)

        # Initialize command handler in selected mode
        self.command_handler = CommandHandler(tts_engine=self.tts, mode=self.mode)

        # Configure OpenAI
        openai.api_key = os.environ.get('OPENAI_API_KEY')

        # Track last recognized user for personalization
        self.last_recognized_user = None

        # Lock to prevent overlapping TTS
        self.tts_lock = threading.Lock()
        self.is_speaking = False

        # Conversation history for context
        self.conversation_history = []
        self.max_history_length = config.get('MAX_HISTORY_LENGTH', 10)

        # Wire up callbacks
        self.audio_processor.set_activation_callback(self.handle_wake_word)
        self.setup_face_recognition_callback()

        # Running flag
        self.running = False

    def setup_face_recognition_callback(self):
        """Register a callback to capture recognized faces"""
        def on_face_recognized(name):
            print(f"Face recognition update: {name}")
            self.last_recognized_user = name
            self.audio_processor.set_last_recognized_user(name)
        self.face_recognizer.set_recognition_callback(on_face_recognized)

    def handle_wake_word(self, message, is_greeting):
        """Handle the wake-word event: greeting vs. command"""
        with self.tts_lock:
            if is_greeting:
                # Play personalized greeting
                print(f"Playing greeting: {message}")
                self.is_speaking = True
                self.tts.speak(message)
                self.is_speaking = False
            else:
                # Process the follow-up command
                print(f"Processing command: {message}")
                # Extract the raw command text
                cmd_text = message
                if message.lower().startswith("you said:"):
                    cmd_text = message[len("you said:"):].strip()
                response = self.process_command(cmd_text)
                # Speak the response
                self.is_speaking = True
                self.tts.speak(response)
                self.is_speaking = False
                print("Command processed, returning to wake word detection")

    def process_command(self, command_text):
        """Determine intent and execute or delegate to OpenAI"""
        try:
            # Parse intent
            cmd_type, cmd_args, raw_text = self.command_handler.parse_command(command_text)

            # If a built-in command, execute it directly
            if cmd_type != 'general_query':
                result = self.command_handler.execute_command(cmd_type, cmd_args, raw_text)
                print(f"Command executed: {cmd_type}, Result: {result}")
                return result

            # Otherwise, build OpenAI messages
            system_prompt = (
                "You are a helpful home assistant. "
                "Keep responses brief and conversational."
            )
            if self.last_recognized_user:
                system_prompt += f" The user's name is {self.last_recognized_user}."

            messages = [{"role": "system", "content": system_prompt}]
            # Append recent history
            for msg in self.conversation_history:
                messages.append(msg)
            # Append current user query
            messages.append({"role": "user", "content": raw_text})

            # Call the OpenAI ChatCompletion API
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            ai_response = completion.choices[0].message.content
            print(f"Assistant response: {ai_response}")

            # Post-process any described actions
            self._execute_ai_actions(ai_response, raw_text)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": raw_text})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            # Trim history
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length*2:]

            return ai_response

        except Exception as e:
            print(f"Error processing command: {e}")
            return "Sorry, I had trouble processing your request."

    def _execute_ai_actions(self, response_text, original_text):
        """
        If the AI response mentions a system action (lights, reminders, timers),
        execute it via the CommandHandler.
        """
        resp = response_text.lower()
        try:
            if 'lights on' in resp or 'turned the lights on' in resp:
                self.command_handler.execute_command('lights', ('on',), original_text)
            elif 'lights off' in resp or 'turned the lights off' in resp:
                self.command_handler.execute_command('lights', ('off',), original_text)
            elif any(kw in resp for kw in ["i'll remind", "i will remind", "remind you"]):
                self.command_handler.execute_command('reminder', (original_text,), original_text)
            elif 'timer' in resp and 'set' in resp:
                self.command_handler.execute_command('timer', (original_text,), original_text)
        except Exception as e:
            print(f"AI action execution error: {e}")

    def start(self):
        """Start the assistant's audio and face pipelines"""
        if self.running:
            print("Assistant is already running")
            return
        self.running = True
        print("Starting audio processing...")
        self.audio_processor.start_audio_processing()
        time.sleep(0.5)
        print("Starting face recognition...")
        self.face_recognizer.run_recognition_loop()
        print("Assistant started")

    def stop(self):
        """Stop audio and face streams"""
        if not self.running:
            print("Assistant is not running")
            return
        self.running = False
        self.audio_processor.stop_audio_processing()
        print("Assistant stopped")

if __name__ == "__main__":
    # Example config; adjust as needed
    config = {
        'VIDEO_STREAM_PORT': 1234,
        'VIDEO_WIDTH': 640,
        'VIDEO_HEIGHT': 480,
        'AUDIO_STREAM_PORT': 1235,
        'AUDIO_SAMPLE_RATE': 16000,
        'AUDIO_CHANNELS': 1,
        'ENCODINGS_FILE': "encodings.pickle",
        'VOICE_LINES_DIR': "voice_lines",
        'RECOGNITION_TIMEOUT': 300,
        'WAKE_WORD': 'hey assistant',
        'TTS_VOICE': 'nova',
        'SILENCE_THRESHOLD': 500,
        'MIN_SPEECH_DURATION': 0.5,
        'PROMPT_SILENCE': 0.5,
        'MODE': 'command',
        'MAX_HISTORY_LENGTH': 10
    }
    assistant = AssistantController(config)
    try:
        assistant.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()

