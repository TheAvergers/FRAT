# assistant_controller.py
import os
import time
import threading
import openai
from video_processor import FaceRecognizer
from command_handler import CommandHandler
from audio_processor import AudioProcessor
from text_to_speech import TextToSpeech
from network_utils import purge_port

class AssistantController:
    def __init__(self, config):
        """Initialize all assistant components"""
        self.config = config
        self.mode = config.get('MODE', 'command')

        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Close existing streams
        purge_port(config['VIDEO_STREAM_PORT'])
        purge_port(config['AUDIO_STREAM_PORT'])

        # Initialize TTS
        self.tts = TextToSpeech(config)
        time.sleep(0.5)

        # Initialize face and audio
        self.face_recognizer = FaceRecognizer(config)
        self.audio_processor = AudioProcessor(config)

        # Command handler
        self.command_handler = CommandHandler(tts_engine=self.tts, mode=self.mode)

        openai.api_key = os.environ.get('OPENAI_API_KEY')

        self.last_recognized_user = None
        self.tts_lock = threading.Lock()
        self.is_speaking = False

        self.conversation_history = []
        self.max_history_length = config.get('MAX_HISTORY_LENGTH', 10)

        # To track how each command was handled
        self.last_command_type = None

        # Callbacks
        self.audio_processor.set_activation_callback(self.handle_wake_word)
        self.setup_face_recognition_callback()

        self.running = False

    def setup_face_recognition_callback(self):
        def on_face_recognized(name):
            print(f"Face recognition update: {name}")
            self.last_recognized_user = name
            self.audio_processor.set_last_recognized_user(name)
        self.face_recognizer.set_recognition_callback(on_face_recognized)

    def handle_wake_word(self, message, is_greeting):
        """Called on wake - greeting or process a single-step command"""
        with self.tts_lock:
            if is_greeting:
                print(f"Playing greeting: {message}")
                self.is_speaking = True
                self.tts.speak(message)
                self.is_speaking = False
            else:
                # message is the full transcribed utterance minus wake-word
                print(f"Received command text: {message}")
                response, cmd_type = self.process_command(message)
                print(f"Handled via: {cmd_type}")
                self.is_speaking = True
                self.tts.speak(response)
                self.is_speaking = False
                print("Command processed, returning to wake word detection")

    def process_command(self, raw_text):
        """
        Parse and execute or delegate to OpenAI.
        Returns (response_text, command_type)
        """
        try:

            # Store the full raw text for scheduler use
            self.last_raw_text = raw_text

            cleaned_text = self.command_handler._convert_to_command_format(raw_text)
            cmd_type, cmd_args, stripped = self.command_handler.parse_command(cleaned_text)

            # track for external inspection
            self.last_command_type = cmd_type

            # Built-in commands take priority
            if cmd_type != 'general_query':
                result = self.command_handler.execute_command(cmd_type, cmd_args, stripped)
                print(f"Command executed: {cmd_type}, Result: {result}")
                return result, cmd_type

            # Prepare system prompt
            system_prompt = (
                "You are a helpful home assistant. Keep responses brief and conversational."
            )
            if self.last_recognized_user:
                system_prompt += f" The user's name is {self.last_recognized_user}."

            # Build message list
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": stripped})

            # Query OpenAI
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            ai_resp = completion.choices[0].message.content
            print(f"Assistant response: {ai_resp}")

            # Execute any declared actions
            self._execute_ai_actions(ai_resp, stripped)

            # Update history
            self.conversation_history.append({"role": "user", "content": stripped})
            self.conversation_history.append({"role": "assistant", "content": ai_resp})
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length*2:]

            return ai_resp, cmd_type

        except Exception as e:
            print(f"Error processing command: {e}")
            return "Sorry, I had trouble processing your request.", 'error'

    def _execute_ai_actions(self, response_text, original_text):
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
        if not self.running:
            print("Assistant is not running")
            return
        self.running = False
        self.audio_processor.stop_audio_processing()
        print("Assistant stopped")

if __name__ == "__main__":
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