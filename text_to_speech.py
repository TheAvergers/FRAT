import os
import tempfile
import pygame
import openai
import subprocess
import time
import requests

class TextToSpeech:
    def __init__(self, config):
        """Initialize text-to-speech system"""
        self.config = config
        # Using OpenAI v0.28.0 syntax
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        pygame.mixer.init()
        
    def speak(self, text):
        """Convert text to speech and play it"""
        if not text.strip():
            print("Empty text, nothing to speak")
            return
            
        print(f"Converting to speech: {text}")
        
        # Create temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # Generate speech using OpenAI API with v0.28.0 syntax
            # Direct API call for v0.28.0
            headers = {
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "tts-1",
                "voice": self.config.get('TTS_VOICE', 'alloy'),
                "input": text
            }
            
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                # Save speech to file
                with open(temp_filename, 'wb') as f:
                    f.write(response.content)
                
                # Play the audio
                print("Playing TTS audio")
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            else:
                print(f"Error generating speech: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


if __name__ == "__main__":
    # For standalone testing
    config = {
        'TTS_VOICE': 'nova',  # Options: alloy, echo, fable, onyx, nova, shimmer
    }
    
    # Make sure OPENAI_API_KEY is set in environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)
        
    tts = TextToSpeech(config)
    tts.speak("Hello! This is a test of the text to speech system.")