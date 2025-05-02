import re
import datetime
import subprocess
import os
import json
import openai
import time
import threading
import requests
import urllib.parse
import webbrowser
import pyaudio
import wave
import math
import random
from datetime import timedelta

class CommandHandler:
    """Handles parsing and execution of user commands"""
    
    def __init__(self, tts_engine=None):
        """Initialize command handler with optional TTS engine"""
        self.tts_engine = tts_engine
        self.openai_client = openai
        
        # Define command patterns
        self.command_patterns = {
            'time': r'\b(what (?:time|hour) is it|tell me the time)\b',
            'date': r'\b(what (?:day|date) is (?:it|today)|tell me the date)\b',
            'weather': r'\b(what\'?s the weather(?: like)?|how\'?s the weather|weather forecast)\b',
            'reminder': r'\b(?:set|create) (?:a |an )?reminder(?: for| to)? (.+)',
            'timer': r'\b(?:set|create) (?:a |an )?timer for (.+)',
            'lights': r'\b(turn|switch) (on|off) (?:the )?lights\b',
            'music': r'\b(?:play|start) (?:some |the )?music\b',
            'stop_music': r'\b(?:stop|pause) (?:the )?music\b',
            'volume': r'\b(?:set|change|adjust) (?:the )?volume(?: to)? (.+)',
            'search': r'\b(?:search|look up|find|google) (.+)',
            'open': r'\b(?:open|launch|start) (.+)',
            'news': r'\b(?:what\'?s|latest|get|tell me) (?:the )?news\b',
            'joke': r'\b(?:tell me a |say a |give me a )?joke\b',
            'calendar': r'\b(?:what\'?s (?:on |in )?my |show |check )(?:calendar|schedule)\b',
            'weather_location': r'\b(?:what\'?s the weather(?: like)? in|how\'?s the weather in) (.+)',
        }
        
        # Set up active timers and reminders
        self.active_timers = {}
        self.active_reminders = {}
        self.timer_counter = 0
        
    def parse_command(self, text, user_context=None):
        """
        Parse the user's command and determine intent
        Returns a tuple of (command_type, command_args, raw_text)
        """
        text = text.lower().strip()
        
        # Check command patterns
        for cmd_type, pattern in self.command_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract any command arguments from the regex groups
                args = match.groups()[1:] if len(match.groups()) > 1 else match.groups()
                return (cmd_type, args, text)
        
        # If no pattern matches, return as a general query
        return ('general_query', None, text)
        
    def execute_command(self, command_type, command_args=None, raw_text="", user_context=None):
        """Execute a command based on its type and arguments"""
        print(f"Executing command: {command_type} with args: {command_args}")
        
        # Choose command execution based on type
        if command_type == 'time':
            return self._handle_time()
            
        elif command_type == 'date':
            return self._handle_date()
            
        elif command_type == 'weather':
            return self._handle_weather()
            
        elif command_type == 'weather_location':
            if command_args and command_args[0]:
                return self._handle_weather_location(command_args[0])
            return "I need to know which location to check the weather for."
            
        elif command_type == 'reminder':
            if command_args and command_args[0]:
                return self._handle_reminder(command_args[0])
            return "I need to know what to remind you about."
            
        elif command_type == 'timer':
            if command_args and command_args[0]:
                return self._handle_timer(command_args[0])
            return "Please specify how long to set the timer for."
            
        elif command_type == 'lights':
            if command_args and command_args[0] in ['on', 'off']:
                return self._handle_lights(command_args[0])
            return "I need to know whether to turn the lights on or off."
            
        elif command_type == 'music':
            return self._handle_music_play()
            
        elif command_type == 'stop_music':
            return self._handle_music_stop()
            
        elif command_type == 'volume':
            if command_args and command_args[0]:
                return self._handle_volume(command_args[0])
            return "Please specify what volume level you want."
            
        elif command_type == 'search':
            if command_args and command_args[0]:
                return self._handle_search(command_args[0])
            return "What would you like me to search for?"

        elif command_type == 'open':
            if command_args and command_args[0]:
                return self._handle_open(command_args[0])
            return "What would you like me to open?"
            
        elif command_type == 'news':
            return self._handle_news()
            
        elif command_type == 'joke':
            return self._handle_joke()
            
        elif command_type == 'calendar':
            return self._handle_calendar()
            
        else:  # general_query
            return self._handle_general_query(raw_text, user_context)
            
    def _handle_time(self):
        """Return the current time"""
        current_time = datetime.datetime.now().strftime('%I:%M %p')
        return f"The current time is {current_time}."
        
    def _handle_date(self):
        """Return today's date"""
        today = datetime.datetime.now().strftime('%A, %B %d, %Y')
        return f"Today is {today}."
        
    def _handle_weather(self):
        """Return a weather forecast placeholder"""
        # In a real implementation, this would call a weather API
        return "I'm sorry, I don't have access to weather data yet. This feature will be implemented soon."
        
    def _handle_weather_location(self, location):
        """Return weather for a specific location"""
        # In a real implementation, this would call a weather API with the location
        return f"I'm sorry, I don't have access to weather data for {location} yet. This feature will be implemented soon."
        
    def _handle_reminder(self, reminder_text):
        """Set a reminder with the specified text"""
        # Extract time if present in the reminder text
        time_match = re.search(r'(?:at|for) (\d{1,2}(?::\d{2})? ?(?:am|pm|a\.m\.|p\.m\.)?)', reminder_text, re.IGNORECASE)
        
        if time_match:
            time_str = time_match.group(1)
            reminder_content = reminder_text.replace(time_match.group(0), "").strip()
            # In a real implementation, parse the time and set a scheduled reminder
            reminder_id = len(self.active_reminders) + 1
            self.active_reminders[reminder_id] = {
                "text": reminder_content,
                "time": time_str
            }
            return f"I'll remind you about {reminder_content} at {time_str}."
        else:
            # For now, just acknowledge the reminder without a specific time
            reminder_id = len(self.active_reminders) + 1
            self.active_reminders[reminder_id] = {
                "text": reminder_text,
                "time": None
            }
            return f"I'll remind you to {reminder_text}."
        
    def _handle_timer(self, duration_text):
        """Set a timer for the specified duration"""
        # Try to parse the duration
        try:
            # Look for patterns like "5 minutes", "30 seconds", etc.
            minutes_match = re.search(r'(\d+)\s*(?:minute|min)s?', duration_text)
            seconds_match = re.search(r'(\d+)\s*(?:second|sec)s?', duration_text)
            hours_match = re.search(r'(\d+)\s*(?:hour|hr)s?', duration_text)
            
            total_seconds = 0
            
            if hours_match:
                total_seconds += int(hours_match.group(1)) * 3600
            if minutes_match:
                total_seconds += int(minutes_match.group(1)) * 60
            if seconds_match:
                total_seconds += int(seconds_match.group(1))
                
            # If no units matched, assume the whole thing is seconds
            if total_seconds == 0 and re.search(r'\d+', duration_text):
                total_seconds = int(re.search(r'(\d+)', duration_text).group(1))
                
            if total_seconds > 0:
                # Create a timer
                timer_id = self.timer_counter + 1
                self.timer_counter = timer_id
                
                # Format time for display
                time_str = ""
                if total_seconds >= 3600:
                    hours = total_seconds // 3600
                    time_str += f"{hours} hour{'s' if hours > 1 else ''} "
                    total_seconds %= 3600
                    
                if total_seconds >= 60:
                    minutes = total_seconds // 60
                    time_str += f"{minutes} minute{'s' if minutes > 1 else ''} "
                    total_seconds %= 60
                    
                if total_seconds > 0:
                    time_str += f"{total_seconds} second{'s' if total_seconds > 1 else ''}"
                    
                # Start timer in separate thread
                thread = threading.Thread(
                    target=self._run_timer,
                    args=(timer_id, time_str),
                    daemon=True
                )
                thread.start()
                
                return f"Timer set for {time_str}."
            else:
                return "I couldn't understand that time format. Please try again."
                
        except Exception as e:
            print(f"Error setting timer: {e}")
            return "I had trouble setting that timer. Please try again."
    
    def _run_timer(self, timer_id, time_str):
        """Run a timer in a separate thread"""
        # In a real implementation, this would actually wait the specified time
        print(f"Timer {timer_id} for {time_str} would run here")
        # When timer completes, would notify the user
        
    def _handle_lights(self, state):
        """Handle turning lights on or off"""
        # In a real implementation, this would control smart home devices
        return f"I've turned the lights {state}."
        
    def _handle_music_play(self):
        """Handle playing music"""
        # In a real implementation, this would control a music system
        return "Playing music now."
        
    def _handle_music_stop(self):
        """Handle stopping music"""
        # In a real implementation, this would control a music system
        return "I've stopped the music."
        
    def _handle_volume(self, level_text):
        """Handle volume adjustment"""
        # Parse the volume level
        if "up" in level_text:
            # Increase volume
            return "I've turned the volume up."
        elif "down" in level_text:
            # Decrease volume
            return "I've turned the volume down."
        elif "max" in level_text or "full" in level_text:
            # Maximum volume
            return "Volume set to maximum."
        elif "min" in level_text or "low" in level_text:
            # Minimum volume
            return "Volume set to minimum."
        else:
            # Try to parse a percentage
            percent_match = re.search(r'(\d+)(?:\s*%|\s*percent)?', level_text)
            if percent_match:
                percent = int(percent_match.group(1))
                if 0 <= percent <= 100:
                    return f"Volume set to {percent}%."
                    
            return "I couldn't understand that volume level."
            
    def _handle_search(self, query):
        """Handle web search queries"""
        # In a real implementation, this would use a search API or open a browser
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.google.com/search?q={encoded_query}"
        
        try:
            # Try to open in browser
            webbrowser.open(search_url)
            return f"I've searched for {query} and opened the results."
        except:
            return f"I tried to search for {query}, but couldn't open the browser."
            
    def _handle_open(self, app_name):
        """Handle opening applications"""
        # Normalize common application names
        app_name = app_name.lower()
        
        # Map of common application names to commands
        app_map = {
            "chrome": "google-chrome" if os.name != "nt" else "start chrome",
            "firefox": "firefox" if os.name != "nt" else "start firefox",
            "browser": "xdg-open https://google.com" if os.name != "nt" else "start https://google.com",
            "file explorer": "nautilus" if os.name != "nt" else "explorer",
            "files": "nautilus" if os.name != "nt" else "explorer",
            "word": "libreoffice --writer" if os.name != "nt" else "start winword",
            "excel": "libreoffice --calc" if os.name != "nt" else "start excel",
            "notepad": "gedit" if os.name != "nt" else "notepad",
            "calculator": "gnome-calculator" if os.name != "nt" else "calc",
            "spotify": "spotify" if os.name != "nt" else "start spotify",
            "terminal": "gnome-terminal" if os.name != "nt" else "start cmd",
            "command prompt": "gnome-terminal" if os.name != "nt" else "start cmd",
            "settings": "gnome-control-center" if os.name != "nt" else "start ms-settings:",
        }
        
        # Try to find the application in our map
        command = None
        for key, value in app_map.items():
            if key in app_name:
                command = value
                break
                
        if command:
            try:
                # Execute in a separate process so it doesn't block
                subprocess.Popen(command, shell=True)
                return f"Opening {app_name}."
            except Exception as e:
                print(f"Error opening application: {e}")
                return f"I had trouble opening {app_name}."
        else:
            return f"I don't know how to open {app_name}."
    
    def _handle_news(self):
        """Handle news requests"""
        # In a real implementation, this would fetch news headlines
        return "Here are today's top headlines: 1. Global economy shows signs of recovery. 2. New breakthrough in renewable energy announced. 3. Local sports team wins championship."
        
    def _handle_joke(self):
        """Tell a joke"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What did the ocean say to the beach? Nothing, it just waved.",
            "Why don't skeletons fight each other? They don't have the guts.",
            "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Why did the scarecrow win an award? Because he was outstanding in his field.",
            "I'm reading a book on anti-gravity. It's impossible to put down!",
            "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them.",
            "How does a penguin build its house? Igloos it together!",
            "Why did the bicycle fall over? It was two-tired."
        ]
        return random.choice(jokes)
        
    def _handle_calendar(self):
        """Handle calendar requests"""
        # In a real implementation, this would fetch calendar events
        now = datetime.datetime.now()
        sample_events = [
            f"Meeting with team at {(now + timedelta(hours=1)).strftime('%I:%M %p')}",
            f"Dinner reservation at {(now + timedelta(hours=5)).strftime('%I:%M %p')}",
            f"Call with client tomorrow at 10:00 AM"
        ]
        return f"Here's what's on your calendar: {'. '.join(sample_events)}."
        
    def _handle_general_query(self, text, user_context=None):
        """Handle general queries by deferring to the OpenAI API"""
        # This should be handled by the parent class through OpenAI
        return None
