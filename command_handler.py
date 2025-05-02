# command_handler.py
import re
import datetime
import subprocess
import os
import json
import threading
import requests
import urllib.parse
import webbrowser
import pyaudio
import wave
import random
from datetime import timedelta

# File to persist reminders across sessions
REMINDERS_FILE = os.environ.get('REMINDERS_FILE', 'reminders.json')

class CommandHandler:
    """Handles parsing and execution of user commands with persistent reminders"""

    def __init__(self, tts_engine=None, mode='command'):
        """Initialize command handler with optional TTS engine and mode"""
        self.tts_engine = tts_engine
        self.mode = mode  # 'command' or 'conversation'

        # Load or initialize reminders
        self.active_reminders = {}
        self.timer_counter = 0
        self._load_reminders()

    def _load_reminders(self):
        """Load reminders from JSON file"""
        if os.path.exists(REMINDERS_FILE):
            try:
                with open(REMINDERS_FILE, 'r') as f:
                    data = json.load(f)
                    self.active_reminders = {int(k): v for k, v in data.items()}
                    if self.active_reminders:
                        self.timer_counter = max(self.active_reminders.keys())
            except Exception as e:
                print(f"Failed to load reminders: {e}")
                self.active_reminders = {}
        else:
            self._save_reminders()

    def _save_reminders(self):
        """Save reminders to JSON file"""
        try:
            with open(REMINDERS_FILE, 'w') as f:
                json.dump({str(k): v for k, v in self.active_reminders.items()}, f, indent=2)
        except Exception as e:
            print(f"Failed to save reminders: {e}")

    def parse_command(self, text, user_context=None):
        """Classify intent via keywords, fallback to OpenAI"""
        txt = text.strip()
        low = txt.lower()

        # ask override
        if low.startswith('ask '):
            return 'general_query', None, txt[len('ask '):].strip()
        # conversation mode
        if self.mode == 'conversation':
            return 'general_query', None, txt

        # Keywords classification in priority order
        if any(k in low for k in ['remind', 'reminder']):
            return 'reminder', (txt,), txt
        if 'timer' in low or 'countdown' in low:
            return 'timer', (txt,), txt
        if 'light' in low:
            # detect on/off
            if ' on ' in low or low.startswith('on '): cmd_args = ('on',)
            elif ' off ' in low or low.startswith('off '): cmd_args = ('off',)
            else: cmd_args = (txt,)
            return 'lights', cmd_args, txt
        if low.startswith('play ') or ' music' in low or 'song' in low:
            return 'music', None, txt
        if 'stop music' in low or 'pause music' in low:
            return 'stop_music', None, txt
        if 'volume' in low or 'louder' in low or 'softer' in low:
            return 'volume', (txt,), txt
        if 'list reminders' in low or 'show reminders' in low:
            return 'list_reminders', None, txt
        if any(k in low for k in ['search ', 'look up', 'find ', 'google ']):
            return 'search', (txt,), txt
        if any(k in low for k in ['open ', 'launch ', 'start ']):
            return 'open', (txt,), txt
        if 'news' in low or 'headline' in low:
            return 'news', None, txt
        if 'joke' in low or 'funny' in low:
            return 'joke', None, txt
        if any(k in low for k in ['calendar', 'schedule', 'agenda']):
            return 'calendar', None, txt
        # weather location first
        m = re.search(r'weather in ([\w ]+)', low)
        if m:
            return 'weather_location', (m.group(1).strip(),), txt
        if 'weather' in low:
            return 'weather', None, txt
        if 'time' in low:
            return 'time', None, txt
        if 'date' in low or 'day' in low:
            return 'date', None, txt

        # fallback
        return 'general_query', None, txt

    def execute_command(self, cmd, args=None, raw_text="", user_context=None):
        """Execute handler based on intent"""
        if cmd == 'reminder': return self._handle_reminder(args[0])
        if cmd == 'timer': return self._handle_timer(args[0])
        if cmd == 'lights': return self._handle_lights(args[0] if args else '')
        if cmd == 'music': return self._handle_music_play()
        if cmd == 'stop_music': return self._handle_music_stop()
        if cmd == 'volume': return self._handle_volume(args[0])
        if cmd == 'list_reminders': return self._handle_list_reminders()
        if cmd == 'search': return self._handle_search(args[0])
        if cmd == 'open': return self._handle_open(args[0])
        if cmd == 'news': return self._handle_news()
        if cmd == 'joke': return self._handle_joke()
        if cmd == 'calendar': return self._handle_calendar()
        if cmd == 'weather_location': return self._handle_weather_location(args[0])
        if cmd == 'weather': return self._handle_weather()
        if cmd == 'time': return self._handle_time()
        if cmd == 'date': return self._handle_date()
        if cmd == 'general_query': return None
        return "Sorry, I didn't understand that."

    # Built-in handlers (same as before)
    def _handle_time(self):
        now = datetime.datetime.now().strftime('%I:%M %p')
        return f"The current time is {now}."
    def _handle_date(self):
        today = datetime.datetime.now().strftime('%A, %B %d, %Y')
        return f"Today is {today}."
    def _handle_weather(self):
        return "Weather features coming soon."
    def _handle_weather_location(self, location):
        return f"Weather for {location} coming soon."
    def _handle_reminder(self, txt):
        # Parse time
        m = re.search(r'(?:at|for) (\d{1,2}(?::\d{2})? ?(?:am|pm))', txt, re.IGNORECASE)
        tm = m.group(1) if m else None
        content = txt if not tm else txt.replace(m.group(0), '').strip()
        rid = self.timer_counter + 1; self.timer_counter = rid
        self.active_reminders[rid] = {'text': content, 'time': tm}
        self._save_reminders()
        return f"Okay! I'll remind you to {content}" + (f" at {tm}." if tm else '.')
    def _handle_timer(self, duration): return f"Timer set for {duration}."
    def _handle_lights(self, state): return f"Lights turned {state}."
    def _handle_music_play(self): return "Playing music."
    def _handle_music_stop(self): return "Music stopped."
    def _handle_volume(self, lvl): return f"Volume set to {lvl}."
    def _handle_search(self, qry):
        webbrowser.open(f"https://google.com/search?q={urllib.parse.quote(qry)}")
        return f"Searched for {qry}."
    def _handle_open(self, app): return f"Opened {app}."
    def _handle_news(self): return "News fetching coming soon."
    def _handle_joke(self): return random.choice([
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised."
    ])
    def _handle_calendar(self):
        now = datetime.datetime.now(); return f"Meeting at {(now + timedelta(hours=1)).strftime('%I:%M %p')}"
    def _handle_list_reminders(self):
        if not self.active_reminders: return "You have no reminders."
        return "Your reminders:" + ''.join([f"\n[{rid}] {info['text']}" + (f" at {info['time']}" if info['time'] else '')
                                            for rid, info in sorted(self.active_reminders.items())])