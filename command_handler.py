import re
import datetime
import threading
import json
import os
import time
import openai
import pygame
from datetime import timedelta

REMINDERS_FILE = os.environ.get('REMINDERS_FILE', 'reminders.json')
COMMANDS_REFERENCE_FILE = os.environ.get('COMMANDS_REFERENCE_FILE', 'commands_reference.txt')
openai.api_key = os.environ.get('OPENAI_API_KEY')

class CommandHandler:
    def __init__(self, tts_engine=None, mode='command'):
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        self.tts_engine = tts_engine
        self.mode = mode
        self.active_reminders = {}
        self.timer_counter = 0
        self._load_reminders()
        self.command_reference = self._load_command_reference()

    def _load_reminders(self):
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
        try:
            with open(REMINDERS_FILE, 'w') as f:
                json.dump({str(k): v for k, v in self.active_reminders.items()}, f, indent=2)
        except Exception as e:
            print(f"Failed to save reminders: {e}")

    def _load_command_reference(self):
        if os.path.exists(COMMANDS_REFERENCE_FILE):
            try:
                with open(COMMANDS_REFERENCE_FILE, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"Failed to load command reference: {e}")
        return ""

    def process_audio_command(self, transcribed_text):
        print(f"Received audio command: {transcribed_text}")
        cleaned_text = self._strip_wake_word(transcribed_text)
        rag_ready_command = self._convert_to_command_format(cleaned_text)
        print(f"RAG processed command: {rag_ready_command}")
        cmd_type, cmd_args, raw_text = self.parse_command(rag_ready_command)
        try:
            result = self.execute_command(cmd_type, cmd_args, raw_text)
            return result
        except Exception as e:
            print(f"Error processing command: {e}")
            return "Sorry, there was an internal error while processing your command."

    def _strip_wake_word(self, text):
        pattern = r'^(alex[:,]?|hey assistant[:,]?|assistant[:,]?)(\s*)'
        return re.sub(pattern, '', text.strip(), flags=re.IGNORECASE)

    def _convert_to_command_format(self, text):
        reference_text = (
            "You are a smart home assistant. Given a user instruction, rewrite it into a command-ready format. "
            "Use the following exact command reference as your guide:\n" + self.command_reference +
            "\nRespond ONLY with the clean command string, no extra commentary. Input: " + text
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Convert user requests into smart home commands."},
                    {"role": "user", "content": reference_text}
                ],
                max_tokens=100,
                temperature=0
            )
            command_text = response['choices'][0]['message']['content'].strip().lower()
            print(f"--- DEBUG RAG RESPONSE ---\nFull API response: {response}\nCleaned command: {command_text}\n--------------------------")
            # Remove potential wrapping like backticks
            command_text = command_text.strip('`').strip()
            return command_text
        except Exception as e:
            print(f"Error during RAG command formatting: {e}")
            return text

    def parse_command(self, text, user_context=None):
        txt = text.strip().lower()
        print(f"DEBUG: Parsing normalized command: '{txt}'")


        # Normalize whitespace
        txt = re.sub(r'\s+', ' ', txt)

        if txt.startswith('set timer ') or txt.startswith('start timer ') or 'timer' in txt:
            return 'timer', (txt,), text
        if txt.startswith('schedule '):
            return 'schedule', (txt,), text
        
        if txt in ('list reminders', 'show reminders', 'list all reminders'):
            return 'list_reminders', None, text

        if txt.startswith('add reminder ') or txt.startswith('set reminder '):
            reminder_text = txt.replace('add reminder ', '').replace('set reminder ', '').strip()
            return 'reminder', (reminder_text,), text

        if 'turn on the lights' in txt:
            return 'lights', ('on',), text
        if 'turn off the lights' in txt or 'lights off' in txt:
            return 'lights', ('off',), text

        if 'play music' in txt:
            return 'music', None, text
        if 'stop music' in txt:
            return 'stop_music', None, text

        if 'what is the time' in txt or txt == 'time':
            return 'time', None, text
        if 'what is the date' in txt or txt == 'date':
            return 'date', None, text


        return 'unknown', None, text




    def execute_command(self, cmd, args=None, raw_text="", user_context=None):
        mapping = {
            'reminder': self._handle_reminder,
            'timer': self._handle_timer,
            'list_reminders': self._handle_list_reminders,
            'lights': self._handle_lights,
            'time': self._handle_time,
            'date': self._handle_date,
            'joke': self._handle_joke,
            'music': self._handle_music_play,
            'stop_music': self._handle_stop_music,
            'volume': self._handle_volume,
            'schedule': self._handle_schedule,

        }
        handler = mapping.get(cmd)
        print(f"DEBUG: Executing command '{cmd}' with args: {args}")
        if handler:
            result = handler(*(args or []))
            return result
        return f"The command '{raw_text}' was not recognized as a valid command in this system. Please try again with a supported request."


    def _handle_reminder(self, reminder_text):
        cleaned_reminder = self._clean_reminder_text(reminder_text)
        self.timer_counter += 1
        self.active_reminders[self.timer_counter] = {
            'reminder': cleaned_reminder,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._save_reminders()
        return f"Reminder set: {cleaned_reminder}"

    def _clean_reminder_text(self, raw_text):
        boilerplate_patterns = [
            r"set (a )?reminder( for me)?( to)?",
            r"remind me to",
            r"create (a )?reminder( to)?"
        ]
        cleaned = raw_text.lower()
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, '', cleaned).strip()
        return cleaned if cleaned else raw_text

    def _handle_timer(self, timer_text):
        duration = self._extract_timer_duration(timer_text)
        if not duration:
            return "Sorry, I couldn't understand the timer duration."
        threading.Thread(target=self._start_timer, args=(duration,), daemon=True).start()
        return f"Timer started for {duration} seconds."

    def _extract_timer_duration(self, text):
        match = re.search(r'(\d+)\s*(seconds|second|minutes|minute|min)', text.lower())
        if not match:
            return None
        value = int(match.group(1))
        unit = match.group(2)
        return value * 60 if 'min' in unit else value

    def _start_timer(self, seconds):
        print(f"Timer running for {seconds} seconds...")
        time.sleep(seconds)

        # Play alarm sound before TTS
        alarm_path = os.path.join(os.getcwd(), 'alarm.wav')
        if os.path.exists(alarm_path):
            print(f"Playing alarm sound from {alarm_path}...")
            try:
                pygame.mixer.music.load(alarm_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error playing alarm sound: {e}")
        else:
            print(f"Alarm sound file not found: {alarm_path}")

        if self.tts_engine:
            self.tts_engine.speak("Time's up")

        print("Timer completed.")

    def _handle_list_reminders(self):
        if not self.active_reminders:
            return "Here are your reminders:\n(You currently have no reminders set.)"
        lines = [f"- {v['reminder']} (Set on {v['timestamp']})" for v in self.active_reminders.values()]
        return "Here are your reminders:\n" + "\n".join(lines)

    def _handle_lights(self, state):
        if state:
            return f"Turning lights {state}. (Placeholder)"
        return "Please specify whether to turn the lights on or off."

    def _handle_time(self):
        now = datetime.datetime.now().strftime('%H:%M')
        return f"The current time is {now}."

    def _handle_date(self):
        today = datetime.datetime.now().strftime('%A, %B %d, %Y')
        return f"Today's date is {today}."

    def _handle_joke(self):
        return "Why did the scarecrow win an award? Because he was outstanding in his field!"

    def _handle_music_play(self):
        return "Playing music. (Placeholder)"

    def _handle_stop_music(self):
        return "Stopping music. (Placeholder)"

    def _handle_volume(self, volume_text):
        return f"Adjusting volume based on your command: {volume_text}. (Placeholder)"
    
    def _handle_schedule(self, schedule_text):
        print(f"Scheduling task: {schedule_text}")

        # Try to extract delay (e.g., "in 30 seconds")
        delay_match = re.search(r'in (\d+)\s*(seconds|second|minutes|minute|min)', schedule_text)
        time_match = re.search(r'at (\d{1,2}:\d{2})(?:\s*(am|pm)?)?', schedule_text)

        if delay_match:
            value = int(delay_match.group(1))
            unit = delay_match.group(2)
            delay_seconds = value * 60 if 'min' in unit else value
            print(f"Task will run in {delay_seconds} seconds.")
            threading.Thread(target=self._schedule_after_delay, args=(delay_seconds, schedule_text), daemon=True).start()
            return f"Task scheduled to run in {delay_seconds} seconds."

        elif time_match:
            time_str = time_match.group(1)
            am_pm = time_match.group(2)
            now = datetime.datetime.now()

            # Parse the time (e.g., 3:30 or 15:30)
            hour, minute = map(int, time_str.split(':'))
            if am_pm:
                if am_pm.lower() == 'pm' and hour < 12:
                    hour += 12
                if am_pm.lower() == 'am' and hour == 12:
                    hour = 0

            scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if scheduled_time < now:
                scheduled_time += datetime.timedelta(days=1)  # if time has passed, schedule for next day

            delay_seconds = (scheduled_time - now).total_seconds()
            print(f"Task will run at {scheduled_time} ({int(delay_seconds)} seconds from now).")
            threading.Thread(target=self._schedule_after_delay, args=(delay_seconds, schedule_text), daemon=True).start()
            return f"Task scheduled to run at {scheduled_time.strftime('%H:%M')}."

        else:
            return "Sorry, I couldn't understand the schedule timing."
    
    def _schedule_after_delay(self, delay_seconds, schedule_text):
        print(f"Scheduled task sleeping for {delay_seconds} seconds...")
        time.sleep(delay_seconds)

        # Clean up the schedule text by removing the 'schedule' keyword
        cleaned_text = schedule_text.strip().lower()
        if cleaned_text.startswith('schedule '):
            cleaned_text = cleaned_text[len('schedule '):].strip()

        print(f"Executing scheduled command (pre-RAG): {cleaned_text}")

        # Pass through RAG to normalize the command
        rag_ready_command = self._convert_to_command_format(cleaned_text)
        print(f"Scheduled command after RAG: {rag_ready_command}")

        # Parse and execute the cleaned + RAG-processed command
        cmd_type, cmd_args, _ = self.parse_command(rag_ready_command)

        print(f"Scheduled task determined to be command type: {cmd_type}")

        if cmd_type and cmd_type != 'unknown':
            result = self.execute_command(cmd_type, cmd_args, rag_ready_command)
        else:
            result = "Scheduled task executed, but no matching command found."

        print(f"Scheduled task executed: {result}")
        if self.tts_engine:
            self.tts_engine.speak(result)


