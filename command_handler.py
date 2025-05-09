import re
import random
import threading
import json
import os
import time
import openai
import pygame
from datetime import datetime, timedelta


REMINDERS_FILE = os.environ.get('REMINDERS_FILE', 'reminders.json')
COMMANDS_REFERENCE_FILE = os.environ.get('COMMANDS_REFERENCE_FILE', 'commands_reference.txt')
openai.api_key = os.environ.get('OPENAI_API_KEY')


class CommandHandler:
    def __init__(self, tts_engine=None, mode='command'):
        # for alarm sound
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        # for Music playback
        self.music_channel = pygame.mixer.Channel(1)
        pygame.mixer.music.set_volume(0.5)
        self.stop_playlist_flag = threading.Event()
        self.tts_engine = tts_engine
        self.mode = mode
        self.active_reminders = {}
        self.timer_counter = 0
        self._reload_scheduled_tasks()
        self._load_reminders()
        self.command_reference = self._load_command_reference()

    def _reload_scheduled_tasks(self):
        schedule_file = "scheduled_tasks.json"
        if not os.path.exists(schedule_file):
            return
        with open(schedule_file, "r") as f:
            tasks = json.load(f)
        now = datetime.now()
        for task in tasks:
            run_at = datetime.strptime(task['run_at'], "%Y-%m-%d %H:%M:%S")
            delay_seconds = (run_at - now).total_seconds()
            if delay_seconds > 0:
                print(f"Re-scheduling: {task['command']} in {delay_seconds:.0f} seconds")
                threading.Thread(
                    target=self._schedule_after_delay,
                    args=(delay_seconds, task['command']),
                    daemon=True
                ).start()
            else:
                print(f"Skipping past task: {task['command']}")

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

    def process_audio_command(self, transcribed_text, do_rag=False):
        #For scheduler use
        self.last_raw_transcribed_text = transcribed_text

        print(f"Received audio command: {transcribed_text}")
        cleaned_text = self._strip_wake_word(transcribed_text)

        if do_rag:
            print("Performing RAG step (manual/test mode).")
            cleaned_text = self._convert_to_command_format(cleaned_text)
        else:
            print(f"Processing cleaned command (no RAG here): {cleaned_text}")

        cmd_type, cmd_args, raw_text = self.parse_command(cleaned_text)

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
                    {"role": "system", "content": "Convert user requests into smart home commands. Ensure you remove any scheduling boilerplate like time expressions and only return the actionable command, exactly as shown in the reference."},
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
        
        # Strip scheduling boilerplate like time specs
        txt = re.sub(r'\bat \d{1,2}:\d{2}(?:\s*(am|pm)?)?\b', '', txt)
        txt = re.sub(r'\bin \d+\s*(seconds|second|minutes|minute|min)\b', '', txt)
        # Strip leading 'playing' or 'schedule' if it’s dangling
        txt = re.sub(r'^(playing|schedule)\s+', '', txt).strip()
        txt = txt.strip()

        print(f"Parser cleaned command for parsing: '{txt}'")


        if re.search(r'\bturn(ing)? on the lights\b', txt):
            return 'lights', ('on',), text
        if re.search(r'\bturn(ing)? off the lights\b', txt) or 'lights off' in txt:
            return 'lights', ('off',), text

        # Play music with optional genre or song name
        music_match = re.search(r'(?:play|start)\s+(?:some\s+)?(?P<target>.+?)?\s*(music|song|songs)?$', txt)
        if music_match:
            target = music_match.group('target')
            if target and target.strip() in ['music', 'song', 'songs']:
                target = None  # Avoid redundancy like "play music music"
            return 'music', (target.strip() if target else None,), text
        
        known_genres = ['jazz', 'rock', 'pop', 'classical', 'hiphop', 'blues', 'metal', 'country']
        if txt in known_genres:
            return 'music', (txt,), text

        
        # Set volume to a specific level
        volume_match = re.search(r'(set volume to|volume)\s*(\d+)\s*%?', txt)
        if volume_match:
            volume_level = int(volume_match.group(2))
            return 'set_volume', (volume_level,), text

        # Increase/decrease volume
        if re.search(r'(increase|turn up|raise).*volume', txt):
            return 'adjust_volume', ('up',), text
        if re.search(r'(decrease|turn down|lower).*volume', txt):
            return 'adjust_volume', ('down',), text

        # Play next song
        if 'next song' in txt or 'skip song' in txt:
            return 'next_song', None, text

        # Shuffle genre
        shuffle_match = re.search(r'shuffle\s+(?P<genre>\w+)', txt)
        if shuffle_match:
            genre = shuffle_match.group('genre')
            return 'shuffle_music', (genre,), text
        
        if re.search(r'\bstop music\b', txt):
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
            'set_volume': self._handle_set_volume,
            'adjust_volume': self._handle_adjust_volume,
            'shuffle_music': self._handle_shuffle_music,
            'next_song': self._handle_next_song,
            'schedule': self._handle_schedule,

        }
        handler = mapping.get(cmd)
        print(f"DEBUG: Executing command '{cmd}' with args: {args}")
        if handler:
            if cmd == 'schedule':
                result = handler(*(args or []), original_text=getattr(self, 'last_raw_transcribed_text', None))
            else:
                result = handler(*(args or []))
            return result

        return f"The command '{raw_text}' was not recognized as a valid command in this system. Please try again with a supported request."


    def _handle_reminder(self, reminder_text):
        cleaned_reminder = self._clean_reminder_text(reminder_text)
        self.timer_counter += 1
        self.active_reminders[self.timer_counter] = {
            'reminder': cleaned_reminder,
            'timestamp': datetime.now().isoformat()
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
        now = datetime.now().strftime('%H:%M')
        return f"The current time is {now}."

    def _handle_date(self):
        today = datetime.now().strftime('%A, %B %d, %Y')
        return f"Today's date is {today}."

    def _handle_joke(self):
        return "Why did the scarecrow win an award? Because he was outstanding in his field!"

    def _handle_music_play(self, genre=None):
        music_root = os.path.join(os.getcwd(), 'music')
        chosen_track = None

        if genre:
            genre_folder = os.path.join(music_root, genre.lower())
            if not os.path.exists(genre_folder):
                return f"Sorry, the genre '{genre}' was not found."
            
            tracks = [f for f in os.listdir(genre_folder)
                    if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
            if not tracks:
                return f"No tracks found in the '{genre}' genre."
            
            chosen_track = os.path.join(genre_folder, random.choice(tracks))
        else:
            tracks = [os.path.join(dp, f) for dp, dn, filenames in os.walk(music_root)
                    for f in filenames if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
            if not tracks:
                return "No music files found in the library."
            
            chosen_track = random.choice(tracks)

        try:
            print(f"Loading and playing: {chosen_track}")
            music_sound = pygame.mixer.Sound(chosen_track)
            self.music_channel.play(music_sound, loops=-1)
            return f"Playing {genre if genre else 'music'} now."
        except Exception as e:
            print(f"Error playing music: {e}")
            return "Sorry, I couldn't play the music."
        
    def _handle_next_song(self):
        if self.music_channel.get_busy():
            self.music_channel.stop()  # This triggers moving to the next song automatically
            return "Skipping to the next song."
        else:
            return "No song is currently playing to skip."
        
    def _handle_shuffle_music(self, genre):
        music_root = os.path.join(os.getcwd(), 'music')
        track_list = []

        if genre.lower() == 'all':
            # Shuffle across all music files
            for dp, dn, filenames in os.walk(music_root):
                for f in filenames:
                    if f.lower().endswith(('.mp3', '.wav', '.ogg')):
                        track_list.append(os.path.join(dp, f))
        else:
            # Shuffle within a specific genre folder
            genre_folder = os.path.join(music_root, genre.lower())
            if not os.path.exists(genre_folder):
                return f"Sorry, the genre '{genre}' was not found."

            track_list = [os.path.join(genre_folder, f)
                        for f in os.listdir(genre_folder)
                        if f.lower().endswith(('.mp3', '.wav', '.ogg'))]

        if not track_list:
            return f"No tracks found to shuffle in the '{genre}' genre."

        random.shuffle(track_list)

        print(f"Shuffling and playing tracks: {track_list}")


        def play_playlist(tracks):
            for track in tracks:
                if self.stop_playlist_flag.is_set():
                    print("Stop flag detected, ending playlist playback.")
                    break
                try:
                    print(f"Playing: {track}")
                    music_sound = pygame.mixer.Sound(track)
                    self.music_channel.play(music_sound)
                    while self.music_channel.get_busy():
                        if self.stop_playlist_flag.is_set():
                            print("Stop flag detected mid-track, stopping playback.")
                            self.music_channel.stop()
                            return
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Error playing {track}: {e}")
                    continue

        self.stop_playlist_flag.clear()
        threading.Thread(target=play_playlist, args=(track_list,), daemon=True).start()

        return f"Shuffling and playing {genre} music."


    def _handle_stop_music(self):
        self.stop_playlist_flag.set()
        if self.music_channel.get_busy():
            self.music_channel.stop()
            return "Music stopped."
        else:
            return "No music is currently playing."


    def _handle_set_volume(self, level):
        level = max(0, min(100, level))  # Clamp between 0 and 100
        self.music_channel.set_volume(level / 100.0)
        return f"Set the volume to {level}%."


    def _handle_adjust_volume(self, direction):
        current_volume = self.music_channel.get_volume() * 100
        delta = 10  # Adjust by 10%
        if direction == 'up':
            new_volume = min(100, current_volume + delta)
        else:
            new_volume = max(0, current_volume - delta)
        self.music_channel.set_volume(new_volume / 100.0)
        return f"{'Increased' if direction == 'up' else 'Decreased'} the volume to {int(new_volume)}%."




    
    def _handle_schedule(self, schedule_text, original_text=None):
        print(f"Scheduling task: {schedule_text}")

        # Try to extract delay (e.g., "in 30 seconds")
        delay_match = re.search(r'in (\d+)\s*(seconds|second|minutes|minute|min)', schedule_text)
        time_match = re.search(r'at (\d{1,2}:\d{2})(?:\s*(am|pm)?)?', schedule_text)

        if delay_match:
            value = int(delay_match.group(1))
            unit = delay_match.group(2)
            delay_seconds = value * 60 if 'min' in unit else value
            print(f"Task will run in {delay_seconds} seconds.")
            threading.Thread(target=self._schedule_after_delay,args=(delay_seconds, schedule_text, original_text),daemon=True).start()


            return f"Task scheduled to run in {delay_seconds} seconds."

        elif time_match:
            time_str = time_match.group(1)
            am_pm = time_match.group(2)
            now = datetime.now()


            # Parse the time (e.g., 3:30 or 15:30)
            hour, minute = map(int, time_str.split(':'))
            if am_pm:
                if am_pm.lower() == 'pm' and hour < 12:
                    hour += 12
                if am_pm.lower() == 'am' and hour == 12:
                    hour = 0

            scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if scheduled_time < now:
                scheduled_time += timedelta(days=1) # if time has passed, schedule for next day

            delay_seconds = (scheduled_time - now).total_seconds()
            print(f"Task will run at {scheduled_time} ({int(delay_seconds)} seconds from now).")
            threading.Thread(target=self._schedule_after_delay, args=(delay_seconds, schedule_text), daemon=True).start()

            # Save schedule to file
            schedule_entry = {"command": schedule_text, "run_at": (datetime.now() + timedelta(seconds=delay_seconds)).strftime("%Y-%m-%d %H:%M:%S")}
            self._save_scheduled_task(schedule_entry)

            return f"Task scheduled to run at {scheduled_time.strftime('%H:%M')}."

        else:
            return "Sorry, I couldn't understand the schedule timing."
        
    def _save_scheduled_task(self, task):
        schedule_file = "scheduled_tasks.json"
        try:
            if os.path.exists(schedule_file):
                with open(schedule_file, "r") as f:
                    data = json.load(f)
            else:
                data = []
            data.append(task)
            with open(schedule_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save scheduled task: {e}")

    def _schedule_after_delay(self, delay_seconds, schedule_text, original_text=None):
        print(f"Scheduled task sleeping for {delay_seconds} seconds...")
        time.sleep(delay_seconds)

        # Prefer using the original raw text for richer context if available
        if original_text:
            print(f"Original raw command: {original_text}")
            cleaned_raw = original_text  # 🛠 DEFINE THIS FIRST
            cleaned_raw = re.sub(r'\bin \d+\s*(seconds|second|minutes|minute|min)\b', '', cleaned_raw, flags=re.IGNORECASE)
            cleaned_raw = re.sub(r'\bat \d{1,2}:\d{2}(?:\s*(am|pm)?)?', '', cleaned_raw, flags=re.IGNORECASE)
            cleaned_raw = cleaned_raw.strip()
            print(f"Cleaned raw command for RAG: {cleaned_raw}")
        else:
            cleaned_raw = schedule_text.strip().lower()
            if cleaned_raw.startswith('schedule '):
                cleaned_raw = cleaned_raw[len('schedule '):].strip()

        print(f"Executing scheduled command (pre-RAG): {cleaned_raw}")

        # Pass through RAG to normalize the command
        rag_ready_command = self._convert_to_command_format(cleaned_raw)

        # Ensure 'schedule' is fully stripped from the RAG response too
        rag_ready_command = rag_ready_command.strip()
        if rag_ready_command.startswith('schedule '):
            rag_ready_command = rag_ready_command[len('schedule '):].strip()

        print(f"Scheduled command after RAG (post-strip): {rag_ready_command}")

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


