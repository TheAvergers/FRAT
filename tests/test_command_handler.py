import os, json, tempfile
import pytest
from command_handler import CommandHandler, REMINDERS_FILE

class DummyTTS:
    def speak(self, text): pass

@pytest.fixture(autouse=True)
def isolate_reminders(tmp_path, monkeypatch):
    # Redirect reminders file to temp for isolation
    f = tmp_path / "reminders.json"
    monkeypatch.setenv("REMINDERS_FILE", str(f))
    yield

def test_time_and_date():
    h = CommandHandler(DummyTTS())
    assert "current time" in h.execute_command('time')
    assert "Today is" in h.execute_command('date')

def test_reminder_and_listing():
    h = CommandHandler(DummyTTS())
    # initially empty
    assert "no reminders" in h._handle_list_reminders().lower()
    # add reminder
    resp = h.execute_command('reminder', ("walk dog at 6pm",), "walk dog at 6pm")
    assert "walk dog" in resp.lower()
    # file should exist
    data = json.loads(open(REMINDERS_FILE).read())
    assert any("walk dog" in v['text'] for v in data.values())
    # listing now shows it
    listing = h.execute_command('list_reminders', None, "")
    assert "walk dog" in listing.lower()

@pytest.mark.parametrize("input_text,expected_type", [
    ("set a timer for 2 minutes", "timer"),
    ("turn off the lights", "lights"),
    ("play music", "music"),
])
def test_parse_command_variants(input_text, expected_type):
    h = CommandHandler(DummyTTS())
    cmd_type, args, raw = h.parse_command(input_text)
    assert cmd_type == expected_type
