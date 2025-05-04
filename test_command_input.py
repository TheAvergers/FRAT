import os
from assistant_controller import AssistantController

# Simple config - adjust paths/ports as needed
config = {
    'VIDEO_STREAM_PORT': 5000,  # dummy value for testing
    'AUDIO_STREAM_PORT': 5001,  # dummy value for testing
    'ENCODINGS_FILE': 'encodings.pickle',
    'VIDEO_WIDTH': 640,
    'VIDEO_HEIGHT': 480,
    'MODE': 'command',
    'TTS_VOICE': 'nova',
    'MAX_HISTORY_LENGTH': 5,
    'WAKE_WORD': 'assistant',
    'AUDIO_SAMPLE_RATE': 16000,  # Added for completeness
    'AUDIO_CHANNELS': 1          # Standard mono channel
}

if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set in environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)

    # Initialize assistant controller
    assistant = AssistantController(config)

    print("Manual Command Test: Type your input and see the response.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        try:
            # 1️⃣ Show what we got from the user
            print("Calling _convert_to_command_format directly for debug...")
            rag_converted = assistant.command_handler._convert_to_command_format(user_input)

            # 2️⃣ Process the command
            response, cmd_type = assistant.process_command(rag_converted)

            print(f"[Assistant Reply] ({cmd_type}): {response}\n")
            print("--- DEBUG INFO ---")
            print(f"Original user input: {user_input}")
            print(f"RAG-converted command: {rag_converted}")
            print(f"Command type: {cmd_type}")
            print(f"Response: {response}")

            # Show the last raw transcribed text if available (for scheduler debugging)
            last_raw = getattr(assistant.command_handler, 'last_raw_transcribed_text', None)
            if last_raw:
                print(f"Last raw transcribed text (for schedule tasks): {last_raw}")

            # For schedule commands, remind tester about delayed execution
            if cmd_type == 'schedule':
                print("Note: This was a schedule command. Watch logs for delayed task execution.\n")
            else:
                print()


        except Exception as e:
            print(f"Exception occurred during command processing: {e}\n")

