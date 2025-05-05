# Face Regonition Assistant Test

Welcome to your Smart AI Home Assistant! This guide walks you through setting up the Raspberry Pi to stream video and audio, configuring your Windows assistant environment, and getting everything running.

---

## Setup

### 1. Finding Your IP Address

You’ll need your Windows PC’s IP to point the Pi streams, and your Pi’s IP to SSH in or check logs.

#### On Raspberry Pi (Raspberry Pi OS)
```bash
hostname -I
# or, ifconfig after installing net-tools:
sudo apt-get update && sudo apt-get install net-tools
ifconfig
```


#### On Windows
Open Command Prompt and run:
```cmd
ipconfig
```
Look under the adapter you're using:
- **Wireless LAN adapter Wi-Fi:** shows your Wi-Fi IP 
- **Ethernet adapter Ethernet:** shows your wired IP 

> Use the appropriate IP in the Pi streaming commands below.

---

### 2. Raspberry Pi: Video Streaming

Install ffmpeg if needed:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Over Wi-Fi** (replace `<WINDOWS_IP>` with your Wi-Fi IP):
```bash
ffmpeg -f v4l2 -input_format yuyv422   -video_size 640x480 -framerate 15   -i /dev/video0   -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -bf 0   -pix_fmt yuv420p -f mpegts udp://<WINDOWS_IP>:1234
```

**Over Ethernet** (replace `<WINDOWS_IP>` with your Ethernet IP):
```bash
ffmpeg -f v4l2 -input_format yuyv422   -video_size 640x480 -framerate 15   -i /dev/video0   -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -bf 0   -pix_fmt yuv420p -f mpegts udp://<WINDOWS_IP>:1234
```

---

### 3. Raspberry Pi: Audio Streaming

**Over Wi-Fi**:
```bash
ffmpeg -f alsa -i dsnoop:1,0   -c:a aac -b:a 128k   -f mpegts udp://<WINDOWS_IP>:1235
```

**Over Ethernet**:
```bash
ffmpeg -f alsa -i dsnoop:1,0   -c:a aac -b:a 128k   -f mpegts udp://<WINDOWS_IP>:1235
```

---

### 4. Auto-Start Streaming on Boot (systemd)

1. **Create startup script** `/usr/local/bin/start-streams.sh`:
   ```bash
   #!/bin/bash
   ffmpeg -f v4l2 -input_format yuyv422 -video_size 640x480 -framerate 15      -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -bf 0      -pix_fmt yuv420p -f mpegts udp://<WINDOWS_IP>:1234 &
   ffmpeg -f alsa -i dsnoop:1,0 -c:a aac -b:a 128k      -f mpegts udp://<WINDOWS_IP>:1235 &
   wait
   ```
   ```bash
   sudo chmod +x /usr/local/bin/start-streams.sh
   ```

2. **Create systemd service** `/etc/systemd/system/streaming.service`:
   ```ini
   [Unit]
   Description=Pi video + audio stream
   After=network.target

   [Service]
   Type=simple
   ExecStart=/usr/local/bin/start-streams.sh
   Restart=always
   User=pi
   Environment=PATH=/usr/bin:/usr/local/bin

   [Install]
   WantedBy=multi-user.target
   ```
3. **Enable and start**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streaming.service
   sudo systemctl start streaming.service
   sudo journalctl -u streaming.service -f
   ```

---

### 5. Windows Assistant Environment

1. **Install Anaconda/Miniconda** and create an environment:
   ```powershell
   conda create -n assistant python=3.9
   conda activate assistant
   ```

2. **Install dependencies** (ensure ffmpeg is in your PATH):
   ```bash
   pip install -r requirements.txt
   ```

---

### 6. Configure OpenAI API Key

Set your API key in the same shell before running:

- **Command Prompt**:
  ```cmd
  set OPENAI_API_KEY=your_api_key_here
  ```
- **PowerShell**:
  ```powershell
  $Env:OPENAI_API_KEY = "your_api_key_here"
  ```

---

### 7. Wake Words

The assistant listens for both:
- **“hey assistant”**
- **“alex”**

After either, it plays a “bing” sound and captures your command.

---

### 8. Testing Streams & CLI

Before full voice setup, test the command logic:

```bash
python test_command_input.py
```

Type commands like:
```
add reminder take out the trash at 8pm
what is the time
play jazz music
```

Ensure you see appropriate responses and debug info.

---

### 9. Preparing Face Encodings

You must supply your own `encodings.pickle`. Example:

```python
import face_recognition, pickle

known_encodings = []
known_names = []

# For each person:
img = face_recognition.load_image_file("alice.jpg")
enc = face_recognition.face_encodings(img)[0]
known_encodings.append(enc)
known_names.append("Alice")

data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)
```

Place `encodings.pickle` in the assistant’s working directory.

---

### 10. Launching the Assistant

```bash
python assistant_controller.py
```

- Listens for face recognition on port **1234** and audio on **1235**.
- Opens an OpenCV window showing detected faces.
- Speak **“Hey Assistant”** or **“Alex”** + your command.
- Press **Ctrl+C** or close the video window to stop.
