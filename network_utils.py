import subprocess
import numpy as np
import os
import time
import sys

def purge_port(port):
    """Kill any process currently using the specified UDP port"""
    print(f"Purging processes on UDP port {port}...")
    if os.name == 'nt':  # Windows
        try:
            output = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True
            ).decode(errors='ignore')
            for line in output.splitlines():
                parts = line.split()
                if parts:
                    pid = parts[-1]
                    subprocess.call(f'taskkill /PID {pid} /F', shell=True)
                    print(f"Killed process {pid} on port {port}")
        except subprocess.CalledProcessError:
            pass
    else:  # Unix/Linux/Mac
        try:
            output = subprocess.check_output(
                f'lsof -i udp:{port} -t', shell=True
            ).decode(errors='ignore')
            for pid in output.split():
                subprocess.call(['kill', '-9', pid])
                print(f"Killed PID {pid} on UDP port {port}")
        except subprocess.CalledProcessError:
            pass

def stream_via_ffmpeg(port, width, height):
    """
    Spawn system FFmpeg to read UDP video stream and pipe raw frames to Python.
    Returns subprocess and reader generator.
    """
    url = f"udp://0.0.0.0:{port}?fifo_size=10000000&overrun_nonfatal=1"
    cmd = [
        'ffmpeg',
        '-i', url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    print(f"Starting FFmpeg video subprocess: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Make sure ffmpeg is running
    time.sleep(0.5)
    if p.poll() is not None:
        print(f"Error: FFmpeg process exited prematurely with code {p.returncode}")
        sys.exit(1)

    def reader():
        frame_size = width * height * 3
        while True:
            raw = p.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3)).copy()
            yield frame
    
    return p, reader()

def stream_via_ffmpeg_audio(port, sample_rate=16000, channels=1):
    """
    Spawn system FFmpeg to read UDP audio stream and pipe raw audio samples to Python.
    Returns subprocess and reader generator.
    """
    url = f"udp://0.0.0.0:{port}?fifo_size=10000000&overrun_nonfatal=1"
    cmd = [
        'ffmpeg',
        '-i', url,
        '-f', 's16le',  # 16-bit signed little-endian PCM
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', str(channels),
        '-'
    ]
    print(f"Starting FFmpeg audio subprocess: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    # Make sure ffmpeg is running
    time.sleep(0.5)
    if p.poll() is not None:
        print(f"Error: FFmpeg audio process exited prematurely with code {p.returncode}")
        sys.exit(1)

    def reader():
        # Buffer size for ~100ms of audio
        buffer_size = int(sample_rate * channels * 0.1) * 2  # 2 bytes per sample for s16le
        while True:
            raw = p.stdout.read(buffer_size)
            if len(raw) < 2:  # Need at least one sample
                break
            audio_chunk = np.frombuffer(raw, dtype=np.int16)
            yield audio_chunk
    
    return p, reader()

def test_udp_receiver(port, is_video=True):
    """Test UDP receiver for video or audio stream"""
    if is_video:
        width, height = 640, 480
        purge_port(port)
        print(f"Testing video stream on port {port}")
        p, reader = stream_via_ffmpeg(port, width, height)
        
        try:
            import cv2
            for i, frame in enumerate(reader):
                cv2.imshow('UDP Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if i > 100:  # Just show first 100 frames for test
                    break
        except Exception as e:
            print(f"Error in video test: {e}")
        finally:
            if p:
                p.kill()
            cv2.destroyAllWindows()
            
    else:
        sample_rate = 16000
        channels = 1
        purge_port(port)
        print(f"Testing audio stream on port {port}")
        p, reader = stream_via_ffmpeg_audio(port, sample_rate, channels)
        
        try:
            # Just print some stats about the audio
            for i, chunk in enumerate(reader):
                print(f"Chunk {i}: shape={chunk.shape}, min={chunk.min()}, max={chunk.max()}")
                if i > 10:  # Just show first 10 chunks for test
                    break
        except Exception as e:
            print(f"Error in audio test: {e}")
        finally:
            if p:
                p.kill()

if __name__ == "__main__":
    # For testing
    import argparse
    parser = argparse.ArgumentParser(description='Test UDP streaming')
    parser.add_argument('--port', type=int, default=1234, help='UDP port to receive on')
    parser.add_argument('--audio', action='store_true', help='Test audio instead of video')
    args = parser.parse_args()
    
    test_udp_receiver(args.port, not args.audio)
