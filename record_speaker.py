# record_speaker.py
import os
import sounddevice as sd
import soundfile as sf
from pathlib import Path

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def record_one(path, duration=3, sr=16000):
    print(f"Recording {duration}s -> {path}")
    try:
        rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        sf.write(path, rec, sr, subtype='PCM_16')
        print("Saved:", path)
    except Exception as e:
        print("Recording failed:", e)
        raise

if __name__ == "__main__":
    root = r"D:\voice_dataset"        # <-- change this to your dataset root
    speaker = "person1"                # <-- change to speaker folder name
    num_takes = 10                     # how many recordings to make
    duration_seconds = 3               # length of each take in seconds

    spk_dir = os.path.join(root, speaker)
    ensure_dir(spk_dir)

    print(f"Speaker folder: {spk_dir}")
    for i in range(1, num_takes + 1):
        fname = f"take_{i:02d}.wav"
        path = os.path.join(spk_dir, fname)
        input(f"\nPress Enter to start take {i}/{num_takes}. Speak for {duration_seconds} seconds...")
        record_one(path, duration=duration_seconds, sr=16000)
    print("\nDone recording.")
