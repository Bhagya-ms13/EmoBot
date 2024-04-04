import tkinter as tk
import pyaudio
import wave
from faster_whisper import WhisperModel


def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def start_recording():
    record_audio()
    status_label.config(text="Recording saved as output.wav")

# Create the GUI
root = tk.Tk()
root.title("Voice Recorder")

record_button = tk.Button(root, text="Record", width=20, command=start_recording)
record_button.pack(pady=20)

status_label = tk.Label(root, text="", width=40)
status_label.pack()

root.mainloop()
def transcribe_audio():

    model_size = "large-v3"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("output.wav", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
       print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    var = segment.text
    return var

transcribe_audio()