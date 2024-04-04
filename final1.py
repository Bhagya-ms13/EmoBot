import tkinter as tk
import pyaudio
import wave
import subprocess
from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from faster_whisper import WhisperModel
from tkinter import scrolledtext

# Load environment variables
load_dotenv()

# Configure generative AI service with API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Please provide the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("EMOBOT-Your Personal emotional Assisstant")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to record audio
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 6
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

    return WAVE_OUTPUT_FILENAME

# Function to transcribe audio
def transcribe_audio():

    model_size = "large-v3"

    # Initialize WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe("output.wav", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    transcribed_texts = []

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcribed_texts.append(segment.text)

    return transcribed_texts

# Initialize Gemini Pro model for chatbot
model_chatbot = genai.GenerativeModel("gemini-pro") 
chat_chatbot = model_chatbot.start_chat(history=[])

# Function to ask question to chatbot and get response
def ask_question_to_chatbot(question):
    response = chat_chatbot.send_message(question, stream=True)
    chat_history = []

    for chunk in response:
        chat_history.append(chunk.text)

    return chat_history

# Button to record audio and ask question
record_button = st.button("Record")

if record_button:
    # Record audio
    audio_file = record_audio()
    # Transcribe audio
    transcribed_texts = transcribe_audio()
    # Pass the transcribed text to the chatbot function
    chat_history = ask_question_to_chatbot(transcribed_texts[0])  # Assuming you're using the first transcribed text
    st.subheader("The Response from Chatbot is")
    for text in chat_history:
        st.write(text)

# Display chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
