import time
from tracemalloc import start
import streamlit as st
import whisper
from pydub import AudioSegment
from io import BytesIO
import tempfile
import torch
import os
from pydub.exceptions import CouldntDecodeError

# Set page config as the first command
st.set_page_config(page_title="Whisper AI Song-to-Lyrics Transcriber")

# Load Whisper model with error handling and GPU support
@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("tiny").to(device)
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        st.stop()  # Stop the app if the model fails to load

model = load_model()

print(f"Model running on: {model.device}")


# Function to transcribe a single segment
def transcribe_segment(segment_buffer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Write the buffer into a temporary file
        temp_file.write(segment_buffer.read())
        temp_file_path = temp_file.name
        
    result = model.transcribe(temp_file_path)
    end = time.time()
    print(f"Segment transcribed in {end - start:.2f} seconds.")
    return result['text']


# Function to check if the audio is empty
def is_audio_empty(audio_file):
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_file)
        
        # Check if the audio has any content
        if audio.duration_seconds == 0:
            return True
        return False
    except CouldntDecodeError:
        st.error("Error: The audio file format is not supported or cannot be decoded. Please upload a valid audio file.")
        return True


# Main function to transcribe audio
def transcribe_audio(audio_file):
    start_time = time.time()
    
    # Check if the audio file is empty
    if is_audio_empty(audio_file):
        st.error("Error: The uploaded audio file is empty or invalid.")
        return ""

    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file)
    
    # Convert the entire audio file to an in-memory buffer
    audio_buffer = BytesIO()
    audio.export(audio_buffer, format="wav")
    audio_buffer.seek(0)  # Rewind the buffer so we can read from the start
    
    # Transcribe the entire audio
    transcription = transcribe_segment(audio_buffer)
    
    return transcription


# Streamlit app
st.title("Whisper AI Song-to-Lyrics Transcriber")
st.write("Upload a song in any language to transcribe it into lyrics using Whisper AI. Supports multiple languages!")

# File uploader
audio_file = st.file_uploader("Upload a Song File", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file, format="audio/wav")

    # Transcribe audio
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_file)
    
    # Display transcription
    st.text_area("Transcribed Lyrics", transcription, height=200)
    
# Footer
st.markdown("<div style='text-align: center; margin-top: 20px;'>Powered by Whisper AI & Streamlit</div>", unsafe_allow_html=True)
