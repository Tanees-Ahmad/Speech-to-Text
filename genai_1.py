import streamlit as st
import tempfile
import whisper
import os
from pydub import AudioSegment
from pathlib import Path
from multiprocessing import Pool
import subprocess

# Function to delete the existing model file
def delete_model_file(model_name):
    model_dir = Path.home() / ".cache" / "whisper" / model_name
    if model_dir.exists():
        for file in model_dir.glob("*"):
            file.unlink()

# Function to check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        st.error(f"ffmpeg error: {e}")
        st.stop()
    except FileNotFoundError:
        st.error("ffmpeg not found. Please install ffmpeg and ensure it is in your system's PATH.")
        st.stop()

# Check if ffmpeg is installed
check_ffmpeg()

# Load Whisper model with error handling
model_name = "tiny"
try:
    model = whisper.load_model(model_name)
except Exception as e:
    st.error(f"Error loading Whisper model: {e}. Retrying...")
    delete_model_file(model_name)
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        st.error(f"Failed to load Whisper model after retry: {e}")
        st.stop()

def transcribe_segment(segment_path):
    result = model.transcribe(segment_path)
    os.remove(segment_path)
    return result['text']

def transcribe_audio(audio_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(temp_file_path)
    
    # Split the audio into 30-second segments
    segment_length = 30 * 1000  # 30 seconds in milliseconds
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    
    # Export segments to temporary files
    segment_paths = []
    for i, segment in enumerate(segments):
        segment_path = f"{temp_file_path}_{i}.wav"
        segment.export(segment_path, format="wav")
        segment_paths.append(segment_path)
    
    # Transcribe segments in parallel
    with Pool() as pool:
        transcriptions = pool.map(transcribe_segment, segment_paths)
    
    os.remove(temp_file_path)
    return " ".join(transcriptions).strip()

# Streamlit app
st.set_page_config(page_title="Whisper AI Song-to-Lyrics Transcriber")
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
