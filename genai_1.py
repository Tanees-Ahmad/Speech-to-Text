import streamlit as st
import whisper
from pydub import AudioSegment
from io import BytesIO
import time
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model with device handling
@st.cache_resource
def load_model():
    try:
        return whisper.load_model("base", device=device)
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        st.stop()

model = load_model()

# Transcribe a single segment of audio
def transcribe_segment(segment_buffer):
    segment_buffer.seek(0)  # Ensure the buffer is at the start
    result = model.transcribe(segment_buffer)
    return result['text']

# Main function to transcribe audio
def transcribe_audio(audio_file):
    start_time = time.time()
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file)
    
    # Split the audio into 30-second segments
    segment_length = 30 * 1000  # 30 seconds in milliseconds
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    
    # Transcribe segments in memory
    transcriptions = []
    for idx, segment in enumerate(segments):
        # Export segment to in-memory buffer
        segment_buffer = BytesIO()
        segment.export(segment_buffer, format="wav")
        segment_buffer.seek(0)
        
        # Transcribe segment
        transcription = transcribe_segment(segment_buffer)
        transcriptions.append(transcription)
    
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
