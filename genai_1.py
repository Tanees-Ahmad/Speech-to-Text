import streamlit as st
import whisper
from pydub import AudioSegment
from io import BytesIO
import time
import tempfile

# Set page config as the first command
st.set_page_config(page_title="Whisper AI Song-to-Lyrics Transcriber")

# Load Whisper model with error handling and caching
@st.cache_resource
def load_model():
    try:
        return whisper.load_model("tiny")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}. Retrying...")
        try:
            return whisper.load_model("tiny")
        except Exception as e:
            st.error(f"Failed to load Whisper model after retry: {e}")
            st.stop()

model = load_model()

# Function to transcribe a single segment
def transcribe_segment(segment_buffer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Write the buffer into a temporary file
        temp_file.write(segment_buffer.read())
        temp_file_path = temp_file.name
        
    result = model.transcribe(temp_file_path)
    return result['text']

# Main function to transcribe audio
def transcribe_audio(audio_file):
    start_time = time.time()
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file)
    
    # Split the audio into 10-second segments (or 15 as per your preference)
    segment_length = 10 * 1000  # 10 seconds in milliseconds
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    
    # Transcribe segments
    transcriptions = []
    for idx, segment in enumerate(segments):
        # Export segment to an in-memory buffer (BytesIO)
        segment_buffer = BytesIO()
        segment.export(segment_buffer, format="wav")
        segment_buffer.seek(0)  # Rewind the buffer for reading
        
        # Transcribe the segment
        transcription = transcribe_segment(segment_buffer)
        transcriptions.append(transcription)
    
    return " ".join(transcriptions).strip()

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
