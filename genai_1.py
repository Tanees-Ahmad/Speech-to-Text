import streamlit as st
import whisper
import tempfile

# Load Whisper model with error handling
try:
    model = whisper.load_model("base")
except Exception as e:
    st.error(f"Error loading Whisper model: {e}")
    st.stop()

def transcribe_audio(audio_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name
    
    # Transcribe the audio using Whisper
    result = model.transcribe(temp_file_path)
    return result['text']

# Streamlit app
st.title("Whisper AI Speech-to-Text Transcriber")
st.write("Upload an audio file to transcribe it into text using Whisper AI. Supports multiple languages!")

# File uploader
audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file, format="audio/wav")

    # Transcribe audio
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_file)
    
    # Display transcription
    st.text_area("Transcribed Text", transcription, height=200)
    
# Footer
st.markdown("<div style='text-align: center; margin-top: 20px;'>Powered by Whisper AI & Streamlit</div>", unsafe_allow_html=True)
