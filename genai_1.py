import streamlit as st
import whisper

# Load Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_file):
    # Transcribe the audio using Whisper
    result = model.transcribe(audio_file)
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
