"""
GEA-6 Multimodal AI Chatbot - Speech Recognition Utilities
Real speech-to-text functionality with multiple backends.

Note: Imports for heavy/optional dependencies are done lazily inside functions
to avoid crashing the whole app if a dependency is missing.
"""

import tempfile
import os
from typing import Any, Optional
import openai
from config import OPENAI_API_KEY


def transcribe_audio(audio_file: Any, language_code: Optional[str] = None) -> str:
    """Transcribe audio file using SpeechRecognition (Google backend).

    language_code: optional BCP-47 code like 'hi-IN', 'mr-IN', etc.
    """
    try:
        try:
            import speech_recognition as sr  # lazy import
        except ModuleNotFoundError:
            return (
                "Speech backend missing. Install it with: "
                "python -m pip install SpeechRecognition pydub"
            )

        recognizer = sr.Recognizer()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name

        # Transcribe using speech_recognition
        with sr.AudioFile(tmp_file_path) as source:
            audio_data = recognizer.record(source)

        if language_code:
            text = recognizer.recognize_google(audio_data, language=language_code)
        else:
            text = recognizer.recognize_google(audio_data)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return text

    except Exception as e:
        # Handle common SR exceptions without importing its symbols at top-level
        msg = str(e)
        if "Could not understand" in msg:
            return "❌ Could not understand the audio. Please try speaking more clearly."
        if "request" in msg.lower():
            return f"❌ Speech recognition service error: {msg}"
        return f"❌ Transcription error: {msg}"


def transcribe_with_whisper(audio_file: Any) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key not configured. Please add your API key in the sidebar."
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Transcribe using Whisper
        with open(tmp_file_path, "rb") as audio_file_obj:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_obj
            )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return transcript.text
        
    except Exception as e:
        return f"❌ Whisper API error: {str(e)}"
