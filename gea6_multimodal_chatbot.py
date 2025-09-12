"""
Exiora AI - Next Gen Multimodal AI Chatbot
Launched by Saad Sadik Shaikh - AI & DS Student from Pune
Advanced conversational AI with text, image, and voice capabilities
"""

import streamlit as st
from utils.api_clients import OpenAIClient, GeminiClient, GrokClient
from utils.speech_utils import transcribe_audio, transcribe_with_whisper
from datetime import datetime
from io import BytesIO
import pandas as pd
import requests
import json
import zipfile
import base64
import tempfile



def _init_state():
    """
    Initialize Streamlit session state variables for the application.
    
    This function sets up the core data structures needed for:
    - Chat history storage
    - Generated images collection
    - Conversation management
    
    Called once at the start of each session to ensure clean state.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

# Configuration Constants
MAX_CHAT_MESSAGES = 500  # Maximum number of chat messages to keep in memory

# AI Model Capability Matrix
# Defines which features each AI provider supports
CAPABILITIES = {
    "OpenAI": {"text": True, "image": True, "vision": True, "stt": True},    # Full feature support
    "Gemini": {"text": True, "image": False, "vision": True, "stt": False},  # No image generation or STT
    "GROK":   {"text": True, "image": False, "vision": False, "stt": False}, # Text generation only
}

def ensure_capability(provider: str, capability: str) -> bool:
    """
    Check if a specific AI provider supports a given capability.
    
    Args:
        provider (str): The AI provider name (OpenAI, Gemini, GROK)
        capability (str): The capability to check (text, image, vision, stt)
    
    Returns:
        bool: True if the provider supports the capability, False otherwise
        
    Side Effects:
        Shows a warning message if the capability is not supported
    """
    supported = CAPABILITIES.get(provider, {}).get(capability, False)
    if not supported:
        st.warning(f"{provider} does not support {capability}. Please switch provider.")
    return supported

def validate_model_features(model_name: str, feature_type: str) -> bool:
    """
    Validate if a model supports a specific feature type.
    
    Args:
        model_name (str): Name of the AI model/provider
        feature_type (str): Type of feature to validate (text, image, vision, stt)
    
    Returns:
        bool: True if the model supports the feature, False otherwise
    """
    return CAPABILITIES.get(model_name, {}).get(feature_type, False)

def handle_api_errors(error_msg: str, model_name: str) -> str:
    """
    Convert technical API errors into user-friendly messages.
    
    Args:
        error_msg (str): The raw error message from the API
        model_name (str): Name of the AI provider that generated the error
    
    Returns:
        str: User-friendly error message with actionable advice
    """
    lower = (error_msg or "").lower()
    
    # Gemini-specific error handling
    if model_name == "Gemini":
        if "timeout" in lower or "deadline" in lower:
            return "Gemini timed out. Please retry, simplify input, or check your network."
        return "Gemini error occurred. Verify API key and try again."
    
    # Groq-specific error handling
    if model_name == "Groq":
        if "invalid" in lower:
            return "Groq request invalid. Check prompt and API permissions."
        return "Groq service error. Please retry or switch model."
    
    # OpenAI-specific error handling
    if model_name == "OpenAI":
        if "rate" in lower:
            return "OpenAI rate-limited. Wait a moment and retry."
        if "key" in lower:
            return "OpenAI API key issue. Verify key in .env or Streamlit secrets."
        return "OpenAI error. Please retry."
    
    # Generic error fallback
    return "An error occurred. Please retry."

def reset_conversation():
    """
    Reset the chat conversation history.
    
    Clears the chat history while preserving generated images
    for potential bulk export functionality.
    """
    st.session_state.chat_history = []
    # Note: Generated images are preserved for bulk export functionality

def show_error(message: str, hint: str = None):
    """
    Display error messages with optional hints to the user.
    
    Args:
        message (str): The main error message to display
        hint (str, optional): Additional helpful hint for the user
    """
    st.error(f"Error: {message}")
    if hint:
        st.caption(f"Hint: {hint}")

def append_chat(role: str, content: str, meta: dict = None):
    """
    Add a new message to the chat history with automatic cleanup.
    
    Args:
        role (str): The role of the message sender ('user' or 'assistant')
        content (str): The message content
        meta (dict, optional): Additional metadata for the message
    
    Side Effects:
        - Adds message to session state chat history
        - Maintains MAX_CHAT_MESSAGES limit by removing oldest messages
        - Stores last assistant message for TTS functionality
    """
    msg = {"role": role, "content": content, "ts": datetime.now().isoformat()}
    if meta:
        msg.update(meta)
    
    st.session_state.chat_history.append(msg)
    
    # Maintain message limit by keeping only the most recent messages
    if len(st.session_state.chat_history) > MAX_CHAT_MESSAGES:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_MESSAGES:]
    
    # Store last assistant message for TTS functionality
    if role == "assistant":
        st.session_state["last_bot"] = content

def synthesize_tts(text: str) -> bytes | None:
    """
    Generate speech audio from text using OpenAI's TTS API.
    
    Args:
        text (str): The text to convert to speech
    
    Returns:
        bytes | None: MP3 audio bytes if successful, None if failed
        
    Note:
        Requires OpenAI API key and openai package installation.
        Falls back gracefully with user-friendly error messages.
    """
    try:
        import openai
        client = openai.OpenAI()
        # Prefer gpt-4o-mini-tts if available; fallback to tts-1
        model = "gpt-4o-mini-tts"
        try:
            resp = client.audio.speech.create(model=model, voice="alloy", input=text)
        except Exception:
            resp = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        # SDK may return bytes or object with content
        audio_bytes = getattr(resp, "content", None)
        if audio_bytes is None:
            try:
                audio_bytes = resp.read()
            except Exception:
                pass
        if not audio_bytes:
            st.warning("TTS created but no audio content returned.")
            return None
        return audio_bytes
    except ModuleNotFoundError:
        st.info("OpenAI SDK not installed for TTS. Install with: python -m pip install openai")
    except Exception as e:
        show_error(f"TTS failed: {e}")
    return None

def synthesize_tts_any(text: str, preferred_engine: str = "auto") -> tuple[bytes | None, str, str]:
    """
    Generate TTS audio using multiple available engines with fallback support.
    
    Args:
        text (str): The text to convert to speech
        preferred_engine (str): Preferred TTS engine ("openai", "gtts", "pyttsx3", or "auto")
    
    Returns:
        tuple[bytes | None, str, str]: (audio_bytes, mime_type, engine_used)
        
    Engine Priority:
        1. Preferred engine (if specified and available)
        2. OpenAI TTS (requires API key)
        3. gTTS (requires internet connection)
        4. pyttsx3 (offline, system-dependent)
        
    Note:
        Each engine has different output formats:
        - OpenAI: MP3
        - gTTS: MP3
        - pyttsx3: WAV
    """
    engines = []
    if preferred_engine != "auto":
        engines.append(preferred_engine)
    engines.extend(["openai", "gtts", "pyttsx3"])

    tried = set()
    for engine in engines:
        if engine in tried:
            continue
        tried.add(engine)
        try:
            if engine == "openai":
                audio_bytes = synthesize_tts(text)
                if audio_bytes:
                    return audio_bytes, "audio/mp3", "openai"
            elif engine == "gtts":
                try:
                    from gtts import gTTS
                except ModuleNotFoundError:
                    continue
                fp = BytesIO()
                gTTS(text=text, lang="en").write_to_fp(fp)
                fp.seek(0)
                return fp.getvalue(), "audio/mp3", "gtts"
            elif engine == "pyttsx3":
                try:
                    import pyttsx3
                except ModuleNotFoundError:
                    continue
                engine_obj = pyttsx3.init()
                # pyttsx3 typically outputs WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    engine_obj.save_to_file(text, tmp.name)
                    engine_obj.runAndWait()
                    tmp.seek(0)
                    data = tmp.read()
                return data, "audio/wav", "pyttsx3"
        except Exception as e:
            # Continue trying other engines; show a lightweight hint
            st.info(f"TTS via {engine} failed: {e}")
            continue
    return None, "", ""

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def export_chat_json() -> bytes:
    return json.dumps({
        "messages": st.session_state.chat_history,
        "exported_at": datetime.now().isoformat()
    }, indent=2).encode()

def export_chat_csv() -> bytes:
    df = pd.DataFrame(st.session_state.chat_history)
    buf = BytesIO(); df.to_csv(buf, index=False)
    return buf.getvalue()

def create_images_zip() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for idx, img_bytes in enumerate(st.session_state.generated_images, start=1):
            zf.writestr(f"image_{idx}.png", img_bytes)
        zf.writestr("metadata.json", json.dumps({
            "count": len(st.session_state.generated_images),
            "exported_at": datetime.now().isoformat()
        }, indent=2))
    buf.seek(0)
    return buf.getvalue()


def _sidebar():
    st.title("GEA-6 Settings")
    model = st.selectbox("Model", ["OpenAI", "Gemini", "GROK"], index=0)
    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    language = st.selectbox(
        "Language",
        [
            "English","Hindi","Marathi","Bengali","Tamil","Telugu","Gujarati",
            "Kannada","Malayalam","Punjabi","Urdu","Odia","Assamese"
        ],
        index=0
    )
    st.divider()
    st.caption("Set API keys via .env or environment vars if needed")
    return model, theme, language

def render_settings_card() -> tuple[str, str, str]:
    """
    Render the top-of-page settings card with modern styling and instant updates.
    
    Creates a visually distinct card containing:
    - AI model selection (OpenAI, Gemini, Grok)
    - Theme selection (Light, Dark)
    - Language selection (13 Indian languages)
    
    Returns:
        tuple[str, str, str]: (selected_model, selected_theme, selected_language)
        
    Features:
        - Modern card design with gradient background and shadows
        - Instant updates across the entire application
        - Persistent settings via session state
        - Responsive column layout
    """
    # Inject minimal CSS once for a modern card look
    if not st.session_state.get("_settings_css_injected"):
        st.markdown(
            """
            <style>
            .gea6-settings-card { 
                background: linear-gradient(180deg, rgba(245,247,250,0.8), rgba(255,255,255,0.9));
                border: 1px solid rgba(0,0,0,0.06);
                box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                border-radius: 12px; 
                padding: 16px 18px; 
                margin: 4px 0 12px 0;
            }
            .gea6-settings-title { 
                font-weight: 700; 
                font-size: 1.1rem; 
                margin-bottom: 6px;
                display: flex; 
                align-items: center;
                gap: 8px;
            }
            .gea6-settings-subtle { color: rgba(0,0,0,0.55); font-size: 0.85rem; margin-bottom: 8px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_settings_css_injected"] = True

    with st.container():
        st.markdown(
            """
            <div class="gea6-settings-card">
              <div class="gea6-settings-title">⚙️ Exiora AI Settings</div>
              <div class="gea6-settings-subtle">Configure your AI model, theme, and language. Changes apply instantly.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([1,1,1])

        # Defaults from state if available
        default_model = st.session_state.get("settings_model", "OpenAI")
        default_theme = st.session_state.get("settings_theme", "Light")
        default_language = st.session_state.get("settings_language", "English")

        with col1:
            model = st.selectbox(
                "Model",
                ["OpenAI", "Gemini", "GROK"],
                index=["OpenAI","Gemini","GROK"].index(default_model) if default_model in ["OpenAI","Gemini","GROK"] else 0,
                key="settings_model",
                help="Choose the AI provider used across all tabs.",
            )
        with col2:
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark"],
                index=["Light","Dark"].index(default_theme) if default_theme in ["Light","Dark"] else 0,
                key="settings_theme",
                help="Switch the app’s color scheme.",
            )
        with col3:
            language = st.selectbox(
                "Language",
                [
                    "English","Hindi","Marathi","Bengali","Tamil","Telugu","Gujarati",
                    "Kannada","Malayalam","Punjabi","Urdu","Odia","Assamese"
                ],
                index=[
                    "English","Hindi","Marathi","Bengali","Tamil","Telugu","Gujarati",
                    "Kannada","Malayalam","Punjabi","Urdu","Odia","Assamese"
                ].index(default_language) if default_language in [
                    "English","Hindi","Marathi","Bengali","Tamil","Telugu","Gujarati",
                    "Kannada","Malayalam","Punjabi","Urdu","Odia","Assamese"
                ] else 0,
                key="settings_language",
                help="UI labels and STT language where applicable.",
            )

        # A subtle divider separating settings from app body
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    return model, theme, language


def _labels(language: str):
    """
    Get localized UI labels based on the selected language.
    
    Args:
        language (str): The selected language name (e.g., "English", "Hindi", "Marathi")
    
    Returns:
        dict: Dictionary containing all UI labels in the selected language
        
    Supported Languages:
        - English (default)
        - Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati
        - Kannada, Malayalam, Punjabi, Urdu, Odia, Assamese
        
    Note:
        Falls back to English if the selected language is not found.
        Each language dictionary contains 30+ UI strings for complete localization.
    """
    en = {
        "tab_text": " Text Chat",
        "tab_img_gen": "Image Generator",
        "tab_img_qa": "Image Q&A",
        "tab_voice": "Voice Chat",
        "refresh": "Refresh",
        "txt_in": "Type your Question :",
        "send": "Send",
        "img_prompt": "Describe the image you want to generate :",
        "generate_image": "Generate Image",
        "download_all_zip": "Download All (ZIP)",
        "download_png": "Download PNG",
        "download_zip": "Download ZIP",
        "no_images_yet": "No images yet.",
        "img_upload": "Upload an Image",
        "img_question": "Ask a question about the image:",
        "get_image_answer": "Get Image Answer",
        "voice_upload": "Upload Audio",
        "use_whisper": "Use OpenAI Whisper",
        "tip_voice": "Tip: You can generate an image from your transcribed text when using OpenAI.",
        "transcribe_and_ask": "Transcribe and Ask",
        "transcribing": "Transcribing...",
        "transcribed_prefix": "Transcribed:",
        "gen_ai_response": "Generating AI response...",
        "gen_image_from_trans": "Generate Image from Transcription",
        "tts_play": "Play AI reply (TTS)",
        "tts_prepare_dl": "Prepare audio download",
        "tts_dl_label": "Download AI reply audio",
        "json_export_btn": "Export Chat (JSON)",
        "csv_export_btn": "Export Chat (CSV)",
        "json_download": "Download JSON",
        "csv_download": "Download CSV",
        "img_only_openai": "Image generation is only supported with OpenAI. Please switch model.",
        "grok_no_vision": "GROK doesn't support vision. Please switch to OpenAI or Gemini.",
        "gen_image_spinner": "Generating image...",
        "analyzing_image": "Analyzing image...",
        "generating_speech": "Generating speech...",
        "generating_speech_file": "Generating speech file...",
    }
    mr = {
        "tab_text": " मजकूर चॅट",
        "tab_img_gen": "प्रतिमा जनरेशन",
        "tab_img_qa": "प्रतिमा प्रश्नोत्तर",
        "tab_voice": "व्हॉइस चॅट",
        "refresh": "रिफ्रेश",
        "txt_in": "Tumcha Prashna Type Kara :",
        "send": "पाठवा",
        "img_prompt": "तुम्हाला तयार करायची प्रतिमा वर्णन करा :",
        "generate_image": "प्रतिमा तयार करा",
        "download_all_zip": "सर्व डाउनलोड (ZIP)",
        "download_png": "PNG डाउनलोड करा",
        "download_zip": "ZIP डाउनलोड करा",
        "no_images_yet": "अजून प्रतिमा नाहीत.",
        "img_upload": "प्रतिमा अपलोड करा",
        "img_question": "प्रतिमेबद्दल प्रश्न विचारा:",
        "get_image_answer": "प्रतिमेचे उत्तर मिळवा",
        "voice_upload": "ऑडिओ अपलोड करा",
        "use_whisper": "OpenAI व्हिस्पर वापरा",
        "tip_voice": "सूचना: OpenAI वापरताना ट्रान्सक्राइब केलेल्या मजकुरावरून प्रतिमा तयार करू शकता.",
        "transcribe_and_ask": "ट्रान्सक्राइब करा आणि विचारा",
        "transcribing": "ट्रान्सक्राइब करत आहे...",
        "transcribed_prefix": "ट्रान्सक्राइब केलेले:",
        "gen_ai_response": "AI उत्तर तयार करत आहे...",
        "gen_image_from_trans": "ट्रान्सक्रिप्शनवरून प्रतिमा तयार करा",
        "tts_play": "AI उत्तर प्ले (TTS)",
        "tts_prepare_dl": "ऑडिओ डाउनलोड तयार करा",
        "tts_dl_label": "AI उत्तर ऑडिओ डाउनलोड",
        "json_export_btn": "चॅट निर्यात (JSON)",
        "csv_export_btn": "चॅट निर्यात (CSV)",
        "json_download": "JSON डाउनलोड करा",
        "csv_download": "CSV डाउनलोड करा",
        "img_only_openai": "प्रतिमा जनरेशन फक्त OpenAI सह समर्थित आहे. कृपया मॉडेल बदला.",
        "grok_no_vision": "GROK व्हिजन समर्थन करत नाही. कृपया OpenAI किंवा Gemini निवडा.",
        "gen_image_spinner": "प्रतिमा तयार करत आहे...",
        "analyzing_image": "प्रतिमा विश्लेषित करत आहे...",
        "generating_speech": "बोली तयार करत आहे...",
        "generating_speech_file": "बोली फाईल तयार करत आहे...",
    }
    hi = {
        "tab_text": " टेक्स्ट चैट",
        "tab_img_gen": "इमेज जनरेटर",
        "tab_img_qa": "इमेज प्रश्नोत्तर",
        "tab_voice": "वॉइस चैट",
        "refresh": "रिफ्रेश",
        "txt_in": "अपना प्रश्न टाइप करें :",
        "send": "भेजें",
        "img_prompt": "जिस छवि को आप उत्पन्न करना चाहते हैं उसका वर्णन करें :",
        "generate_image": "इमेज बनाएं",
        "download_all_zip": "सभी डाउनलोड (ZIP)",
        "download_png": "PNG डाउनलोड करें",
        "download_zip": "ZIP डाउनलोड करें",
        "no_images_yet": "अभी तक कोई इमेज नहीं।",
        "img_upload": "इमेज अपलोड करें",
        "img_question": "इमेज के बारे में प्रश्न पूछें:",
        "get_image_answer": "इमेज का उत्तर प्राप्त करें",
        "voice_upload": "ऑडियो अपलोड करें",
        "use_whisper": "OpenAI व्हिस्पर का उपयोग करें",
        "tip_voice": "टिप: OpenAI में ट्रांसक्राइब्ड टेक्स्ट से इमेज बना सकते हैं।",
        "transcribe_and_ask": "ट्रांसक्राइब करें और पूछें",
        "transcribing": "ट्रांसक्राइब हो रहा है...",
        "transcribed_prefix": "ट्रांसक्राइब्ड:",
        "gen_ai_response": "AI उत्तर बना रहा है...",
        "gen_image_from_trans": "ट्रांसक्रिप्शन से इमेज बनाएं",
        "tts_play": "AI उत्तर प्ले (TTS)",
        "tts_prepare_dl": "ऑडियो डाउनलोड तैयार करें",
        "tts_dl_label": "AI उत्तर ऑडियो डाउनलोड",
        "json_export_btn": "चैट निर्यात (JSON)",
        "csv_export_btn": "चैट निर्यात (CSV)",
        "json_download": "JSON डाउनलोड",
        "csv_download": "CSV डाउनलोड",
        "img_only_openai": "इमेज जनरेशन केवल OpenAI के साथ समर्थित है। कृपया मॉडल बदलें।",
        "grok_no_vision": "GROK विजन सपोर्ट नहीं करता। कृपया OpenAI या Gemini चुनें।",
        "gen_image_spinner": "इमेज बना रहा है...",
        "analyzing_image": "इमेज का विश्लेषण हो रहा है...",
        "generating_speech": "स्पीच बना रहा है...",
        "generating_speech_file": "स्पीच फ़ाइल बना रहा है...",
    }
    bn = {  # Bengali
        "tab_text": " টেক্সট চ্যাট", "tab_img_gen": "ছবি জেনারেটর", "tab_img_qa": "ছবি প্রশ্নোত্তর", "tab_voice": "ভয়েস চ্যাট",
        "refresh": "রিফ্রেশ", "txt_in": "আপনার প্রশ্ন টাইপ করুন:", "send": "পাঠান", "generate_image": "ছবি তৈরি করুন",
        "tts_play": "AI উত্তর প্লে (TTS)", "transcribe_and_ask": "ট্রান্সক্রিপ্ট এবং জিজ্ঞাসা করুন"
    }
    ta = {  # Tamil
        "tab_text": " உரை அரட்டை", "tab_img_gen": "படம் ஜெனரேட்டர்", "tab_img_qa": "படம் கேள்வி பதில்", "tab_voice": "குரல் அரட்டை",
        "refresh": "புதுப்பிக்க", "txt_in": "உங்கள் கேள்வியை தட்டச்சு செய்யுங்கள்:", "send": "அனுப்பு", "generate_image": "படம் உருவாக்கு",
        "tts_play": "AI பதில் இயக்கு (TTS)", "transcribe_and_ask": "எழுத்துருவாக்கம் மற்றும் கேள்வி"
    }
    te = {  # Telugu
        "tab_text": " టెక్స్ట్ చాట్", "tab_img_gen": "చిత్రం జనరేటర్", "tab_img_qa": "చిత్రం ప్రశ్న సమాధానం", "tab_voice": "వాయిస్ చాట్",
        "refresh": "రిఫ్రెష్", "txt_in": "మీ ప్రశ్నను టైప్ చేయండి:", "send": "పంపు", "generate_image": "చిత్రం సృష్టించు",
        "tts_play": "AI సమాధానం ప్లే (TTS)", "transcribe_and_ask": "ట్రాన్స్‌క్రైబ్ మరియు అడుగు"
    }
    gu = {  # Gujarati
        "tab_text": " ટેક્સ્ટ ચેટ", "tab_img_gen": "છબી જનરેટર", "tab_img_qa": "છબી પ્રશ્નોત્તર", "tab_voice": "વૉઇસ ચેટ",
        "refresh": "રિફ્રેશ", "txt_in": "તમારો પ્રશ્ન ટાઇપ કરો:", "send": "મોકલો", "generate_image": "છબી બનાવો",
        "tts_play": "AI જવાબ પ્લે (TTS)", "transcribe_and_ask": "ટ્રાન્સક્રાઇબ અને પૂછો"
    }
    kn = {  # Kannada
        "tab_text": " ಪಠ್ಯ ಚಾಟ್", "tab_img_gen": "ಚಿತ್ರ ಜನರೇಟರ್", "tab_img_qa": "ಚಿತ್ರ ಪ್ರಶ್ನೋತ್ತರ", "tab_voice": "ಧ್ವನಿ ಚಾಟ್",
        "refresh": "ರಿಫ್ರೆಶ್", "txt_in": "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ:", "send": "ಕಳುಹಿಸಿ", "generate_image": "ಚಿತ್ರ ರಚಿಸಿ",
        "tts_play": "AI ಉತ್ತರ ಪ್ಲೇ (TTS)", "transcribe_and_ask": "ಟ್ರಾನ್ಸ್‌ಕ್ರೈಬ್ ಮತ್ತು ಕೇಳಿ"
    }
    ml = {  # Malayalam
        "tab_text": " ടെക്സ്റ്റ് ചാറ്റ്", "tab_img_gen": "ചിത്രം ജനറേറ്റർ", "tab_img_qa": "ചിത്രം ചോദ്യോത്തരം", "tab_voice": "വോയ്സ് ചാറ്റ്",
        "refresh": "റിഫ്രഷ്", "txt_in": "നിങ്ങളുടെ ചോദ്യം ടൈപ്പ് ചെയ്യുക:", "send": "അയയ്ക്കുക", "generate_image": "ചിത്രം സൃഷ്ടിക്കുക",
        "tts_play": "AI ഉത്തരം പ്ലേ (TTS)", "transcribe_and_ask": "ട്രാൻസ്ക്രൈബ് ചെയ്ത് ചോദിക്കുക"
    }
    pa = {  # Punjabi
        "tab_text": " ਟੈਕਸਟ ਚੈਟ", "tab_img_gen": "ਤਸਵੀਰ ਜਨਰੇਟਰ", "tab_img_qa": "ਤਸਵੀਰ ਸਵਾਲ ਜਵਾਬ", "tab_voice": "ਵਾਇਸ ਚੈਟ",
        "refresh": "ਰਿਫਰੈਸ਼", "txt_in": "ਆਪਣਾ ਸਵਾਲ ਟਾਈਪ ਕਰੋ:", "send": "ਭੇਜੋ", "generate_image": "ਤਸਵੀਰ ਬਣਾਓ",
        "tts_play": "AI ਜਵਾਬ ਪਲੇ (TTS)", "transcribe_and_ask": "ਟ੍ਰਾਂਸਕ੍ਰਾਈਬ ਅਤੇ ਪੁੱਛੋ"
    }
    ur = {  # Urdu
        "tab_text": " ٹیکسٹ چیٹ", "tab_img_gen": "تصویر جنریٹر", "tab_img_qa": "تصویر سوال جواب", "tab_voice": "آواز چیٹ",
        "refresh": "ریفریش", "txt_in": "اپنا سوال ٹائپ کریں:", "send": "بھیجیں", "generate_image": "تصویر بنائیں",
        "tts_play": "AI جواب پلے (TTS)", "transcribe_and_ask": "ٹرانسکرائب اور پوچھیں"
    }
    od = {  # Odia
        "tab_text": " ପାଠ୍ୟ ଚାଟ୍", "tab_img_gen": "ଚିତ୍ର ଜେନେରେଟର", "tab_img_qa": "ଚିତ୍ର ପ୍ରଶ୍ନୋତ୍ତର", "tab_voice": "ସ୍ୱର ଚାଟ୍",
        "refresh": "ରିଫ୍ରେସ୍", "txt_in": "ଆପଣଙ୍କ ପ୍ରଶ୍ନ ଟାଇପ୍ କରନ୍ତୁ:", "send": "ପଠାନ୍ତୁ", "generate_image": "ଚିତ୍ର ସୃଷ୍ଟି କରନ୍ତୁ",
        "tts_play": "AI ଉତ୍ତର ପ୍ଲେ (TTS)", "transcribe_and_ask": "ଟ୍ରାନ୍ସକ୍ରାଇବ୍ ଏବଂ ପଚାରନ୍ତୁ"
    }
    as_lang = {  # Assamese
        "tab_text": " পাঠ চেট", "tab_img_gen": "ছবি জেনাৰেটৰ", "tab_img_qa": "ছবি প্ৰশ্নোত্তৰ", "tab_voice": "কণ্ঠস্বৰ চেট",
        "refresh": "ৰিফ্ৰেছ", "txt_in": "আপোনাৰ প্ৰশ্ন টাইপ কৰক:", "send": "পঠিয়াওক", "generate_image": "ছবি সৃষ্টি কৰক",
        "tts_play": "AI উত্তৰ প্লে (TTS)", "transcribe_and_ask": "ট্ৰান্সক্ৰাইব আৰু সোধক"
    }
    
    lang_map = {
        "Marathi": {**en, **mr}, "Hindi": {**en, **hi}, "Bengali": {**en, **bn},
        "Tamil": {**en, **ta}, "Telugu": {**en, **te}, "Gujarati": {**en, **gu},
        "Kannada": {**en, **kn}, "Malayalam": {**en, **ml}, "Punjabi": {**en, **pa},
        "Urdu": {**en, **ur}, "Odia": {**en, **od}, "Assamese": {**en, **as_lang}
    }
    return lang_map.get(language, en)

# Indian languages BCP-47 codes for SpeechRecognition
INDIAN_LANG_CODES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Gujarati": "gu-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
    "Urdu": "ur-IN",
    "Odia": "or-IN",
    "Assamese": "as-IN",
}


def main():
    """
    Main application entry point for Exiora AI chatbot.
    
    Initializes the Streamlit application with:
    - Session state management
    - Page configuration
    - Settings card rendering
    - Theme application
    - Multi-tab interface with localized content
    
    Features:
        - Text Chat with AI models
        - Image Generation (OpenAI DALL-E)
        - Image Q&A with vision models
        - Voice Chat with transcription and TTS
        - Multi-language support (13 languages)
        - Light/Dark theme support
    """
    # Initialize application state
    _init_state()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Exiora AI - Multimodal Chatbot", 
        page_icon="🚀", 
        layout="wide"
    )
    
    # Render settings card and get user preferences
    model, theme, language = render_settings_card()
    labels = _labels(language)

    if theme == "Dark":
        try:
            with open("assets/dark_mode.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            pass
    else:
        try:
            with open("assets/light_mode.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            pass

    st.title("🚀 Exiora AI - Next Gen Multimodal Chatbot")
    st.caption("**Launched by Saad Sadik Shaikh** | AI & DS Student from Pune")
    tab1, tab2, tab3, tab4 = st.tabs([labels["tab_text"], labels["tab_img_gen"], labels["tab_img_qa"], labels["tab_voice"]])

    # Text Chat
    with tab1:
        # Refresh button to reset conversation
        cols_top = st.columns([1,1,6])
        if cols_top[0].button(labels["refresh"]):
            reset_conversation()
            try:
                st.rerun()
            except Exception:
                pass

        with st.form("chat_form"):
            user_input = st.text_area(labels["txt_in"], key="txt_input", height=100)
            submit_chat = st.form_submit_button(labels["send"])
        if submit_chat and user_input.strip():
            append_chat("user", user_input)
            with st.spinner("Generating response..."):
                if model == "OpenAI":
                    resp = OpenAIClient.generate_text(user_input)
                elif model == "Gemini":
                    try:
                        resp = GeminiClient.generate_text(user_input)
                    except Exception as e:
                        resp = handle_api_errors(str(e), "Gemini")
                else:
                    try:
                        resp = GrokClient.generate_text(user_input)
                    except Exception as e:
                        resp = handle_api_errors(str(e), "GROK")
            if isinstance(resp, str) and resp.startswith("❌"):
                show_error(resp)
            else:
                append_chat("assistant", resp)

        for msg in st.session_state.chat_history:
            if msg.get("role") == "user":
                st.markdown(f"<div class='chat-bubble'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble'><b>Exiora AI:</b> {msg['content']}</div>", unsafe_allow_html=True)
        # Text-to-speech for last assistant reply
        if st.session_state.get("last_bot"):
            if st.button("Play last reply (TTS)"):
                audio_bytes = synthesize_tts(st.session_state["last_bot"]) 
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

    # Image Generator
    with tab2:
        img_prompt = st.text_input(labels["img_prompt"], key="img_prompt")
        colg1, colg2 = st.columns([1,1])
        # Restrict image to OpenAI only
        if model != "OpenAI":
            st.info(labels["img_only_openai"])
        if colg1.button(labels["generate_image"], key="img_btn") and img_prompt:
            if model != "OpenAI":
                ensure_capability(model, "image")
                pass
            else:
                with st.spinner(labels["gen_image_spinner"]):
                    if model == "OpenAI":
                        img_url_or_err = OpenAIClient.generate_image(img_prompt)
                    elif model == "Gemini":
                        img_url_or_err = GeminiClient.generate_image(img_prompt)
                    else:
                        img_url_or_err = GrokClient.generate_image(img_prompt)

                if isinstance(img_url_or_err, str) and img_url_or_err.startswith("http"):
                    try:
                        img_bytes = fetch_image_bytes(img_url_or_err)
                        st.image(img_bytes, caption="Generated Image", use_column_width=True)
                        st.session_state.generated_images.append(img_bytes)
                        st.download_button(
                            labels["download_png"],
                            data=img_bytes,
                            file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        show_error(str(e), "Please retry or check network connectivity.")
                else:
                    show_error(str(img_url_or_err))
        if colg2.button(labels["download_all_zip"], key="zip_btn"):
            if st.session_state.generated_images:
                zip_bytes = create_images_zip()
                st.download_button(
                    labels["download_zip"],
                    data=zip_bytes,
                    file_name=f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
            else:
                st.info(labels["no_images_yet"])

    # Image Q&A
    with tab3:
        uploaded_image = st.file_uploader(labels["img_upload"], type=["png", "jpg", "jpeg"])
        img_question = st.text_input(labels["img_question"])
        if st.button(labels["get_image_answer"], key="img_qa_btn") and uploaded_image and img_question:
            if not ensure_capability(model, "vision"):
                pass
            else:
                with st.spinner(labels["analyzing_image"]):
                    if model == "OpenAI":
                        ans = OpenAIClient.image_qa(uploaded_image, img_question)
                    elif model == "Gemini":
                        try:
                            ans = GeminiClient.image_qa(uploaded_image, img_question)
                        except Exception as e:
                            ans = handle_api_errors(str(e), "Gemini")
                    else:
                        ans = labels["grok_no_vision"]
                if isinstance(ans, str) and ans.startswith("❌"):
                    show_error(ans)
                else:
                    st.success(ans)

    # Voice Chat
    with tab4:
        uploaded_audio = st.file_uploader(labels["voice_upload"], type=["wav", "mp3", "m4a", "ogg"], key="aud_upl")
        use_whisper = st.checkbox(labels["use_whisper"], value=True, key="whisper_cb")
        # Unified voice-to-image experience when OpenAI selected
        if model == "OpenAI":
            st.caption(labels["tip_voice"])
        if st.button(labels["transcribe_and_ask"], key="aud_btn") and uploaded_audio:
            with st.spinner(labels["transcribing"]):
                try:
                    if use_whisper and ensure_capability("OpenAI", "stt"):
                        text = transcribe_with_whisper(uploaded_audio)
                    else:
                        lang_code = INDIAN_LANG_CODES.get(language, "en-IN")
                        text = transcribe_audio(uploaded_audio, language_code=lang_code)
                except ModuleNotFoundError:
                    text = "Speech backend missing. Please install SpeechRecognition or disable Whisper."
                except Exception as e:
                    text = f"Transcription failed: {str(e)}"
            if text.startswith("Transcription failed") or text.startswith("Speech backend"):
                show_error(text)
            else:
                st.write(f"{labels['transcribed_prefix']} {text}")
                with st.spinner(labels["gen_ai_response"]):
                    if model == "OpenAI":
                        resp = OpenAIClient.generate_text(text)
                    elif model == "Gemini":
                        resp = GeminiClient.generate_text(text)
                    else:
                        resp = GrokClient.generate_text(text)
                if isinstance(resp, str) and resp.startswith("❌"):
                    show_error(resp)
                else:
                    st.write(resp)
                    st.session_state["last_bot"] = resp
                    # TTS controls directly below AI response
                    col_tts_play, col_tts_download = st.columns([1,1])
                    with col_tts_play:
                        if st.button(labels["tts_play"], key="voice_tts_play"):
                            with st.spinner(labels["generating_speech"]):
                                audio_bytes, mime, used = synthesize_tts_any(st.session_state["last_bot"], preferred_engine="auto")
                            if audio_bytes:
                                st.audio(audio_bytes, format=mime)
                            else:
                                show_error("TTS failed to produce audio.")
                    with col_tts_download:
                        if st.button(labels["tts_prepare_dl"], key="voice_tts_dl_prep"):
                            with st.spinner(labels["generating_speech_file"]):
                                audio_bytes, mime, used = synthesize_tts_any(st.session_state["last_bot"], preferred_engine="auto")
                            if audio_bytes:
                                ext = ".mp3" if mime == "audio/mp3" else ".wav"
                                st.download_button(
                                    labels["tts_dl_label"],
                                    data=audio_bytes,
                                    file_name=f"exiora_ai_reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}",
                                    mime=mime,
                                    key="voice_tts_dl_btn"
                                )
                            else:
                                show_error("Could not generate audio for download.")
                # If OpenAI selected, offer to generate image from transcribed text
                if model == "OpenAI" and st.button("Generate Image from Transcription"):
                    if validate_model_features("OpenAI", "image"):
                        img_url_or_err = OpenAIClient.generate_image(text)
                        if isinstance(img_url_or_err, str) and img_url_or_err.startswith("http"):
                            try:
                                img_bytes = fetch_image_bytes(img_url_or_err)
                                st.image(img_bytes, caption="Generated from Voice", use_column_width=True)
                                st.session_state.generated_images.append(img_bytes)
                                st.download_button(
                                    "Download PNG",
                                    data=img_bytes,
                                    file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                show_error(str(e), "Please retry or check network connectivity.")
                        else:
                            show_error(str(img_url_or_err))
                # Legacy single-button TTS moved above into inline controls

    st.divider()
    # Downloads
    colx1, colx2 = st.columns(2)
    with colx1:
        if st.button(labels["json_export_btn"]) and st.session_state.chat_history:
            st.download_button(
                labels["json_download"],
                data=export_chat_json(),
                file_name="chat_history.json",
                mime="application/json"
            )
    with colx2:
        if st.button(labels["csv_export_btn"]) and st.session_state.chat_history:
            st.download_button(
                labels["csv_download"],
                data=export_chat_csv(),
                file_name="chat_history.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()


