"""
GEA-6 Next Gen Multimodal AI Chatbot (Streamlit App Entrypoint)
Minimal but complete app using updated utils/api_clients and utils/speech_utils
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
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

MAX_CHAT_MESSAGES = 500

# Capability matrix
CAPABILITIES = {
    "OpenAI": {"text": True, "image": True, "vision": True, "stt": True},
    "Gemini": {"text": True, "image": False, "vision": True, "stt": False},
    "GROK":   {"text": True, "image": False, "vision": False, "stt": False},
}

def ensure_capability(provider: str, capability: str) -> bool:
    supported = CAPABILITIES.get(provider, {}).get(capability, False)
    if not supported:
        st.warning(f"{provider} does not support {capability}. Please switch provider.")
    return supported

def validate_model_features(model_name: str, feature_type: str) -> bool:
    """Public helper: whether model supports a feature."""
    return CAPABILITIES.get(model_name, {}).get(feature_type, False)

def handle_api_errors(error_msg: str, model_name: str) -> str:
    """Return user-friendly error messages based on provider and error string."""
    lower = (error_msg or "").lower()
    if model_name == "Gemini":
        if "timeout" in lower or "deadline" in lower:
            return "Gemini timed out. Please retry, simplify input, or check your network."
        return "Gemini error occurred. Verify API key and try again."
    if model_name == "Groq":
        if "invalid" in lower:
            return "Groq request invalid. Check prompt and API permissions."
        return "Groq service error. Please retry or switch model."
    if model_name == "OpenAI":
        if "rate" in lower:
            return "OpenAI rate-limited. Wait a moment and retry."
        if "key" in lower:
            return "OpenAI API key issue. Verify key in .env or Streamlit secrets."
        return "OpenAI error. Please retry."
    return "An error occurred. Please retry."

def reset_conversation():
    st.session_state.chat_history = []
    # do not clear generated images by default; it's useful for bulk export

def show_error(message: str, hint: str=None):
    st.error(f"Error: {message}")
    if hint:
        st.caption(f"Hint: {hint}")

def append_chat(role: str, content: str, meta: dict=None):
    msg = {"role": role, "content": content, "ts": datetime.now().isoformat()}
    if meta:
        msg.update(meta)
    st.session_state.chat_history.append(msg)
    if len(st.session_state.chat_history) > MAX_CHAT_MESSAGES:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_MESSAGES:]
    if role == "assistant":
        st.session_state["last_bot"] = content

def synthesize_tts(text: str) -> bytes | None:
    """Create speech audio from text using OpenAI TTS if available.

    Returns mp3 bytes or None on failure (with a user-facing error).
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
    """Generate TTS audio bytes and mime using available engines.

    Order: preferred -> OpenAI -> gTTS -> pyttsx3. Returns (audio_bytes, mime, engine_used).
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
    """Top-of-page, visually distinct settings card with instant updates.

    Returns (model, theme, language)
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
              <div class="gea6-settings-title">⚙️ GEA-6 Settings</div>
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
    # Centralized UI labels per selected language. Defaults to English.
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
    if language == "Marathi":
        base = {**en, **mr}
    elif language == "Hindi":
        base = {**en, **hi}
    else:
        base = en
    return base

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
    _init_state()
    st.set_page_config(page_title="GEA-6 Multimodal AI", page_icon="🤖", layout="wide")
    # Render the new settings card at the top
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

    st.title("GEA-6 Next Gen Multimodal AI Chatbot")
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
                st.markdown(f"<div class='chat-bubble'><b>GEA6:</b> {msg['content']}</div>", unsafe_allow_html=True)
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
                                    file_name=f"gea6_reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}",
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


