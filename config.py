"""
GEA-6 Multimodal AI Chatbot Configuration
Enhanced configuration management for hackathon deployment
"""

from dotenv import load_dotenv
import os
import logging
from typing import Optional

# Load environment variables
load_dotenv()

# API Keys Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Prefer GROK_API_KEY if present, fallback to GROQ_API_KEY for compatibility
GROK_API_KEY = os.getenv('GROK_API_KEY')
GROQ_API_KEY = GROK_API_KEY or os.getenv('GROQ_API_KEY')

# Application Configuration
APP_NAME = "GEA-6 Multimodal AI Chatbot"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Next Generation Multimodal AI Chatbot with Text, Image, and Voice Capabilities"

# Model Configuration
DEFAULT_MODELS = {
    "openai": {
        "text": "gpt-3.5-turbo",
        "image": "dall-e-3",
        "vision": "gpt-4-vision-preview",
        "whisper": "whisper-1"
    },
    "gemini": {
        "text": "gemini-pro",
        "vision": "gemini-pro-vision"
    },
    "grok": {
        "text": "llama2-70b-4096"
    }
}

# Image Generation Settings
IMAGE_SETTINGS = {
    "default_size": "1024x1024",
    "quality": "standard",
    "max_images": 10,
    "supported_formats": ["png", "jpg", "jpeg"]
}

# Audio Processing Settings
AUDIO_SETTINGS = {
    "supported_formats": ["wav", "mp3", "m4a", "ogg"],
    "max_duration": 300,  # 5 minutes
    "sample_rate": 16000
}

# Session Management
SESSION_SETTINGS = {
    "max_conversations": 100,
    "max_messages_per_conversation": 1000,
    "auto_save_interval": 5  # messages
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    "max_response_time": 30,  # seconds
    "cache_duration": 3600,  # 1 hour
    "rate_limit_per_minute": 60
}

# UI/UX Settings
UI_SETTINGS = {
    "default_theme": "light",
    "default_language": "English",
    "enable_animations": True,
    "mobile_breakpoint": 768
}

# Logging Configuration
def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gea6_chatbot.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Validation Functions
def validate_api_key(api_key: str, provider: str) -> bool:
    """Validate API key format"""
    if not api_key:
        return False
    
    if provider == "openai":
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider == "gemini":
        return len(api_key) > 20
    elif provider == "groq":
        return api_key.startswith("gsk_") and len(api_key) > 20
    
    return False

def get_config_summary() -> dict:
    """Get configuration summary for debugging"""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "api_keys_configured": {
            "openai": bool(OPENAI_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
            "grok": bool(GROQ_API_KEY)
        },
        "default_models": DEFAULT_MODELS,
        "image_settings": IMAGE_SETTINGS,
        "audio_settings": AUDIO_SETTINGS,
        "session_settings": SESSION_SETTINGS
    }

# Initialize logging
logger = setup_logging()
