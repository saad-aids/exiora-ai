"""
GEA-6 Multimodal AI Chatbot - API Clients
Real API integrations for OpenAI, Gemini, and Groq
"""

import openai
import google.generativeai as genai
from groq import Groq
import base64
import requests
import time
from typing import Any, Dict
from config import OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY


class OpenAIClient:
    """OpenAI API client with real implementations"""
    
    def __init__(self):
        if OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = None
    
    @classmethod
    def generate_text(cls, prompt: str) -> str:
        """Generate text response using OpenAI"""
        if not OPENAI_API_KEY:
            return "❌ OpenAI API key not configured. Please add your API key in the sidebar."
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ OpenAI API Error: {str(e)}"

    @classmethod
    def generate_image(cls, prompt: str) -> str:
        """Generate image using DALL-E"""
        if not OPENAI_API_KEY:
            return "❌ OpenAI API key not configured. Please add your API key in the sidebar."
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return response.data[0].url
        except Exception as e:
            return f"❌ DALL-E API Error: {str(e)}"

    @classmethod
    def image_qa(cls, image_file: Any, question: str) -> str:
        """Answer questions about uploaded images using current OpenAI vision models."""
        if not OPENAI_API_KEY:
            return "❌ OpenAI API key not configured. Please add your API key in the sidebar."
        
        try:
            # Convert uploaded file to base64
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode()
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            # Use gpt-4o (multimodal) or fallback to gpt-4o-mini
            model_candidates = ["gpt-4o", "gpt-4o-mini"]
            last_err = None
            response = None
            for m in model_candidates:
                try:
                    response = client.chat.completions.create(
                        model=m,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=800
                    )
                    break
                except Exception as e:
                    last_err = e
                    continue
            if response is None:
                raise last_err or Exception("OpenAI vision request failed")
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ OpenAI Vision API Error: {str(e)}"


class GeminiClient:
    """Google Gemini API client with real implementations"""
    
    def __init__(self):
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Text
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            # Vision (multimodal)
            try:
                self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception:
                self.vision_model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            self.model = None
            self.vision_model = None
    
    @classmethod
    def generate_text(cls, prompt: str) -> str:
        """Generate text response using Gemini"""
        if not GEMINI_API_KEY:
            return "❌ Gemini API key not configured. Please add your API key in the sidebar."
        
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {str(e)}"

    @classmethod
    def generate_image(cls, prompt: str) -> str:
        """Gemini doesn't support image generation"""
        return "❌ Gemini doesn't support image generation. Please use OpenAI for image generation."

    @classmethod
    def image_qa(cls, image_file: Any, question: str) -> str:
        """Answer questions about uploaded images using Gemini Vision"""
        if not GEMINI_API_KEY:
            return "❌ Gemini API key not configured. Please add your API key in the sidebar."
        
        try:
            from PIL import Image
            import io
            
            # Convert uploaded file to PIL Image
            image = Image.open(image_file)
            
            genai.configure(api_key=GEMINI_API_KEY)
            # Prefer 1.5-flash, fallback to 1.5-pro
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content([question, image])
            except Exception:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content([question, image])
            return response.text
        except Exception as e:
            return f"❌ Gemini Vision API Error: {str(e)}"


class GrokClient:
    """GROK (Groq) API client with real implementations"""
    
    def __init__(self):
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            self.client = None
    
    @classmethod
    def generate_text(cls, prompt: str) -> str:
        """Generate text response using GROK (Groq).

        Tries a list of recommended models to avoid 'model_not_found' errors.
        """
        if not GROQ_API_KEY:
            return "❌ GROK API key not configured. Please add your API key in the sidebar."

        # Prefer currently supported GROK (Groq) models
        candidate_models = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]

        last_err = None
        for model_name in candidate_models:
            try:
                client = Groq(api_key=GROQ_API_KEY)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                err_str = str(e)
                # Continue to next candidate for known model errors
                retriable_hints = [
                    "model_not_found",
                    "does not exist",
                    "model_decommissioned",
                    "no longer supported",
                    "deprecated",
                ]
                if not any(h in err_str for h in retriable_hints):
                    break

        friendly = "GROK model unavailable. Try again later or switch model (OpenAI/Gemini)."
        if last_err:
            friendly += f" Details: {last_err}"
        return f"❌ {friendly}"

    @classmethod
    def generate_image(cls, prompt: str) -> str:
        """GROK doesn't support image generation"""
        return "❌ GROK doesn't support image generation. Please use OpenAI for image generation."

    @classmethod
    def image_qa(cls, image_file: Any, question: str) -> str:
        """GROK doesn't support vision capabilities"""
        return "❌ GROK doesn't support image analysis. Please use OpenAI or Gemini for image Q&A."
