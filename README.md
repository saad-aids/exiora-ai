# 🚀 Exiora AI - Next Gen Multimodal Chatbot

**Launched by Saad Sadik Shaikh** | AI & DS Student from Pune

A powerful, multilingual AI chatbot with advanced text, image, and voice capabilities. Built with Streamlit and supporting 13 Indian languages.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🤖 **Multi-Model AI Support**
- **OpenAI GPT-4**: Text generation, image creation, vision analysis, speech-to-text
- **Google Gemini**: Text generation and vision analysis
- **Grok AI**: Fast text generation

### 🌐 **13 Language Support**
- English, Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati
- Kannada, Malayalam, Punjabi, Urdu, Odia, Assamese
- Complete UI localization with native language support

### 🎯 **Core Capabilities**
- **Text Chat**: Conversational AI with context awareness
- **Image Generation**: Create stunning images from text prompts
- **Image Q&A**: Analyze and answer questions about uploaded images
- **Voice Chat**: Speech-to-text transcription with AI responses
- **Text-to-Speech**: Convert AI responses to audio (OpenAI TTS, gTTS, pyttsx3)

### 🎨 **Modern UI/UX**
- Beautiful, responsive design with light/dark themes
- Real-time settings with instant updates
- Intuitive tab-based navigation
- Professional card-based layout

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for GPT-4, image generation, TTS)
- Google AI API key (for Gemini)
- Groq API key (for Grok AI)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/exiora-ai.git
cd exiora-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the application**
```bash
streamlit run gea6_multimodal_chatbot.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
exiora-ai/
├── gea6_multimodal_chatbot.py    # Main Streamlit application
├── utils/
│   ├── api_clients.py            # AI model clients (OpenAI, Gemini, Grok)
│   ├── speech_utils.py           # Speech-to-text utilities
│   └── __init__.py
├── assets/
│   ├── light_mode.css            # Light theme styles
│   └── dark_mode.css             # Dark theme styles
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── DEPLOYMENT_GUIDE.md           # Deployment instructions
```

## 🎮 Usage Guide

### 1. **Text Chat**
- Select your preferred AI model from settings
- Type your question in the text area
- Get instant AI responses with context awareness

### 2. **Image Generation**
- Switch to "Image Generator" tab
- Describe the image you want to create
- Download generated images as PNG files

### 3. **Image Q&A**
- Upload an image (PNG, JPG, JPEG)
- Ask questions about the image
- Get detailed AI analysis and answers

### 4. **Voice Chat**
- Upload audio files (WAV, MP3, M4A, OGG)
- Choose between OpenAI Whisper or local transcription
- Get AI responses and play them back with TTS

### 5. **Settings**
- **Model**: Choose between OpenAI, Gemini, or Grok
- **Theme**: Switch between light and dark modes
- **Language**: Select from 13 supported languages

## 🔧 Configuration

### API Keys Setup

1. **OpenAI API Key**
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Add to `.env` file as `OPENAI_API_KEY`

2. **Google AI API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate API key
   - Add to `.env` file as `GOOGLE_AI_API_KEY`

3. **Groq API Key**
   - Visit [Groq Console](https://console.groq.com/keys)
   - Create API key
   - Add to `.env` file as `GROQ_API_KEY`

### Language Configuration
The app supports 13 Indian languages with complete UI localization. Language codes are automatically mapped for speech recognition:

- English: `en-IN`
- Hindi: `hi-IN`
- Marathi: `mr-IN`
- Bengali: `bn-IN`
- Tamil: `ta-IN`
- Telugu: `te-IN`
- Gujarati: `gu-IN`
- Kannada: `kn-IN`
- Malayalam: `ml-IN`
- Punjabi: `pa-IN`
- Urdu: `ur-IN`
- Odia: `or-IN`
- Assamese: `as-IN`

## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy with environment variables set in the dashboard

### Docker Deployment
```bash
docker build -t exiora-ai .
docker run -p 8501:8501 --env-file .env exiora-ai
```

### Local Production
```bash
streamlit run gea6_multimodal_chatbot.py --server.port 8501 --server.address 0.0.0.0
```

## 🛠️ Development

### Adding New Languages
1. Add language to `INDIAN_LANG_CODES` dictionary
2. Create translation dictionary in `_labels()` function
3. Add to language selection in settings

### Adding New AI Models
1. Create client class in `utils/api_clients.py`
2. Add to `CAPABILITIES` matrix
3. Update model selection in settings

### Customizing UI
- Modify CSS files in `assets/` directory
- Update theme injection in main app
- Customize card styling in `render_settings_card()`

## 📊 Performance

- **Response Time**: < 2 seconds for text generation
- **Image Generation**: 10-30 seconds depending on complexity
- **Voice Transcription**: Real-time processing
- **Memory Usage**: ~200MB base + model-specific overhead

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Saad Sadik Shaikh**
- AI & Data Science Student from Pune
- GitHub: [@saad-aids](https://github.com/saad-aids)
- LinkedIn: [Saad Sadik Shaikh](https://linkedin.com/in/saad-sadik-shaikh)

## 🙏 Acknowledgments

- OpenAI for GPT-4 and DALL-E APIs
- Google for Gemini AI
- Groq for fast inference
- Streamlit for the amazing framework
- The open-source community for inspiration

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/saad-aids/exiora-ai/issues) page
2. Create a new issue with detailed description
3. Contact: [shaikhmsaadmsadik@gmail.com]

---

⭐ **Star this repository if you found it helpful!**

Made with ❤️ by Saad Sadik Shaikh
