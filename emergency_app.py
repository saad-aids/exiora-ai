"""
Emergency Streamlit App - Minimal Version for Deployment
This version removes all problematic dependencies and focuses on core functionality
"""

import streamlit as st
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="GEA-6 AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Simple OpenAI client
class SimpleOpenAIClient:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_text(self, prompt):
        if not self.api_key:
            return "❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"❌ API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Main app
def main():
    st.title("🤖 GEA-6 AI Chatbot")
    st.markdown("**Emergency Deployment Version** - Minimal dependencies for successful deployment")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.error("❌ OpenAI API Key: Not found")
            st.info("Set OPENAI_API_KEY environment variable")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    st.header("💬 Chat with AI")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = SimpleOpenAIClient()
                response = client.generate_text(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("**GEA-6 Multimodal AI Chatbot** | Created by Saad Sadik Shaikh")
    st.markdown("*Emergency deployment version - Full features available in main app*")

if __name__ == "__main__":
    main()
