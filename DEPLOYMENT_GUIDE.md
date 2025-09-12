# 🚀 Exiora AI - Deployment Guide

**Launched by Saad Sadik Shaikh** | AI & DS Student from Pune

This guide covers various deployment options for your Exiora AI chatbot.

## 📋 Prerequisites

- Python 3.8 or higher
- Git installed
- API keys for OpenAI, Google AI, and Groq
- Basic knowledge of cloud platforms

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended for Beginners)

**Pros:** Free, easy setup, automatic updates
**Cons:** Limited customization, public repos only

#### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit - Exiora AI"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `gea6_multimodal_chatbot.py`
   - Add secrets in the dashboard:
     ```
     OPENAI_API_KEY = "your_key_here"
     GOOGLE_AI_API_KEY = "your_key_here"
     GROQ_API_KEY = "your_key_here"
     ```

3. **Deploy**
   - Click "Deploy!"
   - Your app will be live at `https://your-app-name.streamlit.app`

### 2. Heroku

**Pros:** Easy scaling, add-ons available
**Cons:** Paid plans for production

#### Setup:
1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create exiora-ai
   ```

3. **Create Procfile**
   ```bash
   echo "web: streamlit run gea6_multimodal_chatbot.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set OPENAI_API_KEY=your_key_here
   heroku config:set GOOGLE_AI_API_KEY=your_key_here
   heroku config:set GROQ_API_KEY=your_key_here
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

### 3. Railway

**Pros:** Modern platform, good free tier
**Cons:** Newer platform

#### Setup:
1. **Connect GitHub**
   - Visit [railway.app](https://railway.app)
   - Sign in with GitHub
   - Create new project from GitHub repo

2. **Configure Environment**
   - Add environment variables in Railway dashboard
   - Set start command: `streamlit run gea6_multimodal_chatbot.py --server.port=$PORT --server.address=0.0.0.0`

3. **Deploy**
   - Railway auto-deploys on git push

### 4. Docker Deployment

**Pros:** Consistent across environments
**Cons:** Requires Docker knowledge

#### Create Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "gea6_multimodal_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
# Build image
docker build -t exiora-ai .

# Run container
docker run -p 8501:8501 --env-file .env exiora-ai
```

### 5. AWS EC2

**Pros:** Full control, scalable
**Cons:** Requires AWS knowledge, costs money

#### Setup:
1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - t2.micro for free tier
   - Open port 8501 in security groups

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and pip
   sudo apt install python3 python3-pip -y
   
   # Clone repository
   git clone https://github.com/yourusername/exiora-ai.git
   cd exiora-ai
   
   # Install dependencies
   pip3 install -r requirements.txt
   
   # Set environment variables
   export OPENAI_API_KEY="your_key_here"
   export GOOGLE_AI_API_KEY="your_key_here"
   export GROQ_API_KEY="your_key_here"
   
   # Run application
   streamlit run gea6_multimodal_chatbot.py --server.port=8501 --server.address=0.0.0.0
   ```

3. **Use PM2 for Process Management**
   ```bash
   # Install PM2
   sudo npm install -g pm2
   
   # Create ecosystem file
   cat > ecosystem.config.js << EOF
   module.exports = {
     apps: [{
       name: 'exiora-ai',
       script: 'streamlit',
       args: 'run gea6_multimodal_chatbot.py --server.port=8501 --server.address=0.0.0.0',
       cwd: '/home/ubuntu/exiora-ai',
       env: {
         OPENAI_API_KEY: 'your_key_here',
         GOOGLE_AI_API_KEY: 'your_key_here',
         GROQ_API_KEY: 'your_key_here'
       }
     }]
   }
   EOF
   
   # Start with PM2
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```

### 6. Google Cloud Platform

**Pros:** Good integration with Google AI
**Cons:** Complex setup

#### Setup:
1. **Create App Engine App**
   ```bash
   gcloud app create --region=us-central
   ```

2. **Create app.yaml**
   ```yaml
   runtime: python39
   
   env_variables:
     OPENAI_API_KEY: "your_key_here"
     GOOGLE_AI_API_KEY: "your_key_here"
     GROQ_API_KEY: "your_key_here"
   
   handlers:
   - url: /.*
     script: auto
   ```

3. **Deploy**
   ```bash
   gcloud app deploy
   ```

## 🔧 Environment Variables

### Required Variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Optional Variables:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=light
```

## 🚀 Performance Optimization

### 1. Enable Caching
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_function():
    # Your expensive operations here
    pass
```

### 2. Optimize Images
- Compress images before upload
- Use appropriate formats (PNG for graphics, JPEG for photos)
- Implement image resizing

### 3. Database Integration
```python
# Add to requirements.txt
sqlite3  # Built-in
# or
psycopg2-binary  # PostgreSQL
pymongo  # MongoDB
```

### 4. Load Balancing
- Use multiple instances behind a load balancer
- Implement session affinity for Streamlit apps

## 🔒 Security Considerations

### 1. API Key Security
- Never commit API keys to version control
- Use environment variables or secret management
- Rotate keys regularly

### 2. Input Validation
```python
def validate_input(user_input):
    if len(user_input) > 10000:
        raise ValueError("Input too long")
    # Add more validation as needed
    return user_input
```

### 3. Rate Limiting
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 60.0 / calls_per_minute - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator
```

## 📊 Monitoring and Logging

### 1. Application Logs
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exiora_ai.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Health Checks
```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now()}
```

### 3. Metrics Collection
- Use services like DataDog, New Relic, or Prometheus
- Monitor API usage, response times, and error rates

## 🆘 Troubleshooting

### Common Issues:

1. **Port Already in Use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   # Kill process
   kill -9 <PID>
   ```

2. **API Key Not Working**
   - Check environment variables are set correctly
   - Verify API key format and permissions
   - Check API quotas and billing

3. **Memory Issues**
   - Increase container memory limits
   - Implement memory-efficient image processing
   - Use streaming for large files

4. **Slow Performance**
   - Enable caching
   - Optimize database queries
   - Use CDN for static assets

## 📞 Support

For deployment issues:
1. Check the [Issues](https://github.com/yourusername/exiora-ai/issues) page
2. Create a new issue with deployment details
3. Contact: [your-email@example.com]

---

**Happy Deploying! 🚀**

Made with ❤️ by Saad Sadik Shaikh