# 🚨 Streamlit Deployment Troubleshooting Guide

## Common Streamlit Installation Failures & Solutions

### 1. **Memory/Timeout Issues**
**Problem**: Installation fails due to memory constraints or timeout
**Solutions**:
```bash
# Use minimal requirements first
pip install -r requirements-minimal.txt

# Install with no cache and timeout increase
pip install --no-cache-dir --timeout 1000 -r requirements.txt

# For Docker builds, increase memory
docker build --memory=4g -t your-app .
```

### 2. **System Dependencies Missing**
**Problem**: `pyaudio`, `opencv-python` fail to install
**Solutions**:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio libasound2-dev

# For Alpine (Docker)
apk add --no-cache portaudio-dev alsa-lib-dev

# Use headless versions
pip install opencv-python-headless  # Instead of opencv-python
```

### 3. **Version Conflicts**
**Problem**: Package version incompatibilities
**Solutions**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# or
fresh_env\Scripts\activate  # Windows

# Install with conflict resolution
pip install --upgrade pip
pip install --force-reinstall --no-deps streamlit
pip install -r requirements.txt
```

### 4. **Python Version Issues**
**Problem**: Wrong Python version for deployment platform
**Solutions**:
```bash
# Check Python version
python --version

# Use specific Python version
pyenv install 3.11.6
pyenv local 3.11.6

# For Heroku, create runtime.txt
echo "python-3.11.6" > runtime.txt
```

### 5. **Platform-Specific Issues**

#### **Streamlit Cloud**
```bash
# Use requirements.txt (not requirements-dev.txt)
# Pin all versions
# Remove system-dependent packages
# Use opencv-python-headless instead of opencv-python
```

#### **Heroku**
```bash
# Create Procfile
echo "web: streamlit run gea6_multimodal_chatbot.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Add buildpacks
heroku buildpacks:add heroku/python
heroku buildpacks:add https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git
```

#### **Railway**
```bash
# Use railway.json for configuration
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run gea6_multimodal_chatbot.py --server.port=$PORT --server.address=0.0.0.0",
    "healthcheckPath": "/_stcore/health"
  }
}
```

### 6. **Dependency Resolution Commands**

```bash
# Check for conflicts
pip check

# Show dependency tree
pip show streamlit

# Install with dependency resolution
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# Alternative: Use pip-tools
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

### 7. **Environment-Specific Fixes**

#### **Docker Deployment**
```dockerfile
# Multi-stage build to reduce image size
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501
CMD ["streamlit", "run", "gea6_multimodal_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Local Development**
```bash
# Use conda for better dependency management
conda create -n exiora-ai python=3.11
conda activate exiora-ai
conda install -c conda-forge streamlit
pip install -r requirements.txt
```

### 8. **Debugging Installation Issues**

```bash
# Verbose installation
pip install -v -r requirements.txt

# Check what's being installed
pip list

# Test individual packages
pip install streamlit --no-deps
pip install openai --no-deps

# Check system requirements
python -c "import sys; print(sys.version)"
python -c "import platform; print(platform.platform())"
```

### 9. **Alternative Package Sources**

```bash
# Use different package index
pip install -i https://pypi.org/simple/ -r requirements.txt

# Use conda-forge for problematic packages
conda install -c conda-forge opencv
pip install -r requirements.txt
```

### 10. **Emergency Fallback**

If all else fails, use this minimal setup:

```bash
# Create emergency requirements
cat > requirements-emergency.txt << EOF
streamlit==1.28.1
openai==1.3.5
python-dotenv==1.0.0
requests==2.31.0
EOF

# Install and test
pip install -r requirements-emergency.txt
streamlit run gea6_multimodal_chatbot.py
```

## Quick Fix Checklist

- [ ] Check Python version compatibility (3.8-3.11)
- [ ] Use pinned versions instead of ranges
- [ ] Remove system-dependent packages (pyaudio, opencv-python)
- [ ] Use headless versions (opencv-python-headless)
- [ ] Create fresh virtual environment
- [ ] Test with minimal requirements first
- [ ] Check platform-specific requirements
- [ ] Use appropriate buildpacks for cloud platforms
- [ ] Increase memory/timeout limits
- [ ] Use multi-stage Docker builds

## Platform-Specific Requirements

### Streamlit Cloud
- Use `requirements.txt` (not `requirements-dev.txt`)
- Pin all versions
- No system dependencies
- File size limits apply

### Heroku
- `runtime.txt` for Python version
- `Procfile` for start command
- Buildpacks for system dependencies
- Memory limits: 512MB free tier

### Railway
- `railway.json` for configuration
- Automatic Python detection
- Good for ML applications
- Generous free tier

### Docker
- Multi-stage builds recommended
- Use slim Python images
- Install system dependencies in build stage
- Optimize for production

## Getting Help

1. Check platform-specific documentation
2. Use `pip check` to identify conflicts
3. Test locally with same Python version
4. Use minimal requirements for initial deployment
5. Gradually add dependencies back
6. Monitor deployment logs for specific errors
