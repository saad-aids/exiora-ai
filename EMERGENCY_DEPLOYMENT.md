# 🚨 EMERGENCY DEPLOYMENT GUIDE
## Get Your App Deployed in 5 Minutes

### Step 1: Use Emergency Files
Replace your current files with these minimal versions:

```bash
# Copy emergency files
cp emergency_app.py gea6_multimodal_chatbot.py
cp requirements-emergency.txt requirements.txt
```

### Step 2: Test Locally First
```bash
# Install minimal requirements
pip install -r requirements.txt

# Test the app
streamlit run gea6_multimodal_chatbot.py
```

### Step 3: Deploy to Streamlit Cloud

#### Option A: Streamlit Cloud (Recommended)
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Emergency deployment - minimal version"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `gea6_multimodal_chatbot.py`
   - Add secrets:
     ```
     OPENAI_API_KEY = your_openai_key_here
     ```
   - Click "Deploy!"

#### Option B: Railway (Alternative)
1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Create new project from your repo

2. **Configure**:
   - Add environment variable: `OPENAI_API_KEY`
   - Set start command: `streamlit run gea6_multimodal_chatbot.py --server.port=$PORT --server.address=0.0.0.0`

### Step 4: If Still Failing

#### Check These Common Issues:

1. **API Key Not Set**:
   ```bash
   # Test locally with API key
   export OPENAI_API_KEY="your_key_here"
   streamlit run gea6_multimodal_chatbot.py
   ```

2. **Python Version Issues**:
   ```bash
   # Create runtime.txt
   echo "python-3.11.6" > runtime.txt
   ```

3. **File Path Issues**:
   - Make sure `gea6_multimodal_chatbot.py` is in the root directory
   - Check that all files are committed to git

4. **Memory Issues**:
   - Use the emergency requirements (only 3 packages)
   - Remove any large files from your repo

### Step 5: Verify Deployment

Your app should be accessible at:
- Streamlit Cloud: `https://your-app-name.streamlit.app`
- Railway: `https://your-app-name.railway.app`

### Step 6: Gradually Add Features Back

Once the basic app is deployed:

1. **Add one feature at a time**
2. **Test each addition**
3. **Use the optimized requirements**:
   ```bash
   cp requirements-optimized.txt requirements.txt
   ```

### Emergency Commands

If nothing works, try these commands:

```bash
# Create completely fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# or
fresh_env\Scripts\activate  # Windows

# Install only essential packages
pip install streamlit==1.28.1
pip install python-dotenv==1.0.0
pip install requests==2.31.0

# Test
streamlit run emergency_app.py
```

### Platform-Specific Fixes

#### Streamlit Cloud
- Use `requirements.txt` (not `requirements-dev.txt`)
- Main file must be in root directory
- No system dependencies allowed

#### Railway
- Add `railway.json`:
  ```json
  {
    "build": {
      "builder": "NIXPACKS"
    },
    "deploy": {
      "startCommand": "streamlit run gea6_multimodal_chatbot.py --server.port=$PORT --server.address=0.0.0.0"
    }
  }
  ```

#### Heroku
- Add `Procfile`:
  ```
  web: streamlit run gea6_multimodal_chatbot.py --server.port=$PORT --server.address=0.0.0.0
  ```
- Add `runtime.txt`:
  ```
  python-3.11.6
  ```

### Debugging Commands

```bash
# Check what's installed
pip list

# Check for conflicts
pip check

# Test individual components
python -c "import streamlit; print('Streamlit OK')"
python -c "import requests; print('Requests OK')"
python -c "import dotenv; print('Dotenv OK')"
```

### Success Checklist

- [ ] Emergency app runs locally
- [ ] Only 3 packages in requirements.txt
- [ ] API key is set correctly
- [ ] Files are in root directory
- [ ] All files committed to git
- [ ] Platform-specific files added (if needed)

### If All Else Fails

Use this ultra-minimal version:

```python
# ultra_minimal.py
import streamlit as st

st.title("Hello World")
st.write("This is the most basic Streamlit app possible")

if st.button("Click me"):
    st.write("Button clicked!")
```

```bash
# ultra_minimal_requirements.txt
streamlit==1.28.1
```

This will definitely deploy. Once it works, gradually add features back.

### Getting Help

1. Check the deployment logs on your platform
2. Test locally first
3. Use the emergency files
4. Start with the most basic version possible
5. Add complexity gradually

**Remember**: It's better to have a working simple app than a broken complex one!
