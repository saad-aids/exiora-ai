# 🤝 Contributing to Exiora AI

**Launched by Saad Sadik Shaikh** | AI & DS Student from Pune

Thank you for your interest in contributing to Exiora AI! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git installed
- Basic knowledge of Streamlit and AI/ML concepts

### Development Setup

1. **Fork the Repository**
   ```bash
   # Click the "Fork" button on GitHub, then clone your fork
   git clone https://github.com/yourusername/exiora-ai.git
   cd exiora-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

5. **Run the Application**
   ```bash
   streamlit run gea6_multimodal_chatbot.py
   ```

## 📋 How to Contribute

### 1. **Bug Reports**
- Use the GitHub issue tracker
- Include detailed steps to reproduce
- Specify your environment (OS, Python version, etc.)
- Attach relevant logs or screenshots

### 2. **Feature Requests**
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Consider implementation complexity
- Discuss with maintainers before major changes

### 3. **Code Contributions**

#### Pull Request Process:
1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create Pull Request on GitHub
   ```

## 📝 Coding Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write descriptive docstrings for functions
- Keep functions focused and small

### Example Function Documentation:
```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1 (str): Description of parameter 1
        param2 (int, optional): Description of parameter 2. Defaults to 10.
    
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> example_function("test", 5)
        True
    """
    # Implementation here
    pass
```

### Streamlit Best Practices
- Use `st.cache_data` for expensive operations
- Implement proper error handling
- Use session state for persistent data
- Follow Streamlit's component guidelines

### File Organization
```
exiora-ai/
├── gea6_multimodal_chatbot.py    # Main application
├── utils/                        # Utility modules
│   ├── api_clients.py           # AI model clients
│   ├── speech_utils.py          # Speech processing
│   └── __init__.py
├── assets/                      # Static assets
│   ├── light_mode.css
│   └── dark_mode.css
├── tests/                       # Test files
├── docs/                        # Documentation
└── requirements.txt
```

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests
- Create test files in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external API calls

### Example Test:
```python
import pytest
from utils.api_clients import OpenAIClient

def test_openai_client_initialization():
    """Test OpenAI client initialization."""
    client = OpenAIClient()
    assert client is not None

def test_generate_text_success():
    """Test successful text generation."""
    client = OpenAIClient()
    result = client.generate_text("Hello, world!")
    assert isinstance(result, str)
    assert len(result) > 0
```

## 🌐 Adding New Languages

### 1. **Update Language Codes**
Add to `INDIAN_LANG_CODES` in the main file:
```python
"NewLanguage": "new-lang-code",
```

### 2. **Add Translations**
Create a new language dictionary in `_labels()`:
```python
new_lang = {
    "tab_text": "Translated Text Chat",
    "tab_img_gen": "Translated Image Generator",
    # ... add all required keys
}
```

### 3. **Update Language Map**
Add to the `lang_map` dictionary:
```python
"NewLanguage": {**en, **new_lang},
```

### 4. **Test the Implementation**
- Verify all UI elements are translated
- Test language switching
- Check speech recognition integration

## 🔧 Adding New AI Models

### 1. **Create Client Class**
Add to `utils/api_clients.py`:
```python
class NewAIClient:
    def __init__(self):
        # Initialize client
        
    def generate_text(self, prompt: str) -> str:
        # Implement text generation
        pass
```

### 2. **Update Capabilities Matrix**
Add to `CAPABILITIES` in main file:
```python
"NewAI": {"text": True, "image": False, "vision": False, "stt": False},
```

### 3. **Add to Settings**
Update the model selection in `render_settings_card()`:
```python
["OpenAI", "Gemini", "GROK", "NewAI"]
```

### 4. **Implement Integration**
Add model handling in the main application logic.

## 📚 Documentation

### Code Documentation
- Use docstrings for all functions and classes
- Include type hints
- Add inline comments for complex logic
- Update README.md for new features

### API Documentation
- Document all public functions
- Include usage examples
- Specify parameter types and return values
- Note any side effects or exceptions

## 🐛 Bug Fixing

### Before Fixing
1. **Reproduce the Bug**
   - Create a minimal test case
   - Document the expected vs actual behavior
   - Check if it's already reported

2. **Investigate**
   - Check recent changes
   - Look at error logs
   - Test with different configurations

### Fixing Process
1. **Create Fix Branch**
   ```bash
   git checkout -b fix/bug-description
   ```

2. **Implement Fix**
   - Make minimal changes
   - Add tests to prevent regression
   - Update documentation if needed

3. **Test Thoroughly**
   - Test the specific bug
   - Run existing tests
   - Test edge cases

## 🚀 Release Process

### Version Numbering
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared

## 💬 Communication

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas
- **Email**: [your-email@example.com] for private matters

### Code Review Process
1. **Automated Checks**
   - Tests must pass
   - Code style compliance
   - Security scan results

2. **Human Review**
   - At least one maintainer approval
   - Code quality assessment
   - Documentation review

3. **Merge Process**
   - Squash commits if needed
   - Update version numbers
   - Tag releases

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📄 License

By contributing to Exiora AI, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Your contributions help make Exiora AI better for everyone. Whether you're fixing bugs, adding features, improving documentation, or just reporting issues, your help is greatly appreciated!

---

**Happy Contributing! 🚀**

Made with ❤️ by Saad Sadik Shaikh
