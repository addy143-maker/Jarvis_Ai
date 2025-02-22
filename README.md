# Jarvis AI Assistant 🤖

A sophisticated voice-controlled AI assistant powered by Google's Gemini API that brings natural language interaction to your macOS system.

## 🌟 Features

### Core Capabilities
- Voice-controlled system operations
- Natural language processing using Google's Gemini AI
- Contextual conversation management
- Secure authentication system
- Comprehensive command logging

### System Controls
- Application management (open/close applications)
- System settings control (brightness, volume)
- Screen capture (screenshots, webcam photos)
- System operations (lock, sleep, restart, shutdown)

### Smart Features
- Weather updates
- News briefings
- Wikipedia searches
- YouTube playback
- Apple Music control
- Mood analysis through voice patterns

### Security & Privacy
- Password protection
- Encrypted command processing
- Secure conversation history
- Environment variable management

## 🔧 Requirements

- Python 3.8+
- macOS (optimized for macOS environment)
- Required API Keys:
  - Google Gemini API
  - News API
  - WeatherAPI

## 📦 Dependencies

```python
pyttsx3
speech_recognition
wikipedia
requests
librosa
numpy
soundfile
opencv-python
google-generativeai
pvrecorder
cryptography
python-dotenv
torch
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jarvis-ai-assistant.git
cd jarvis-ai-assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GENAI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_news_api_key
ENCRYPTION_KEY=your_encryption_key
```

## 💡 Usage

1. Run the assistant:
```bash
python jarvis.py
```

2. Authenticate with your passcode
3. Start giving voice commands

## 🎯 Common Commands

- "Open [application_name]"
- "Play [song/video] on YouTube"
- "What's the weather in [city]?"
- "Tell me about [topic]"
- "Take a screenshot"
- "Analyze my mood"

## 🔐 Security

The system includes multiple security features:
- Password protection
- Command encryption
- Secure logging
- Environment variable protection

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
