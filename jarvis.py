import os
from datetime import datetime, timedelta
import webbrowser
import pyttsx3
import speech_recognition as sr
import wikipedia
import requests
import subprocess
import librosa
import numpy as np
import soundfile as sf
import cv2
import getpass
import json
import time
import google.generativeai as genai
import sys
from pvrecorder import PvRecorder
import shlex
import logging
import base64
import hashlib
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import torch
from dataclasses import dataclass
import re
from typing import Optional, List, Tuple, Dict, Any
import json

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(
    filename="jarvis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=GENAI_API_KEY)

def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

class ConversationManager:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_window = timedelta(minutes=10)
        
    def add_exchange(self, user_input: str, assistant_response: str):
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "assistant_response": assistant_response
        })
        
    def get_recent_context(self) -> List[Dict[str, str]]:
        current_time = datetime.now()
        recent_messages = []
        
        for exchange in reversed(self.conversation_history):
            if current_time - exchange["timestamp"] <= self.context_window:
                recent_messages.append({
                    "role": "user",
                    "content": exchange["user_input"]
                })
                recent_messages.append({
                    "role": "model",
                    "content": exchange["assistant_response"]
                })
            else:
                break
                
        return list(reversed(recent_messages))

class ContextManager:
    def __init__(self):
        self.expecting_follow_up = False
        self.last_command = None
        self.context = {}

    def set_last_command(self, command, follow_up_needed):
        self.last_command = command
        self.expecting_follow_up = follow_up_needed

    def clear_context(self):
        self.last_command = None
        self.expecting_follow_up = False
        self.context = {}

    def __enter__(self):
        print("Entering Context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting Context")

class CommandLogger:
    def __init__(self):
        self.history = []

    def log_command(self, command):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append((timestamp, command))

    def get_history(self):
        return self.history

class EnhancedChatbot:
    def __init__(self, history_file="/Users/jeel/Desktop/Jarvis/conversation_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
        self.conversation_manager = ConversationManager()
        self.command_patterns = {
            r'\b(what|who|how|why|where|when|tell me about)\b': self._handle_query,
            r'\b(play|open|search|find)\b': self._handle_action,
            r'\b(set|adjust|change)\b': self._handle_setting,
        }
        
        # Initialize Gemini model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print("Warning: JSON file is corrupted. Creating a new one.")
                return []
        return []

    def save_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=4)
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    def log_conversation(self, user_input, bot_response):
        conversation_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "bot": bot_response
        }
        self.history.append(conversation_entry)
        self.save_history()

    def _handle_query(self, user_input: str) -> str:
        try:
            chat = self.model.start_chat(history=[])
            if self.conversation_manager.conversation_history:
                recent_history = self.conversation_manager.get_recent_context()
                for msg in recent_history:
                    if msg["role"] == "user":
                        chat.send_message(msg["content"])
                        response = chat.send_message(user_input)
                        return response.text
        
        except Exception as e:
                    logging.error(f"Error in Gemini query: {e}")
                    return "I apologize, but I'm having trouble processing that request."

    def _handle_action(self, user_input: str) -> str:
        return self._handle_query(user_input)

    def _handle_setting(self, user_input: str) -> str:
        return self._handle_query(user_input)

    def process_input(self, user_input: str) -> str:
        user_input = user_input.lower().strip()
        
        if user_input in ["what do you mean?", "explain that", "tell me more", "how?", "why?"]:
            if self.conversation_manager.conversation_history:
                last_exchange = self.conversation_manager.conversation_history[-1]
                context = f"Regarding the previous response: '{last_exchange['assistant_response']}', please explain further."
                response = self._handle_query(context)
            else:
                response = "I'm not sure what to explain. Could you please ask your question again?"
        else:
            for pattern, handler in self.command_patterns.items():
                if re.search(pattern, user_input):
                    response = handler(user_input)
                    break
            else:
                response = self._handle_query(user_input)

        self.conversation_manager.add_exchange(user_input, response)
        self.log_conversation(user_input, response)
        
        return response

def take_command(command_logger=None):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"🗣️ You said: {command}")
        
        if command_logger:
            command_logger.log_command(f"User Command: {command}")
        
        return command
    except sr.UnknownValueError:
        print("⚠️ Sorry, I didn't catch that. Please try again.")
        return ""
    except sr.RequestError as e:
        print(f"🚨 Could not request results, error: {e}")
        return ""

def analyze_command_for_followup(command: str) -> Tuple[bool, str]:
    followup_patterns = {
        r"\bsearch\b": "What would you like me to search for?",
        r"\bopen\b": "Which application would you like me to open?",
        r"\bplay\b": "What would you like me to play?",
        r"\bset\b\s+(?!an alarm|a timer)": "What value would you like me to set?",
        r"\bremind\b": "What would you like me to remind you about?",
        r"\bschedule\b": "What would you like me to schedule?",
        r"\bweather\b(?!.*in\s+\w+)": "For which location would you like the weather?",
    }

    for pattern, prompt in followup_patterns.items():
        if re.search(pattern, command, re.IGNORECASE):
            return True, prompt

    return False, ""

class SecurityManager:
    def __init__(self):
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _get_encryption_key(self):
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            key = Fernet.generate_key()
            logging.info("Generated new encryption key")
        return key
    
    def encrypt_data(self, data: str) -> bytes:
        try:
            return self.cipher_suite.encrypt(data.encode())
        except Exception as e:
            logging.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        try:
            return self.cipher_suite.decrypt(encrypted_data).decode()
        except Exception as e:
            logging.error(f"Decryption failed: {e}")
            raise

def authenticate_user():
    """Authenticate user with password (with retries and security improvements)."""
    max_attempts = 3  # Set maximum retry attempts
    attempts = 0

    speak("Please authenticate yourself to proceed.")

    while attempts < max_attempts:
        speak("Enter your passcode.")
        password = getpass.getpass("Enter your passcode: ")  # Hides input

        if password == "147369":
            speak("Authentication successful!")
            print("Authentication successful!")
            return True  # Exit function on success
        else:
            attempts += 1
            remaining_attempts = max_attempts - attempts
            if remaining_attempts > 0:
                speak(f"Authentication failed! You have {remaining_attempts} attempts left.")
                print(f"Authentication failed!")
            else:
                speak("Authentication failed! Too many incorrect attempts.")
                print("Too many failed attempts. Try again later.")
                print("Exiting program.")
                time.sleep(5)  # Add a cooldown after max attempts
                return False

    return False

def open_application(command):
    """Opens common macOS applications based on voice command."""
    apps = {
        "safari": "Safari",
        "notes": "Notes",
        "terminal": "Terminal",
        "spotify": "Spotify",
        "messages": "Messages",
        "calendar": "Calendar",
        "photos": "Photos",
        "preview": "Preview",
        "mail": "Mail",
        "facetime": "FaceTime",
        "reminders": "Reminders",
        "music": "Music",
        "maps": "Maps",
        "whatsapp": "WhatsApp"
    }
    for key in apps:
        if key in command:
            speak(f"Opening {apps[key]}")
            os.system(f"open -a {shlex.quote(apps[key])}")
            return
    speak("Sorry, I couldn't find that application.")

def set_brightness(level):
    """Sets Mac brightness (level: 0.0 to 1.0)"""
    try:
        level = max(0.0, min(1.0, level))
        os.system(f"osascript -e 'tell application \"System Events\" to set value of slider 1 of group 1 of tab group 1 of window 1 of application process \"System Preferences\" to {level}'")
        speak(f"Brightness set to {int(level * 100)} percent.")
    except:
        speak("Sorry, I couldn't adjust the brightness.")

def decrease_brightness():
    """Decreases screen brightness."""
    os.system("ddcctl -d 1 -b 50")

def increase_volume():
    """Increases Mac volume by 10%"""
    os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
    speak("Increasing volume.")

def decrease_volume():
    """Decreases Mac volume by 10%"""
    os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
    speak("Decreasing volume.")

def take_picture():
    """Takes a picture using the webcam."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return

    ret, frame = cam.read()
    if ret:
        filename = f"/Users/jeel/Desktop/photo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"Photo saved: {filename}")
    else:
        print("Error: Could not capture image")

    cam.release()
    cv2.destroyAllWindows()

def get_greeting():
    """Returns a greeting based on the time of day"""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

def get_news():
    """Fetches top news headlines"""
    api_key = NEWS_API_KEY
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={api_key}"
    try:
        response = requests.get(url).json()
        headlines = [article["title"] for article in response["articles"][:3]]
        return "Here are the top news headlines: " + " | ".join(headlines)
    except:
        return "I couldn't fetch the latest news."

def get_daily_briefing():
    """Provides a daily briefing"""
    greeting = get_greeting()
    speak(f"{greeting}, Jeel! Have a great day!")

def control_apple_music(command):
    """Controls Apple Music playback."""
    if "play music" in command:
        speak("Playing music.")
        os.system("osascript -e 'tell application \"Music\" to play'")
    elif "pause music" in command:
        speak("Pausing music.")
        os.system("osascript -e 'tell application \"Music\" to pause'")

def play_on_youtube(command):
    """Opens YouTube with the search query."""
    search_query = command.replace("play", "").strip()
    url = f"https://www.youtube.com/results?search_query={search_query}"
    webbrowser.open(url)
    speak(f"Playing {search_query} on YouTube.")

def tell_time():
    """Tells the current time."""
    now = datetime.now().strftime("%I:%M %p")
    speak(f"The current time is {now}")

def search_wikipedia(command):
    """Searches Wikipedia for the given query."""
    query = command.replace("who is", "").replace("what is", "").strip()
    try:
        summary = wikipedia.summary(query, sentences=2)
        speak(summary)
    except wikipedia.exceptions.PageError:
        speak("Sorry, I couldn't find any information on that topic.")

def get_weather(command):
    """Fetches weather details using WeatherAPI.com."""
    city = command.replace("weather in", "").strip()
    api_key = "6561910c40ca4c229a864742251402"  # Directly using the API key
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for HTTP issues
        data = response.json()
        
        if "current" in data and "location" in data:
            temp = data['current']['temp_c']
            desc = data['current']['condition']['text']
            speak(f"The temperature in {city} is {temp} degrees Celsius with {desc}.")
        else:
            speak("Sorry, I couldn't fetch the weather details. Please check the city name.")
    except requests.exceptions.RequestException as e:
        speak("Sorry, I couldn't fetch the weather details. There was a network issue.")
        print(e)


def close_all_windows():
    """Closes all open windows on macOS."""
    speak("Closing all windows.")
    os.system("osascript -e 'tell application \"System Events\" to keystroke \"w\" using {command down}'")

def analyze_mood(audio_file):
    """Analyzes the mood of the user based on voice tone"""
    try:
        y, sr = librosa.load(audio_file)
        energy = np.mean(librosa.feature.rms(y=y))
        pitch = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        if pitch > 3000 and energy > 0.05:
            mood = "excited"
        elif pitch < 1500 and energy < 0.02:
            mood = "sad"
        else:
            mood = "neutral"
        
        speak(f"I sense that you're feeling {mood} today.")
    except:
        speak("Sorry, I couldn't analyze your mood.")

def record_and_analyze():
    """Records user's voice and analyzes mood"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Tell me how you're feeling today.")
        audio = recognizer.listen(source , timeout=3 , phrase_time_limit=5)
        file_path = "mood_analysis.wav"
        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        analyze_mood(file_path)

def take_screenshot():
    """Takes a screenshot and saves it to the desktop."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"/Users/jeel/Desktop/screenshot_{timestamp}.png"
    os.system(f"screencapture {filename}")
    speak("Screenshot saved on your desktop.")

def lock_mac():
    """Locks the macOS screen."""
    speak("Locking your Mac.")
    os.system("pmset displaysleepnow")

def sleep_mac():
    """Puts macOS to sleep."""
    speak("Putting your Mac to sleep.")
    os.system("osascript -e 'tell application \"System Events\" to sleep'")

def restart_mac():
    """Restarts macOS."""
    speak("Restarting your Mac.")
    os.system("osascript -e 'tell app \"System Events\" to restart'")

def shutdown_mac():
    """Shuts down macOS."""
    speak("Shutting down your Mac.")
    os.system("sudo shutdown -h now")

def unlock_mac():
    """Unlocks the Mac using an AppleScript."""
    speak("Unlocking your Mac.")
    os.system("bash ~/unlock_mac.sh")

def run_jarvis():
    """Updated run_jarvis function with working follow-up system"""
    security = SecurityManager()
    command_logger = CommandLogger()
    chatbot = EnhancedChatbot()
    context_manager = ContextManager()
    
    if not authenticate_user():
        return
    
    speak("Hello! I am JARVIS. How can I assist you today?")
    
    while True:
        # If we're expecting a follow-up, get it
        if context_manager.expecting_follow_up:
            speak(context_manager.context.get('follow_up_prompt', 'Please provide more details.'))
            follow_up_command = take_command(command_logger=command_logger)
            if follow_up_command:
                # Combine the original command with the follow-up
                full_command = f"{context_manager.last_command} {follow_up_command}"
                context_manager.clear_context()
            else:
                speak("I didn't catch that. Let's start over.")
                context_manager.clear_context()
                continue
        else:
            command = take_command(command_logger=command_logger)
            if not command:
                continue
            
            # Check if command needs follow-up
            needs_followup, prompt = analyze_command_for_followup(command)
            if needs_followup:
                context_manager.set_last_command(command, True)
                context_manager.context['follow_up_prompt'] = prompt
                continue
            
            full_command = command

        try:
            encrypted_command = security.encrypt_data(full_command)
            logging.info(f"Processing encrypted command: {encrypted_command}")

            print("Command Processing".center(60))
            print(f"User Query: {full_command}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")

            # Process the command with specific handlers first
            if "play" in full_command and "music" not in full_command:
                play_on_youtube(full_command)
            elif "play music" in full_command or "pause music" in full_command:
                control_apple_music(full_command)

            elif "time" in command:
                    tell_time()
            elif "who is" in command or "what is" in command:
                    search_wikipedia(command)
            elif "weather in" in command:
                    get_weather(command)
            elif "open" in command:
                    open_application(command)
            elif "take a screenshot" in command:
                    take_screenshot()
            elif "lock my mac" in command:
                    lock_mac()
            elif "close all windows" in command:
                    close_all_windows()
            elif "take a picture" in command:
                    take_picture()
            elif "put my mac to sleep" in command:
                    sleep_mac()
            elif "restart my mac" in command:
                    restart_mac()
            elif "shut down my mac" in command:
                    shutdown_mac()
            elif "unlock my mac" in command:
                    unlock_mac()
            elif "analyze my mood" in command:
                    record_and_analyze()
            elif "increase brightness" in command:
                    set_brightness(1.0)
            elif "decrease brightness" in command:
                    set_brightness(0.2)
            elif "increase volume" in command:
                    increase_volume()
            elif "decrease volume" in command:
                    decrease_volume()
            elif "exit" in command or "stop" in command:
                    speak("Goodbye! Have a great day.")
                    print("\nFinal Command History:")
                    for time, cmd in command_logger.get_history():
                        print(f"{time}: {cmd}")
                    os._exit(0)
            else:
                response = chatbot.process_input(full_command)
                speak(response)

        except Exception as e:
            logging.error(f"Error processing command: {e}")
            speak("I encountered an error processing that command.")
            context_manager.clear_context()

if __name__ == "__main__":
    run_jarvis()