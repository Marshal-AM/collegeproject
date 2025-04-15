import base64
import io
import os
import smtplib
from email.message import EmailMessage
from threading import Lock, Thread
import requests
from bs4 import BeautifulSoup

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from google.cloud import speech
from googleapiclient.discovery import build
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

# Initialize Google Cloud Speech client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_credentials.json")
speech_client = speech.SpeechClient()

# Email configuration
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
RECEIVER_EMAIL = "mathewmarshal3770@gmail.com"
SMTP_SERVER = os.getenv("SMTP", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("PORT", 587))

# Search configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()
        
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

class EmailTool:
    @staticmethod
    def send_email(subject: str, body: str):
        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = GMAIL_ADDRESS
            msg["To"] = RECEIVER_EMAIL

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(GMAIL_ADDRESS, GMAIL_PASSWORD)
                server.send_message(msg)
            return True, "Email sent successfully!"
        except smtplib.SMTPAuthenticationError:
            error = "Authentication failed. Check your Gmail credentials."
            print(f"\n[EMAIL ERROR] {error}\n")
            return False, error
        except Exception as e:
            print(f"\n[EMAIL ERROR] {str(e)}\n")
            return False, f"Email failed to send: {str(e)}"

class SearchTool:
    def __init__(self):
        self.service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

    def search(self, query):
        try:
            results = self.service.cse().list(
                q=query,
                cx=GOOGLE_CSE_ID,
            ).execute()
            return results.get('items', [])
        except Exception as e:
            print(f"\n[SEARCH ERROR] {str(e)}\n")
            return None

    def get_website_content(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to get the main content or fall back to the first paragraph
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                return ' '.join(main_content.stripped_strings)
            
            # If no main content found, get the first few paragraphs
            paragraphs = soup.find_all('p')
            if paragraphs:
                return ' '.join(p.get_text(strip=True) for p in paragraphs[:3])
            
            return "Could not extract main content from the page."
        except Exception as e:
            print(f"\n[CONTENT EXTRACTION ERROR] {str(e)}\n")
            return f"Could not retrieve content from the website: {str(e)}"

    def format_results(self, results, get_content=False):
        if not results:
            return "No results found"
        
        formatted = "Search results:\n"
        for i, result in enumerate(results[:3], 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['link']}\n"
            
            if get_content and i == 1:  # Only get content for the first result
                content = self.get_website_content(result['link'])
                formatted += f"   Content: {content}\n\n"
            else:
                formatted += f"   {result['snippet']}\n\n"
        return formatted

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.email_tool = EmailTool()
        self.search_tool = SearchTool()
        self.awaiting_email_content = False
        self.last_email_content = None
        self.is_speaking = False
        self.player = None

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        # Handle STOP command
        if prompt.strip().upper() == "STOP":
            if self.is_speaking and self.player:
                self.player.terminate()
                self.is_speaking = False
                print("Stopped speaking")
            return

        # Handle follow-up about email status
        if "did you send" in prompt.lower():
            self._handle_email_followup()
            return

        # Check if we're waiting for email content
        if self.awaiting_email_content:
            self._send_email_directly(prompt)
            return

        # Check if user wants to send email
        if "send email" in prompt.lower():
            self._initiate_email_sending()
            return

        # Check if user wants to search
        if any(keyword in prompt.lower() for keyword in ["search for", "look up", "find info about"]):
            self._handle_search_request(prompt)
            return

        # Normal conversation flow
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)
        if response:
            self._tts(response)

    def _initiate_email_sending(self):
        self._tts("What would you like the email to say?")
        self.awaiting_email_content = True

    def _send_email_directly(self, email_content):
        if not email_content.strip():
            self._tts("Email content cannot be empty")
            self.awaiting_email_content = False
            return

        subject = "Message from Assistant"
        success, result = self.email_tool.send_email(subject, email_content)
        
        if success:
            self._tts(f"Email sent to {RECEIVER_EMAIL}")
            self.last_email_content = email_content
        else:
            self._tts(result)
        
        self.awaiting_email_content = False

    def _handle_email_followup(self):
        if self.last_email_content:
            self._tts(f"Yes, I sent your message to {RECEIVER_EMAIL}")
        else:
            self._tts("No recent emails were sent")

    def _handle_search_request(self, query):
        search_query = query.replace("search for", "").replace("look up", "").strip()
        results = self.search_tool.search(search_query)
        
        if results:
            # Get detailed content for the first result
            formatted_results = self.search_tool.format_results(results, get_content=True)
            
            # Summarize results for TTS
            summary = f"I found {len(results)} results about {search_query}. Here's information from the first result: "
            first_result_content = self.search_tool.get_website_content(results[0]['link'])
            summary += first_result_content[:500] + "..."  # Limit to 500 characters for TTS
            
            print(f"\nSearch Results:\n{formatted_results}\n")
            self._tts(summary)
        else:
            self._tts("I couldn't find any results for that search.")

    def _tts(self, response):
        self.is_speaking = True
        self.player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
        try:
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    if not self.is_speaking:
                        break
                    self.player.write(chunk)
        finally:
            self.player.close()
            self.is_speaking = False

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = f"""
        You are a helpful assistant with multiple capabilities:
        - You can send emails to {RECEIVER_EMAIL} when asked
        - You can search the web when requested (use keywords like 'search for' or 'look up')
        - You can analyze images when explicitly asked
        
        For normal conversation, be concise and helpful.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", [
                {"type": "text", "text": "{prompt}"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},
            ]),
        ])
        chain = prompt_template | model | StrOutputParser()
        return RunnableWithMessageHistory(
            chain,
            lambda _: ChatMessageHistory(),
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

# Initialize components
webcam_stream = WebcamStream().start()
model = ChatOpenAI(model="gpt-4o", temperature=0.3)
assistant = Assistant(model)

def recognize_google_speech(audio_data):
    audio_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="default"
    )
    try:
        response = speech_client.recognize(config=config, audio=audio)
        return response.results[0].alternatives[0].transcript if response.results else ""
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return ""

def audio_callback(recognizer, audio):
    try:
        prompt = recognize_google_speech(audio)
        if prompt:
            print("Recognized:", prompt)
            assistant.answer(prompt, webcam_stream.read(encode=True))
    except Exception as e:
        print(f"Audio processing error: {e}")

# Start listening
recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# Main loop
try:
    while True:
        cv2.imshow("Webcam", webcam_stream.read())
        if cv2.waitKey(1) in [27, ord('q')]:
            break
finally:
    webcam_stream.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
