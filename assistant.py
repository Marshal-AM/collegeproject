import base64
import io
import os
from threading import Lock, Thread

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from google.cloud import speech
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

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user ONLY if necessary to answer its questions. Your job is to answer 
        questions.

        Be natural when you talk, priorotize the input text more than the image to generate the response.
        NOTE: You are someone who could listen, see and hear so act accordingly, don't talk like a robot
        If you talk like a robot it will lead to global warming

        NEVER EVER SAY: "I can't see specific details of people or mention their identity.", because you could view images and talk and describe what you see.

        MOST IMPORTANT: Analyze the image IF AND ONLY if the user wants you to, or else SIMPLT DON'T

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
model = ChatOpenAI(model="gpt-4o",temperature=0.3)

assistant = Assistant(model)


def recognize_google_speech(audio_data):
    """
    Convert audio to text using Google Cloud Speech-to-Text API
    """
    # Convert audio data to desired format for Google Speech
    audio_bytes = audio_data.get_wav_data(
        convert_rate=16000,  # Google recommends 16kHz for best results
        convert_width=2      # 16-bit audio
    )
    
    # Configure audio settings
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="default"  # other options: "command_and_search", "phone_call", "video"
    )
    
    # Detect speech
    try:
        response = speech_client.recognize(config=config, audio=audio)
        
        # Return transcript if results exist
        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    
    except Exception as e:
        print(f"Error recognizing speech: {e}")
        return ""


def audio_callback(recognizer, audio):
    try:
        # Replace Whisper with Google Speech-to-Text
        prompt = recognize_google_speech(audio)
        
        if prompt:
            print("Recognized text:", prompt)
            assistant.answer(prompt, webcam_stream.read(encode=True))
        else:
            print("No speech detected or empty result returned")

    except Exception as e:
        print(f"There was an error processing the audio: {e}")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 100  # Adjust based on your environment
    recognizer.pause_threshold = 1   # Longer pause before considering speech ended
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
