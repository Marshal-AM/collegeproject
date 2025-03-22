# Voice Assistant with Google Cloud Speech-to-Text

This application is a voice assistant that uses:
- Google Cloud Speech-to-Text for speech recognition
- GPT-4o for natural language understanding and response generation
- OpenAI's TTS-1 for voice synthesis

## Setup Instructions

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Set up Google Cloud Speech-to-Text API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the Speech-to-Text API for your project
4. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name and description
   - Grant the "Speech-to-Text User" role
   - Create a JSON key and download it
5. Place the downloaded JSON key file in your project directory
6. Set the environment variable in your .env file:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your-credentials-file.json
   ```

### 3. Set up OpenAI API credentials
Add your OpenAI API key to your .env file:
```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Application
```
python assistant.py
```

## Usage
- Start the application and allow webcam access
- Speak naturally after the application starts
- The assistant will process your speech, respond textually, and speak back to you
- Press 'q' or ESC to exit

## Configuration
You can adjust the following parameters in the code:
- Speech recognition settings in the `recognize_google_speech` function
- Speech energy threshold and pause threshold in the microphone setup
- Assistant personality in the system prompt