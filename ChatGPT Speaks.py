import os
import requests
import pyaudio
import wave
import openai
import soundfile as sf
import sounddevice as sd

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
AudioSegment.ffmpeg = "C:/Users/waxxx/OneDrive/Documents/Development/Projects/LearningVSCode/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/Users/waxxx/OneDrive/Documents/Development/Projects/LearningVSCode/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe"
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the .env file
credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

chatgpt_api_key = "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"

def record_audio(file_name, duration=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = file_name

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



def transcribe_audio(file_name):
    client = speech.SpeechClient()

    with open(file_name, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        return result.alternatives[0].transcript

    return None

def chat_with_gpt(prompt, api_key):
    openai.api_key = api_key
    
    model_engine = "text-davinci-002"  # You can choose another model if you prefer
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=150,  # You can adjust this value based on your requirements
        n=1,
        stop=None,
        temperature=0.3,
    )
    
    if response and response.choices:
        return response.choices[0].text.strip()
    else:
        return None

def text_to_speech(text, voice_name='en-AU-Neural2-C'):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-au',
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print("Audio content written to file 'output.mp3'")

    return "output.mp3"

def play_audio(file_name):
    # Open the MP3 file using soundfile
    audio_file = 'C:/Users/waxxx/OneDrive/Documents/Development/Projects/LearningVSCode/output.mp3'
    audio, sample_rate = sf.read(audio_file)

    # Play the audio using sounddevice
    sd.play(audio, sample_rate)
    sd.wait()



audio_file = "audio.wav"
record_audio(audio_file)
transcription = transcribe_audio(audio_file)

if transcription:
    print(f"Transcribed Text: {transcription}")
    response = chat_with_gpt(transcription, chatgpt_api_key)
    if response:
        print(f"ChatGPT Response: {response}")
        output_file = text_to_speech(response, voice_name="en-AU-Neural2-C")
        
        play_audio( 'C:/Users/waxxx/OneDrive/Documents/Development/Projects/LearningVSCode/output.mp3')
else:
    print("Couldn't transcribe the audio.")