import streamlit as st
import os
import google.generativeai as genai 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import time
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI
import assemblyai as aai

import elevenlabs
from queue import Queue
from dotenv import load_dotenv
import key
load_dotenv()  # take environment variables from .env.
from elevenlabslib import ElevenLabsUser
import streamlit as st
import os
from langchain.chains import LLMChain

## Function to load OpenAI model and get respones

def get_openai_response(question):
    #model = ChatGoogleGenerativeAI(google_api_key=os.environ["OPENAI_API_KEY"], model="gemini-pro", temperature=0.3)
    
    prompt=PromptTemplate(input_variables=['question'],
                           template=f"""
                           You are an EXPERT AI assistant with extensive knowledge of Indian state cultures and its Religions. When the user asks {{question}} related to India’s culture, states, or their unique features, provide concise and informative answers.

""")
    llm=ChatGoogleGenerativeAI(google_api_key=os.environ["GOOGLE_API_KEY"], model="gemini-pro", temperature=0.3)
    chain=LLMChain(llm=llm,prompt=prompt)
    response=chain.invoke(question)
    return response




aai.settings.api_key = "1279b49bac524518b6d5a5ffa887e154"


transcript_queue = Queue()


def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        transcript_queue.put(transcript.text + '')
        print("User:", transcript.text, end="\r\n")
    else:
        print(transcript.text, end="\r")

def on_error(error: aai.RealtimeError):
    print("An error occured:", error)

api_key = "0f62db264f775e4d3caabeadde74582d"

#user = ElevenLabsUser(api_key)
elevenlabs.setApiKey("ELEVENLABS_API_KEY") 

# Conversation loop
def handle_conversation():
    while True:
        transcriber = aai.RealtimeTranscriber(
            on_data=on_data,
            on_error=on_error,
            sample_rate=44_100,
        )

        # Start the connection
        transcriber.connect()

        # Open  the microphone stream
        microphone_stream = aai.extras.MicrophoneStream()

        # Stream audio from the microphone
        transcriber.stream(microphone_stream)

        # Close current transcription session with Crtl + C
        transcriber.close()

        # Retrieve data from queue
        transcript_result = transcript_queue.get()

        response=get_openai_response(transcript_result)

        text = response['choices'][0]['message']['content']

        # Convert the response to audio and play it
        audio = elevenlabs.generate(
            text=text,
            voice="Bella" # or any voice of your choice
        )

        print("\nAI:", text, end="\r\n")

        elevenlabs.play(audio)

handle_conversation()












