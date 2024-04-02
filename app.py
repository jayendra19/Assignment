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

from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
from langchain.chains import LLMChain
import pyttsx3
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
    return response.get('text', '')


engine = pyttsx3.init()
# Set the speaking rate (adjust as desired)
engine.setProperty('rate', 180)

# Set the volume level (between 0 and 1)
engine.setProperty('volume', 1.0)


'''
# Streamlit app
def main():
    st.title("Data Science AI Assistant")
    
    # Button to start voice recognition
    if st.button("Click to Speak"):
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Capture voice command
        with sr.Microphone() as source:
            st.write("Listening for a question about data science...")
            audio = recognizer.listen(source)

        # Recognize speech using Google Web Speech API
        try:
            question = recognizer.recognize_google(audio)
            st.write(f"Question: {question}")
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
            return
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return

        # Get the response from the AI model
        response = get_openai_response(question)
        st.write(f"Response: {response}")

        # Convert the response to speech
        engine.say(response)


if __name__ == "__main__":
    main()'''




def recognize_speech_and_respond():
    # initialized the recognizer
    recognizer = sr.Recognizer()

    # capturing voice command
    with sr.Microphone() as source:
        print("Listening for a question about India and Its Culture...")
        audio = recognizer.listen(source)

    # Recognize speech using Google Web Speech API
    try:
        question = recognizer.recognize_google(audio)
        print(f"Question: {question}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return

    
    


    # Get the response from the AI model
    response = get_openai_response(question)
    print(f"Response: {response}")

    engine.say(response)
    engine.runAndWait()
    engine.stop()


# Example usage
if __name__ == "__main__":
    recognize_speech_and_respond()

'''
import speech_recognition as sr
from gtts import gTTS
import os
def text_to_speech(text):
    gTTS(text = text, lang = 'en', slow = False)

def recognize_speech_and_respond():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Capture voice command
    with sr.Microphone() as source:
        print("Listening for a question about India and Its Culture...")
        audio = recognizer.listen(source)

    try:
        question = recognizer.recognize_google(audio)
        print(f"Question: {question}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return

    # Get the response from the AI model (replace with your actual response logic)
    response = get_openai_response(question)
    print(f"Response: {response}")

    # Convert the response to speech using gTTS
    text_to_speech(response)


    return response


# Example usage
if __name__ == "__main__":
    recognize_speech_and_respond()

'''

