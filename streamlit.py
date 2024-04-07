# voice_chatbot.py
import streamlit as st
import pyttsx3
import speech_recognition as sr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit_webrtc as webrtc
from streamlit_webrtc import AudioProcessorBase
def get_openai_response(question):
    
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
# Initialize the speech recognition engine
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define the wake word
WAKE_WORD = "chatbot"
STOP_WORD = "stop"
# Set Streamlit app title
st.title("Voice Chatbot")

# Record audio and convert to text
def recognize_speech():
    with sr.Microphone() as source:
        st.write("Listening for a question...")
        audio = recognizer.listen(source)
    try:
        question = recognizer.recognize_google(audio)
        st.write(f"Question: {question}")
        return question
    except sr.UnknownValueError:
        st.write("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.write(f"Error accessing Google Speech Recognition service: {e}")
        return None


#Main streamlit app
if __name__ == "__main__":
    while True:
        text = recognize_speech()
        if text and WAKE_WORD in text:
            st.write("Activated. What's your question?")
            question = recognize_speech()
            if question:
                response = get_openai_response(question)  # Replace with your actual model
                st.write(f"Response: {response}")

                # Convert response to speech
                engine.say(response)
                engine.runAndWait()
                break
                
'''# Function to stop the response
def stop_response():
    engine.stop()

# Main Streamlit app
if __name__ == "__main__":
    while True:
        text = recognize_speech()
        if text and WAKE_WORD in text:
            st.write("Activated. What's your question?")
            question = recognize_speech()
            if question:
                if STOP_WORD in question:
                    stop_response()
                    st.write("Response stopped by user.")
                    break
                else:
                    response = get_openai_response(question)
                    st.write(f"Response: {response}")

                    # Convert response to speech
                    engine.say(response)
                    engine.runAndWait()
                    break'''


'''user_question=recognize_speech()
    if user_question:
        response = get_openai_response(user_question)
        st.write(f"Response: {response}")
        # Convert response to speech
        engine.say(response)
        engine.runAndWait()'''
