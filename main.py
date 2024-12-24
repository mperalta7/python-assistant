import pyttsx3
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
import queue
from llama_cpp import Llama
import os

#Vicuna Model
model_path = os.environ.get("LLM_MODEL_PATH")  # Path for your LLM Model
llm = Llama(model_path=model_path,n_ctx=4096)

#Vosk STT
q = queue.Queue()
stt_model_path = os.environ.get("STT_MODEL_PATH")
model = Model(stt_model_path)  # Path for your STT model
recognizer = KaldiRecognizer(model, 16000)

def chat_bot(user_prompt):
    print(f"\nBot received prompt: {user_prompt}")
    prompt = f"Instructions: You are a friendly and helpful assistant that provides factual answers. {user_prompt} \n Answer:"
    response = llm(prompt, max_tokens=150, temperature=0.9, top_p=.9, )
    answer = response['choices'][0]['text']
    print(f"Bot answer:{answer}")
    text_to_speech(answer)


# Speech-to-Text Callback Function
def user_speech_to_text():
    trigger_word = "command"  # Every word said after this is used for your prompt
    exit_words = ["exit","close","end"] # Say this after trigger word to exit program (ex. command end)
    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        q.put(bytes(indata))

    stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback)
    stream.start()
    try:
        print("Listening... Press Ctrl+C to stop.")
        while True:
            try:
                data = q.get(timeout=1)
                if recognizer.AcceptWaveform(data):
                    user_phrase = json.loads(recognizer.Result())["text"]
                    word_search_index = str(user_phrase).lower().find(trigger_word)
                    print(f"User spoke: {user_phrase}")
                    if word_search_index != -1:  # If the trigger word is used
                        prompt = user_phrase[word_search_index + len(trigger_word):].strip()
                        if prompt.strip().lower() in exit_words:
                            exit()
                        stream.stop()  # Stop the microphone
                        chat_bot(prompt)
                else:
                    print(json.loads(recognizer.PartialResult())["partial"], end="\r")
            except queue.Empty:
                pass  # Ignore empty queue errors
    finally:
        stream.stop()
        stream.close()

# Text-to-Speech Function
def text_to_speech(text):
    if text:
        engine = pyttsx3.init()
        engine.setProperty("voice", engine.getProperty("voices")[1].id)
        engine.say(text)
        engine.runAndWait()
        user_speech_to_text()

user_speech_to_text()