import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "DuckyBlender/racist-phi3", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "DuckyBlender/racist-phi3",
    torch_dtype="float16",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def ask_racistphi(prompt):
    inputs = tokenizer(prompt, return_tensors="pt",
                       padding=True, truncation=True)

    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            speak("speak up you idiot")
            return None
        except sr.RequestError:
            speak("speak up you idiot")
            return None
        except Exception as e:
            speak("ur code broke bozo")
            print(f"Error: {e}")
            return None


speak("Hi! I am the racist dinosaur, How is your day going?")
while True:
    user_input = listen()
    if user_input:
        if "bye" in user_input.lower():
            speak("Bye bye.")
            break
        else:
            response = ask_racistphi(user_input)
            print(f"Model says: {response}")
            speak(response)
