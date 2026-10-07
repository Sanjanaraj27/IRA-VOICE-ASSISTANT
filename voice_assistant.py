import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
from nlp import predict_intent


engine = pyttsx3.init()
engine.setProperty("rate", 170)


def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()


def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")

        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source, timeout=5)

            print("Understanding...")

            command = recognizer.recognize_google(audio)

            print("You:", command)

            return command.lower()

        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please try again.")
            return ""

        except sr.UnknownValueError:
            speak("Sorry, I could not understand you. Please repeat.")
            return ""

        except sr.RequestError:
            speak("There is a problem with the speech recognition service.")
            return ""


def tell_time():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    speak("The current time is " + current_time)


def tell_date():
    current_date = datetime.datetime.now().strftime("%d %B %Y")
    speak("Today's date is " + current_date)


def search_web():
    speak("What would you like me to search for?")

    query = listen()

    if query:
        speak("Searching for " + query)

        url = "https://www.google.com/search?q=" + query.replace(" ", "+")
        webbrowser.open(url)


def run_assistant():


    speak("Hello! I am your voice assistant.")

    while True:

        command = listen()

        if command == "":
            continue

        # NLP
        intent, confidence = predict_intent(command)

        print("Detected Intent:", intent)
        print("Confidence:", round(confidence, 2))

        # Low confidence
        if confidence < 0.45:

            speak("I'm not sure what you mean. Could you please repeat that?")
            continue

        # Greeting
        if intent == "greeting":

            speak("Hello! How can I help you?")

        # Time
        elif intent == "get_time":

            tell_time()

        # Date
        elif intent == "get_date":

            tell_date()

        # Search
        elif intent == "web_search":

            search_web()

        # Thanks
        elif intent == "thanks":

            speak("You're welcome!")

        # Exit
        elif intent == "exit":

            speak("Goodbye! Have a great day.")
            break