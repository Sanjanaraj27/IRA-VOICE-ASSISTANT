from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Training data
training_sentences = [

    # GREETING
    "hello",
    "hi",
    "hey",
    "hey assistant",
    "hello assistant",
    "good morning",
    "good evening",
    "how are you",
    "how are you doing",
    "nice to meet you",

    # GET TIME
    "what time is it",
    "tell me the time",
    "what is the current time",
    "can you tell me the time",
    "do you know what time it is",
    "what time is it right now",
    "could you tell me the current time",
    "I want to know the time",
    "please tell me what time it is",
    "how late is it",

    # GET DATE
    "what is today's date",
    "tell me today's date",
    "what date is it",
    "what is the date today",
    "can you tell me today's date",
    "I want to know today's date",
    "what day is today",
    "tell me the current date",

    # WEB SEARCH
    "search for something",
    "search the web",
    "google something",
    "look this up",
    "find information about something",
    "search for information",
    "can you search for me",
    "I want to search something",
    "find this on Google",

    # THANKS
    "thank you",
    "thanks",
    "thanks a lot",
    "thank you so much",
    "I appreciate it",

    # EXIT
    "bye",
    "goodbye",
    "exit",
    "quit",
    "stop",
    "close the assistant",
    "I want to leave",
    "see you later"
]


training_labels = [

    # GREETING - 10
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",

    # TIME - 10
    "get_time",
    "get_time",
    "get_time",
    "get_time",
    "get_time",
    "get_time",
    "get_time",
    "get_time",
    "get_time",
    "get_time",

    # DATE - 8
    "get_date",
    "get_date",
    "get_date",
    "get_date",
    "get_date",
    "get_date",
    "get_date",
    "get_date",

    # SEARCH - 9
    "web_search",
    "web_search",
    "web_search",
    "web_search",
    "web_search",
    "web_search",
    "web_search",
    "web_search",
    "web_search",

    # THANKS - 5
    "thanks",
    "thanks",
    "thanks",
    "thanks",
    "thanks",

    # EXIT - 8
    "exit",
    "exit",
    "exit",
    "exit",
    "exit",
    "exit",
    "exit",
    "exit"
]


# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(training_sentences)


# Create NLP classifier
model = LogisticRegression()

# Train the model
model.fit(X, training_labels)


def predict_intent(sentence):

    # Convert new sentence into TF-IDF features
    sentence_vector = vectorizer.transform([sentence])

    # Predict intent
    intent = model.predict(sentence_vector)[0]

    # Calculate confidence
    confidence = max(
        model.predict_proba(sentence_vector)[0]
    )

    return intent, confidence
