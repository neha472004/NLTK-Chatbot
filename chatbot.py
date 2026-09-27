import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Chatbot knowledge base
corpus = [
    "Hello! How can I help you?",
    "Hi! Nice to meet you.",
    "What is your name?",
    "I am a Python chatbot.",
    "I am a chatbot built using Python and NLTK.",
    "Python is a popular programming language.",
    "Python is used for web development, AI, data science and automation.",
    "Tell me a joke.",
    "Why did the programmer quit his job? Because he didn't get arrays!",
    "What can you do?",
    "I can answer basic questions using NLP and text similarity.",
    "Thank you.",
    "You're welcome!",
    "Goodbye!"
]

lemmer = nltk.stem.WordNetLemmatizer()


def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text.lower())

    return [
        lemmer.lemmatize(token)
        for token in tokens
        if token not in string.punctuation
    ]


def respond(user_input):
    data = corpus + [user_input]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_lemmatize,
        token_pattern=None
    )

    matrix = vectorizer.fit_transform(data)

    similarity = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    )

    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.15:
        return "Sorry, I don't understand that yet."

    return corpus[index]


print("=" * 50)
print("       Python NLTK Chatbot")
print("=" * 50)
print("ChatBot: Ask me something!")
print("ChatBot: Type 'bye' to exit.")
print()

while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "bye":
        print("ChatBot: Goodbye! Have a great day!")
        break

    if not user_input:
        print("ChatBot: Please type something.")
        continue

    print("ChatBot:", respond(user_input))