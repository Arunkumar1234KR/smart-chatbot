import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ArunkumarBot"
context = {"last_tag": None}  # Better context tracking


def get_response(msg):
    sentence = tokenize(msg.lower())  # Case-insensitive
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Confidence threshold
    CONF_THRESHOLD = 0.75  

    if prob.item() >= CONF_THRESHOLD:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Context-aware follow-ups
                if tag == "yes_followup" and context["last_tag"] in ["education", "internship", "certifications", "projects_overview"]:
                    return random.choice(intent['responses'])
                # Save last tag for follow-up handling
                context["last_tag"] = tag
                return random.choice(intent['responses'])

    # If uncertain, return fallback
    return random.choice([resp for resp in intents['intents'] if resp["tag"] == "fallback"][0]["responses"])


if __name__ == "__main__":
    print("Let's chat with ArunkumarBot! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}") 
