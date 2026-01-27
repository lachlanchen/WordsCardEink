import os
import re
import json
from pprint import pprint

from openai import OpenAI

from env_loader import load_env

load_env()

client = OpenAI(
    organization=os.environ.get("OPENAI_ORG_ID"),
)
# print(*client.models.list(), sep="\n")

# keep trying 
# Step 1: Request to generate 50 random advanced words
num_words = 10
words_request = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an assistant with a vast vocabulary and creativity."},
        {"role": "user", "content": f"Think wildly and provide me with a python list of {num_words} unique advanced words that are often used in formal readings. Give me only the plain python list compatible with json.loads and start with [ and end with ] WITHOUT any other words. "}
  ]
)



words_response = words_request.choices[0].message.content
print(words_response)

# untill parsable
words_list = json.loads(words_response)
len(words_list)

unique_words = list(set(filter(None, words_list)))

unique_words



# keep trying 
# Example words and their details
examples = [
    {
        "word": "abstraction",
        "syllable_word": "ab·strac·tion",
        "phonetic": "ˈæb·stræk·ʃən",
        "japanese_synonym": "抽象（ちゅうしょう）"
    },
    {
        "word": "paradox",
        "syllable_word": "par·a·dox",
        "phonetic": "ˈpær·ə·dɒks",
        "japanese_synonym": "逆説（ぎゃくせつ）"
    }
]

# Formatting examples into the desired string format
formatted_examples = ', '.join([f"'{{\"word\": \"{example['word']}\", \"syllable_word\": \"{example['syllable_word']}\", \"phonetic\": \"{example['phonetic']}\", \"japanese_synonym\": \"{example['japanese_synonym']}\"}}'" for example in examples[1:2]])

# Step 3: Format the message for the detailed list request
detailed_list_message = (
    "I need each word to come with its syllable_word (separated by central dots), "
    "phonetic transcription (also separated by central dots), and a Japanese synonym. "
    "The Japanese synonym should have its Chinese characters followed by the hiragana reading in parentheses. "
    "Ensure to use central dots for both syllable and phonetic separation, and correctly place hiragana in the Japanese synonyms. "
    "The output should be like [" + formatted_examples + "]. "
    "This is the list of words: " + ', '.join(unique_words[:5]) + ". " # make it random to choose 5 words
)

# Step 2: Request to generate the detailed list using the words from Step 1
detailed_list_request = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words."},
    {"role": "user", "content": detailed_list_message}
  ]
)


print(detailed_list_request.choices[0].message.content)
# untill this is parsable
word_phonetics = json.loads(detailed_list_request.choices[0].message.content)

print(word_phonetics)


# random choose two word to update the example
# save the word_phonetics
