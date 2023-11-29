import pandas as pd
import nltk
nltk.download('cmudict')

from nltk.corpus import cmudict
import eng_to_ipa as ipa

# List of commonly used English words
words = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "as", "was", "with", "be", "by", "on", "not", "he", "I", "this",
    "are", "or", "his", "from", "at", "which", "but", "have", "an", "had",
    "they", "you", "were", "their", "one", "all", "we", "can", "her", "has",
    "there", "been", "if", "more", "when", "will", "no", "out", "do", "so",
    "up", "what", "about", "who", "get", "which", "go", "me", "make", "can",
    "like", "time", "just", "him", "know", "take", "people", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way", "even"
]

# Create a DataFrame to store the data
df = pd.DataFrame(columns=["Word", "Phonetic Symbols", "Japanese Synonym", "English Sentence", "Japanese Sentence"])

# Populate the DataFrame with the words
df["Word"] = words

# df.head()  # Display the first few rows to check the DataFrame structure


# Initialize the CMU Pronouncing Dictionary
d = cmudict.dict()

# Function to get phonetic symbols with dot for syllable separation
def get_phonetic_symbols(word):
    try:
        phonetic = ipa.convert(word)
        # Replace spaces with dots for syllable separation
        phonetic = phonetic.replace(' ', '·')
        return phonetic
    except Exception as e:
        return "N/A"

# Function to get Japanese synonyms (placeholder function)
def get_japanese_synonym(word):
    # This is a placeholder function. In a real application, this should be replaced with a reliable translation method
    return "日本語同義語"

# Function to create English and Japanese sentences (placeholder function)
def create_sentences(word):
    # Placeholder sentences
    english_sentence = f"This is a sentence with the word {word}."
    japanese_sentence = f"これは「{word}」という言葉を含む文です。"
    return english_sentence, japanese_sentence

# Apply the functions to the first 10 words
for index, row in df.head(10).iterrows():
    word = row['Word']
    df.at[index, 'Phonetic Symbols'] = get_phonetic_symbols(word)
    df.at[index, 'Japanese Synonym'] = get_japanese_synonym(word)
    english_sentence, japanese_sentence = create_sentences(word)
    df.at[index, 'English Sentence'] = english_sentence
    df.at[index, 'Japanese Sentence'] = japanese_sentence

# df.head(10)  # Display the first 10 rows with the completed information

print(df.head(10))