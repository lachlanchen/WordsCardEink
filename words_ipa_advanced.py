import pandas as pd
import nltk
nltk.download('cmudict')

from nltk.corpus import cmudict
import eng_to_ipa as ipa
import pyphen

# Initialize Pyphen for English syllabification
dic = pyphen.Pyphen(lang='en')

# Function to syllabify words
def syllabify(word):
    return dic.inserted(word)


# Advanced list of English words
words = [
    "analyze", "concept", "context", "derive", "empirical", "framework", "hypothesis", 
    "implement", "infer", "mechanism", "parameter", "protocol", "quantitative", 
    "refine", "simulate", "theoretical", "validate", "variable", "aggregate", "coherent",
    "complement", "constrain", "decentralize", "dichotomy", "disseminate", "fluctuate", 
    "heuristic", "innovate", "integrate", "mediate", "normative", "paradigm", 
    "pertinent", "proliferate", "synthesize", "transcend", "underlying", "viability", 
    "volatility", "ameliorate", "catalyze", "converge", "delineate", "demarcate", 
    "dichotomize", "equivocate", "extrapolate", "facilitate", "galvanize", "harmonize", 
    "ideate", "juxtapose", "legitimize", "marginalize", "nuance", "obfuscate", 
    "permeate", "reconcile", "saturate", "triangulate", "underscore", "venerate", 
    "articulate", "bolster", "corroborate", "depict", "elucidate", "foster", "garner", 
    "hinder", "illuminate", "jostle", "kaleidoscope", "liaise", "mitigate", "narrate", 
    "oscillate", "prolific", "quintessential", "resilient", "stagnate", "thrive", 
    "ubiquitous", "vindicate", "wane", "xenophile", "yield", "zealot"
]



# Create a DataFrame to store the data
df = pd.DataFrame(columns=["Word", "Syllabified", "Phonetic Symbols", "Syllables", "Japanese Synonym", "English Sentence", "Japanese Sentence"])

# Populate the DataFrame with the words
df["Word"] = words

# df.head()  # Display the first few rows to check the DataFrame structure


# Initialize the CMU Pronouncing Dictionary
d = cmudict.dict()

# Function to syllabify words
def syllabify(word):
    return dic.inserted(word).replace("-", "·")

# Function to get phonetic symbols with dot for syllable separation
def get_phonetic_symbols(word):
    try:
        phonetic = ipa.convert(word)
        # Replace spaces with dots for syllable separation
        phonetic = phonetic.replace(' ', '·')
        return phonetic
    except Exception as e:
        return "N/A"

# Function to get phonetic symbols with dot for syllable separation
def get_syllable(word):
    if word.lower() in d:
        phonetics = d[word.lower()][0]  # Take the first pronunciation variant
        phonetic = ' '.join(phonetics)
        phonetic = phonetic.replace('0', '').replace('1', '').replace('2', '')  # Remove stress markers
        phonetic = phonetic.replace(' ', '·')  # Replace spaces with dots for syllable separation
        return phonetic
    else:
        return "N/A"

# Function to get Japanese synonyms (placeholder function)
def get_japanese_synonym(word):
    # Example Japanese synonyms for the first 10 words
    japanese_synonyms = {
        "analyze": "分析（ぶんせき）する",
        "concept": "概念 (がいねん) ",
        "context": "文脈 (ぶんみゃく) ",
        "derive": "導出（どうしゅつ）する",
        "empirical": "経験的(けいけんてき）",
        "framework": "枠組（わくぐ）み",
        "hypothesis": "仮説 (かせつ)",
        "implement": "実装 (じっそう）する",
        "infer": "推測 (すいそく）する",
        "mechanism": "メカニズム（めかにずむ) ）"
        # Add more synonyms for other words
    }
    return japanese_synonyms.get(word, "日本語同義語")

# Function to create English and Japanese sentences (placeholder function)
def create_sentences(word):
    # Placeholder sentences
    english_sentence = f"This is a sentence with the word {word}."
    japanese_sentence = f"これは「{word}」という言葉を含む文です。"
    return english_sentence, japanese_sentence

# Apply the functions to the first 10 words
for index, row in df.iterrows():
    word = row['Word']
    df.at[index, "Syllabified"] = syllabify(word)
    df.at[index, 'Phonetic Symbols'] = get_phonetic_symbols(word)
    df.at[index, 'Syllables'] = get_syllable(word)
    df.at[index, 'Japanese Synonym'] = get_japanese_synonym(word)
    english_sentence, japanese_sentence = create_sentences(word)
    df.at[index, 'English Sentence'] = english_sentence
    df.at[index, 'Japanese Sentence'] = japanese_sentence

# df.head(10)  # Display the first 10 rows with the completed information

print(df.head(10))

df.to_excel("words.xlsx")