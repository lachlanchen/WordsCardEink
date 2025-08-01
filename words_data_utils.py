"""
Eink Words GPT Project
----------------------

Project Name: Eink Words GPT
Author: Lachlan CHEN
Website: https://lazying.art
GitHub: https://github.com/lachlanchen/

Description:
The Eink Words GPT project integrates the cutting-edge e-ink technology with the power of OpenAI's GPT models. 
Designed and developed by Lachlan CHEN, this project represents a unique and innovative approach to word learning. 
It features a dynamic word display system using a Raspberry Pi 5 and a Waveshare 7-color 7.3-inch e-ink display, 
selecting interesting and relevant words dynamically from OpenAI. This system is a part of the 'Art of Lazying' theme, 
reflecting a philosophy of efficient and enjoyable learning. The Eink Words GPT project is open-source, inviting 
contributions from the community to further enrich this learning experience.

"""


import json
import json5
import random
from openai import OpenAI

import sqlite3
import os

from datetime import datetime
import pytz
import csv
from pykakasi import kakasi
import re
import nltk
from nltk.corpus import words
from pprint import pprint
import traceback
import numpy as np
import pandas as pd
import opencc

# from phonetic_checker import PhoneticRechecker




class JSONParsingError(Exception):
    """Exception raised for errors in the JSON parsing."""
    def __init__(self, error_message, json_string, error_pos=None):
        self.error_message = error_message
        self.json_string = json_string.lower()
        self.error_pos = error_pos
        self.error_details = f"{self.error_message}\nError Position: {self.error_pos}" if self.error_pos else f"{self.error_message}"
        self.message = f"JSON String: {self.json_string}\n{self.error_details}"
        super().__init__(self.message)

class NotEnoughUniqueWordsError(Exception):
    """Exception raised when not enough unique words are fetched."""
    def __init__(self, required_num, fetched_num, unique_words, duplicated_words, json_string):
        self.required_num = required_num
        self.fetched_num = fetched_num
        self.unique_words = [word.lower() for word in unique_words]
        self.duplicated_words = duplicated_words
        self.json_string = json_string.lower()
        self.error_details = (
            f"Error: Required {self.required_num} unique words, "
            f"but only {self.fetched_num} non-duplicated words comparing to my local database were fetched. "
            # f"Duplicated words in local database: {', '.join(self.duplicated_words)}."
            f"These non-duplicated words I am yearning for are: "
            f"{', '.join(self.unique_words)}. "
        )
        self.message = f"JSON String: {self.json_string}\n{self.error_details}"
        super().__init__(self.message)


def random_shuffle(input_list):
    # Convert the input list to a NumPy array
    array = np.array(input_list, dtype=object)

    # Shuffle the array in-place
    np.random.shuffle(array)

    # Convert the shuffled array back to a list
    shuffled_list = array.tolist()

    return shuffled_list

def random_sample(input_list, sample_size):
    # Ensure the sample size is not larger than the list size
    sample_size = min(sample_size, len(input_list))

    # Convert the input list to a NumPy array if it's not already one
    if not isinstance(input_list, np.ndarray):
        input_array = np.array(input_list)
    else:
        input_array = input_list

    # Randomly sample elements without replacement
    sampled_elements = np.random.choice(input_array, size=sample_size, replace=False)

    return list(sampled_elements)

# Function to count syllables based on dots
def count_syllables(word):
    # Count dots and stress symbols, subtract one if the first character is a stress symbol
    count = word.count('·') + word.count('ˈ') + word.count('ˌ')
    if word.startswith('ˈ') or word.startswith('ˌ'):
        count -= 1
    return count + 1



def clean_english(text):
    return text.replace(".", "·").replace(" · ", " ").replace("·ˈ", "ˈ").replace("·ˌ", "ˌ").replace("ˈ·", "ˈ").replace("ˌ·", "ˌ")#.replace(" ", "")

def clean_japanese(text):
    return text.replace(".", "").replace("·", "").replace("(", "（").replace(")", "）").replace(" ", "")




def split_word(text):
    # Split the text into words by space
    words = text.split(' ')
    
    all_syllables = []
    for word in words:
        # Clean each word
        cleaned_word = clean_english(word)

        # Apply the existing logic to each word
        if cleaned_word.startswith('ˈ') or cleaned_word.startswith('ˌ'):
            syllable_word = cleaned_word[0] + cleaned_word[1:].replace('ˈ', '·ˈ').replace('ˌ', '·ˌ').replace(" ", " ·")
        else:
            syllable_word = cleaned_word.replace('ˈ', '·ˈ').replace('ˌ', '·ˌ').replace("-", "·-").replace(" ", " ·")

        # Split into syllables
        syllables = syllable_word.split('·')
        syllables[-1] = syllables[-1] + " "
        all_syllables.extend(syllables)

    return all_syllables


# Function to split words into syllables and get color for each syllable
def split_word_with_color(word, colors):
        # Replace stress symbols with a preceding dot, except at the beginning

        syllables = split_word(word)
        
        color_syllables = [(syllable, colors[i % len(colors)]) for i, syllable in enumerate(syllables)]
        return color_syllables




def extract_kanji(japanese_text):
    """
    Remove parentheses, hiragana, and katakana from Japanese text, leaving only kanji.
    """
    # Regex to match hiragana, katakana, and characters in parentheses
    regex = u"[\u3040-\u309F\u30A0-\u30FF]|[（(].*?[)）]"
    kanji = re.sub(regex, '', japanese_text)
    if kanji.startswith("、"):
        kanji = kanji[1:]
    return kanji

def remove_second_parentheses(text):
    regex = re.compile(r'(（[^）]*）)(（[^）]*）)')
    return re.sub(regex, lambda match: match.group(1), text)


# Function to remove text inside parentheses
def remove_text_including_parentheses(text):
    while '（' in text and '）' in text:
        start = text.find('（')
        end = text.find('）') + 1
        text = text[:start] + text[end:]
    return text


def remove_text_inside_parentheses(text):
    new_text = ""
    in_parentheses = False

    for char in text:
        if char == '（':
            in_parentheses = True
            new_text += char
        elif char == '）' and in_parentheses:
            in_parentheses = False
            new_text += char
        elif not in_parentheses:
            new_text += char

    return new_text


def remove_content_inside_parentheses(text):
    # This regex matches anything inside parentheses and removes it, including the parentheses
    return re.sub(r'（[^）]*）', '', text)


# pattern = r'[ぁ-んァ-ンヴゔーヵヶㇰ-ㇿガ-ドヰヱヵヶ]'
# [一-龠]+|[ぁ-ゔ]+|[ァ-ヴー]+|[a-zA-Z0-9]+|[ａ-ｚＡ-Ｚ０-９]+|[々〆〤ヶ]+

def remove_japanese_letter_including_parentheses(text):
    # This regex matches hiragana or katakana inside parentheses and removes them, keeping the parentheses
    return re.sub(r'（[ぁ-ゔァ-ヴガ-ドㇰ-ㇿヵヶ々ヰヱ〆〤ー\-]+）', '', text)
    # return re.sub(r'(?<=（)[ぁ-ゔ々ー\-]+(?=）)', '', text)


def remove_japanese_letter_inside_parentheses(text):
    # This regex matches hiragana or katakana inside parentheses and removes them, keeping the parentheses
    return re.sub(r'(?<=（)[ぁ-ゔァ-ヴガ-ドㇰ-ㇿヵヶ々ヰヱ〆〤ー\-]+(?=）)', '', text)
    # return re.sub(r'(?<=（)[ぁ-ゔ々ー\-]+(?=）)', '', text)



def remove_hiragana_including_parentheses(text):
    # Comprehensive regex pattern for Japanese characters
    
    # return re.sub(r'（[ぁ-ゔァ-ヴガ-ドㇰ-ㇿヵヶ々ー\-（）]+）', '', text)
    return re.sub(r'（[ぁ-ゔ々ー\-（）]+）', '', text)
    # return remove_text_including_parentheses(text)


def remove_hiragana_inside_parentheses(text):
    # This regex matches hiragana or katakana inside parentheses and removes them, keeping the parentheses

    # return re.sub(r'(?<=（)[ぁ-ゔァ-ヴガ-ドㇰ-ㇿヵヶ々ー\-（）]+(?=）)', '', text)
    return re.sub(r'(?<=（)[ぁ-ゔ々ー\-（）]+(?=）)', '', text)
    # return remove_content_inside_parentheses(text)


def remove_hiragana_and_parentheses(text):
    """
    Remove all Hiragana, related letters, and parentheses from the text.

    Parameters:
    text (str): The input string from which Hiragana and parentheses will be removed.

    Returns:
    str: The text with Hiragana and parentheses removed.
    """
    # Regex pattern to remove Hiragana, related letters, and full-width parentheses
    pattern = r'[（）ぁ-ゔ-゚〜ー\-]'

    return re.sub(pattern, '', text)



def transcribe_japanese(text):
    from pykakasi import kakasi

    text = remove_hiragana_including_parentheses(text)

    kks = kakasi()
    kks.setMode("J", "H")  # Japanese to Hiragana
    kks.setMode("K", "H")  # Katakana to Hiragana
    conv = kks.getConverter()

    result = ""
    current_chunk = ""
    last_kanji_hiragana = ""
    is_kanji = False
    is_katakana = False

    for char in text:
        if '\u4E00' <= char <= '\u9FFF':  # Kanji
            if is_katakana:  # Close katakana chunk if open
                result += f"({current_chunk})"
                current_chunk = ""
                is_katakana = False

            hiragana = conv.do(char)
            last_kanji_hiragana = hiragana  # Store the hiragana of the current kanji
            if not is_kanji:
                is_kanji = True
                current_chunk = ""
            current_chunk += hiragana
            result += char
        elif char == '々':  # Ideographic Iteration Mark
            if is_katakana:  # Close katakana chunk if open
                result += f"({current_chunk})"
                current_chunk = ""
                is_katakana = False

            if not is_kanji:
                is_kanji = True
                current_chunk = ""
            current_chunk += last_kanji_hiragana
            result += char
        elif '\u30A0' <= char <= '\u30FF':  # Katakana
            if is_kanji:  # Close kanji chunk if open
                result += f"({current_chunk})"
                current_chunk = ""
                is_kanji = False

            hiragana = conv.do(char)
            if not is_katakana:
                is_katakana = True
                current_chunk = ""
            current_chunk += hiragana
            result += char
        else:  # Hiragana or others
            if is_kanji or is_katakana:
                result += f"({current_chunk}){char}"
                is_kanji = False
                is_katakana = False
            else:
                result += char

    if is_kanji or is_katakana:  # Remaining kanji or katakana chunk at the end
        result += f"({current_chunk})"


    return clean_japanese(result)


def count_hiragana_repetitions(japanese_string):
    # Regex pattern to find kanji followed by parentheses containing hiragana
    # pattern = r'([一-龠]+)\（([ぁ-んァ-ン]+)\）'
    pattern = r'([一-龠ァ-ヴガ-ドㇰ-ㇿヵヶヰヱ々〆〤ー\-]+)（([ぁ-ゔー\-]+)）'
    
    results = []
    prev_end = 0
    for match in re.finditer(pattern, japanese_string):
        kanji, hiragana_in_parentheses = match.groups()
        kanji_start, _ = match.span(1)

        # Extracting the preceding hiragana substring
        preceding_hiragana = japanese_string[prev_end:kanji_start]

        # Counting the overlap of hiragana before kanji and hiragana inside the parenthesis
        overlap_count = 0
        for i in range(1, min(len(preceding_hiragana), len(hiragana_in_parentheses)) + 1):
            if preceding_hiragana[-i:] == hiragana_in_parentheses[:i]:
                overlap_count = i

        results.append((kanji, overlap_count))

        prev_end = kanji_start + len(kanji)

    return results

def smallest_non_zero_repetition(japanese_string):
    """
    Find the smallest non-zero repetition count in the results.

    :param repetition_results: List of tuples with kanji and repetition count.
    :return: The smallest non-zero repetition count, or None if all are zero.
    """
    repetition_results = count_hiragana_repetitions(japanese_string)

    print("repetition_results: ", repetition_results)


    smallest_non_zero = None
    for _, count in repetition_results:
        if count > 0 and (smallest_non_zero is None or count < smallest_non_zero):
            smallest_non_zero = count
    return 0 if smallest_non_zero is None else smallest_non_zero

def compare_repetition_results(string1, string2):
    """
    Compares two sets of hiragana repetition results and identifies discrepancies.

    :param results1: First set of repetition results.
    :param results2: Second set of repetition results.
    :return: A list of tuples indicating discrepancies. Each tuple contains the kanji and the two different counts.
    """

    # string1 = clean_japanese(string1)
    # string2 = clean_japanese(string2)

    print("string1: ", string1)
    print("string2: ", string2)

    results1 = count_hiragana_repetitions(string1)
    results2 = count_hiragana_repetitions(string2)


    discrepancies = []

    # Convert results to dictionaries for easier comparison
    dict1 = {kanji: count for kanji, count in results1}
    dict2 = {kanji: count for kanji, count in results2}

    # Compare the two dictionaries
    for kanji in set(dict1.keys()).union(dict2.keys()):
        count1 = dict1.get(kanji, None)
        count2 = dict2.get(kanji, None)

        # if count1 != count2:
        if count1 != count2:
            if (not count1 is None) and (not count2 is None):
                if count1 < count2:
                    continue
            discrepancies.append((kanji, count1, count2))

    return discrepancies


def clean_and_transcribe(word_details):
    word_details_new = word_details.copy()

    for word in word_details_new:
        # Update phonetic field if it exists
        if "phonetic" in word:
            word["phonetic"] = clean_english(word["phonetic"])

        # Update syllable_word field if it exists
        if "syllable_word" in word:
            word["syllable_word"] = clean_english(word.get("syllable_word", ""))

        # Update japanese_synonym field if it exists
        if "japanese_synonym" in word:
            # Clean and transcribe japanese_synonym
            # word['japanese_synonym'] = clean_japanese(word['japanese_synonym'])
            # cleaned_synonym = remove_hiragana_including_parentheses(clean_japanese(word["japanese_synonym"]))
            cleaned_synonym = remove_japanese_letter_including_parentheses(clean_japanese(word["japanese_synonym"]))


            # word["japanese_synonym"] = clean_japanese(remove_second_parentheses(transcribe_japanese(cleaned_synonym)))  # Replace with your transcription function
            word["japanese_synonym"] = clean_japanese(transcribe_japanese(cleaned_synonym))  # Replace with your transcription function

    return word_details_new


def clean_word_details(word_details):
    for word in word_details:
        # Update phonetic field if it exists
        if "phonetic" in word:
            word["phonetic"] = clean_english(word.get("phonetic", ""))

        # Update syllable_word field if it exists
        if "syllable_word" in word:
            word["syllable_word"] = clean_english(word.get("syllable_word", ""))

        # Update japanese_synonym field if it exists
        if "japanese_synonym" in word:
            word["japanese_synonym"] = clean_japanese(word["japanese_synonym"])

        

    return word_details




# Example usage
# words_db = WordsDatabase(db_path)
# word_fetcher = AdvancedWordFetcher()
# chooser = OpenAiChooser(words_db, word_fetcher)

# chosen_word = chooser.choose()
# print(chosen_word)

class OpenAiChooser:
    def __init__(self, db, word_fetcher, words_list=None):
        self.db = db
        self.word_fetcher = word_fetcher
        self.original_words_list = words_list
        self.current_words = []
        self.words_iterator = iter([])
        self.enable_openai = False

        # Process the provided words_list only if it's not None
        if self.original_words_list:
            self.process_words_list()

    def process_words_list(self):
        self.current_words = []
        for word in self.original_words_list:
            word_details = self.db.find_word_details(word)
            if not word_details:
                word_details = self.word_fetcher.fetch_word_details([word], self.db)[0]
            self.current_words.append(word_details)
        self.words_iterator = iter(self.current_words)

    def update_words_list(self, words_list):
        self.original_words_list = words_list
        self.process_words_list()
        self.words_iterator = iter(self.current_words)

    def _is_daytime_in_hk(self, start=9, end=22):
        hk_timezone = pytz.timezone('Asia/Hong_Kong')
        hk_time = datetime.now(hk_timezone)
        # if self.enable_openai:
        #     return 9 <= hk_time.hour < 22  # Daytime hours in Hong Kong
        # else:
        #     return False

        print("hk hour: ", hk_time.hour)

        return start <= hk_time.hour < end  # Daytime hours in Hong Kong
        # return True
        # return False

    def fetch_new_words(self):
        # If original_words_list is None, fetch new words dynamically
        if not self.original_words_list:
            if self._is_daytime_in_hk() and self.enable_openai:
                words = self.word_fetcher.fetch_words(
                    1, 
                    self.db, 
                    include_existing=False
                )
                openai_words = self.word_fetcher.fetch_word_details(
                    words, 
                    self.db, 
                    num_words_phonetic=10
                )
                db_words = self.db.fetch_random_words(10)
            else:
                db_words = self.db.fetch_random_words(10)
                # db_words = []
                openai_words = []

            self.current_words = openai_words + db_words
            # random.shuffle(self.current_words)
            self.current_words= random_shuffle(self.current_words)
        else:
            # Repopulate current_words using original_words_list
            self.process_words_list()

        self.words_iterator = iter(self.current_words)

    def choose(self):

        word = None
        try:
            word = next(self.words_iterator)
        except StopIteration:
            # print("StopIteration encountered in choose method.")
            if self.original_words_list:
                # print("Restarting iterator from the beginning.")
                self.words_iterator = iter(self.current_words)
                word = next(self.words_iterator)
            else:
                # print("Fetching new words as original_words_list is None.")
                self.fetch_new_words()
                word = next(self.words_iterator)
        # word = clean_and_transcribe([word])[0]
        return word

    def get_current_words(self):
        return self.current_words


class EmojiWordChooser:
    def __init__(self, csv_file_path='data/words_emoji.csv'):
        self.csv_file_path = csv_file_path
        self.current_words = []
        self.words_iterator = iter([])
        self.load_words_from_csv()

    def load_words_from_csv(self):
        try:
            # Load words from the CSV file
            df = pd.read_csv(self.csv_file_path)
            # Remove leading and trailing spaces from column names
            df.columns = [col.strip() for col in df.columns]
            # Convert DataFrame rows to dictionaries and clean values
            self.current_words = [self.clean_row(row) for row in df.to_dict(orient='records')]
            self.words_iterator = iter(self.current_words)
        except Exception as e:
            print(f"Error loading CSV file: {e}")

    def clean_row(self, row):
        """Strip spaces and quotes from each value in the row, depending on its type."""
        cleaned_row = {}
        for key, value in row.items():
            if isinstance(value, str):
                cleaned_row[key] = value.strip().strip('"')
            else:
                cleaned_row[key] = value
        return cleaned_row

    def choose(self):
        """Returns the next word from the iterator, or raises StopIteration if exhausted."""
        try:
            return next(self.words_iterator)
        except StopIteration:
            raise StopIteration

    def get_current_words(self):
        """Returns the current list of words."""
        return self.current_words