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
import opencc




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
    return text.replace(".", "·").replace("·ˈ", "ˈ").replace("·ˌ", "ˌ")#.replace(" ", "")

def clean_japanese(text):
    return text.replace(".", "").replace("·", "").replace("(", "（").replace(")", "）").replace(" ", "")


def split_word(word):
    word = clean_english(word)

    if word.startswith('ˈ') or word.startswith('ˌ'):
        word = word[0] + word[1:].replace('ˈ', '·ˈ').replace('ˌ', '·ˌ')
    else:
        word = word.replace('ˈ', '·ˈ').replace('ˌ', '·ˌ').replace("-", "·-").replace(" ", "· ")

    syllables = word.split('·')

    return syllables


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

# def transcribe_japanese(text):
#     from pykakasi import kakasi

#     text = remove_hiragana_including_parentheses(text)

#     kks = kakasi()
#     kks.setMode("J", "H")  # Japanese to Hiragana
#     kks.setMode("K", "H")  # Katakana to Hiragana
#     conv = kks.getConverter()

#     result = ""
#     current_chunk = ""
#     last_kanji_hiragana = ""
#     is_kanji_or_katakana = False

#     for char in text:
#         if '\u4E00' <= char <= '\u9FFF':  # Kanji
#             hiragana = conv.do(char)
#             last_kanji_hiragana = hiragana  # Store the hiragana of the current kanji
#             if not is_kanji_or_katakana:
#                 is_kanji_or_katakana = True
#                 current_chunk = ""
#             current_chunk += hiragana
#             result += char
#         elif char == '々':  # Ideographic Iteration Mark
#             if not is_kanji_or_katakana:
#                 is_kanji_or_katakana = True
#                 current_chunk = ""
#             current_chunk += last_kanji_hiragana
#             result += char
#         elif '\u30A0' <= char <= '\u30FF':  # Katakana
#             hiragana = conv.do(char)
#             if not is_kanji_or_katakana:
#                 is_kanji_or_katakana = True
#                 current_chunk = ""
#             current_chunk += hiragana
#             result += char
#         else:  # Hiragana or others
#             if is_kanji_or_katakana:
#                 result += f"({current_chunk}){char}"
#                 is_kanji_or_katakana = False
#             else:
#                 result += char

#     if is_kanji_or_katakana:  # Remaining kanji or katakana chunk at the end
#         result += f"({current_chunk})"

#     return result

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

class WordsDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        if os.path.exists(db_path):
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

            self.create_field_if_not_exists("words_phonetics", "kanji_synonym", "TEXT")
            self.create_field_if_not_exists("words_phonetics", "chinese_synonym", "TEXT")
            self.create_field_if_not_exists("words_phonetics", "simplified_chinese_synonym", "TEXT")
            self.create_field_if_not_exists("words_phonetics", "arabic_synonym", "TEXT")
            self.create_field_if_not_exists("words_phonetics", "french_synonym", "TEXT")

    def log_history_update(self, old_new_word_details_pairs, history_csv_path='data/words_update_history.csv'):
        with open(history_csv_path, 'a', newline='', encoding='utf-8') as history_file:
            history_writer = csv.writer(history_file)
            for old_details, new_details in old_new_word_details_pairs:
                for key in old_details.keys():
                    old_value = old_details[key]
                    new_value = new_details[key]
                    if old_value != new_value:
                        history_writer.writerow([key, old_value, "→", new_value])

    # def word_exists(self, word):
    #     if self.conn:
    #         self.cursor.execute("SELECT COUNT(*) FROM words_phonetics WHERE word = ?", (word.strip(),))
    #         return self.cursor.fetchone()[0] > 0
    #     return False

    def word_exists(self, word):
        if self.conn:
            # print("Checking existence: ", word)

            # Prepare the word in lowercase
            word_lower = word.strip().lower()
            
            # Update the SQL query to check for both exact and lowercase match
            # self.cursor.execute("SELECT COUNT(*) FROM words_phonetics WHERE word = ? OR LOWER(word) = ?", (word.strip(), word_lower))
            self.cursor.execute("SELECT COUNT(*) FROM words_phonetics WHERE word = ? OR word = ?", (word.strip(), word_lower))
            
            return self.cursor.fetchone()[0] > 0
        return False


    def create_field_if_not_exists(self, table_name, field_name, field_type, default_value=None):
        """
        Add a new field to a table if it doesn't already exist, with an optional default value.

        Args:
        - table_name (str): Name of the table to alter.
        - field_name (str): Name of the new field to add.
        - field_type (str): Data type of the new field. Common SQLite data types include:
            * INTEGER: Whole numbers. Default: 0.
            * TEXT: Text strings. Default: '' (empty string).
            * REAL: Floating point numbers. Default: 0.0.
            * BLOB: Binary data. Default: NULL.
            * NUMERIC: Flexible type for numbers, dates, booleans. Default: 0 or NULL.
            * DATE: Dates in 'YYYY-MM-DD' format. Default: NULL or specific date like '1970-01-01'.
            * BOOLEAN: Stored as INTEGER 0 (false) or 1 (true). Default: 0.
        - default_value (optional): The default value for the new field. If not provided, SQLite uses NULL.

        Example usage:
        create_field_if_not_exists('my_table', 'new_column', 'TEXT', 'default text')
        """
        if self.conn:
            try:
                # Check if the column exists
                self.cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [row[1] for row in self.cursor.fetchall()]
                if field_name not in columns:
                    default_clause = f"DEFAULT {default_value}" if default_value is not None else ""
                    alter_query = f"ALTER TABLE {table_name} ADD COLUMN {field_name} {field_type} {default_clause};"
                    self.cursor.execute(alter_query)
                    self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")
                traceback.print_exc()

    def delete_column_if_exists(self, table_name, column_name):
        """
        Remove a field from a table if it exists.

        Args:
        - table_name (str): Name of the table to alter.
        - column_name (str): Name of the field to remove.

        Example usage:
        delete_column_if_exists('my_table', 'column_to_remove')
        """
        if self.conn:
            try:
                # Check if the column exists
                self.cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [row[1] for row in self.cursor.fetchall()]
                if column_name in columns:
                    # Create a list of columns to retain (excluding the one to be deleted)
                    columns.remove(column_name)
                    retained_columns = ', '.join(columns)

                    # Create a new table with the same structure minus the column
                    temp_table_name = f"{table_name}_temp"
                    self.cursor.execute(f"CREATE TABLE {temp_table_name} AS SELECT {retained_columns} FROM {table_name};")

                    # Drop the old table
                    self.cursor.execute(f"DROP TABLE {table_name};")

                    # Rename the new table to the old table's name
                    self.cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name};")

                    self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")
                traceback.print_exc()
    


    def insert_word_details(self, word_details, force=False):
        if self.conn:
            # Extracting word details
            word = word_details.get('word', '').lower()
            fields_to_update = ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 
                                'french_synonym', 'chinese_synonym', 'simplified_chinese_synonym', 'kanji_synonym']

            # Cleaning and preparing data for insertion
            data_to_insert = [word]
            columns = ['word']
            placeholders = ['?']
            update_parts = []

            for field in fields_to_update:
                if field in word_details:
                    # value = clean_english(word_details[field]) if field != 'japanese_synonym' else clean_japanese(word_details[field])
                    # Choose the appropriate cleaning function based on the field
                    if field == 'japanese_synonym':
                        value = clean_japanese(word_details[field])
                    elif field in ['syllable_word', 'phonetic']:
                        value = clean_english(word_details[field])
                    # elif field == "word":
                    #     value = word_details[field].lower()
                    else:
                        value = word_details[field]  # No cleaning function for other fields
                    data_to_insert.append(value)
                    columns.append(field)
                    placeholders.append('?')
                    update_parts.append(f"{field} = excluded.{field}")

            try:
                if force:
                    # Dynamically constructing the UPSERT query
                    query = f"""
                        INSERT INTO words_phonetics ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                        ON CONFLICT(word) DO UPDATE SET
                            {', '.join(update_parts)};
                    """
                    self.cursor.execute(query, data_to_insert)
                else:
                    # Insert new record, ignore on duplicate
                    query = f"""
                        INSERT INTO words_phonetics ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)});
                    """
                    self.cursor.execute(query, data_to_insert)

                self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")
                traceback.print_exc()


    def update_word_details(self, word_details, force=False):
        if self.conn:
            word = word_details.get('word', '').lower()  # Ensure the word is in lowercase

            # Prepare data and query for dynamic update
            data_to_update = []
            update_parts = []

            for key in ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 'french_synonym', 'chinese_synonym', 'simplified_chinese_synonym', 'kanji_synonym']:
                if key in word_details:
                    # Choose the appropriate cleaning function based on the field
                    if key == 'japanese_synonym':
                        cleaned_value = clean_japanese(word_details[key])
                    elif key in ['syllable_word', 'phonetic']:
                        cleaned_value = clean_english(word_details[key])
                    # elif key == "word":
                    #     cleaned_value = word_details[key].lower()
                    else:
                        cleaned_value = word_details[key]  # No cleaning function for other fields

                    if cleaned_value:
                        data_to_update.append(cleaned_value)
                        update_parts.append(f"{key} = ?")

            if not update_parts:
                # No data to update
                return

            query = f"UPDATE words_phonetics SET {', '.join(update_parts)} WHERE word = ?"
            data_to_update.append(word)

            try:
                # Execute the query with the values
                self.cursor.execute(query, data_to_update)
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")
                traceback.print_exc()



    def get_total_word_count(self):
        if self.conn:
            self.cursor.execute("SELECT COUNT(*) FROM words_phonetics")
            return self.cursor.fetchone()[0]
        return 0

    def process_and_store_kanji(self, word):
        """
        For a given word, if it exists in the database, create and populate the 'kanji' field.
        """
        if self.conn:
            # Check if word exists
            if self.word_exists(word):
                # Extract and process the Japanese synonym to obtain the kanji
                self.cursor.execute("SELECT japanese_synonym FROM words_phonetics WHERE word = ?", (word,))
                japanese_synonym = self.cursor.fetchone()[0]
                kanji = extract_kanji(japanese_synonym)

                # Update the kanji field
                update_query = "UPDATE words_phonetics SET kanji = ? WHERE word = ?"
                try:
                    self.cursor.execute(update_query, (kanji, word))
                    self.conn.commit()
                except sqlite3.Error as e:
                    print(f"SQLite Error: {e}")
                    traceback.print_exc()

 

    def update_kanji_for_all_words(self):
        """
        Loop through all words in the database and update the 'kanji' field.
        """

        self.create_field_if_not_exists("words_phonetics", "kanji_synonym", "TEXT")
        if self.conn:
            self.cursor.execute("SELECT word, japanese_synonym FROM words_phonetics")
            rows = self.cursor.fetchall()
            for word, japanese_synonym in rows:
                kanji = extract_kanji(japanese_synonym)
                update_query = "UPDATE words_phonetics SET kanji = ? WHERE word = ?"
                try:
                    self.cursor.execute(update_query, (kanji, word))
                except sqlite3.Error as e:
                    print(f"SQLite Error: {e}")
                    traceback.print_exc()

            self.conn.commit()


    def fetch_and_clean_word_details(self, words):
        updated_words = []
        for word in words:
            word_details = self.find_word_details(word)

            if word_details:
                original_word_details = word_details.copy()

                # Clean English and Japanese text
                word_details['syllable_word'] = clean_english(word_details.get('syllable_word', ''))
                word_details['phonetic'] = clean_english(word_details.get('phonetic', ''))
                word_details['japanese_synonym'] = clean_japanese(word_details.get('japanese_synonym', ''))

                # Check if cleaning resulted in changes
                if word_details != original_word_details:
                    self.update_word_details(word_details, force=True)
                    updated_words.append(word_details)

        return updated_words

    def update_all_words(self, batch_size=10):
        total_words = self.get_total_word_count()
        processed = 0

        while processed < total_words:
            # Fetch a batch of words from the database
            words_batch = self.fetch_words_batch(processed, batch_size)
            words_to_update = []

            for word_detail in words_batch:
                cleaned_word_detail = self.fetch_and_clean_word_details([word_detail["word"]])
                if cleaned_word_detail != word_detail:
                    words_to_update.extend(cleaned_word_detail)

            # Update the database with cleaned and updated word details
            for updated_word in words_to_update:
                # print("updateing words: ", updated_word)
                self.update_word_details(updated_word, force=True)

            processed += len(words_batch)

    def convert_and_update_chinese_synonyms(self):
        if self.conn:
            # self.create_field_if_not_exists("words_phonetics", "simplified_chinese_synonym", "TEXT")

            # Initialize OpenCC converters
            s2t_converter = opencc.OpenCC('s2t')  # Simplified to Traditional
            t2s_converter = opencc.OpenCC('t2s')  # Traditional to Simplified

            self.cursor.execute("SELECT word, chinese_synonym, simplified_chinese_synonym FROM words_phonetics")
            rows = self.cursor.fetchall()

            for word, chinese_synonym, simplified_chinese_synonym in rows:
                if chinese_synonym:
                    # Check and convert to traditional Chinese if it's simplified
                    traditional_chinese = s2t_converter.convert(chinese_synonym)

                    # Check if conversion actually happened, update if different
                    if traditional_chinese != chinese_synonym:
                        update_query = "UPDATE words_phonetics SET chinese_synonym = ? WHERE word = ?"
                        try:
                            self.cursor.execute(update_query, (traditional_chinese, word))
                        except sqlite3.Error as e:
                            print(f"SQLite Error: {e}")
                            traceback.print_exc()

                    # Convert to simplified Chinese if simplified_chinese_synonym is empty
                    if not simplified_chinese_synonym:
                        simplified_chinese = t2s_converter.convert(traditional_chinese)
                        update_simplified_query = "UPDATE words_phonetics SET simplified_chinese_synonym = ? WHERE word = ?"
                        try:
                            self.cursor.execute(update_simplified_query, (simplified_chinese, word))
                        except sqlite3.Error as e:
                            print(f"SQLite Error: {e}")
                            traceback.print_exc()

            self.conn.commit()
    

    def process_word_rows(self, rows, field_names):

        # print("rows:", rows)
        # print("field_names: ", field_names)
        """
        Process rows from the database into a list of dictionaries with word details.
        Process specific fields and keep other fields unprocessed.
        """
        processed_rows = []
        for row in [rows]:
            processed_row = {}
            for field, value in zip(field_names, row):
                if field in ["word", "syllable_word", "phonetic"]:
                    processed_row[field] = clean_english(value)
                else:
                    processed_row[field] = value
            processed_rows.append(processed_row)
        return processed_rows[0]

    def get_table_fields(self, table_name, excluded=None):
        if self.conn:
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [row[1] for row in self.cursor.fetchall()]
            if excluded:
                columns = [column for column in columns if column not in excluded]
            return columns

    def find_word_details(self, word, fields=None, excluded=["id"]):
        if self.conn:
            all_fields = self.get_table_fields('words_phonetics', excluded)
            selected_fields = ', '.join(fields if fields else all_fields)

            self.cursor.execute(f"SELECT {selected_fields} FROM words_phonetics WHERE word = ?", (word,))
            result = self.cursor.fetchone()


            # print("all_fields: ", all_fields)
            if result:
                field_names = fields if fields else all_fields
                return self.process_word_rows(result, field_names)
        return None


    def fetch_random_words(self, num_words, fields=None, excluded=["id"]):
        if self.conn:
            all_fields = self.get_table_fields('words_phonetics', excluded)
            selected_fields = ', '.join(fields if fields else all_fields)

            # Fetch all rows
            query = f"SELECT {selected_fields} FROM words_phonetics ORDER BY RANDOM()"
            # query = f"SELECT {selected_fields} FROM words_phonetics ORDER BY RANDOM() LIMIT ?"
            self.cursor.execute(query)
            # self.cursor.execute(query, (num_words*10,))
            rows = self.cursor.fetchall()

            # Convert rows to a NumPy array and shuffle
            rows_array = np.array(rows)
            # np.random.shuffle(rows_array)
            rows_array = random_shuffle(rows_array)

            # Select the first 'num_words' rows
            selected_rows = rows_array[:num_words]

            # Convert each row back to the original format if needed
            return [self.process_word_rows(tuple(row), fields if fields else all_fields) for row in selected_rows]

    def fetch_last_10_words(self, fields=None, excluded=["id"]):
        if self.conn:
            all_fields = self.get_table_fields('words_phonetics', excluded)
            selected_fields = ', '.join(fields if fields else all_fields)

            query = f"SELECT {selected_fields} FROM words_phonetics ORDER BY rowid DESC LIMIT 10"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return [self.process_word_rows(row, fields if fields else all_fields) for row in rows]

    def fetch_words_batch(self, offset, limit, fields=None, excluded=["id"]):
        if self.conn:
            all_fields = self.get_table_fields('words_phonetics', excluded)
            selected_fields = ', '.join(fields if fields else all_fields)

            # print("all_fields: ", all_fields)

            query = f"SELECT {selected_fields} FROM words_phonetics LIMIT ? OFFSET ?"
            self.cursor.execute(query, (limit, offset))
            rows = self.cursor.fetchall()

            # print("rows: ", rows)
            return [self.process_word_rows(row, fields if fields else all_fields) for row in rows]





    def update_from_word_details_correction_csv(self, word_details_correction_csv_path):
        history_csv_path = 'data/words_update_history.csv'

        with open(word_details_correction_csv_path, 'r+', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            lines = list(reader)
            remaining_lines = []

            for line in lines:
                # Fetch old details from the database
                old_detail = self.find_word_details(line['word'])

                # Update word details in the database
                self.insert_word_details(line, force=True)

                # Fetch updated details for logging
                updated_detail = self.find_word_details(line['word'])

                if old_detail and updated_detail:
                    # Log changes to history if there were any updates
                    old_new_pairs = [(old_detail, updated_detail)]
                    self.log_history_update(old_new_pairs, history_csv_path)
                else:
                    remaining_lines.append(line)

            # Remove updated items from the error CSV
            file.seek(0)
            file.truncate()
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(remaining_lines)

    def update_from_word_list_csv(self, words_update_csv_path, fetcher):
        words = []

        # Extract words from data/words_update.csv
        with open(words_update_csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                words.append(row[0])

        word_details = []
        for word in words:
            word_detail = self.find_word_details(word)
            if word_detail is None:
                # Fetch word details if not found in the database
                word_detail = fetcher.fetch_word_details([word], self)[0]
            word_details.append(word_detail)

        # Recheck and fetch details for the words
        rechecked_details = fetcher.recheck_word_details(word_details, self)

        

        # Update the database with these details
        self.update_from_list(word_details_list, words_update_csv_path)

    def update_from_list(self, word_details_list, words_update_csv_path):
        history_csv_path = 'data/words_update_history.csv'

        for new_details in word_details_list:
            # Fetch old details from the database
            old_detail = self.find_word_details(new_details['word'])

            # Update the word details in the database
            self.insert_word_details(new_details, force=True)

            # Fetch updated details for logging
            updated_detail = self.find_word_details(new_details['word'])

            if old_detail and updated_detail:
                # Log changes to history if there were any updates
                old_new_pairs = [(old_detail, updated_detail)]
                self.log_history_update(old_new_pairs, history_csv_path)

        # Remove updated words from data/words_update.csv
        self.remove_words_from_csv(words_update_csv_path, word_details_list)


    def remove_words_from_csv(self, csv_path, word_details_list):
        updated_words = {details['word'] for details in word_details_list}

        with open(csv_path, 'r+', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            remaining_words = [row[0] for row in reader if row[0] not in updated_words]

            file.seek(0)
            file.truncate()

            writer = csv.writer(file)
            for word in remaining_words:
                writer.writerow([word])




    def close(self):
        if self.conn:
            self.conn.close()


class AdvancedWordFetcher:
    def __init__(self, client, max_retries=3):
        self.client = client
        self.max_retries = max_retries
        self.examples = self.load_examples()
        self.model_name = ["gpt-3.5-turbo", "gpt-4-1106-preview"]

    def load_examples(self):
        examples_file_path = 'data/word_examples.csv'
        if os.path.exists(examples_file_path):
            with open(examples_file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                example_list = list(reader)

                # print("example_list: ", example_list)

                if len(example_list) > 1:
                   return example_list

        return [
            {"word": "abstraction", "syllable_word": "ab·strac·tion", "phonetic": "ˈæb·stræk·ʃən", "japanese_synonym": "抽象（ちゅうしょう）"},
            {"word": "paradox", "syllable_word": "par·a·dox", "phonetic": "ˈpær·ə·dɒks", "japanese_synonym": "逆説（ぎゃくせつ）"}
        ]

    def save_examples(self):
        examples_file_path = 'data/word_examples.csv'
        with open(examples_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.examples[0].keys())
            writer.writeheader()
            writer.writerows(self.examples)

    def extract_and_parse_json(self, text):
        bracket_pattern = r'\[.*?\]'
        matches = re.findall(bracket_pattern, text, re.DOTALL)

        if not matches:
            raise JSONParsingError("No JSON string found in text", text)

        json_string = matches[0]

        try:
            parsed_json = json5.loads(json_string)
            if len(parsed_json) == 0:
                raise JSONParsingError("Parsed JSON string is empty", json_string)
            
            return parsed_json
        # except json.JSONDecodeError as e:
        except ValueError as e:  # Catching ValueError for json5
            traceback.print_exc()
            raise JSONParsingError(f"JSON Decode Error: {e}", json_string)


    def extract_and_parse_words(self, text, num_words, word_database, not_enough_words_list=[]):
        bracket_pattern = r'\[.*?\]'
        matches = re.findall(bracket_pattern, text, re.DOTALL)

        if not matches:
            raise JSONParsingError("No JSON string found in text", text)

        json_string = matches[0]

        try:
            parsed_json = json5.loads(json_string) + not_enough_words_list
            if len(parsed_json) == 0:
                raise JSONParsingError("Parsed JSON string is empty", json_string)

            unique_words = [word for word in parsed_json if not word_database.word_exists(word)]
            if len(unique_words) < num_words // 2:
                duplicated_words = set(parsed_json) - set(unique_words)

                fetched_num = len(unique_words)
                raise NotEnoughUniqueWordsError(len(parsed_json), fetched_num, unique_words, list(duplicated_words), json_string)

            return unique_words
        # except json.JSONDecodeError as e:
        except ValueError as e:  # Catching ValueError for json5
            traceback.print_exc()
            raise JSONParsingError(f"JSON Decode Error: {e}", json_string)


    def load_propensities(self):
        propensities_file_path = 'data/words_propensity.txt'
        propensities = []
        if os.path.exists(propensities_file_path):
            with open(propensities_file_path, 'r') as file:
                # Only include lines that are not empty and do not start with '#'
                propensities = [line.strip() for line in file if line.strip() and not line.startswith('#')]
        return propensities

    def fetch_words(self, num_words, word_database):

        # return self.fetch_words_local(num_words*5, word_database)
        return self.fetch_words_openai(num_words, word_database)


    def fetch_words_local(self, num_words, word_database):
        # Ensure NLTK words are downloaded
        # nltk.download('words', quiet=True)

        # Load words from NLTK
        nltk_word_list = words.words()

        # print("nltk words length: ", len(nltk_word_list))



        # Filter out words that already exist in the database
        unique_words = [word for word in nltk_word_list if not word_database.word_exists(word)]

        # random.shuffle(unique_words)
        unique_words = random_shuffle(unique_words)

        # Check if there are enough unique words
        if len(unique_words) < num_words:
            raise ValueError(f"Not enough unique words. Only {len(unique_words)} unique words found.")

        # Return the specified number of unique words
        return unique_words[:num_words]

    def save_unused_words(self, words, random_words, file_path='data/unused_words.csv'):
        """
        Saves the words that were not selected by random_sample to a CSV file.

        :param random_words: The list of words selected by random_sample.
        :param file_path: The path to the CSV file where the unused words will be saved.
        """
        # Calculate the unused words
        unused_words = set(words) - set(random_words)

        # Write the unused words to a CSV file
        with open(file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for word in unused_words:
                writer.writerow([word])


    def fetch_words_openai(self, num_words, word_database):
        propensities = self.load_propensities()
        unique_words = []
        messages = []

        words_number_scale_factor = 5
        num_words_scaled = num_words * words_number_scale_factor


        local_words = self.fetch_words_local(num_words, word_database)

        # Choose the appropriate prompt based on whether propensities are available
        if propensities:
            criteria_list = "\n".join([f"{i+1}) {propensity}" for i, propensity in enumerate(propensities)])
            user_message = (
                f"Generate a python list of {num_words_scaled} unique advanced words that meet one or more of the following criteria:\n"
                f"{criteria_list}\n"
                # "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
                # "and include the word's tendency or special characteristic next to each word."
                "The output should be like ['word 1', 'word 2', ..., 'word N']."
            )
        else:
            user_message = (
                 "Choose English words from most common to advanced that are often used in daily expression or formal readings in various areas. "
                f"Could you take a deep breath, think widly and give me a list of {num_words_scaled} words with similar list (compatible to json.loads) FORMAT: \n"
                f"\n{json.dumps(local_words)}. "
            )

        messages = [
            {"role": "system", "content": "You are an assistant with a vast vocabulary and creativity. You almost know every English words in this universe. "},
            {"role": "user", "content": user_message}
        ]

        not_enough_words_list = []

        for try_num in range(self.max_retries):
            try:
                

                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )

                words_list = self.extract_and_parse_words(response.choices[0].message.content, num_words, word_database, not_enough_words_list=not_enough_words_list)

                words_list.extend(not_enough_words_list)

                # print("words_list: ", words_list)
                unique_words = [word for word in words_list if not word_database.word_exists(word)]
                if unique_words:
                    break

            except JSONParsingError as jpe:
                print(f"JSON parsing failed: {jpe.error_details}")
                
                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})

                # messages.insert(0, {"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})
                # messages.insert(0, {"role": "system", "content": jpe.json_string})

                # print(f"Retrying: try the {try_num+2} times...")

                traceback.print_exc()

                print('Ignoring the error, continuing...')

                continue

            except NotEnoughUniqueWordsError as not_enough_error:

                print(f"Not enough word: {not_enough_error.error_details}")


                # not_enough_words_list.extend(json5.loads(not_enough_error.json_string))
                not_enough_words_list.extend(not_enough_error.unique_words)

                messages.append({"role": "system", "content": not_enough_error.json_string})
                messages.append({"role": "user", "content": f"{not_enough_error.error_details} Please take a deep breath and use your imagination to think more widely. "})

                # print(f"Retrying: try the {try_num+2} times...")

                traceback.print_exc()

                print('Ignoring the error, continuing...')

                continue


            except Exception as e:
                print(f"An unexpected error occurred: {e}")

                traceback.print_exc()

                raise e

        if not unique_words:
            raise RuntimeError("Failed to fetch unique words after maximum retries.")

        return unique_words

    def fetch_word_details(self, words, word_database=None, num_words_phonetic=10):
        # random_words = random.sample(words, min(num_words_phonetic, len(words)))
        # random_words = random_sample(words, num_words_phonetic)
        random_words = words

        self.save_unused_words(words, random_words)

        # Directly create the mismatch message here
        example_word = self.examples[0].get("word", "")
        syllables = split_word(self.examples[0].get("syllable_word", ""))
        phonetics = split_word(self.examples[0].get("phonetic", ""))
        mappings = self.map_syllables_phonetics(syllables, phonetics)


        example_string = json.dumps(self.examples, ensure_ascii=False, separators=(',', ':'))
        words_string = ', '.join(random_words).lower()
        

        

        detailed_list_message = (
            "Could you provide a detailed syllable (using ·) and phonetic separation (also using ·) "
            "ensuring a one-to-one correspondence between syllable_word and its IPA phonetic? "
            "Please adjust the syllable or phonetic divisions if necessary "
            "to ensure each syllable directly matches/aligns with its corresponding phonetic element, "
            "even if this means altering the conventional syllable breakdown."
            f"For example, the separation of {example_word} should reflect the correspondence: \n {mappings}. \n"
            "Also provide the japanese_synonym with hiragana pronounciation of kanji inside parentheses, pure Japanese kanji_synonym, arabic_synonym, traditional chinese_synonym in Taiwan language habits, simplified_chinese_synonym and french_synonym. "
            # "For the traditional and simplified Chinese, please consider habits and context of HK, Taiwan, Macau and  China mainland. "
            f"Similar to this json.loads compatible FORMAT: \n {example_string}. "
            f"Could you provide me the linguistic details for words [ {words_string} ] ?"
        )

        messages = [
                {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words."},
                {"role": "user", "content": detailed_list_message}
            ]

        

        for _ in range(self.max_retries):
            try:

                # print(f"Querying {random_words} from OpenAI...")

                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )
                
                word_phonetics = self.extract_and_parse_json(response.choices[0].message.content)



                # Save word details to database
                for detail in word_phonetics:
                    if word_database:
                        word_database.insert_word_details(detail)
                
                # self.recheck_syllable_and_phonetic(word_phonetics, word_database)
                # self.recheck_japanese_synonym(word_phonetics, word_database)

                words_list = [word["word"] for word in word_phonetics]

                print("Starting comparing separation...")
                self.split_and_compare_phonetic_syllable(word_phonetics.copy(), word_database)
                print("Starting check Japanese...")
                self.recheck_japanese_synonym_with_conditions(word_phonetics.copy(), word_database)
                print("Generating kanji...")
                word_database.update_kanji_for_all_words()
                print("Starting check pure kanji...")
                # self.fetch_pure_kanji_synonyms(words_list.copy(), word_database)
                self.recheck_pure_kanji_synonym(word_phonetics.copy(), word_database)
                print("Starting check Arabic...")
                word_database.convert_and_update_chinese_synonyms()
                # self.fetch_arabic_synonyms(words_list.copy(), word_database)
                self.recheck_arabic_synonym(word_phonetics.copy(), word_database)

                # word_phonetics = [word_database.find_word_details(word["word"]) for word in word_phonetics]
                word_phonetics = [word_database.find_word_details(word) for word in words_list]

                self.examples= word_phonetics[0:2]
                self.save_examples()
                return word_phonetics
            except JSONParsingError as jpe:
                print(f"JSON parsing failed: {jpe.error_details}")
                traceback.print_exc()

                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})

                print("Ignoring the error, continuing...")
                

                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

                traceback.print_exc()

                raise e
            else:
                # print("Fetched word details successfully.")
                return word_phonetics

        raise RuntimeError("Failed to parse response after maximum retries.")

    def recheck_word_details(self, word_details, word_database=None, num_words_phonetic=10, recheck=False):


        word_details = clean_and_transcribe(word_details)

        words = [{k: v for k, v in word.items() if k in ['word']} for word in word_details]

        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # print("Rechking whole details: ", word_details)

        detailed_list_message = (
            "Let's recheck each word's details and ensure they are correctly formatted. "
            "Please ensure the syllable separation in the 'phonetic' transcription aligns with the 'syllable_word' separation. "
            "Each syllable should be marked with a central dot (·) in both fields. "
            "Adjust the 'phonetic' field to match the syllable divisions in 'syllable_word'. "
            "That ˈ and ˌ are always treated as separator regardless if '·' exists or not. "
            "Could you correct the furigana/hiragana of kanji/katakana inside the parentheses as needed? \n {}"
        ).format(json.dumps(word_details, ensure_ascii=False, separators=(',', ':')))


        messages = [
            {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words."},
            {"role": "user", "content": detailed_list_message}
        ]

        for _ in range(self.max_retries):
            try:
                # print(f"Rechecking {words} from OpenAI...")
                
                

                response = self.client.chat.completions.create(
                    # model="gpt-3.5-turbo",
                    # model="gpt-4",
                    model=self.model_name[1],
                    messages=messages
                )
                word_phonetics = self.extract_and_parse_json(response.choices[0].message.content)
                # print("Parsed rechecked result: ", word_phonetics)
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # Save word details to database
                for detail in word_phonetics:
                    if word_database:
                        # word_database.insert_word_details(detail)
                        word_database.update_word_details(detail)

                # self.recheck_syllable_and_phonetic(word_phonetics, word_database)
                # self.recheck_japanese_synonym(word_phonetics, word_database)
                return word_phonetics

            except JSONParsingError as jpe:
                print(f"JSON parsing failed: {jpe.error_details}")

                traceback.print_exc()

                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})


                
                print("Ignoring the error, continuing...")

                continue
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
                raise e
            else:
                # print("Fetched and rechecked word details successfully.")
                return word_phonetics
        raise RuntimeError("Failed to parse response after maximum retries.")


    def recheck_syllable_and_phonetic(self, word_details, word_database=None, messages=""):
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Rechecking syllable and phonetics: ")
        pprint([word["word"] for word in word_details])
        print("\n")



        # # Prepare examples excluding the Japanese synonyms
        # example_list = [{k: v for k, v in example.items() if k != 'japanese_synonym'} for example in self.examples]
        
        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'syllable_word', 'phonetic']} for word in word_details]
        
        words = [{k: v for k, v in word.items() if k in ['word']} for word in word_details]

        word_details = clean_word_details(word_details)


        if not messages:
            # Prepare message for rechecking details
            detailed_list_message = (
                "That ˈ and ˌ are always treated as separator regardless if '·' exists or not. "
                "Please ensure the syllable separation in the 'phonetic' transcription aligns with the 'syllable_word' separation. "
                "Each syllable should be marked with a central dot (·) in both fields. "
                "Adjust the 'phonetic' field or the syllable divisions in 'syllable_word' to be matched. "
                "Here is the word detail for correction and output SAME format: {}."
            ).format(json.dumps(word_details, ensure_ascii=False, separators=(',', ':')))

            messages = [
                {
                    "role": "system", 
                    "content":(
                            "You are an assistant skilled in linguistics, "
                            "capable of providing accurate and detailed phonetic and linguistic attributes for given words. "
                            "You are excellent in separate words and their phonetics into consistent and accurate separations with '·'."
                        )
                },
                {"role": "user", "content": detailed_list_message}
            ]

        for _ in range(self.max_retries):
        # for _ in range(1):
            try:
                # print(f"Rechecking syllable and phonetic for {words} from OpenAI...")
                
                
                # print("messages for recheck_syllable_and_phonetic: ")
                # pprint(messages)
                # print('\n')

                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )

                # print("Rechecked result: ", response.choices[0].message.content)
                word_phonetics = self.extract_and_parse_json(response.choices[0].message.content)
                word_phonetics = clean_word_details(word_phonetics)

                # print("rechecked result from openai: ")
                # pprint(word_phonetics)
                # print('\n')

                # print("Parsed result: ", word_phonetics)
                print("OpenAI: ")
                pprint(word_phonetics)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                # Save word details to database
                for detail in word_phonetics:
                    if word_database:
                        print(f"Updating syllable and phonetic of word with", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                        print("\n")
                        word_database.update_word_details(detail)
                return word_phonetics
            except JSONParsingError as jpe:
                print(f"JSON parsing failed: {jpe.error_details}")

                traceback.print_exc()

                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})

                
                print("Ignoring the error, continuing...")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

                traceback.print_exc()

                raise e
            else:
                # print("Fetched and rechecked word details successfully.")
                return word_phonetics


        raise RuntimeError("Failed to parse response after maximum retries.")

    
    

    def split_and_compare_phonetic_syllable(self, word_details, word_database=None):
        rechecked_word_details = []

        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'syllable_word', 'phonetic']} for word in word_details]

        # Initial system message for context
        basic_message = {
            "role": "system", 
            "content": (
                    "You are an assistant skilled in linguistics with flexibility, "
                    "capable of providing accurate, flexible and detailed phonetic and linguistic attributes for given words. "
                    "You excel in separating words and their phonetics into consistent and accurate separations with '·'."
                )
        }

        for word_detail in word_details:
            word = word_detail.get('word', '')

            example_format = json.dumps(
                [{'word': word, 'syllable_word': '', 'phonetic': ''}], 
                ensure_ascii=False, 
                separators=(',', ':')
            )

            detailed_list_message = (
                "Syncronize each syllable with a corresponding phonetic component. "
                # "Adjust the 'phonetic' field or the syllable divisions in 'syllable_word' to be matched. "
                "That ˈ and ˌ are always treated as separator regardless if '·' exists or not. "
                # "Please ensure the syllable separation in the English 'phonetic' transcription aligns with the 'syllable_word' separation. "
                "Each syllable should be marked with a central dot (·) in both fields. "
                # "Here is the word detail for correction and output SAME format AS: {}."
            )#.format(example_format)

            messages = [basic_message]


            for n_try in range(self.max_retries):
                print(f"Trying the {n_try} time(s)...")

                
                syllable_word = word_detail.get('syllable_word', '')
                phonetic = word_detail.get('phonetic', '')
                

                

                if not syllable_word or not phonetic:
                    # Message for missing data
                    message = (
                        "The syllable_word or phonetic of word '{}' is missing. "
                        "Please provide the missing data in this json.loads compatible format: \n"
                        "{}"
                    ).format(word, example_format)

                    if n_try == 0:
                        messages.append({"role": "user", "content": message + detailed_list_message})
                    else:
                        messages.append({"role": "user", "content": message})

                    word_detail = self.recheck_syllable_and_phonetic([word_detail], word_database, messages)[0]
                    word_detail = clean_word_details([word_detail])[0]
                    
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(',', ':'))})
                
                elif len(split_word(syllable_word)) != len(split_word(phonetic)):
                    # Directly create the mismatch message here
                    syllables = split_word(syllable_word)
                    phonetics = split_word(phonetic)
                    mappings = self.map_syllables_phonetics(syllables, phonetics)

                    mismatch_message = (
                        "You can merge syllables or add divisions in either sides to make it align, "
                        "even if it diverges from typical linguistic patterns. "
                        # f"Mismatch of word syllable and phonetic separation in '{word}':\n"
                        f"THE OUTPUT JSON FORMAT: {example_format}. "
                        f"Make the mismatched Syllable-Phonetic Pairs of {word} match: \n {mappings}. \n"
                    )

                    if n_try == 0:
                        messages.append({"role": "user", "content": detailed_list_message + mismatch_message})
                    else:
                        messages.append({"role": "user", "content": mismatch_message})

                    word_detail = self.recheck_syllable_and_phonetic([word_detail], word_database, messages)[0]
                    
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(',', ':'))})
                
                else:
                    if n_try == 0:
                        print("No need to update: ", word_detail["word"])
                    else: 
                        print("Successfully updated: ", word_detail["word"])

                    rechecked_word_details.append(word_detail)
                    break

                if n_try == self.max_retries - 1:
                    print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

        return rechecked_word_details


    def map_syllables_phonetics(self, syllables, phonetics):
        # Adjust the lists to make them equal in length for mapping
        max_length = max(len(syllables), len(phonetics))
        syllables.extend([''] * (max_length - len(syllables)))
        phonetics.extend([''] * (max_length - len(phonetics)))

        # Create a mapping of syllable to phonetic
        mapping = ', '.join([f"{syl} ↔ {phon}" for syl, phon in zip(syllables, phonetics)])
        return mapping


    def recheck_japanese_synonym(self, word_details, word_database=None, messages=''):
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Rechecking Japanese synonym: ")
        # pprint(word_details)
        pprint([word["word"] for word in word_details])
        print("\n")

        

        # Prepare examples including only the word and Japanese synonym
        # example_list = [{k: v for k, v in example.items() if k in ['word', 'japanese_synonym']} for example in self.examples]

        # words = [{k: v for k, v in word.items() if k in ['word']} for word in word_details]

        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'japanese_synonym']} for word in word_details]
        word_details = clean_and_transcribe(word_details)
        
        # print("Clean and transribed: ", word_details)

        # detailed_list_message = (
        #     "{}"
        #     "Could you correct the furigana/hiragana of kanji/katakana inside the parentheses as needed and output same format? \n {}"
        # ).format(message, json.dumps(word_details_formatted, ensure_ascii=False, separators=(',', ':')))

        if not messages:
            detailed_list_message = (
                "Could you correct the furigana/hiragana of kanji/katakana inside the parentheses as needed and output same format? \n {}"
            ).format(json.dumps(word_details, ensure_ascii=False, separators=(',', ':')))

            messages = [
                {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words. You are excellent in providing hiragana (furigana) for consecutive kanji/katakana. "},
                {"role": "user", "content": detailed_list_message}
            ]

        for _ in range(self.max_retries):
        # for _ in range(1):
            try:
                # print(f"Fetching Japanese synonyms for {word_details_formatted} from OpenAI...")
                
                # print("messages for recheck_japanese_synonym:")
                # pprint(messages)
                # print('\n')

                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )

                # print("Rechecked Japanese synonym: ", response.choices[0].message.content)
                word_phonetics = self.extract_and_parse_json(response.choices[0].message.content)
                word_phonetics = clean_word_details(word_phonetics)
                # print("fetched result from openai:")
                # pprint(word_phonetics)
                # print('\n')

                print("OpenAI: ")
                print(word_phonetics)
                # print("\n")
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                # Save word details to database
                for detail in word_phonetics:
                    if word_database:
                        print("Updating japanese synonym of word with ", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                        print("\n")
                        word_database.update_word_details(detail)

                return word_phonetics
            except JSONParsingError as jpe:
                print(f"JSON parsing failed: {jpe.error_details}")
                traceback.print_exc()

                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})

                
                print("Ignoring the error, continuing...")

                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

                traceback.print_exc()

                raise e
            else:
                # print("Fetched and rechecked word details successfully.")
                return word_phonetics                

        raise RuntimeError("Failed to parse response after maximum retries.")



    def recheck_japanese_synonym_with_conditions(self, word_details, word_database=None):
        rechecked_word_details = []

        word_details = [{k: v for k, v in word.items() if k in ['word', 'japanese_synonym']} for word in word_details]

        basic_message = {
            "role": "system", 
            "content": (
                "You are an assistant skilled in linguistics, "
                "knowing Japanese extremly well. "
                "You are excellent in providing hiragana (furigana) for consecutive kanji/katakana."
            )
        }

        detailed_list_message = (
            "Treat 々 as kanji. "
            "Return the same back if japanese_synonym are all hiragana without parentheses. "
            "If not found, you can make up some homonym. "
            
        )

        # word_details = clean_and_transcribe(word_details)

        for word_detail in word_details:
            
            messages = [basic_message]



            for n_try in range(self.max_retries):
                print("###############")
                print(f"Trying the {n_try} time(s)...")
                print("###############")

                japanese_synonym = word_detail.get('japanese_synonym', '')
                # print("original japanese_synonym: ", japanese_synonym)


                # Remove all hiragana and parentheses
                cleaned_synonym = remove_hiragana_and_parentheses(japanese_synonym).replace("（", "").replace("）", "")

                if cleaned_synonym == '':
                    print("All hiragana!")
                    cleaned_hiragana = remove_hiragana_including_parentheses(japanese_synonym)
                    word_detail['japanese_synonym'] = cleaned_hiragana
                    
                    if word_database:
                        word_database.update_word_details(word_detail)
                    
                    rechecked_word_details.append(word_detail)
                    break
                else:

                    without_hiragana_original = remove_hiragana_inside_parentheses(japanese_synonym)
                    # without_hiragana_original = remove_japanese_letter_inside_parentheses(japanese_synonym)
                    # n_hiragana_repetitions_original = smallest_non_zero_repetition(japanese_synonym)

                    word_detail_transribed = clean_and_transcribe([word_detail])[0]
                    japanese_synonym_cleaned_and_transcribed = word_detail_transribed["japanese_synonym"]
                    # n_hiragana_repetitions_transcribed = smallest_non_zero_repetition(japanese_synonym_cleaned_and_transcribed)
                    without_hiragana_transcribed = remove_hiragana_inside_parentheses(japanese_synonym_cleaned_and_transcribed)

                    discrepancies = compare_repetition_results(japanese_synonym, japanese_synonym_cleaned_and_transcribed)
                    # n_hiragana_repetitions_original = discrepancies[0][1]
                    # n_hiragana_repetitions_transcribed = discrepancies[0][2]

                    print(
                        "\n"
                        f"Rechecking for word '{word_detail['word']}':\n"
                        f"Full: {json.dumps(japanese_synonym, ensure_ascii=False, separators=(',', ':'))}\n"
                        f"Original: {without_hiragana_original}\n"
                        f"Kakasi: {without_hiragana_transcribed}\n"
                    )


                    # print(without_hiragana_original, " ? ", without_hiragana_transcribed)

                    # if (n_hiragana_repetitions_original > n_hiragana_repetitions_transcribed) or (without_hiragana_original != without_hiragana_transcribed) or (not japanese_synonym):
                    if (not japanese_synonym) or (without_hiragana_original != without_hiragana_transcribed) or (len(discrepancies) > 0):

                        word_detail_blanked = [{
                                k: (v if k != 'japanese_synonym' else remove_hiragana_inside_parentheses(v))
                                for k, v in word_detail_transribed.items() 
                            }
                        ]


                        if not japanese_synonym:
                            message = (
                                "No Japanese synonym provided. Please provide one anyway even if it's incorrect."
                            )

                            print("No Japanese synonym provided. ")
                        elif without_hiragana_original != without_hiragana_transcribed:

                            message = (
                                "The hiragana inside the parentheses is generated by pykakasi. Please recheck."
                                "Could you provide the furigana/hiragana of kanji/katakana inside the parentheses as needed and output SAME format? \n {}"
                            ).format(
                                json.dumps(word_detail_blanked, ensure_ascii=False, separators=(',', ':')),
                            )

                            print("Furigana position incorrect. ")

                        else:
                            if (len(discrepancies) > 0):
                                print("discrepancies: \n")
                                pprint(discrepancies)

                                message = (
                                    "There are possible repetitions of hiragana before the kanji inside the parentheses: {} ."
                                    "Like pykakasi, please remove the possible repeated hiragana inside the parentheses: {} . "
                                    "Could you provide the correct furigana/hiragana of kanji/katakana inside the parentheses as needed and output SAME format? \n {}"
                                ).format(
                                    word_detail["japanese_synonym"],
                                    word_detail_transribed["japanese_synonym"],
                                    json.dumps(word_detail_blanked, ensure_ascii=False, separators=(',', ':'))
                                )

                                print("Repetition inside parentheses. ")
                            else:
                                message = (
                                    "The hiragana inside the parentheses is generated by pykakasi. Please recheck."
                                    "Could you provide the furigana/hiragana of kanji/katakana inside the parentheses as needed and output SAME format? \n {}"
                                ).format(
                                    json.dumps(word_detail_blanked, ensure_ascii=False, separators=(',', ':')),
                                )

                                print("Unkown transcription error. ")

                        if n_try == 0:
                            messages.append({"role": "user", "content": detailed_list_message + message})
                        else:
                            messages.append({"role": "user", "content": message})

                        # print(
                        #     "###############"
                        #     f"Furigana position incorrect, rechecking for word '{word_detail['word']}':\n"
                        #     f"Full: {json.dumps(japanese_synonym, ensure_ascii=False, separators=(',', ':'))}\n"
                        #     f"Original: {without_hiragana_original}\n"
                        #     f"Kakasi: {without_hiragana_transcribed}\n"
                        # )

                        word_detail = self.recheck_japanese_synonym([word_detail], word_database, messages)[0]
                        word_detail = clean_word_details([word_detail])[0]
                        messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(",", ":"))})
                        # rechecked_word_details.append(word_detail)
                    else:
                        if n_try == 0:
                            print("No need to update: ", word_detail["word"])
                        else: 
                            print("Successfully updated: ", word_detail["word"])
                        rechecked_word_details.append(word_detail)
                        # print("Updated word: ", word_detail["word"])
                        break


            if n_try == self.max_retries - 1:
                print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

        return rechecked_word_details

    def fetch_pure_kanji_synonyms(self, words, word_database=None, messages=None):
        # Prepare examples in the required format
        # example_list = [
        #     {"word": "sun", "kanji_synonym": "太陽", "chinese_synonym": "太陽", "simplified_chinese_synonym": "太阳"},
        #     {"word": "moon", "kanji_synonym": "月", "chinese_synonym": "月亮", "simplified_chinese_synonym": "月亮"}
        # ]

        example_list = [
            {
                "word": "computer",
                "kanji_synonym": "電子計算機",  # Japanese Kanji: Literally "Electronic Calculating Machine"
                "chinese_synonym": "計算機",  # Traditional Chinese: "Calculating Machine"
                "simplified_chinese_synonym": "电脑"  # Simplified Chinese: "Electric Brain"
            },
            {
                "word": "cellphone",
                "kanji_synonym": "携帯電話",  # Japanese Kanji: Literally "Portable Telephone"
                "chinese_synonym": "手機",  # Traditional Chinese: "Hand Machine"
                "simplified_chinese_synonym": "手机"  # Simplified Chinese: "Hand Machine"
            }
        ]


        examples_json = json.dumps(example_list, ensure_ascii=False, separators=(',', ':'))

        if not messages:
            # Prepare the prompt with examples
            detailed_list_message = (
                "Based on the following examples, provide the pure Japanese kanji synonym with as least Japanese letters as possible for each word in the list. "
                "If it's hard just give some loosely related or use traditional Chinese as kanji_synonym field. "
                # "The output should be in a plain JSON format, as a list of dictionaries, "
                # "where each dictionary contains 'word', 'kanji_synonym' and 'chinese_synonym'. "
                "Each dictionary contains 'word', 'kanji_synonym', (traditional) 'chinese_synonym' and 'simplified_chinese_synonym'. "
                # "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
                "Output the json.loads compatible format as the examples: {}\n"
                "The words to process are: {}."
            ).format(examples_json, ', '.join(words))

            messages = [
                {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. You are excellent in providing pure kanji synonyms."},
                {"role": "user", "content": detailed_list_message}
            ]

        # Fetch Arabic synonyms using OpenAI
        return self.attempt_to_fetch_synonyms(messages, word_database)

    def recheck_pure_kanji_synonym(self, word_details, word_database=None):

        print('Checking pure kanji synonym for these words: ')
        # pprint(word_details)
        pprint([word["word"] for word in word_details])
        print("\n")

        rechecked_word_details = []

        basic_message = {
            "role": "system", 
            "content": (
                "You are an assistant skilled in linguistics, "
                "knowledgeable in Japanese, traditional Chinese and simplified Chinese language. "
                "You are excellent in providing pure kanji and Traditional Chinese synonyms for English words."
            )
        }

        for word_detail in word_details:
            if 'word' not in word_detail:
                raise ValueError(f"Missing 'word' key in word_detail: {word_detail}")

            word = word_detail['word']

            detailed_list_message = (
                f"Please provide the pure kanji synonym and Chinese synonym for the word '{word}' as it is currently missing. "
                "Try you best to get pure kanji, if no pure kanji found, you can make up some homonym or use traditional Chinese as kanji_synonym field. "
                # "Format the output in a plain JSON format, as a list of dictionaries, "
                # "where each dictionary contains 'word', 'kanji_synonym' and 'chinese_synonym'. "
                "Each dictionary contains 'word', 'kanji_synonym', (traditional) 'chinese_synonym' and 'simplified_chinese_synonym'. "
                "Output the json.loads compatible format as [{'word':'', 'kanji_synonym': '', 'chinese_synonym': '', 'simplified_chinese_synonym':''}]"
            )

            # break

            messages = [
                basic_message, 
                {"role": "user", "content": detailed_list_message}
            ]


            for n_try in range(self.max_retries):
                print(f"Trying the {n_try} time(s)...")
                
                kanji_synonym = word_detail.get('kanji_synonym', '')
                chinese_synonym = word_detail.get('chinese_synonym', '')

                # Check if kanji synonym is empty
                if not (kanji_synonym and chinese_synonym):
                    # print("n_try: ", n_try, "word_detail: ", word_detail)

                    # print("n_try: ", n_try)
                    # print(word_detail)
                    
                    

                    print("Updating pure kanji of word: ", word)
                    word_detail = self.fetch_pure_kanji_synonyms([word], word_database, messages=messages)[0]
                    # print("n_try: ", n_try, "word_detail: ", word_detail)
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(",", ":"))})
                else:
                    if n_try == 0:
                        print("No need to update: ", word)
                    else:
                        print("Successfully updated: ", word)
                    rechecked_word_details.append(word_detail)
                    break

                if n_try == self.max_retries - 1:
                    print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

        return rechecked_word_details


    def fetch_arabic_synonyms(self, words, word_database, messages=None):
        print("Words to fetch arabic: ", words)
        # Prepare examples in the required format
        example_list = [
            {"word": "sun", "arabic_synonym": "شمس"},
            {"word": "moon", "arabic_synonym": "قمر"}
        ]
        examples_json = json.dumps(example_list, ensure_ascii=False, separators=(',', ':'))

        if not messages:
            # Prepare the prompt with examples
            detailed_list_message = (
                "Based on the following examples, provide the Arabic synonym for each word in the list. "
                # "The output should be in a plain JSON format, as a list of dictionaries, "
                # "where each dictionary contains 'word' and 'arabic_synonym'. "
                "Each dictionary contains 'word' and 'arabic_synonym'. "
                # "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
                "Output the json.loads compatible format as examples: {}\n"
                "The words to process are: {}."
            ).format(examples_json, ', '.join(words))

            messages = [
                {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. You are excellent in providing Arabic synonyms."},
                {"role": "user", "content": detailed_list_message}
            ]

        # Fetch Arabic synonyms using OpenAI
        return self.attempt_to_fetch_synonyms(messages, word_database)


    def recheck_arabic_synonym(self, word_details, word_database=None):

        print('Checking arabic synonym for these words: ')
        # pprint(word_details)
        pprint([word["word"] for word in word_details])
        print("\n")

        rechecked_word_details = []

        basic_message = {
            "role": "system", 
            "content": (
                "You are an assistant skilled in linguistics, "
                "knowledgeable in Arabic language. "
                "You are excellent in providing Arabic synonyms for English words."
            )
        }

        for word_detail in word_details:
            if 'word' not in word_detail:
                raise ValueError(f"Missing 'word' key in word_detail: {word_detail}")

            word = word_detail['word']

            detailed_list_message = (
                f"Please provide the Arabic synonym ANYWAY for the word '{word}' as it is currently missing. "
                "If not found, you can make up some homonym. "
                # "Format the output in a plain JSON format, as a list of dictionaries, "
                # "where each dictionary contains 'word' and 'arabic_synonym'. "
                "Each dictionary contains 'word' and 'arabic_synonym'. "
                "Output in this json.loads compatible format [{'word':'', 'arabic_synonym': ''}]"
            )

            # break

            messages = [
                basic_message, 
                {
                    "role": "user", 
                    "content": detailed_list_message
                }
            ]


            for n_try in range(self.max_retries):
                # print(word_detail)
                arabic_synonym = word_detail.get('arabic_synonym', '')

                # Check if Arabic synonym is empty
                if not arabic_synonym:

                    # print("n_try: ", n_try)
                    # print(word_detail)
                    

                    print("Updating arabic of word: ", word)
                    word_detail = self.fetch_arabic_synonyms([word], word_database, messages)[0]
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(",", ":"))})
                else:
                    if n_try == 0:
                        print("No need to update: ", word)
                    else: 
                        print("Successfully updated: ", word)
                    rechecked_word_details.append(word_detail)
                    break

                if n_try == self.max_retries - 1:
                    print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

        return rechecked_word_details



    def fetch_chinese_synonyms(self, words, word_database=None):
        # Prepare examples in the required format
        example_list = [
            {"word": "freedom", "chinese_synonym": "自由"},
            {"word": "happiness", "chinese_synonym": "幸福"}
        ]
        examples_json = json.dumps(example_list, ensure_ascii=False, separators=(',', ':'))

        # Prepare the prompt with examples
        detailed_list_message = (
            "Based on the following examples, provide the Chinese synonym for each word in the list. "
            "The output should be in a plain JSON format, as a list of dictionaries, "
            "where each dictionary contains 'word' and 'chinese_synonym'. "
            "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
            "Examples: {}\n"
            "The words to process are: {}."
        ).format(examples_json, ', '.join(words))

        messages = [
            {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. You are excellent in providing Chinese synonyms."},
            {"role": "user", "content": detailed_list_message}
        ]

        # Fetch Chinese synonyms using OpenAI
        return self.attempt_to_fetch_synonyms(messages, word_database)


    def fetch_french_synonyms(self, words, word_database=None, model_name=None):
        # Prepare examples in the required format
        example_list = [
            {"word": "peace", "french_synonym": "paix"},
            {"word": "love", "french_synonym": "amour"}
        ]
        examples_json = json.dumps(example_list, ensure_ascii=False, separators=(',', ':'))

        # Prepare the prompt with examples
        detailed_list_message = (
            "Based on the following examples, provide the French synonym for each word in the list. "
            "The output should be in a plain JSON format, as a list of dictionaries, "
            "where each dictionary contains 'word' and 'french_synonym'. "
            # "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
            "Output the COMPLETE json.loads compatible format as examples: {}\n"
            "The words to process are: {}."
        ).format(examples_json, ', '.join(words))

        messages = [
            {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. You are excellent in providing French synonyms."},
            {"role": "user", "content": detailed_list_message}
        ]

        # Fetch French synonyms using OpenAI
        return self.attempt_to_fetch_synonyms(messages, word_database, model_name=model_name)


    def recheck_french_synonym(self, word_details, word_database=None):

        print('Checking french synonym for these words: ')
        # pprint(word_details)
        pprint([word["word"] for word in word_details])
        print("\n")

        rechecked_word_details = []

        basic_message = {
            "role": "system", 
            "content": (
                "You are an assistant skilled in linguistics, "
                "knowledgeable in French language. "
                "You are excellent in providing French synonyms for English words."
            )
        }

        for word_detail in word_details:
            if 'word' not in word_detail:
                raise ValueError(f"Missing 'word' key in word_detail: {word_detail}")

            word = word_detail['word']

            detailed_list_message = (
                f"Please provide the French synonym ANYWAY for the word '{word}' as it is currently missing. "
                "If not found, you can make up some homonym. "
                # "Format the output in a plain JSON format, as a list of dictionaries, "
                # "where each dictionary contains 'word' and 'arabic_synonym'. "
                "Each dictionary contains 'word' and 'french_synonym'. "
                "Output the COMPLETE list in this json.loads compatible format [{'word':'', 'french_synonym': ''}]"
            )

            # break

            messages = [
                basic_message, 
                {
                    "role": "user", 
                    "content": detailed_list_message
                }
            ]


            for n_try in range(self.max_retries):
                # print(word_detail)
                french_synonym = word_detail.get('french_synonym', '')

                # Check if French synonym is empty
                if not french_synonym:

                    # print("n_try: ", n_try)
                    # print(word_detail)
                    

                    print("Updating arabic of word: ", word)
                    word_detail = self.fetch_french_synonyms([word], word_database, messages)[0]
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(",", ":"))})
                else:
                    if n_try == 0:
                        print("No need to update: ", word)
                    else: 
                        print("Successfully updated: ", word)
                    rechecked_word_details.append(word_detail)
                    break

                if n_try == self.max_retries - 1:
                    print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

        return rechecked_word_details

    def attempt_to_fetch_synonyms(self, messages, word_database=None, model_name=None):
        for _ in range(self.max_retries):
            try:
                # pprint(messages)
                response = self.client.chat.completions.create(
                    model= model_name if model_name else self.model_name[1],
                    messages=messages
                )


                synonyms = self.extract_and_parse_json(response.choices[0].message.content)

                # pprint(synonyms)
                # Save synonyms to database
                for detail in synonyms:
                    if word_database:
                        word_database.update_word_details(detail)

                return synonyms
            except JSONParsingError as jpe:
                print(f"JSON parsing failed: {jpe.error_details}")

                traceback.print_exc()

                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})

                
                print("Ignoring the error, continuing...")

                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

                traceback.print_exc()

                raise e
            else:
                # print("Fetched synonyms successfully.")
                return synonyms

        raise RuntimeError("Failed to parse response after maximum retries.")







# Example usage
# words_db = WordsDatabase(db_path)
# word_fetcher = AdvancedWordFetcher(client)
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

    def _is_daytime_in_hk(self):
        hk_timezone = pytz.timezone('Asia/Hong_Kong')
        hk_time = datetime.now(hk_timezone)
        if self.enable_openai:
            return 9 <= hk_time.hour < 22  # Daytime hours in Hong Kong
        else:
            return False
        # return True
        # return False

    def fetch_new_words(self):
        # If original_words_list is None, fetch new words dynamically
        if not self.original_words_list:
            if self._is_daytime_in_hk():
                words = self.word_fetcher.fetch_words(10, self.db)
                openai_words = self.word_fetcher.fetch_word_details(words, self.db, num_words_phonetic=10)
                db_words = self.db.fetch_random_words(10)
            else:
                db_words = self.db.fetch_random_words(20)
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

if __name__ == "__main__":


    # Usage example
    client = OpenAI()


    # Database path
    db_path = 'words_phonetics.db'

    # Initialize database class
    words_db = WordsDatabase(db_path)

    # Initialize word fetcher
    word_fetcher = AdvancedWordFetcher(client)

    # words = word_fetcher.fetch_words(50, words_db)
    # print("words: ", words)


    # # Fetch word details
    # word_details = word_fetcher.fetch_word_details(words, words_db)
    # print("words: ", word_details)


    

    # Example usage
    words_db = WordsDatabase(db_path)
    word_fetcher = AdvancedWordFetcher(client)
    chooser = OpenAiChooser(words_db, word_fetcher)

    # chosen_word = chooser.choose()
    # print(chosen_word)

    chooser.get_current_words()

    # Close the database connection
    words_db.close()


