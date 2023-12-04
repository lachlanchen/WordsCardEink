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



class JSONParsingError(Exception):
    """Exception raised for errors in the JSON parsing."""
    def __init__(self, error_message, json_string, error_pos=None):
        self.error_message = error_message
        self.json_string = json_string
        self.error_pos = error_pos
        self.error_details = f"{self.error_message}\nError Position: {self.error_pos}" if self.error_pos else f"{self.error_message}"
        self.message = f"JSON String: {self.json_string}\n{self.error_details}"
        super().__init__(self.message)

class NotEnoughUniqueWordsError(Exception):
    """Exception raised when not enough unique words are fetched."""
    def __init__(self, required_num, fetched_num, duplicated_words, json_string):
        self.required_num = required_num
        self.fetched_num = fetched_num
        self.duplicated_words = duplicated_words
        self.json_string = json_string
        self.error_details = (f"Error: Required {self.required_num} unique words, but only {self.fetched_num} non-duplicated words were fetched. "
                        f"Duplicated words in local database: {', '.join(self.duplicated_words)}.")
        self.message = f"JSON String: {self.json_string}\n{self.error_details}"
        super().__init__(self.message)




# Function to count syllables based on dots
def count_syllables(word):
    # Count dots and stress symbols, subtract one if the first character is a stress symbol
    count = word.count('·') + word.count('ˈ') + word.count('ˌ')
    if word.startswith('ˈ') or word.startswith('ˌ'):
        count -= 1
    return count + 1

# Function to split words into syllables and get color for each syllable
def extract_kanji(japanese_text):
    """
    Remove parentheses, hiragana, and katakana from Japanese text, leaving only kanji.
    """
    # Regex to match hiragana, katakana, and characters in parentheses
    regex = u"[\u3040-\u309F\u30A0-\u30FF]|[（(].*?[)）]"
    return re.sub(regex, '', japanese_text)


def remove_second_parentheses(text):
    regex = re.compile(r'(（[^）]*）)(（[^）]*）)')
    return re.sub(regex, lambda match: match.group(1), text)


def transcribe_japanese(text):
    from pykakasi import kakasi

    kks = kakasi()
    kks.setMode("J", "H")  # Japanese to Hiragana
    kks.setMode("K", "H")  # Katakana to Hiragana
    conv = kks.getConverter()

    result = ""
    current_chunk = ""
    last_kanji_hiragana = ""
    is_kanji_or_katakana = False

    for char in text:
        if '\u4E00' <= char <= '\u9FFF':  # Kanji
            hiragana = conv.do(char)
            last_kanji_hiragana = hiragana  # Store the hiragana of the current kanji
            if not is_kanji_or_katakana:
                is_kanji_or_katakana = True
                current_chunk = ""
            current_chunk += hiragana
            result += char
        elif char == '々':  # Ideographic Iteration Mark
            if not is_kanji_or_katakana:
                is_kanji_or_katakana = True
                current_chunk = ""
            current_chunk += last_kanji_hiragana
            result += char
        elif '\u30A0' <= char <= '\u30FF':  # Katakana
            hiragana = conv.do(char)
            if not is_kanji_or_katakana:
                is_kanji_or_katakana = True
                current_chunk = ""
            current_chunk += hiragana
            result += char
        else:  # Hiragana or others
            if is_kanji_or_katakana:
                result += f"({current_chunk}){char}"
                is_kanji_or_katakana = False
            else:
                result += char

    if is_kanji_or_katakana:  # Remaining kanji or katakana chunk at the end
        result += f"({current_chunk})"

    return result


# Function to remove text inside parentheses
def remove_text_inside_parentheses(text):
    while '（' in text and '）' in text:
        start = text.find('（')
        end = text.find('）') + 1
        text = text[:start] + text[end:]
    return text

def clean_english(text):
    return text.replace(".", "·").replace("·ˈ", "ˈ").replace("·ˌ", "ˌ") #.replace(" ", "")

def clean_japanese(text):
    return text.replace(".", "").replace("·", "").replace("(", "（").replace(")", "）").replace(" ", "")

def remove_hiragana_inside_parentheses(text):
    # This regex matches hiragana or katakana inside parentheses and removes them, keeping the parentheses
    return re.sub(r'(?<=（)[ぁ-んァ-ンー-]+(?=）)', '', text)

def remove_content_inside_parentheses(text):
    # This regex matches anything inside parentheses and removes it, including the parentheses
    return re.sub(r'（[^）]*）', '', text)


def split_word(word):
    word = clean_english(word)

    if word.startswith('ˈ') or word.startswith('ˌ'):
        word = word[0] + word[1:].replace('ˈ', '·ˈ').replace('ˌ', '·ˌ')
    else:
        word = word.replace('ˈ', '·ˈ').replace('ˌ', '·ˌ').replace("-", "·-")

    syllables = word.split('·')

    return syllables


def split_word_with_color(word, colors):
        # Replace stress symbols with a preceding dot, except at the beginning

        syllables = split_word(word)
        
        color_syllables = [(syllable, colors[i % len(colors)]) for i, syllable in enumerate(syllables)]
        return color_syllables



def clean_and_transcribe(word_details):
    for word in word_details:
        # Update phonetic field if it exists
        if "phonetic" in word:
            word["phonetic"] = clean_english(word["phonetic"])

        # Update syllable_word field if it exists
        if "syllable_word" in word:
            word["syllable_word"] = clean_english(word.get("syllable_word", ""))

        # Update japanese_synonym field if it exists
        if "japanese_synonym" in word:
            word['japanese_synonym'] = clean_japanese(word['japanese_synonym'])

            # Clean and transcribe japanese_synonym
            clean_synonym = re.sub(r'（[ぁ-んァ-ンー-]+）', '', word["japanese_synonym"])  # Remove hiragana in parentheses
            word["japanese_synonym"] = clean_japanese(remove_second_parentheses(transcribe_japanese(clean_synonym)))  # Replace with your transcription function

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

    def word_exists(self, word):
        if self.conn:
            self.cursor.execute("SELECT COUNT(*) FROM words_phonetics WHERE word = ?", (word,))
            return self.cursor.fetchone()[0] > 0
        return False

    
    def insert_word_details(self, word_details, force=False):
        if self.conn:
            # print("insert words: ", word_details)
            word = word_details['word'] # .lower()
            syllable_word = clean_english(word_details['syllable_word']) # .lower())
            phonetic = clean_english(word_details['phonetic'])
            japanese_synonym = remove_second_parentheses(clean_japanese(word_details['japanese_synonym']))

            try:
                if force:
                    # UPSERT operation: Update if exists, insert if not
                    self.cursor.execute("""
                        INSERT INTO words_phonetics (word, syllable_word, phonetic, japanese_synonym)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(word) DO UPDATE SET
                            syllable_word = excluded.syllable_word,
                            phonetic = excluded.phonetic,
                            japanese_synonym = excluded.japanese_synonym;
                    """, (word, syllable_word, phonetic, japanese_synonym))
                else:
                    # Insert new record, ignore on duplicate
                    self.cursor.execute("""
                        INSERT INTO words_phonetics (word, syllable_word, phonetic, japanese_synonym)
                        VALUES (?, ?, ?, ?);
                    """, (word, syllable_word, phonetic, japanese_synonym))

                self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")

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




    def update_word_details(self, word_details, force=False):
        if self.conn:
            word = word_details.get('word', '') # .lower()

            # Prepare data and query for dynamic update
            data_to_update = []
            update_parts = []

            for key in ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 'french_synonym', 'chinese_synonym', 'kanji_synonym']:
                if key in word_details:
                    cleaned_value = clean_english(word_details[key]) if key != 'japanese_synonym' else clean_japanese(word_details[key])
                    data_to_update.append(cleaned_value)
                    update_parts.append(f"{key} = ?")
                    # print(f"Updated {word}. ")

            if not update_parts:
                # print("No data to update.")
                return

            query = f"UPDATE words_phonetics SET {', '.join(update_parts)} WHERE word = ?"
            data_to_update.append(word)

            try:
                # Execute the query with the values
                self.cursor.execute(query, data_to_update)
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")




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

    
    
    def process_word_rows(self, rows):
        """
        Process rows from the database into a list of dictionaries with word details.
        """
        processed_rows = []
        for row in rows:
            processed_row = {
                "word": clean_english(row[0]),
                "syllable_word": clean_english(row[1]),
                "phonetic": clean_english(row[2]),
                "japanese_synonym": row[3]  # Not applying clean_english to Japanese synonym
            }
            processed_rows.append(processed_row)
        return processed_rows


    def find_word_details(self, word):
        if self.conn:
            self.cursor.execute("SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics WHERE word = ?", (word,))
            result = self.cursor.fetchone()
            if result:
                return self.process_word_rows([result])[0]  # Process the single row
        return None

    def fetch_random_words(self, num_words):
        if self.conn:
            query = "SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics ORDER BY RANDOM() LIMIT ?"
            self.cursor.execute(query, (num_words,))
            rows = self.cursor.fetchall()
            return self.process_word_rows(rows)  # Process all fetched rows
        else:
            return []

    def fetch_last_10_words(self):
        if self.conn:
            query = "SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics ORDER BY rowid DESC LIMIT 10"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return self.process_word_rows(rows)  # Process all fetched rows
        else:
            return []

    def fetch_words_batch(self, offset, limit):
        if self.conn:
            query = "SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics LIMIT ? OFFSET ?"
            self.cursor.execute(query, (limit, offset))
            rows = self.cursor.fetchall()
            return self.process_word_rows(rows)  # Process all fetched rows
        else:
            return []




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
            parsed_json = json.loads(json_string)
            if len(parsed_json) == 0:
                raise JSONParsingError("Parsed JSON string is empty", json_string)
            
            return parsed_json
        except json.JSONDecodeError as e:
            raise JSONParsingError(f"JSON Decode Error: {e}", json_string)

    def extract_and_parse_words(self, text, num_words, word_database):
        bracket_pattern = r'\[.*?\]'
        matches = re.findall(bracket_pattern, text, re.DOTALL)

        if not matches:
            raise JSONParsingError("No JSON string found in text", text)

        json_string = matches[0]

        try:
            parsed_json = json.loads(json_string)
            if len(parsed_json) == 0:
                raise JSONParsingError("Parsed JSON string is empty", json_string)

            unique_words = [word for word in parsed_json if not word_database.word_exists(word)]
            if len(unique_words) < num_words // 2:
                duplicated_words = set(parsed_json) - set(unique_words)
                fetched_num = len(unique_words)
                raise NotEnoughUniqueWordsError(len(parsed_json), fetched_num, list(duplicated_words), json_string)

            return parsed_json
        except json.JSONDecodeError as e:
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

        random.shuffle(unique_words)

        # Check if there are enough unique words
        if len(unique_words) < num_words:
            raise ValueError(f"Not enough unique words. Only {len(unique_words)} unique words found.")

        # Return the specified number of unique words
        return unique_words[:num_words]


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
                "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
                # "and include the word's tendency or special characteristic next to each word."
                "The output should be like ['word 1', 'word 2', ..., 'word N']."
            )
        else:
            user_message = (
                # f"Think wildly and provide me with a python list of {num_words_scaled} unique advanced lowercase words that are often used in formal readings. "
                f"Take a deep breath and think wildly with your imagination and provide me with {num_words_scaled} words. "
                "Choose commonly used to advanced that are often used in daily expression or formal readings in various areas. "
                # "The output plain json should be like ['word 1', 'word 2', ..., 'word N'] starts with [ and end with ]. "
                f"To provide some randomness, a list of words not existing in our local database is provided below. "
                "You can ignore or use it if your output have duplications. "
                f"Output non-capitalized wordes if not necessary and same format as: \n{json.dumps(local_words)}."
            )

        messages = [
            {"role": "system", "content": "You are an assistant with a vast vocabulary and creativity. You almost know every English words in this universe. "},
            {"role": "user", "content": user_message}
        ]

        for try_num in range(self.max_retries):
            try:
                

                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )

                words_list = self.extract_and_parse_words(response.choices[0].message.content, num_words, word_database)

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

                continue

            except NotEnoughUniqueWordsError as not_enough_error:

                print(f"Not enough word: {not_enough_error.error_details}")
                
                # messages.append({"role": "system", "content": response.choices[0].message.content})

                messages.append({"role": "system", "content": not_enough_error.json_string})
                messages.append({"role": "user", "content": f"{not_enough_error.error_details}. Please take a deep breath and use your imagination to think more widely. "})

                # messages.insert(0, {"role": "user", "content": f"{not_enough_error.error_details}. Please take a deep breath and use your imagination to think more widely. "})
                # messages.insert(0, {"role": "system", "content": not_enough_error.json_string})

                # print(f"Retrying: try the {try_num+2} times...")

                continue


            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise e

        if not unique_words:
            raise RuntimeError("Failed to fetch unique words after maximum retries.")

        return unique_words

    def fetch_word_details(self, words, word_database, num_words_phonetic=10):
        random_words = random.sample(words, min(num_words_phonetic, len(words)))
        

        detailed_list_message = (
            "For each word, we need to correctly format the syllable_word (with · separating syllables), phonetic transcription (phonemes also separated by ·), and the Japanese synonym. "
            "Ensure the word syllables and phonetic separation are syncronized. "
            "In the case of Japanese synonyms, the hiragana (furigana) should follow directly after the kanji and katakana. For example, 'その後' should be followed by its furigana '（ご)', instead of repeating the kanji as in 'その後（そのご)'. "
            "Consider '容易にする' – the correct form is '容易（ようい）にする', placing 'する' outside the parentheses to align with standard formatting. "
            "In 'もの悲しい', the proper format is 'もの悲（かな）しい', where the hiragana directly follows its respective kanji. "
            "Remember, no dots in the Japanese synonym and hiragana should be placed inside parentheses right after the kanji/katakana. "
            "The output plain json format should RESEMBLE: {}. "
            "The words to process are: {}."
        ).format(json.dumps(self.examples, ensure_ascii=False, separators=(',', ':')), ', '.join(random_words))

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
                    word_database.insert_word_details(detail)
                
                # self.recheck_syllable_and_phonetic(word_phonetics, word_database)
                # self.recheck_japanese_synonym(word_phonetics, word_database)

                print("Starting comparing separation...")
                self.split_and_compare_phonetic_syllable(word_phonetics.copy(), word_database)
                print("Starting check Japanese...")
                self.recheck_japanese_synonym_with_conditions(word_phonetics.copy(), word_database)
                print("Starting fetching Arabic...")
                self.fetch_arabic_synonyms([word["word"] for word in word_phonetics], word_database)

                word_phonetics = [word_database.find_word_details(word["word"]) for word in word_phonetics]

                self.examples= word_phonetics[0:2]
                self.save_examples()
                return word_phonetics
            except JSONParsingError as jpe:
                # print(f"JSON parsing failed: {jpe.error_details}")
                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})
                continue
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
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
            "Each syllable should be marked with a central dot (·) in both fields. Adjust the 'phonetic' field to match the syllable divisions in 'syllable_word'. "
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
                        word_database.insert_word_details(detail)

                # self.recheck_syllable_and_phonetic(word_phonetics, word_database)
                # self.recheck_japanese_synonym(word_phonetics, word_database)
                return word_phonetics

            except JSONParsingError as jpe:
                # print(f"JSON parsing failed: {jpe.error_details}")
                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})
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
        pprint(word_details)
        print("\n")



        # Prepare examples excluding the Japanese synonyms
        example_list = [{k: v for k, v in example.items() if k != 'japanese_synonym'} for example in self.examples]
        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'syllable_word', 'phonetic']} for word in word_details]
        words = [{k: v for k, v in word.items() if k in ['word']} for word in word_details]

        word_details = clean_and_transcribe(word_details)


        if not messages:
            # Prepare message for rechecking details
            detailed_list_message = (
                "Please ensure the syllable separation in the 'phonetic' transcription aligns with the 'syllable_word' separation. "
                "Each syllable should be marked with a central dot (·) in both fields. Adjust the 'phonetic' field or the syllable divisions in 'syllable_word' to be matched. "
                "Here is the word detail for correction and output SAME format: {}."
            ).format(json.dumps(word_details, ensure_ascii=False, separators=(',', ':')))

            messages = [
                {"role": "system", "content": " \
                    You are an assistant skilled in linguistics, \
                    capable of providing accurate and detailed phonetic and linguistic attributes for given words. \
                    You are excellent in separate words and their phonetics into consistent and accurate separations with '·'."},
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
                        print(f"Updating word with", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                        print("\n")
                        word_database.update_word_details(detail)
                return word_phonetics
            except JSONParsingError as jpe:
                # print(f"JSON parsing failed: {jpe.error_details}")
                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})
                continue
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
                raise e
            else:
                # print("Fetched and rechecked word details successfully.")
                return word_phonetics


        raise RuntimeError("Failed to parse response after maximum retries.")

    
    # def split_and_compare_phonetic_syllable(self, word_details, word_database):
    #     rechecked_word_details = []

    #     # Extract the relevant parts from word_details
    #     word_details = [{k: v for k, v in word.items() if k in ['word', 'syllable_word', 'phonetic']} for word in word_details]

    #     # Initial system message for context
    #     basic_message = {
    #         "role": "system", 
    #         "content": "You are an assistant skilled in linguistics, capable of providing accurate and detailed phonetic and linguistic attributes for given words. You excel in separating words and their phonetics into consistent and accurate separations with '·'."
    #     }

        



    #     for word_detail in word_details:

    #         # Example format for output with empty syllable_word and phonetic values
    #         example_format = json.dumps(
    #             [{'word': word_detail['word'], 'syllable_word': '', 'phonetic': ''}], 
    #             ensure_ascii=False, 
    #             separators=(',', ':')
    #         )
            

    #         for n_try in range(self.max_retries):

    #             word = word_detail.get('word', '')
    #             syllable_word = word_detail.get('syllable_word', '')
    #             phonetic = word_detail.get('phonetic', '')
    #             messages = [basic_message]

    #             detailed_list_message = (
    #                 "Please ensure the syllable separation in the 'phonetic' transcription aligns with the 'syllable_word' separation. "
    #                 "Each syllable should be marked with a central dot (·) in both fields. Adjust the 'phonetic' field or the syllable divisions in 'syllable_word' to be matched. "
    #                 "Here is the word detail for correction and output SAME format AS: {}."
    #             ).format(example_format)

    #             if not syllable_word or not phonetic:
    #                 # Message for missing data
    #                 message = ("The syllable_word or phonetic of word '{}' is missing. Please provide the missing data. REMEMBER THE OUTPUT JSON FORMAT: {}. ").format(word, example_format)
    #                 # print(message)
    #                 if n_try == 0:
    #                     messages.append({"role": "user", "content": message + detailed_list_message})
    #                 else:
    #                     messages.append({"role": "user", "content": message})
    #                 word_detail = self.recheck_syllable_and_phonetic([word_detail], word_database, messages)[0]
    #                 messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(',', ':'))})
    #             elif len(split_word(syllable_word)) != len(split_word(phonetic)):
    #                 # Constructing and inserting the first mismatch message
    #                 mismatch_message = self.create_mismatch_message(word, syllable_word, phonetic) + ("REMEMBER THE OUTPUT JSON FORMAT: {}. ").format(example_format)
    #                 # print("mismatched: ", mismatch_message)
    #                 if n_try == 0:
    #                     messages.append({"role": "user", "content": mismatch_message + detailed_list_message})
    #                 else:
    #                     messages.append({"role": "user", "content": mismatch_message})
    #                 word_detail = self.recheck_syllable_and_phonetic([word_detail], word_database, messages)[0]
    #                 messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(',', ':'))})
    #             else:
    #                 rechecked_word_details.append(word_detail)
    #                 # print("Updated word: ", word)
    #                 break

    #         if n_try == self.max_retries - 1:
    #             print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

    #     return rechecked_word_details


    # def create_mismatch_message(self, word, syllable_word, phonetic):
    #     syllables = split_word(syllable_word)
    #     phonetics = split_word(phonetic)
    #     mappings = self.map_syllables_phonetics(syllables, phonetics)

    #     message = (f"Mismatch of word syllable and phonetic separation in '{word}':\n"
    #                # f"Syllables: {syllable_word}\n"
    #                # f"Phonetics: {phonetic}\n"
    #                f"Mapped Syllable-Phonetic Pairs:\n{mappings}\n"
    #                "Sync each syllable with a corresponding phonetic component, even if it diverges from typical linguistic patterns. You can merge and increase syllables in either sides to make it consistent. ")
    #     return message

    def split_and_compare_phonetic_syllable(self, word_details, word_database):
        rechecked_word_details = []

        # Extract the relevant parts from word_details
        word_details = [{k: v for k, v in word.items() if k in ['word', 'syllable_word', 'phonetic']} for word in word_details]

        # Initial system message for context
        basic_message = {
            "role": "system", 
            "content": "\
                You are an assistant skilled in linguistics, \
                capable of providing accurate and detailed phonetic and linguistic attributes for given words. \
                You excel in separating words and their phonetics into consistent and accurate separations with '·'."
        }

        for word_detail in word_details:
            for n_try in range(self.max_retries):
                word = word_detail.get('word', '')
                syllable_word = word_detail.get('syllable_word', '')
                phonetic = word_detail.get('phonetic', '')
                messages = [basic_message]

                example_format = json.dumps(
                    [{'word': word, 'syllable_word': '', 'phonetic': ''}], 
                    ensure_ascii=False, 
                    separators=(',', ':')
                )

                detailed_list_message = (
                    # "Adjust the 'phonetic' field or the syllable divisions in 'syllable_word' to be matched. "
                    "Please ensure the syllable separation in the 'phonetic' transcription aligns with the 'syllable_word' separation. "
                    "Each syllable should be marked with a central dot (·) in both fields. "
                    # "Here is the word detail for correction and output SAME format AS: {}."
                ).format(example_format)

                if not syllable_word or not phonetic:
                    # Message for missing data
                    message = ("The syllable_word or phonetic of word '{}' is missing. Please provide the missing data. REMEMBER THE OUTPUT JSON FORMAT: {}. ").format(word, example_format)
                    if n_try == 0:
                        messages.append({"role": "user", "content": message + detailed_list_message})
                    else:
                        messages.append({"role": "user", "content": message})
                    word_detail = self.recheck_syllable_and_phonetic([word_detail], word_database, messages)[0]
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(',', ':'))})
                elif len(split_word(syllable_word)) != len(split_word(phonetic)):
                    # Directly create the mismatch message here
                    syllables = split_word(syllable_word)
                    phonetics = split_word(phonetic)
                    mappings = self.map_syllables_phonetics(syllables, phonetics)

                    mismatch_message = (
                        "Sync each syllable with a corresponding phonetic component, "
                        "even if it diverges from typical linguistic patterns. "
                        "You can merge and increase syllables in either sides to make it align. "
                        f"Mismatch of word syllable and phonetic separation in '{word}':\n"
                        f"Mapped Syllable-Phonetic Pairs:\n{mappings}\n"
                        f"THE OUTPUT JSON FORMAT: {example_format}. "
                    )
                    if n_try == 0:
                        messages.append({"role": "user", "content": detailed_list_message + mismatch_message})
                    else:
                        messages.append({"role": "user", "content": mismatch_message})
                    word_detail = self.recheck_syllable_and_phonetic([word_detail], word_database, messages)[0]
                    messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(',', ':'))})
                else:
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
        mapping = '\n'.join([f"{syl} ↔ {phon}" for syl, phon in zip(syllables, phonetics)])
        return mapping


    def recheck_japanese_synonym(self, word_details, word_database=None, messages=''):
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("Rechecking Japanese synonym: ")
        pprint(word_details)
        print("\n")

        

        # Prepare examples including only the word and Japanese synonym
        example_list = [{k: v for k, v in example.items() if k in ['word', 'japanese_synonym']} for example in self.examples]

        words = [{k: v for k, v in word.items() if k in ['word']} for word in word_details]

        # Extract the relevant parts from word_details
        word_details_formatted = [{k: v for k, v in word.items() if k in ['word', 'japanese_synonym']} for word in word_details]
        word_details = clean_and_transcribe(word_details)
        # print("Clean and transribed: ", word_details)

        # detailed_list_message = (
        #     "{}"
        #     "Could you correct the furigana/hiragana of kanji/katakana inside the parentheses as needed and output same format? \n {}"
        # ).format(message, json.dumps(word_details_formatted, ensure_ascii=False, separators=(',', ':')))

        if not messages:
            detailed_list_message = (
                "Could you correct the furigana/hiragana of kanji/katakana inside the parentheses as needed and output same format? \n {}"
            ).format(json.dumps(word_details_formatted, ensure_ascii=False, separators=(',', ':')))

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
                        print("Updating word with ", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                        print("\n")
                        word_database.update_word_details(detail)

                return word_phonetics
            except JSONParsingError as jpe:
                # print(f"JSON parsing failed: {jpe.error_details}")
                # messages.append({"role": "system", "content": response.choices[0].message.content})
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})
                continue
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
                raise e
            else:
                # print("Fetched and rechecked word details successfully.")
                return word_phonetics                

        raise RuntimeError("Failed to parse response after maximum retries.")



    def recheck_japanese_synonym_with_conditions(self, word_details, word_database=None):
        rechecked_word_details = []

        basic_message = {
            "role": "system", 
            "content": "\
                You are an assistant skilled in linguistics, \
                knowing Japanese extremly well. \
                You are excellent in providing hiragana (furigana) for consecutive kanji/katakana."
        }

        for word_detail in word_details:
            

            for n_try in range(self.max_retries):

                japanese_synonym = word_detail.get('japanese_synonym', '')
                # print("original japanese_synonym: ", japanese_synonym)

                word_detail_transribed = clean_and_transcribe([word_detail])
                # print("word detail transribed:")
                # pprint(word_detail_transribed)


                # Extract the relevant parts from word_details
                word_detail_transribed = [{k: v for k, v in word.items() if k in ['word', 'japanese_synonym']} for word in word_detail_transribed]


                word_detail_example = [{
                    k: (v if k != 'japanese_synonym' else re.sub(r'（[ぁ-んァ-ンー-]+）', '（）', v))
                    for k, v in word.items() 
                    if k in ['word', 'japanese_synonym']
                } for word in word_detail_transribed]

                detailed_list_message = (
                    "Treat 々 as kanji. Return the same back if japanese_synonym are all hiragana without parentheses. Could you provide the furigana/hiragana of kanji/katakana inside the parentheses as needed and output SAME format? \n {}"
                ).format(json.dumps(word_detail_example, ensure_ascii=False, separators=(',', ':')))

                messages = [basic_message]

                # Remove all hiragana and parentheses
                cleaned_synonym = re.sub(r'（[ぁ-んァ-ンー-]+）', '', japanese_synonym).replace("（", "").replace("）", "")

                if cleaned_synonym == '':
                    # print("All hiragana!")
                    clean_synonym = re.sub(r'（[ぁ-んァ-ンー-]+）', '', japanese_synonym)
                    word_detail['japanese_synonym'] = clean_synonym
                    word_database.update_word_details(word_detail)
                    
                    rechecked_word_details.append(word_detail)
                    break
                else:
                    without_hiragana_original = remove_hiragana_inside_parentheses(japanese_synonym)
                    cleaned_transcribed = word_detail_transribed[0]["japanese_synonym"]
                    without_hiragana_transcribed = remove_hiragana_inside_parentheses(cleaned_transcribed)

                    # print(without_hiragana_original, " ? ", without_hiragana_transcribed)

                    if without_hiragana_original != without_hiragana_transcribed or not japanese_synonym:
                        message = "No Japanese synonym provided. Please provide one anyway even if it's incorrect." if not japanese_synonym \
                                else "The hiragana inside the parentheses is generated by pykakasi. Please recheck."
                        if n_try == 0:
                            messages.append({"role": "user", "content": message + detailed_list_message})
                        else:
                            messages.append({"role": "user", "content": message})

                        print(
                            f"Furigana position incorrect, rechecking for word '{word_detail['word']}':\n"
                            f"Full: {json.dumps(japanese_synonym, ensure_ascii=False, separators=(',', ':'))}\n"
                            f"Original: {without_hiragana_original}\n"
                            f"Kakasi: {without_hiragana_transcribed}\n"
                        )

                        word_detail = self.recheck_japanese_synonym([word_detail], word_database, messages)[0]
                        messages.append({"role": "system", "content": json.dumps([word_detail], ensure_ascii=False, separators=(",", ":"))})
                        # rechecked_word_details.append(word_detail)
                    else:
                        rechecked_word_details.append(word_detail)
                        # print("Updated word: ", word_detail["word"])
                        break


            if n_try == self.max_retries - 1:
                print(f"Max retries reached for word '{word_detail['word']}'. Final attempt may still have issues.")

        return rechecked_word_details


    def fetch_arabic_synonyms(self, words, word_database):
        # Prepare examples in the required format
        example_list = [
            {"word": "sun", "arabic_synonym": "شمس"},
            {"word": "moon", "arabic_synonym": "قمر"}
        ]
        examples_json = json.dumps(example_list, ensure_ascii=False, separators=(',', ':'))

        # Prepare the prompt with examples
        detailed_list_message = (
            "Based on the following examples, provide the Arabic synonym for each word in the list. "
            "The output should be in a plain JSON format, as a list of dictionaries, "
            "where each dictionary contains 'word' and 'arabic_synonym'. "
            "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
            "Examples: {}\n"
            "The words to process are: {}."
        ).format(examples_json, ', '.join(words))

        messages = [
            {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. You are excellent in providing Arabic synonyms."},
            {"role": "user", "content": detailed_list_message}
        ]

        # Fetch Arabic synonyms using OpenAI
        return self.attempt_to_fetch_synonyms(messages, word_database)


    def fetch_chinese_synonyms(self, words, word_database):
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


    def fetch_french_synonyms(self, words, word_database):
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
            "Format the list for compatibility with json.loads, starting with [ and ending with ]. "
            "Examples: {}\n"
            "The words to process are: {}."
        ).format(examples_json, ', '.join(words))

        messages = [
            {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed linguistic attributes for given words. You are excellent in providing French synonyms."},
            {"role": "user", "content": detailed_list_message}
        ]

        # Fetch French synonyms using OpenAI
        return self.attempt_to_fetch_synonyms(messages, word_database)

    def attempt_to_fetch_synonyms(self, messages, word_database):
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )

                synonyms = self.extract_and_parse_json(response.choices[0].message.content)
                # Save synonyms to database
                for detail in synonyms:
                    if word_database:
                        word_database.update_word_details(detail)

                return synonyms
            except JSONParsingError as jpe:
                # print(f"JSON parsing failed: {jpe.error_details}")
                messages.append({"role": "system", "content": jpe.json_string})
                messages.append({"role": "user", "content": f"JSON parsing failed: {jpe.error_details}"})
                continue
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
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
        # return 8 <= hk_time.hour < 24  # Daytime hours in Hong Kong
        return True
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
                openai_words = []

            self.current_words = openai_words + db_words
            random.shuffle(self.current_words)
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
        word = clean_and_transcribe([word])[0]
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


