
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

from words_data_utils import (
    # Exception Classes
    JSONParsingError,
    NotEnoughUniqueWordsError,
    
    # Utility Functions - Random/Sampling
    random_shuffle,
    random_sample,
    
    # Text Processing Functions - General
    count_syllables,
    clean_english,
    clean_japanese,
    split_word,
    split_word_with_color,
    
    # Japanese Text Processing Functions
    extract_kanji,
    remove_second_parentheses,
    remove_text_including_parentheses,
    remove_text_inside_parentheses,
    remove_content_inside_parentheses,
    remove_japanese_letter_including_parentheses,
    remove_japanese_letter_inside_parentheses,
    remove_hiragana_including_parentheses,
    remove_hiragana_inside_parentheses,
    remove_hiragana_and_parentheses,
    transcribe_japanese,
    
    # Japanese Analysis Functions
    count_hiragana_repetitions,
    smallest_non_zero_repetition,
    compare_repetition_results,
    
    # Word Details Processing Functions
    clean_and_transcribe,
    clean_word_details,
    
    # Chooser Classes
    OpenAiChooser,
    EmojiWordChooser,
)

class WordsDatabase:
    # def __init__(self, db_path):
    #     self.db_path = db_path
    #     self.conn = None
    #     if os.path.exists(db_path):
    #         self.conn = sqlite3.connect(db_path)
    #         self.cursor = self.conn.cursor()

    #         self.create_field_if_not_exists("words_phonetics", "kanji_synonym", "TEXT")
    #         self.create_field_if_not_exists("words_phonetics", "chinese_synonym", "TEXT")
    #         self.create_field_if_not_exists("words_phonetics", "simplified_chinese_synonym", "TEXT")
    #         self.create_field_if_not_exists("words_phonetics", "arabic_synonym", "TEXT")
    #         self.create_field_if_not_exists("words_phonetics", "french_synonym", "TEXT")

    def __init__(self, db_path, table_name='words_phonetics', fields=None):
        self.db_path = db_path
        self.table_name = table_name
        self.fields = fields
        self.conn = None

        if os.path.exists(db_path):
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

            self.create_table_if_not_exists()
            if fields:
                self.create_fields_if_not_exists()

            # Ensure all synonym and phonetic fields exist
            self.ensure_all_synonym_fields()

    def ensure_all_synonym_fields(self):
        """Ensure all synonym and phonetic fields exist in the database"""
        synonym_fields = [
            ("kanji_synonym", "TEXT"),
            ("chinese_synonym", "TEXT"),
            ("simplified_chinese_synonym", "TEXT"), 
            ("arabic_synonym", "TEXT"),
            ("arabic_phonetic", "TEXT"),
            ("arabic_transliteration", "TEXT"),
            ("french_synonym", "TEXT"),
            ("french_phonetic", "TEXT")
        ]
    
        for field_name, field_type in synonym_fields:
            self.create_field_if_not_exists("words_phonetics", field_name, field_type)

    def sync_database_to_csv(self, csv_file_path=None):
        """Export the database table to a CSV file."""
        if not csv_file_path:
            base_name = os.path.splitext(os.path.basename(self.db_path))[0]
            csv_file_path = f"data/{base_name}/{self.table_name}.csv"
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        if self.conn:
            try:
                query = f"SELECT * FROM {self.table_name}"
                df = pd.read_sql_query(query, self.conn)
                df.to_csv(csv_file_path, index=False)
                print(f"Database exported to {csv_file_path}")
            except Exception as e:
                print(f"Error during database export: {e}")


    def update_database_from_csv(self, csv_file_path=None):
        """Update the database using data from a CSV file."""
        if not csv_file_path:
            base_name = os.path.splitext(os.path.basename(self.db_path))[0]
            csv_file_path = f"data/{base_name}/{self.table_name}.csv"
        if self.conn:
            try:
                # Load CSV data into a DataFrame
                df = pd.read_csv(csv_file_path)

                # Add columns from CSV to database if they don't exist
                for column in df.columns:
                    self.cursor.execute(f"PRAGMA table_info({self.table_name});")
                    existing_columns = [row[1] for row in self.cursor.fetchall()]
                    if column not in existing_columns:
                        self.create_field_if_not_exists(self.table_name, column, "TEXT")

                # Insert or update rows from the CSV
                for _, row in df.iterrows():
                    columns = ', '.join(row.index)
                    placeholders = ', '.join(['?'] * len(row))
                    update_clause = ', '.join([f"{col}=excluded.{col}" for col in row.index])

                    query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        ON CONFLICT(id) DO UPDATE SET
                        {update_clause};
                    """
                    self.cursor.execute(query, tuple(row))
                    self.conn.commit()
            except Exception as e:
                print(f"Error during database update: {e}")

    def sync_from_csv_to_database(self, csv_file_path=None, confirm=False):
        """Sync the database with the structure and content of a CSV file."""
        if not csv_file_path:
            base_name = os.path.splitext(os.path.basename(self.db_path))[0]
            csv_file_path = f"data/{base_name}/{self.table_name}.csv"

        if not confirm:
            print("Sync operation requires confirmation. Set confirm=True to proceed.")
            return

        if self.conn:
            try:
                # Load CSV data
                df_csv = pd.read_csv(csv_file_path)
                csv_columns = set(df_csv.columns)

                # Get database columns
                self.cursor.execute(f"PRAGMA table_info({self.table_name});")
                db_columns = {row[1] for row in self.cursor.fetchall()}

                # Add new columns from CSV to database
                for column in csv_columns - db_columns:
                    self.create_field_if_not_exists(self.table_name, column, "TEXT")

                # Delete columns from database not in CSV
                for column in db_columns - csv_columns:
                    self.delete_column_if_exists(self.table_name, column)

                # Update or insert rows
                for _, row in df_csv.iterrows():
                    columns = ', '.join(row.index)
                    placeholders = ', '.join(['?'] * len(row))
                    update_clause = ', '.join([f"{col}=excluded.{col}" for col in row.index])

                    query = f"""
                        INSERT INTO {self.table_name} ({columns})
                        VALUES ({placeholders})
                        ON CONFLICT(id) DO UPDATE SET
                        {update_clause};
                    """
                    self.cursor.execute(query, tuple(row))

                # Delete rows not in CSV
                df_db = pd.read_sql_query(f"SELECT * FROM {self.table_name}", self.conn)
                db_ids = set(df_db['id'])
                csv_ids = set(df_csv['id'])
                ids_to_delete = db_ids - csv_ids
                for id in ids_to_delete:
                    self.cursor.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (id,))

                self.conn.commit()
                print("Database successfully synced with CSV.")

            except Exception as e:
                print(f"Error during syncing database with CSV: {e}")




    def create_table_if_not_exists(self):
        if self.conn:
            try:
                self.cursor.execute(f"PRAGMA table_info({self.table_name});")
                if not self.cursor.fetchall():
                    # If fields are provided, use them; otherwise, create default columns
                    if self.fields:
                        columns = ', '.join([f"{name} {type}" for name, type in self.fields])
                    else:
                        columns = "id INTEGER PRIMARY KEY, create_date TEXT, update_date TEXT"
                    
                    self.cursor.execute(f"CREATE TABLE {self.table_name} ({columns});")
                    self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")
                traceback.print_exc()

    



    def create_fields_if_not_exists(self):
        if self.conn:
            try:
                for field_name, field_type in self.fields:
                    self.create_field_if_not_exists(self.table_name, field_name, field_type)
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")
                traceback.print_exc()


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
    


    def insert_word_details(self, word_details, force=False):
        if self.conn:
            # Extracting word details
            word = word_details.get('word', '').lower()
            # fields_to_update = ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 
            #                     'french_synonym', 'chinese_synonym', 'simplified_chinese_synonym', 'kanji_synonym']

            fields_to_update = ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 
                                'french_synonym', 'chinese_synonym', 'simplified_chinese_synonym', 'kanji_synonym',
                                'arabic_phonetic', 'arabic_transliteration', 'french_phonetic']

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
        fields_to_update = ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 
                            'french_synonym', 'chinese_synonym', 'simplified_chinese_synonym', 'kanji_synonym',
                            'arabic_phonetic', 'arabic_transliteration', 'french_phonetic']

        if self.conn:
            word = word_details.get('word', '').lower()  # Ensure the word is in lowercase

            # Prepare data and query for dynamic update
            data_to_update = []
            update_parts = []

            # for key in ['syllable_word', 'phonetic', 'japanese_synonym', 'arabic_synonym', 'french_synonym', 'chinese_synonym', 'simplified_chinese_synonym', 'kanji_synonym']:
            for key in fields_to_update:
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
                print(f"Successfully updated word '{word}' with fields: {[key for key in word_details.keys() if key != 'word']}")
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