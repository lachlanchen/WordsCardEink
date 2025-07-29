import json
import traceback  # Make sure to import traceback for error handling
from openai import OpenAI

from pprint import pprint

from words_data import WordsDatabase, AdvancedWordFetcher
# from words_data import AdvancedWordFetcher
import csv

import os

from datetime import datetime

class PhoneticRechecker(AdvancedWordFetcher):
    def __init__(self, max_retries=3):
        super().__init__(**{"max_retries": 3})
        self.processed_csv = "word_phonetics_processed.csv"
        self.log_folder = "logs-word-phonetics"

        # self.client = OpenAI()
        # # self.model_name = model_name
        # self.model_name = "gpt-4-0125-preview"
        # self.max_retries = max_retries
        # self.word_database = word_database  # Assuming there's a word_database for updating word details

    def save_log(self, words, prompt, response_content):
        """Save the prompt and response content to a JSON file, adjusting for multiple words."""
        # If 'words' is a list, join the word entries; else, use the word directly
        if isinstance(words, list):
            word_list = [word_dict['word'] for word_dict in words]
            word_str = ",".join(word_list)
        else:
            word_str = words  # Assuming 'words' is a string when not a list

        # Ensure the log folder exists
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        
        # Format the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Construct the file path with words joined by commas
        filepath = os.path.join(self.log_folder, f"{word_str}-{timestamp}.json")
        
        # Data to save
        data_to_save = {
            "prompt": prompt,
            "response": response_content
        }
        
        # Save the data to a JSON file
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data_to_save, file, indent=4, ensure_ascii=False)

    def word_exists(self, word):
        """Check if a word exists in the processed CSV."""
        if not os.path.exists(self.processed_csv):
            return False
        with open(self.processed_csv, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['word'] == word:
                    return True
        return False

    def save_to_csv(self, data):
        filename = self.processed_csv
        """Save processed data to a CSV file."""
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['word', 'syllable_word', 'phonetic']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for item in data:
                writer.writerow(item)

    def sanitize_input_and_normalize_syllables(self, data):
        """
        Sanitize the input by removing dots and slashes from all values, from the word, phonetics,
        to each syllable. Then, join the word_syllables to check if it equals the complete word.
        If not, find the repetition in the word syllable, then remove the repeated in the later syllable.
        
        Args:
        - data: A JSON-like list of dictionaries with 'word', 'phonetics', and 'syllables' keys.
        
        Returns:
        - The modified data with updates on whether the sanitized syllables match the original word
          and any necessary corrections applied.
        """
        for item in data:
            # Remove dots and slashes from word and phonetics
            item['word'] = item['word'].replace('.', '').replace('/', '')
            # item['phonetics'] = item['phonetics'].replace('.', '').replace('/', '')
            
            # Sanitize syllables
            sanitized_syllables = []
            for syl in item['syllables']:
                sanitized = {
                    'word_syllable': syl['word_syllable'].replace('.', '').replace('/', ''),
                    'phonetic_syllable': syl['phonetic_syllable'].replace('.', '').replace('/', '')
                }
                sanitized_syllables.append(sanitized)
            item['syllables'] = sanitized_syllables
            
            # Join the word_syllables and compare with the sanitized word
            joined_syllables = ''.join([syl['word_syllable'] for syl in sanitized_syllables])
            if joined_syllables == item['word']:
                item['match'] = True
            else:
                # Find and remove repeated letters in later syllables if the initial comparison fails
                corrected_syllables = []
                for syl in sanitized_syllables:
                    if corrected_syllables:
                        # Check for repetition with the last character of the previous syllable
                        if corrected_syllables[-1]['word_syllable'][-1] == syl['word_syllable'][0]:
                            corrected_syllable = syl['word_syllable'][1:]  # Remove the repeated character
                        else:
                            corrected_syllable = syl['word_syllable']
                        corrected_syllables.append({'word_syllable': corrected_syllable, 'phonetic_syllable': syl['phonetic_syllable']})
                    else:
                        corrected_syllables.append(syl)  # First syllable, add without checking
                
                corrected_word = ''.join([syl['word_syllable'] for syl in corrected_syllables])
                item['corrected_word'] = corrected_word
                item['match'] = corrected_word == item['word']
        
        return data

    def convert_to_dot_separated_json(self, data):
        """
        Convert the given data with sanitized and potentially corrected syllables into a new format
        where the syllable representation of the word is separated by central dots, and phonetics are
        consolidated into a single string.
        
        Args:
        - data: A list of dictionaries with 'word', 'phonetics', and 'syllables' keys, where syllables
                are already sanitized and potentially corrected.
        
        Returns:
        - A new list of dictionaries in the format [{'word': '', 'syllable_word': '', 'phonetic': ''}].
        """
        result = []
        for item in data:
            # Join syllables with a central dot for the word representation
            syllable_word = '·'.join([syl['word_syllable'] for syl in item['syllables']]).replace("· ·", " ")
            
            # The phonetics are already provided as a single consolidated string, so we use it directly
            phonetic = '·'.join([syl['phonetic_syllable'] for syl in item['syllables']]).replace("· ·", " ")
            
            result.append({
                'word': item['word'],
                'syllable_word': syllable_word,
                'phonetic': phonetic
            })
        return result
        # Implementation remains the same as provided earlier.

    def recheck_word_phonetics_with_paired_tuple(self, words, word_database, force=False):

        words_filtered = []
        for word in words:
            if self.word_exists(word) and not force:
                print(f"Skipping '{word}' as it exists in {self.processed_csv}.")
                continue
            else:
                words_filtered.append(word)

        if len(words_filtered) == 0:
            return []



        words_string = json.dumps(words_filtered, indent=2, ensure_ascii=False)



        prompt = (
            "Provide the correct and standard phonetics of the word, "
            "and the syllable of this word with its phonetics in pair. "
            "Make sure the join of syllables will be exact the original word and phonetics. "
            "Treat space as a syllable. "
            f"The word: {words} "
            "Output only json format: "
            "[{"
            "\"word\": \"\","
            "\"phonetics\": \"\","
            "\"syllables\": ["
            "{\"word_syllable\": \"\", \"phonetic_syllable\":\" \"},"
            "]"
            "},]"
        )

        messages = [{"role": "system", "content": prompt}]
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name[1],
                    messages=messages
                )

                result = response.choices[0].message.content
                print("result: ", result)

                

                # Assuming `response` is the desired JSON data structure from OpenAI
                word_phonetics = self.extract_and_parse_json(result)

                pprint(word_phonetics)

                # break

                # Process the response through the provided functions
                data_sanitized = self.sanitize_input_and_normalize_syllables(word_phonetics)
                converted_data = self.convert_to_dot_separated_json(data_sanitized)

                self.save_to_csv(converted_data)

                self.save_log(word_phonetics, prompt, result)

                print("OpenAI: ")
                pprint(converted_data)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                # Save word details to database
                for detail in converted_data:
                    if word_database:
                        print(f"Updating syllable and phonetic of word with", json.dumps(detail, ensure_ascii=False, separators=(",", ":")))
                        word_database.update_word_details(detail)  # Assuming update_word_details is a method of word_database
                return converted_data
            except Exception as e:  # Generic exception handling, consider specific ones as needed
                print(f"An unexpected error occurred: {e}")
                traceback.print_exc()
                # You may want to modify the message or prompt based on the error for a retry
                continue

        raise RuntimeError("Failed to recheck phonetics after maximum retries.")

# Note: Make sure to replace `client`, `model_name`, and `word_database` with actual objects and implementations.


if __name__ == "__main__":
    # Database path and initialization
    db_path = 'words_phonetics.db'
    # words_db = WordsDatabase(db_path)
    # word_fetcher = AdvancedWordFetcher()

    # phonetic_checker = PhoneticRechecker()
    # phonetic_checker.recheck_word_phonetics_with_paired_tuple(["hello", "underlying", "hack", "in time for"], words_db)


