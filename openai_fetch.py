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

def transcribe_japanese(text):
    kks = kakasi()
    kks.setMode("J", "H")  # Japanese to Hiragana
    kks.setMode("K", "H")  # Katakana to Hiragana
    conv = kks.getConverter()

    result = ""
    current_chunk = ""
    is_kanji_or_katakana = False

    for char in text:
        hiragana = conv.do(char)

        if '\u4E00' <= char <= '\u9FFF':  # Kanji
            if not is_kanji_or_katakana:
                is_kanji_or_katakana = True
                current_chunk = ""
            current_chunk += hiragana
            result += char
        elif '\u30A0' <= char <= '\u30FF':  # Katakana
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

def clean_and_transcribe(word_details):
    # Function to remove text inside parentheses
    def remove_text_inside_parentheses(text):
        while '(' in text and ')' in text:
            start = text.find('(')
            end = text.find(')') + 1
            text = text[:start] + text[end:]
        return text

    for word in word_details:
        # Update phonetic field
        word["phonetic"] = word["phonetic"].replace(".", "·").replace(" ", "")

        # Update word_details field
        word["japanese_synonym"] = word["japanese_synonym"].replace(".", "").replace("·", "").replace(" ", "").replace("(", "（").replace(")", "）")

        # Clean and transcribe japanese_synonym
        if "japanese_synonym" in word:
            # clean_synonym = remove_text_inside_parentheses(word["japanese_synonym"])
            clean_synonym = re.sub(r'（[ぁ-んァ-ン]+）', '', word["japanese_synonym"])  # Remove hiragana in parentheses
            word["japanese_synonym"] = transcribe_japanese(clean_synonym)  # Replace with your transcription function

    return word_details

class WordsDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        if os.path.exists(db_path):
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

    def word_exists(self, word):
        if self.conn:
            self.cursor.execute("SELECT COUNT(*) FROM words_phonetics WHERE word = ?", (word,))
            return self.cursor.fetchone()[0] > 0
        return False

    # def insert_word_details(self, word_details, force=False):
    #     if self.conn:
    #         word = word_details['word'].lower()
    #         syllable_word = word_details['syllable_word'].lower()
    #         phonetic = word_details['phonetic']
    #         japanese_synonym = word_details['japanese_synonym']

    #         try:
    #             if force:
    #                 # Update existing record
    #                 self.cursor.execute("""
    #                     UPDATE words_phonetics 
    #                     SET syllable_word = ?, phonetic = ?, japanese_synonym = ?
    #                     WHERE word = ?;
    #                 """, (syllable_word, phonetic, japanese_synonym, word))
    #             else:
    #                 # Insert new record, ignore on duplicate
    #                 self.cursor.execute("""
    #                     INSERT INTO words_phonetics (word, syllable_word, phonetic, japanese_synonym)
    #                     VALUES (?, ?, ?, ?);
    #                 """, (word, syllable_word, phonetic, japanese_synonym))

    #             self.conn.commit()
    #         except sqlite3.Error as e:
    #             print(f"SQLite Error: {e}")

    def insert_word_details(self, word_details, force=False):
        if self.conn:
            word = word_details['word'].lower()
            syllable_word = word_details['syllable_word'].lower()
            phonetic = word_details['phonetic'].replace(".", "·").replace(" ", "")
            japanese_synonym = word_details['japanese_synonym'].replace(".", "").replace("·", "").replace(" ", "").replace("(", "（").replace(")", "）")


            try:
                # UPSERT operation: Update if exists, insert if not
                self.cursor.execute("""
                    INSERT INTO words_phonetics (word, syllable_word, phonetic, japanese_synonym)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(word) DO UPDATE SET
                        syllable_word = excluded.syllable_word,
                        phonetic = excluded.phonetic,
                        japanese_synonym = excluded.japanese_synonym;
                """, (word, syllable_word, phonetic, japanese_synonym))

                self.conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite Error: {e}")


    def fetch_random_words(self, num_words):
        if self.conn:
            query = "SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics ORDER BY RANDOM() LIMIT ?"
            self.cursor.execute(query, (num_words,))
            rows = self.cursor.fetchall()
            # Convert each row to a dictionary
            return [{"word": row[0], "syllable_word": row[1], "phonetic": row[2], "japanese_synonym": row[3]} for row in rows]
        else:
            return []

    def update_from_word_details_correction_csv(self, word_details_correction_csv_path):
        history_csv_path = 'words_update_history.csv'

        # Read from error CSV and update database
        with open(word_details_correction_csv_path, 'r+', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            lines = list(reader)
            remaining_lines = []

            for line in lines:
                self.insert_word_details(line, force=True)

                # Log to history
                with open(history_csv_path, 'a', newline='', encoding='utf-8') as history_file:
                    history_writer = csv.writer(history_file)
                    history_writer.writerow([line['word'], line['syllable_word'], line['phonetic'], line['japanese_synonym']])
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

        # Extract words from words_update.csv
        with open(words_update_csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                words.append(row[0])

        # Recheck and fetch details for the words
        word_details_list = fetcher.recheck_word_details(words, self)

        # Update the database with these details
        self.update_from_list(word_details_list, words_update_csv_path)

    def update_from_list(self, word_details_list, words_update_csv_path):
        history_csv_path = 'words_update_history.csv'

        # Update the database and log changes
        for details in word_details_list:
            self.insert_word_details(details, force=True)

            # Log to history
            with open(history_csv_path, 'a', newline='', encoding='utf-8') as history_file:
                history_writer = csv.writer(history_file)
                history_writer.writerow([details['word'], details['syllable_word'], details['phonetic'], details['japanese_synonym']])

        # Remove updated words from words_update.csv
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

    def fetch_last_10_words(self):
        if self.conn:
            query = "SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics ORDER BY rowid DESC LIMIT 10"
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return [{"word": row[0], "syllable_word": row[1], "phonetic": row[2], "japanese_synonym": row[3]} for row in rows]
        else:
            return []

    def get_total_word_count(self):
        if self.conn:
            self.cursor.execute("SELECT COUNT(*) FROM words_phonetics")
            return self.cursor.fetchone()[0]
        return 0

    def fetch_words_batch(self, offset, limit):
        if self.conn:
            query = "SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics LIMIT ? OFFSET ?"
            self.cursor.execute(query, (limit, offset))
            rows = self.cursor.fetchall()
            return [{"word": row[0], "syllable_word": row[1], "phonetic": row[2], "japanese_synonym": row[3]} for row in rows]
        else:
            return []

    def close(self):
        if self.conn:
            self.conn.close()


class AdvancedWordFetcher:
    def __init__(self, client, max_retries=3):
        self.client = client
        self.max_retries = max_retries
        self.examples = self.load_examples()

    def load_examples(self):
        examples_file_path = 'word_examples.csv'
        if os.path.exists(examples_file_path):
            with open(examples_file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                example_list = list(reader)

                print("example_list: ", example_list)

                if len(example_list) > 1:
                   return example_list

        return [
            {"word": "abstraction", "syllable_word": "ab·strac·tion", "phonetic": "ˈæb·stræk·ʃən", "japanese_synonym": "抽象（ちゅうしょう）"},
            {"word": "paradox", "syllable_word": "par·a·dox", "phonetic": "ˈpær·ə·dɒks", "japanese_synonym": "逆説（ぎゃくせつ）"}
        ]

    def save_examples(self):
        examples_file_path = 'word_examples.csv'
        with open(examples_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.examples[0].keys())
            writer.writeheader()
            writer.writerows(self.examples)

    def fetch_words(self, num_words, word_database):
        unique_words = []

        for _ in range(self.max_retries):
            try:

                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant with a vast vocabulary and creativity."},
                        {"role": "user", "content": f"Think wildly and provide me with a python list of {num_words} unique advanced words that are often used in formal readings. Give me only the plain python list compatible with json.loads and start with [ and end with ] WITHOUT any other words."}
                    ]
                )

                # print(response)

                words_list = json.loads(response.choices[0].message.content.lower())
                unique_words = [word for word in words_list if not word_database.word_exists(word)]
                if unique_words:
                    break
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                continue
            except Exception as e:  # General exception catch
                print(f"An unexpected error occurred: {e}")
            else:
                print("Fetched unique words successfully.")
                break
        if not unique_words:
            raise RuntimeError("Failed to fetch unique words after maximum retries.")
        return unique_words

    def fetch_word_details(self, words, word_database, num_words_phonetic=10):
        random_words = random.sample(words, min(num_words_phonetic, len(words)))
        new_examples = random.sample(random_words, min(2, len(random_words)))
        
        example_word = random.choice(new_examples)

        # detailed_list_message = (
        #     "I need each word to come with its syllable_word (separated by central dots), "
        #     "phonetic transcription (also separated by central dots), and a Japanese synonym. "
        #     "Ensure to use central dots for both syllable and phonetic separation, and correctly place hiragana only after kanji characters and katakana in the Japanese synonyms like その後（ご), 実装 (じっそう）する， 押（お）し合（あ）う, プロトコル（ぷろとこる）to avoid repetitive hiragana."
        #     f"The output should be like {json.dumps(self.examples, ensure_ascii=False, separators=(',', ':'))}. "
        #     f"This is the list of words: {', '.join(random_words)}. "
        #     )

        detailed_list_message = (
            "For each word, include syllable_word (with central dots separating syllables), "
            "corret phonetic transcription (phonemes also separated by central dots), and a correct Japanese synonym. "
            "Also, include a correct Japanese synonym, ensuring hiragana is placed only after kanji and katakana, as in 'その後（ご)', '実装 (じっそうする)', '押（お）し合（あ）う', 'プロトコル（ぷろとこる）', to prevent repetitive hiragana. "
            "For example, like 容易にする（よういにする）, もの悲しい (ものかなしい）, 美しい（うつくしい）, 極めて悪い（きわめてわるい）, 旧式の（きゅうしきの）and その後（そのご) are WRONG. The する like thing should be moved after the parenthesis 容易（ようい）にする. The CORRECT one for latter are その後（ご), 美（うつく）しい, 極（きわ）めて悪（わる）い, 旧式（きゅうしき）のand もの悲(かな）しい . "
            "Ensure NO hiragana between parenthesis and kanji or katakana. "
            "Ensure NO dots in Japanese synonym. "
            "Ensure equal correct separation between words and phonetic symbols. "
            "The output format should resemble: {}."
            "The words to process are: {}."
        ).format(json.dumps(self.examples, ensure_ascii=False, separators=(',', ':')), ', '.join(random_words))

        for _ in range(self.max_retries):
            try:
                print(f"Querying {random_words} from OpenAI...")
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words."},
                        {"role": "user", "content": detailed_list_message}
                    ]
                )
                word_phonetics = json.loads(response.choices[0].message.content)
                # Save word details to database
                for detail in word_phonetics:
                    word_database.insert_word_details(detail)

                self.examples= word_phonetics[0:2]
                self.save_examples()
                return word_phonetics
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                continue
            except Exception as e:  # General exception catch
                print(f"An unexpected error occurred: {e}")
                raise Exception(f"An unexpected error occurred: {e}")
            else:
                print("Fetched word details successfully.")
                return word_phonetics
        raise RuntimeError("Failed to parse response after maximum retries.")

    def recheck_word_details(self, words, word_database, word_details=[], num_words_phonetic=10, recheck=False):
        random_words = words
        

        # detailed_list_message = (
        #     "Please recheck and provide detailed information for each word, including its syllable_word (separated by central dots), "
        #     "phonetic transcription (also separated by central dots), and correctly placed hiragana right after Chinese characters and katakana in the Japanese synonyms like その後（ご), 実装 (じっそう）する， 押（お）し合（あ）う, プロトコル（ぷろとこる）to avoid repetitive hiragana. "
        #     "Ensure to use central dots for both syllable and phonetic separation. "
        #     "The expected output format should be like {}."
        #     "This is the list of words for rechecking: {}."
        # ).format(json.dumps(self.examples, ensure_ascii=False, separators=(',', ':')), ', '.join(random_words))

        word_details = clean_and_transcribe(word_details)

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(word_details)

        detailed_list_message = (
            "Please recheck and provide detailed information for each word. Include syllable_word (with central dots separating syllables), "
            "correct phonetic transcription (phonemes also separated by central dots), and a correct Japanese synonym. "
            # "For Japanese synonyms, place HIRAGANA ONLY AFTER KANJI AND KATAKANA, like 'その後（ご)', '実装 (じっそうする)', '押（お）し合（あ）う', 'プロトコル（ぷろとこる）', avoiding repetitive hiragana. "
            # "For example, like 容易にする（よういにする）, もの悲しい (ものかなしい）, 美しい（うつくしい）, 極めて悪い（きわめてわるい）, 旧式の（きゅうしきの）and その後（そのご) are WRONG. The する like thing should be moved after the parenthesis 容易（ようい）にする. The CORRECT one for latter are その後（ご), 美（うつく）しい, 極（きわ）めて悪（わる）い, 旧式（きゅうしき）のand もの悲(かな）しい . "
            # "For example,  する like thing should be moved after the parenthesis 容易（ようい）にする. The CORRECT examples are その後（ご), 美（うつく）しい, 極（きわ）めて悪（わる）い, 旧式（きゅうしき）のand もの悲(かな）しい. "
            # "ONLY and ALL kanji and katakana should be followed with hiragana in parentheses. "
            "ALL kanji and katakana and ONLY THEM should be transcripted with hiragana in parentheses. "
            "ENSURE NO repeated hiragana before '（' and inside parentheses. Divide and distribute the hiragana into its corresponding kanji like this 極（きわ）めて悪（わる）い. "
            # "Ensure NO hiragana between parenthesis and kanji or katakana. "
            "Ensure NO dots in Japanese synonym. "
            "Ensure equal correct separation between words and phonetic symbols. "
            "Make NO changes if CORRECT. "
            "Words for rechecking: {}."
            "The expected output should be the same format as: {}."
        ).format(', '.join(random_words), json.dumps(word_details, ensure_ascii=False, separators=(',', ':')))
        # ).format(', '.join(random_words))

        # print(detailed_list_message)

        for _ in range(self.max_retries):
            try:
                print(f"Rechecking {random_words} from OpenAI...")
                response = self.client.chat.completions.create(
                    # model="gpt-3.5-turbo",
                    # model="gpt-4",
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are an assistant skilled in linguistics, capable of providing detailed phonetic and linguistic attributes for given words."},
                        {"role": "user", "content": detailed_list_message}
                    ]
                )
                word_phonetics = json.loads(response.choices[0].message.content)
                print(word_phonetics)
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # Save word details to database
                for detail in word_phonetics:
                    word_database.insert_word_details(detail)
                return word_phonetics
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                continue
            except Exception as e:  # General exception catch
                # print(f"An unexpected error occurred: {e}")
                raise Exception(f"An unexpected error occurred: {e}")
            else:
                print("Fetched and rechecked word details successfully.")
                return word_phonetics
        raise RuntimeError("Failed to parse response after maximum retries.")



# Example usage
# words_db = WordsDatabase(db_path)
# word_fetcher = AdvancedWordFetcher(client)
# chooser = OpenAiChooser(words_db, word_fetcher)

# chosen_word = chooser.choose()
# print(chosen_word)

class OpenAiChooser:
    def __init__(self, db, word_fetcher):
        self.db = db
        self.word_fetcher = word_fetcher
        self.current_words = []
        self.fetch_new_words()

    def _is_daytime_in_hk(self):
        hk_timezone = pytz.timezone('Asia/Hong_Kong')
        hk_time = datetime.now(hk_timezone)
        # return 8 <= hk_time.hour < 24  # Daytime hours in Hong Kong
        return False


    def fetch_new_words(self):
        if self._is_daytime_in_hk():
            # Fetch 10 words from OpenAI and 10 from the database
            words = self.word_fetcher.fetch_words(10, self.db)
            openai_words = self.word_fetcher.fetch_word_details(words, self.db)
            db_words = self.db.fetch_random_words(10)

            print("openai_words: ", openai_words)
            print("db_words: ", db_words)
        else:
            # Fetch 20 words from the database at night
            db_words = self.db.fetch_random_words(20)
            openai_words = []

        self.current_words = openai_words + db_words
        # self.current_words = openai_words
        random.shuffle(self.current_words)

    def get_current_words(self):
        return self.current_words

    def choose(self):
        if not self.current_words:
            self.fetch_new_words()

        return self.current_words.pop()



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


