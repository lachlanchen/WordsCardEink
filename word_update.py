from openai import OpenAI
from openai_fetch import WordsDatabase, AdvancedWordFetcher
import csv
from pykakasi import kakasi

import re

def remove_second_parentheses(text):
    regex = re.compile(r'(（[^）]*）)(（[^）]*）)')
    return re.sub(regex, lambda match: match.group(1), text)


def update_last_10_words(database, fetcher):
    # Fetch the last 10 words from the database
    last_10_words = database.fetch_last_10_words()

    print(last_10_words)

    # Extract just the words for rechecking
    words_to_recheck = [word_detail['word'] for word_detail in last_10_words]

    # Recheck word details using AdvancedWordFetcher
    rechecked_details = fetcher.recheck_word_details(words_to_recheck, database)

    print(rechecked_details)
    
    # Update the database with rechecked details
    for details in rechecked_details:
        database.insert_word_details(details, force=True)

def log_updated_words(updated_words, log_file='words_updated.csv'):
    with open(log_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for word in updated_words:
            writer.writerow([word])

def get_logged_words(log_file='words_updated.csv'):
    logged_words = set()
    try:
        with open(log_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                logged_words.add(row[0])
    except FileNotFoundError:
        pass  # File not found, return empty set
    return logged_words

def remove_consecutive_parenthesis_in_batches(database, fetcher, batch_size=10):
    logged_words = get_logged_words()
    total_words = database.get_total_word_count()
    processed = 0

    while processed < total_words:
        words_batch = database.fetch_words_batch(processed, batch_size)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # print("before update: ", words_batch)
        words_to_recheck = [word_detail['word'] for word_detail in words_batch if word_detail['word'] not in logged_words]

        if len(words_to_recheck) > 0:
            # print("The words are all processed. ")
            # continue

            # rechecked_details = fetcher.recheck_word_details(words_to_recheck, database, word_details=words_batch)
            # print("after update: ", rechecked_details)
            rechecked_details = words_batch
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            updated_words = []
            for details in rechecked_details:
                details["japanese_synonym"] = remove_second_parentheses(details["japanese_synonym"])
                database.insert_word_details(details, force=True)
                updated_words.append(details['word'])

            log_updated_words(updated_words)
        processed += len(words_batch)

def update_database_in_batches(database, fetcher, batch_size=10):
    logged_words = get_logged_words()
    total_words = database.get_total_word_count()
    processed = 0

    while processed < total_words:
        words_batch = database.fetch_words_batch(processed, batch_size)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # print("before update: ", words_batch)
        words_to_recheck = [word_detail['word'] for word_detail in words_batch if word_detail['word'] not in logged_words]

        if len(words_to_recheck) > 0:
            # print("The words are all processed. ")
            # continue

            rechecked_details = fetcher.recheck_word_details(words_to_recheck, database, word_details=words_batch)
            # print("after update: ", rechecked_details)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            updated_words = []
            for details in rechecked_details:
                database.insert_word_details(details, force=True)
                updated_words.append(details['word'])

            log_updated_words(updated_words)
        processed += len(words_batch)

# Usage example
client = OpenAI()  # Assuming you have initialized OpenAI client

# Database path
db_path = 'words_phonetics.db'

# Initialize database and word fetcher
words_db = WordsDatabase(db_path)
word_fetcher = AdvancedWordFetcher(client)

# # Update the last 10 words
# update_last_10_words(words_db, word_fetcher)

# Update the database in batches
update_database_in_batches(words_db, word_fetcher, 5)

# Close the database connection
words_db.close()
