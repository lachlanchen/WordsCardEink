import sqlite3
import random

# Establishing a connection to the SQLite database
# If the file doesn't exist, SQLite will create it
conn = sqlite3.connect('words_phonetics.db')

# Creating a cursor object to interact with the database
cursor = conn.cursor()

# Creating the table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS words_phonetics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT UNIQUE,
        syllable_word TEXT,
        phonetic TEXT,
        japanese_synonym TEXT
    )
''')

# Committing the creation of the table
conn.commit()

# Words and phonetics data
# words_phonetics = [
#     {"word":"analyze", "syllable_word":"an·a·lyze", "phonetic":"ˈæn·ə·ˌlaɪz", "japanese_synonym":"分析（ぶんせき）する"},
#     {"word":"concept", "syllable_word":"con·cept", "phonetic":"ˈkɑn·sɛpt", "japanese_synonym":"概念 (がいねん) "},
#     {"word":"context", "syllable_word":"con·text", "phonetic":"ˈkɑn·tɛkst", "japanese_synonym":"文脈 (ぶんみゃく) "}
# ]

from grossary import words_phonetics

# Inserting data into the database
for item in words_phonetics:
    try:
        cursor.execute('''
            INSERT INTO words_phonetics (word, syllable_word, phonetic, japanese_synonym)
            VALUES (?, ?, ?, ?)
        ''', (item['word'], item['syllable_word'], item['phonetic'], item['japanese_synonym']))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Record for {item['word']} already exists. Skipping insert.")

# Function to fetch a random word
def fetch_random_word():
    cursor.execute('SELECT * FROM words_phonetics')
    records = cursor.fetchall()
    return random.choice(records) if records else None

# Fetching a random word
random_word = fetch_random_word()
print(random_word)

# Closing the database connection
conn.close()
