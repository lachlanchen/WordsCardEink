import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('words_phonetics.db')
cursor = conn.cursor()

# Step 1: Create a new table with an auto-increment ID
cursor.execute('''
    CREATE TABLE IF NOT EXISTS words_phonetics_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT UNIQUE,
        syllable_word TEXT,
        phonetic TEXT,
        japanese_synonym TEXT
    )
''')

# Step 2: Copy data from the original table to the new table
cursor.execute('''
    INSERT INTO words_phonetics_new (word, syllable_word, phonetic, japanese_synonym)
    SELECT word, syllable_word, phonetic, japanese_synonym FROM words_phonetics
''')

# Step 3: Rename the original table
cursor.execute('ALTER TABLE words_phonetics RENAME TO words_phonetics_old')

# Step 4: Rename the new table to the original table's name
cursor.execute('ALTER TABLE words_phonetics_new RENAME TO words_phonetics')

# Step 5: Drop the old table
cursor.execute('DROP TABLE IF EXISTS words_phonetics_old')

# Commit the changes and close the connection
conn.commit()
conn.close()
