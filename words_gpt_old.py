#!/usr/bin/python
# -*- coding:utf-8 -*-

# import os
# os.environ['GPIOZERO_PIN_FACTORY'] = os.environ.get('GPIOZERO_PIN_FACTORY', 'mock')
# import gpiozero
# from gpiozero.pins.mock import MockFactory
# gpiozero.Device.pin_factory = MockFactory()

import sys
import os
pic_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pic')
lib_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
font_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'font')
if os.path.exists(lib_root):
    sys.path.append(lib_root)

import logging
from waveshare_epd import epd7in3f
import time
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling
import traceback
import itertools

import random

from grossary import words_phonetics

logging.basicConfig(level=logging.DEBUG)

from openai_fetch import WordsDatabase, AdvancedWordFetcher, OpenAiChooser

import json
from openai import OpenAI

import sqlite3
import os

from datetime import datetime
import pytz
import re




# Usage example
client = OpenAI()


# Database path
db_path = 'words_phonetics.db'

# Initialize database class
words_db = WordsDatabase(db_path)

# Initialize word fetcher
word_fetcher = AdvancedWordFetcher()

# Example usage
words_db = WordsDatabase(db_path)
word_fetcher = AdvancedWordFetcher()



# Function to count syllables based on dots
def count_syllables(word):
    # Count dots and stress symbols, subtract one if the first character is a stress symbol
    count = word.count('·') + word.count('ˈ') + word.count('ˌ')
    if word.startswith('ˈ') or word.startswith('ˌ'):
        count -= 1
    return count + 1

# Function to split words into syllables and get color for each syllable

def split_word(word, colors):
    # Replace stress symbols with a preceding dot, except at the beginning
    if word.startswith('ˈ') or word.startswith('ˌ'):
        word = word[0] + word[1:].replace('ˈ', '·ˈ').replace('ˌ', '·ˌ')
    else:
        word = word.replace('ˈ', '·ˈ').replace('ˌ', '·ˌ')

    syllables = word.split('·')
    color_syllables = [(syllable, colors[i % len(colors)]) for i, syllable in enumerate(syllables)]
    return color_syllables

def find_suitable_font_size(text, font_path, max_width, start_size=60, step=2):
    """
    Find the largest font size that allows the text to fit within max_width.
    """
    font_size = start_size
    font = ImageFont.truetype(font_path, font_size)
    print("text: ", text)
    # text_width = font.getsize(text)[0]

    # Create a dummy image and draw object to measure text
    dummy_image = Image.new('RGB', (100, 100))
    draw = ImageDraw.Draw(dummy_image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # Calculate the text width

    while text_width > max_width and font_size > 0:
        font_size -= step
        font = ImageFont.truetype(font_path, font_size)
        # text_width = font.getsize(text)[0]

        # dummy_image = Image.new('RGB', (100, 100))
        # draw = ImageDraw.Draw(dummy_image)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Calculate the text width

    return font_size

def get_text_size(text, font):
    # Create a dummy image and draw object to measure text
    dummy_image = Image.new('RGB', (100, 100))
    draw = ImageDraw.Draw(dummy_image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # Calculate the text width
    text_height = text_bbox[3] - text_bbox[1]

    return text_width, text_height


def draw_japanese_with_hiragana(draw, text, jp_font_path, small_width, y):
    text = text.replace(" ", "").replace("(", "（").replace(")", "）")
    # Updated regex to include Katakana
    regex = re.compile(r'([一-龠々ァ-ンー]+)（([ぁ-んァ-ンー]+)）')

    # Determine the font size for Japanese text (excluding hiragana in parentheses)
    plain_text = re.sub(r'（[ぁ-んァ-ンー]+）', '', text)  # Remove hiragana in parentheses
    font_size = find_suitable_font_size(plain_text, jp_font_path, small_width)
    font = ImageFont.truetype(jp_font_path, font_size)

    pos_x = (small_width - get_text_size(plain_text, font)[0]) // 2

    # Draw Japanese text with hiragana
    last_match_end = 0
    for match in regex.finditer(text):
        kanji_or_katakana, hiragana = match.groups()
        start, end = match.span()

        # Draw preceding text
        preceding_text = text[last_match_end:start]
        draw.text((pos_x, y), preceding_text, font=font, fill=(0, 0, 0))
        pos_x += get_text_size(preceding_text, font)[0]

        # Draw kanji or katakana
        draw.text((pos_x, y), re.sub(r'（[ぁ-んァ-ンー]+）', '', kanji_or_katakana), font=font, fill=(0, 0, 0))
        kanji_or_katakana_width = get_text_size(kanji_or_katakana, font)[0]

        # Adjust hiragana font size to fit the width of kanji or katakana
        hiragana_font_size = find_suitable_font_size(hiragana, jp_font_path, kanji_or_katakana_width, start_size=font_size)
        hiragana_font = ImageFont.truetype(jp_font_path, hiragana_font_size)

        # Draw hiragana above kanji or katakana
        hiragana_x = pos_x
        hiragana_y = y - get_text_size(hiragana, hiragana_font)[1] - 2
        draw.text((hiragana_x, hiragana_y), hiragana, font=hiragana_font, fill=(0, 0, 0))

        pos_x += kanji_or_katakana_width
        last_match_end = end

    # Draw remaining text after last match
    remaining_text = text[last_match_end:]
    draw.text((pos_x, y), re.sub(r'（[ぁ-んァ-ンー]+）', '', remaining_text), font=font, fill=(0, 0, 0))

    return y + get_text_size(plain_text, font)[1] + 20  # Adjust the vertical space after the Japanese text


class WeightedRandomChooser:
    def __init__(self, items):
        self.items = items
        self.initial_weights = [1] * len(items)
        self.weights = self.initial_weights.copy()
        self.chosen_flags = [False] * len(items)

    def choose(self):
        chosen_item = random.choices(self.items, weights=self.weights, k=1)[0]
        index = self.items.index(chosen_item)
        
        # Update chosen flags and weights
        self.chosen_flags[index] = True
        self.weights[index] *= 0.5

        # Check if all items have been chosen, and reset if so
        if all(self.chosen_flags):
            self.reset_weights()

        return chosen_item

    def reset_weights(self):
        self.weights = self.initial_weights.copy()
        self.chosen_flags = [False] * len(self.items)

class RandomSortChooser:
    def __init__(self, items):
        self.items = items
        self.shuffle_items()

    def shuffle_items(self):
        self.shuffled_items = self.items.copy()
        random.shuffle(self.shuffled_items)
        self.index = 0

    def choose(self):
        if self.index >= len(self.shuffled_items):
            self.shuffle_items()

        chosen_item = self.shuffled_items[self.index]
        self.index += 1
        return chosen_item



# Example usage
# chooser = WeightedRandomChooser(words_phonetics)
# Example usage
# chooser = RandomSortChooser(words_phonetics)

chooser = OpenAiChooser(words_db, word_fetcher)


# Sort words by syllable count
# words_phonetics.sort(key=lambda x: count_syllables(x['syllable_word']), reverse=True)


try:
    logging.info("epd7in3f Demo")



    # # Define the text and colors
    # text = "lazying.art"

    # Colors: Black, Red, Green, Blue, Red, Yellow, Orange
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 165), (0, 255, 165)]

    # Increase the font size to make it larger
    font_size = 60  # Adjust the font size as needed

    font_path = os.path.join(font_root, 'Font.ttc')
    ipa_font_path = os.path.join(font_root, 'arial.ttf')
    jp_font_path = os.path.join(font_root, 'HolidayMDJP.otf')

    # print("font_dir: ", font_dir)
    # font = ImageFont.truetype(font_dir, font_size)
    # # ipa_font_dir = os.path.join(fontdir, 'NotoSerif-unhinted/NotoSerif-Bold.ttf')
    # # ipa_font_dir = os.path.join(fontdir, 'ipa_font.ttf')
    # # ipa_font_dir = os.path.join(fontdir, 'SILMIPA_.TTF')
    # # ipa_font_dir = "/Library/Fonts/Arial Unicode.ttf"
    # print("font_dir: ", ipa_font_dir)
    # ipa_font = ImageFont.truetype(ipa_font_dir, 40)
    # print("font_dir: ", jp_font_dir)
    # jp_font = ImageFont.truetype(jp_font_dir, 40)

    epd = epd7in3f.EPD()
    logging.info("init and Clear")
    epd.init()
    # epd.Clear()
    

    # Smaller image dimensions
    small_width, small_height = epd.width // 2, epd.height // 2

    # Drawing on the image
    logging.info("Drawing on the image...")
    



    # for item in itertools.cycle(words_phonetics):
    while True:
        # item = random.choice(words_phonetics)

        item = chooser.choose()
        # item = chooser.choose()
        print(item)


        small_image = Image.new('RGB', (small_width, small_height), (255,255,255))  # 255: clear the frame (white background)
        draw = ImageDraw.Draw(small_image)

        # image = Image.new('RGB', (epd.width, epd.height), (255, 255, 255))
        # draw = ImageDraw.Draw(image)

        
        y = 1  # Adjust starting y position
        # Determine the font size for the phonetic symbols
        phonetic_text = ' '.join(syllable for syllable, _ in split_word(item['phonetic'].replace(".", "·"), colors))
        font_size = find_suitable_font_size(phonetic_text, ipa_font_path, small_width)
        font = ImageFont.truetype(ipa_font_path, font_size)

        # Draw phonetic symbols above the word
        # phonetic_width = sum(font.getsize(syllable)[0]*0.5 for syllable, _ in split_word(item['phonetic'], colors))
        phonetic_width = sum(get_text_size(syllable, font)[0] for syllable, _ in split_word(item['phonetic'], colors))
        print("small_width: ", small_width)
        print("phonetic_width: ", phonetic_width)
        phonetic_x = (small_width - phonetic_width) // 2
        for syllable, color in split_word(item['phonetic'], colors):
            draw.text((phonetic_x, y), syllable.strip(), font=font, fill=color)
            phonetic_x += get_text_size(syllable, font)[0]
        
        # y += font.getsize(syllable)[1] + 0  # Space between phonetic symbols and word
        y += get_text_size(syllable, font)[1] + 20  # Space between phonetic symbols and word
        # Determine the font size for the phonetic symbols
        phonetic_text = ' '.join(syllable for syllable, _ in split_word(item['syllable_word'], colors))
        font_size = find_suitable_font_size(phonetic_text, font_path, small_width)
        font = ImageFont.truetype(font_path, font_size)

        # Calculate text width and adjust x coordinate for centering
        word_width = sum(get_text_size(syllable, font)[0] for syllable, _ in split_word(item['syllable_word'], colors))
        word_x = (small_width - word_width) // 2
        # Draw each syllable in word
        for syllable, color in split_word(item['syllable_word'], colors):
            draw.text((word_x, y), syllable, font=font, fill=color)
            word_x += get_text_size(syllable, font)[0]

        # how to place japanese

        y += get_text_size(syllable, font)[1] + 45  # Space for Japanese translation
        # Determine the font size for the phonetic symbols
        # phonetic_text = ' '.join(syllable for syllable, _ in split_word(item['syllable_word'], colors))
        text = item["japanese_synonym"]

        # font_size = find_suitable_font_size(text, jp_font_path, small_width)
        # font = ImageFont.truetype(jp_font_path, font_size)
        # # Draw Japanese translation (centered)
        # # japanese_text = japanese_synonyms[item['word']]
        # japanese_text = item["japanese_synonym"]
        # japanese_text_width = get_text_size(japanese_text, font)[0]
        # japanese_text_x = (small_width - japanese_text_width) // 2
        # draw.text((japanese_text_x, y), japanese_text, font=font, fill=(0, 0, 0))

        draw_japanese_with_hiragana(draw, text.replace("・", "").replace("·", ""), jp_font_path, small_width, y)

        # Displaying the image
        # Scale the image up
        # Himage = small_image.resize((epd.width, epd.height), Image.ANTIALIAS)
        # Himage = small_image.resize((epd.width, epd.height), Image.LANCZOS)
        Himage = small_image.resize((epd.width, epd.height), Resampling.LANCZOS)
        epd.display(epd.getbuffer(Himage))
        time.sleep(300)  # Display each word for 10 minutes

    
    # Clear and go to sleep
    logging.info("Clear...")
    epd.Clear()

    logging.info("Goto Sleep...")
    epd.sleep()

except IOError as e:
    logging.info(e)

except KeyboardInterrupt:
    logging.info("ctrl + c:")
    epd7in3f.epdconfig.module_exit()
    exit()

# Close the database connection
words_db.close()