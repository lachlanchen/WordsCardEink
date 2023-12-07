#!/usr/bin/python
# -*- coding:utf-8 -*-

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



# import os
# os.environ['GPIOZERO_PIN_FACTORY'] = os.environ.get('GPIOZERO_PIN_FACTORY', 'mock')
# import gpiozero
# from gpiozero.pins.mock import MockFactory
# gpiozero.Device.pin_factory = MockFactory()

import argparse

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
logging.basicConfig(level=logging.DEBUG)

# from grossary import words_phonetics
import json
import json5
from openai import OpenAI
from words_data import WordsDatabase, AdvancedWordFetcher, OpenAiChooser
from words_data import split_word, split_word_with_color, count_syllables

import sqlite3
import os

from datetime import datetime
import pytz
import re


import numpy as np

import arabic_reshaper
from bidi.algorithm import get_display


# Usage example
client = OpenAI()
# Database path
db_path = 'words_phonetics.db'
# Initialize database class
words_db = WordsDatabase(db_path)
# Initialize word fetcher
words_db = WordsDatabase(db_path)
word_fetcher = AdvancedWordFetcher(client)


class EPaperHardware:
    def __init__(self, epd_module):
        self.epd = epd_module.EPD()
        self.init_display()

    def init_display(self):
        self.epd.init()

    def clear_and_sleep(self):
        self.epd.Clear()
        self.epd.sleep()

    def get_display_size(self):
        return self.epd.width, self.epd.height

    def display_image(self, image):
        self.epd.display(self.epd.getbuffer(image))

    def clear_display(self):
        self.epd.Clear()


class GradientTextureGenerator:
    def __init__(self, width=1024, height=1024, base_color=(255, 255, 255), noise_intensity_range=(0.01, 0.05)):
        self.width = width
        self.height = height
        self.base_color = base_color  # Base color for the gradient, default is white
        self.noise_intensity_range = noise_intensity_range

    def create_gradient(self):
        # Start with a base color array
        Z = np.full((self.height, self.width, 3), self.base_color, dtype=np.uint8)

        # Create a slight linear gradient
        for i in range(3):  # For each color channel
            gradient = np.linspace(0, 1, self.width) * 10  # Adjust the factor for the gradient effect
            Z[:, :, i] = np.clip(Z[:, :, i] - gradient[None, :], 0, 255).astype(np.uint8)

        # Add noise with varying intensity to each channel
        for i in range(3):  # For each color channel
            noise_intensity = np.random.uniform(*self.noise_intensity_range)
            noise = np.random.normal(0, noise_intensity * 255, (self.height, self.width))
            Z[:, :, i] = np.clip(Z[:, :, i] + noise, 0, 255).astype(np.uint8)

        return Z

    def get_pil_image(self, Z):
        try:
            # Create a PIL Image from the numpy array
            image = Image.fromarray(Z, 'RGB')
            return image
        except Exception as e:
            print(f"Error in creating PIL image: {e}")

            traceback.print_exc()

            return None

    def save_image(self, Z, file_path):
        """
        Save the generated gradient as an image using PIL.

        :param Z: The numpy array representing the gradient texture.
        :param file_path: Path to save the image.
        """
        # Convert the numpy array to an image and save it
        image = self.get_pil_image(Z)
        if image:
            image.save(file_path)


# # Example usage
# generator = GradientTextureGenerator()
# Z = generator.create_gradient()
# image = generator.get_pil_image(Z)  # Now you have a PIL Image object

# # Displaying the image for demonstration
# image.show()



class EPaperDisplay:
    # def __init__(self, hardware, font_root, scale_factor=2):
    def __init__(self, hardware, font_root, scale_factor=1, background_texture="white", content_type='japanese_synonym', image_folder='images', emoji_path=None):
        self.hardware = hardware
        self.scale_factor = scale_factor
        self.width, self.height = [dim // scale_factor for dim in hardware.get_display_size()]
        self.font_root = font_root
        # Colors: Black, Red, Green, Blue, Red, Yellow, Orange
        # self.pallete = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 165), (0, 255, 165)]
        # crimson   #DC143C (220,20,60)
        # dark orange   #FF8C00 (255,140,0)
        # sandy brown   #F4A460 (244,164,96)
        # dark violet #9400D3 (148,0,211)
        # saddle brown  #8B4513 (139,69,19)
        self.pallete = [
            (0, 0, 0), # Black
            # (135, 37, 24), # R
            # (160, 37, 24), # R
            (220, 20, 60), # R
            # (55, 84, 6), # G
            (55, 120, 6), # G
            # (68, 48, 108), # Blue
            # (68, 48, 160), # B
            # (153, 92, 233), # B
            # (101, 40, 66), # P
            # (153, 151, 255), # Purple
            (148, 0, 211), # P
            # (144, 63, 22), # Y
            # (255, 178, 102), # Y
            (244, 164, 96), # Y
            (139, 69, 19), # Brown
        ]
        self.setup_fonts()


        self.background_texture = background_texture
        self.content_type = content_type
        print("content_type: ", content_type)
        # Format the current date and time to use in the folder name
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime("%Y%m%d")
        # self.image_folder = os.path.join(image_folder, timestamp)
        # self.image_folder = f"{image_folder}-{timestamp}"
        # self.image_folder = image_folder

        if emoji_path:
            self.image_folder = os.path.join(emoji_path, timestamp)
        else:
            # self.image_folder = os.path.join("images", image_folder, timestamp)
            # self.image_folder = os.path.join("images", timestamp)
            self.image_folder = os.path.join("images", self.background_texture)


        self.word = None
        self.image = None
        self.intermediate_images = []

    def save_intermediate_image(self):
        # Save a copy of the current image
        self.intermediate_images.append(self.image.copy())

    def save_all_images(self):
        word = self.word
        """Saves all intermediate images to files."""
        # print("word_path: ", word)
        word_path = os.path.join(self.image_folder, f"{word}-{self.content_type}")
        if not os.path.exists(word_path):
            os.makedirs(word_path)
        for i, img in enumerate(self.intermediate_images):
            img = img.resize((self.width * self.scale_factor, self.height * self.scale_factor), Resampling.LANCZOS)
            img.save(os.path.join(word_path, f"{i+1:02d}.jpg"))
            if i == len(self.intermediate_images) - 1:
                img.save(os.path.join(self.image_folder, f"{word}-{self.content_type}.jpg"))


    def clear_intermediate_images(self):
        self.intermediate_images = []

    def setup_fonts(self):
        self.jp_font_path = os.path.join(self.font_root, 'HolidayMDJP.otf')
        self.jp_font_path_fallback = os.path.join(self.font_root, 'KouzanMouhituFontOTF.otf')
        self.ipa_font_path = os.path.join(self.font_root, 'arial.ttf')
        self.arabic_font_path = os.path.join(self.font_root, 'arial.ttf')
        # self.chinese_font_path = os.path.join(self.font_root, 'HanyiSentySeaSpray.ttf')
        # self.chinese_font_path = os.path.join(self.font_root, 'HanyiSentyDew.ttf')
        # self.chinese_font_path = os.path.join(self.font_root, 'HanyiSentyBubbleTea.ttf')
        self.chinese_font_path = os.path.join(self.font_root, 'HanyiSenty Candy-color-mono.ttf')
        # self.chinese_font_path = os.path.join(self.font_root, 'HanyiSentyLotus.ttf')
        # self.chinese_font_path = os.path.join(self.font_root, 'KouzanMouhituFontOTF.otf')
        self.default_font_path = os.path.join(self.font_root, 'Font.ttc')

    def create_content_layout(self, item):
        self.word = item["word"]

        if self.background_texture == "white":
            self.image = Image.new('RGB', (self.width, self.height), (255,255,255))
        else:
            generator = GradientTextureGenerator(width=self.width, height=self.height, base_color=(230, 230, 230))
            Z = generator.create_gradient()
            self.image = generator.get_pil_image(Z)

        draw = ImageDraw.Draw(self.image)

        # Divide the display into 4 rows
        row_height = self.height // 4

        self.used_height = 2

        if self.content_type != "japanese_and_arabic":
            self.draw_phonetic(draw, item['phonetic'], 0, row_height)
            self.save_intermediate_image()
            self.draw_word(draw, item['syllable_word'], row_height, row_height)
            self.save_intermediate_image()

            self.used_height = 2

        else:

            self.draw_phonetic(draw, item['phonetic'], 0, 0.85 * row_height)
            self.save_intermediate_image()
            self.draw_word(draw, item['syllable_word'], 0.8 * row_height, 0.85 * row_height)
            self.save_intermediate_image()

            self.used_height = 1.7


        # self.draw_japanese(draw, item['japanese_synonym'], 2 * row_height, 2 * row_height)
        # self.save_intermediate_image()


        # Using the selected content type
        if self.content_type == 'japanese_synonym':
            self.draw_japanese(draw, item['japanese_synonym'], 2 * row_height, 2 * row_height, start_size=120)
        elif self.content_type == 'kanji':
            self.draw_kanji_only(draw, item['kanji'], 2 * row_height, 2 * row_height)
        elif self.content_type == 'kanji_synonym':
            self.draw_kanji_only(draw, item['kanji_synonym'], 2 * row_height, 2 * row_height)
        elif self.content_type == 'arabic_synonym':
            self.draw_arabic_synonym(draw, item['arabic_synonym'], 2 * row_height, 2 * row_height)
        elif self.content_type == "chinese_synonym":
            self.draw_chinese_synonym(draw, item['chinese_synonym'], 2 * row_height, 2 * row_height)
        elif self.content_type == "japanese_and_arabic":
            # print("Drawing japanese and arabic...")
            self.draw_japanese(draw, item['japanese_synonym'], 1.7 * row_height, 1.8 * row_height, start_size=120)
            self.draw_arabic_synonym(draw, item['arabic_synonym'], 2.9 * row_height, 1 * row_height)
        self.save_intermediate_image()


        self.save_all_images()
        self.clear_intermediate_images()

        # Scale the image up to fit the display
        return self.image.resize((self.width * self.scale_factor, self.height * self.scale_factor), Resampling.LANCZOS)

    def find_font_size(self, text, font_path, max_width, max_height, start_size=120, step=2):
        font_size = start_size
        font = ImageFont.truetype(font_path, font_size)
        while True:
            text_width, text_height = self.get_text_size(text, font)
            if text_width <= max_width and text_height <= max_height:
                break
            font_size -= step
            if font_size <= 0:
                break
            font = ImageFont.truetype(font_path, font_size)
        return font_size

    def get_text_size(self, text, font):
        dummy_image = Image.new('RGB', (100, 100))
        draw = ImageDraw.Draw(dummy_image)
        return draw.textbbox((0, 0), text, font=font)[2:]


    def draw_phonetic(self, draw, phonetic_text, start_y, row_height):
        phonetic_text_cleaned = phonetic_text.replace('·', '')
        font = ImageFont.truetype(self.ipa_font_path, self.find_font_size(phonetic_text_cleaned, self.ipa_font_path, self.width, row_height))
        syllables = split_word_with_color(phonetic_text, self.pallete)
        
        # Calculate the total width of the line
        total_width = sum(self.get_text_size(syllable.replace('·', ''), font)[0] for syllable, _ in syllables)
        
        # Calculate Y position for the entire line
        line_height = self.get_text_size(phonetic_text_cleaned, font)[1]
        line_y = start_y + (row_height - line_height) / 2
        
        x = (self.width - total_width) / 2
        for syllable, color in syllables:
            draw.text((x, line_y), syllable.replace('·', ''), font=font, fill=color)
            self.save_intermediate_image()
            x += self.get_text_size(syllable.replace('·', ''), font)[0]

    def draw_word(self, draw, word_text, start_y, row_height):
        word_text_cleaned = word_text.replace('·', '')
        font = ImageFont.truetype(self.default_font_path, self.find_font_size(word_text_cleaned, self.default_font_path, self.width, row_height))
        syllables = split_word_with_color(word_text, self.pallete)
        
        # Calculate the total width of the line
        total_width = sum(self.get_text_size(syllable.replace('·', ''), font)[0] for syllable, _ in syllables)
        
        # Calculate Y position for the entire line
        line_height = self.get_text_size(word_text_cleaned, font)[1]
        line_y = start_y + (row_height - line_height) / 2
        
        x = (self.width - total_width) / 2
        for syllable, color in syllables:
            draw.text((x, line_y), syllable.replace('·', ''), font=font, fill=color)
            self.save_intermediate_image()
            x += self.get_text_size(syllable.replace('·', ''), font)[0]



    def draw_japanese(self, draw, japanese_text, start_y, row_height, start_size=120):
        self.draw_japanese_with_hiragana(draw, japanese_text, self.jp_font_path, self.width, start_y, row_height, start_size=start_size)



    def draw_kanji_only(self, draw, kanji_text, start_y, row_height):
        """
        Draws kanji characters, stripping away any non-kanji characters.
        """
        if not kanji_text:  # Skip if no kanji characters are present
            return

        kanji_text = re.sub(r'[^\u4e00-\u9faf]', '', kanji_text)  # Strip non-kanji characters

        if not kanji_text:  # Skip if no kanji characters are present
            return

        font_path = self.jp_font_path  # Assuming this is set in setup_fonts
        font_size = self.find_font_size(kanji_text, font_path, self.width, row_height)
        font = ImageFont.truetype(font_path, font_size)

        text_width, text_height = self.get_text_size(kanji_text, font)
        x = (self.width - text_width) / 2
        y = start_y + (row_height - text_height) / 2

        # draw.text((x, y), kanji_text, font=font, fill=(0, 0, 0))
        font_paths = [self.jp_font_path, self.jp_font_path_fallback]
        self.draw_kanji_char(draw, kanji_text, x, y, font_paths, font_size)
        self.save_intermediate_image()


    def draw_kanji_synonym_only(self, draw, kanji_text, start_y, row_height):
        """
        Draws kanji characters, stripping away any non-kanji characters.
        """
        if not kanji_text:  # Skip if no kanji characters are present
            return
        kanji_text = re.sub(r'[^\u4e00-\u9faf]', '', kanji_text)  # Strip non-kanji characters
        if not kanji_text:  # Skip if no kanji characters are present
            return

        font_path = self.jp_font_path  # Assuming this is set in setup_fonts
        font_size = self.find_font_size(kanji_text, font_path, self.width, row_height)
        font = ImageFont.truetype(font_path, font_size)

        text_width, text_height = self.get_text_size(kanji_text, font)
        x = (self.width - text_width) / 2
        y = start_y + (row_height - text_height) / 2

        # draw.text((x, y), kanji_text, font=font, fill=(0, 0, 0))
        font_paths = [self.jp_font_path, self.jp_font_path_fallback]
        self.draw_kanji_char(draw, kanji_text, x, y, font_paths, font_size)
        self.save_intermediate_image()

    def draw_chinese_synonym(self, draw, chinese_text, start_y, row_height):
        """
        Draws kanji characters, stripping away any non-kanji characters.
        """
        # kanji_text = re.sub(r'[^\u4e00-\u9faf]', '', kanji_text)  # Strip non-kanji characters
        if not chinese_text:  # Skip if no kanji characters are present
            return

        font_path = self.chinese_font_path  # Assuming this is set in setup_fonts
        font_size = self.find_font_size(chinese_text, font_path, self.width, row_height, start_size=180)
        font = ImageFont.truetype(font_path, font_size)

        text_width, text_height = self.get_text_size(chinese_text, font)
        x = (self.width - text_width) / 2
        y = start_y + (row_height - text_height) / 2

        draw.text((x, y), chinese_text, font=font, fill=(0, 0, 0))
        self.save_intermediate_image()




    def is_char_supported(self, character, font_path, background_color=(255, 255, 255)):
        font = ImageFont.truetype(font_path, 20)
        image = Image.new('RGB', (40, 40), background_color)
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), character, font=font, fill=(0, 0, 0))

        for x in range(image.width):
            for y in range(image.height):
                if image.getpixel((x, y)) != background_color:
                    return True
        return False

    def draw_kanji_char(self, draw, text, x, y, font_paths, font_size):

        get_text_size = self.get_text_size

        for char in text:
            for font_path in font_paths:
                if self.is_char_supported(char, font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
                else:
                    # If no font supports the character, use the last font in the list
                    font = ImageFont.truetype(font_paths[-1], font_size)

            draw.text((x, y), char, font=font, fill=(0, 0, 0))
            # print("text: ", text)
            # x += font.getsize(char)[0]  # Update x position for next character
            x += get_text_size(char, font)[0]  # Update x position for next character


    def draw_japanese_with_hiragana(self, draw, text, jp_font_path, max_width, y, max_height, start_size=120):
        find_font_size = self.find_font_size
        get_text_size = self.get_text_size

        text = text.replace(" ", "").replace("(", "（").replace(")", "）")
        regex = re.compile(r'([一-龠ァ-ヴガ-ドㇰ-ㇿヵヶヰヱ々〆〤ー\-]+)（([ぁ-ゔー\-]+)）')

        plain_text = re.sub(r'（[ぁ-ゔー\-ー\-]+）', '', text)
        font_size = find_font_size(plain_text, jp_font_path, max_width, max_height, start_size=start_size)
        font = ImageFont.truetype(jp_font_path, font_size)

        pos_x = (max_width - get_text_size(plain_text, font)[0]) / 2
        y += (max_height - get_text_size(plain_text, font)[1]) / 2  # Center vertically in the row

        last_match_end = 0
        for match in regex.finditer(text):
            kanji_or_katakana, hiragana = match.groups()
            start, end = match.span()

            preceding_text = text[last_match_end:start]
            draw.text((pos_x, y), preceding_text, font=font, fill=(0, 0, 0))
            pos_x += get_text_size(preceding_text, font)[0]

            # draw.text((pos_x, y), re.sub(r'（[ぁ-んァ-ンー-]+）', '', kanji_or_katakana), font=font, fill=(0, 0, 0))
            font_paths = [self.jp_font_path, self.jp_font_path_fallback]
            self.draw_kanji_char(draw, re.sub(r'（[ぁ-ゔー\-（）]+）', '', kanji_or_katakana), pos_x, y, font_paths, font_size)

            kanji_or_katakana_width = get_text_size(kanji_or_katakana, font)[0]
            kanji_or_katakana_height = get_text_size(kanji_or_katakana, font)[1]

            hiragana_font_size = find_font_size(hiragana, jp_font_path, kanji_or_katakana_width, (max_height - kanji_or_katakana_height) / 2)
            hiragana_font = ImageFont.truetype(jp_font_path, hiragana_font_size)
            hiragana_x = pos_x + (kanji_or_katakana_width - get_text_size(hiragana, hiragana_font)[0]) / 2
            hiragana_y = y - get_text_size(hiragana, hiragana_font)[1] - self.used_height
            draw.text((hiragana_x, hiragana_y), hiragana, font=hiragana_font, fill=(0, 0, 0))

            pos_x += kanji_or_katakana_width
            last_match_end = end

        remaining_text = text[last_match_end:]
        draw.text((pos_x, y), re.sub(r'（[ぁ-ゔー\-]+）', '', remaining_text), font=font, fill=(0, 0, 0))

    # def draw_arabic_synonym(self, draw, arabic_text, start_y, row_height):
    #     """
    #     Draws Arabic text with different colors from the palette.
    #     """

    #     if not arabic_text:  # Skip if no kanji characters are present
    #         return
    #     # Define the font path for Arabic text
    #     arabic_font_path = self.arabic_font_path  # Replace with your Arabic font file

    #     # Determine the font size
    #     font_size = self.find_font_size(arabic_text, arabic_font_path, self.width, row_height)
    #     font = ImageFont.truetype(arabic_font_path, font_size)

    #     # Initialize color cycle
    #     color_cycle = itertools.cycle(self.pallete)

    #     # Split the text into characters
    #     characters = list(arabic_text)
    #     total_width = sum(self.get_text_size(char, font)[0] for char in characters)

    #     # Calculate starting x position
    #     x = (self.width - total_width) / 2
    #     y = start_y + (row_height - self.get_text_size(arabic_text, font)[1]) / 2

    #     # Draw each character with a color from the palette
    #     for char in characters:
    #         draw.text((x, y), char, font=font, fill=next(color_cycle))
    #         self.save_intermediate_image()
    #         x += self.get_text_size(char, font)[0]

    # def draw_arabic_synonym(self, draw, arabic_text, start_y, row_height):
    #     """
    #     Draws Arabic text with different colors from the palette, accounting for right-to-left writing.
    #     """

    #     if not arabic_text:  # Skip if no Arabic characters are present
    #         return

    #     # Define the font path for Arabic text
    #     arabic_font_path = self.arabic_font_path  # Replace with your Arabic font file

    #     # Determine the font size
    #     font_size = self.find_font_size(arabic_text, arabic_font_path, self.width, row_height)
    #     font = ImageFont.truetype(arabic_font_path, font_size)

    #     # Initialize color cycle
    #     color_cycle = itertools.cycle(self.pallete)

    #     # Split the text into characters
    #     characters = list(arabic_text)
    #     total_width = sum(self.get_text_size(char, font)[0] for char in characters)

    #     # Calculate starting x position (starting from right)
    #     x = (self.width + total_width) / 2

    #     # Draw each character with a color from the palette, moving right to left
    #     for char in reversed(characters):
    #         char_width, char_height = self.get_text_size(char, font)
    #         x -= char_width
    #         y = start_y + (row_height - char_height) / 2
    #         draw.text((x, y), char, font=font, fill=next(color_cycle))
    #         self.save_intermediate_image()

    # def draw_arabic_synonym(self, draw, arabic_text, start_y, row_height):
    #     """
    #     Draws Arabic text with different colors from the palette, respecting the correct form of each character.
    #     """

    #     if not arabic_text:  # Skip if no Arabic characters are present
    #         return

    #     # Reshape and apply RTL to the Arabic text
    #     reshaped_text = arabic_reshaper.reshape(arabic_text)
    #     bidi_text = get_display(reshaped_text)

    #     # Define the font path for Arabic text
    #     arabic_font_path = self.arabic_font_path  # Replace with your Arabic font file

    #     # Determine the font size
    #     font_size = self.find_font_size(bidi_text, arabic_font_path, self.width, row_height)
    #     font = ImageFont.truetype(arabic_font_path, font_size)

    #     # Initialize color cycle
    #     color_cycle = itertools.cycle(self.pallete)

    #     # Split the reshaped and reordered text into characters
    #     characters = list(bidi_text)
    #     total_width = sum(self.get_text_size(char, font)[0] for char in characters)

    #     # Calculate starting x position (starting from right)
    #     x = (self.width + total_width) / 2

    #     # Draw each character with a color from the palette, moving right to left
    #     for char in reversed(characters):
    #         char_width, char_height = self.get_text_size(char, font)
    #         x -= char_width
    #         y = start_y + (row_height - char_height) / 2
    #         draw.text((x, y), char, font=font, fill=next(color_cycle))
    #         self.save_intermediate_image()

    def draw_arabic_synonym(self, draw, arabic_text, start_y, row_height):
        """
        Draws Arabic text with different colors from the palette, with characters aligned at the same height.
        """

        if not arabic_text:  # Skip if no Arabic characters are present
            return

        # Reshape and apply RTL to the Arabic text
        reshaped_text = arabic_reshaper.reshape(arabic_text)
        bidi_text = get_display(reshaped_text)

        # Define the font path for Arabic text
        arabic_font_path = self.arabic_font_path  # Replace with your Arabic font file

        # Determine the font size
        font_size = self.find_font_size(bidi_text, arabic_font_path, self.width, row_height)
        font = ImageFont.truetype(arabic_font_path, font_size)

        # Initialize color cycle
        color_cycle = itertools.cycle(self.pallete)

        # Split the reshaped and reordered text into characters
        characters = list(bidi_text)
        total_width = sum(self.get_text_size(char, font)[0] for char in characters)

        # Find the maximum character height to align all characters
        max_char_height = max(self.get_text_size(char, font)[1] for char in characters)

        # Calculate starting x position (starting from right)
        x = (self.width + total_width) / 2

        # Fixed y-position based on maximum character height
        y = start_y + (row_height - max_char_height) / 2

        # # Draw each character with a color from the palette, moving right to left
        # for char in reversed(characters):
        #     char_width, _ = self.get_text_size(char, font)
        #     x -= char_width
        #     draw.text((x, y), char, font=font, fill=next(color_cycle))
        #     self.save_intermediate_image()

        # Draw each character with a color from the palette, moving right to left
        for char in reversed(characters):
            char_width, _ = self.get_text_size(char, font)
            x -= char_width
            if char == ' ':  # Reset the color cycle when encountering a space
                color_cycle = itertools.cycle(self.pallete)
            draw.text((x, y), char, font=font, fill=next(color_cycle))
            self.save_intermediate_image()



if __name__=="__main__":
    # import logging
    # from waveshare_epd import epd7in3f
    # import time

    # Create the parser
    parser = argparse.ArgumentParser(description='Run script with OpenAI option.')

    # Add an argument
    parser.add_argument('--enable_openai', action='store_true', help='Enable OpenAI features')
    parser.add_argument('--make_emoji', action='store_true', default=False, help='Enable emoji making feature')

    # Parse the arguments
    args = parser.parse_args()

    # Set variable based on the argument
    enable_openai = args.enable_openai

    # make_emoji = False
    make_emoji = args.make_emoji

    logging.basicConfig(level=logging.DEBUG)

    

    epd_module = epd7in3f
    epd_hardware = EPaperHardware(epd_module)
    # epd_display = EPaperDisplay(epd_hardware, font_root, content_type="japanese_synonym", image_folder="images-japanese-synonym")
    epd_display = EPaperDisplay(epd_hardware, font_root, content_type="japanese_and_arabic", image_folder="images-japanese-synonym")
    epd_display.pallete = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 165), (0, 255, 165)]
    if make_emoji:
        epd_display_virtual_kanji = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="kanji", image_folder="virtual-images-grey-kanji", emoji_path="emoji")
        epd_display_virtual_kanji_synonym = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="kanji_synonym", image_folder="virtual-images-grey-kanji-synonym", emoji_path="emoji")
        epd_display_virtual_japanese = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="japanese_synonym", image_folder="virtual-images-grey-japanese-synonym", emoji_path="emoji")
        epd_display_virtual_arabic = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="arabic_synonym", image_folder="virtual-images-grey-arabic-synonym", emoji_path="emoji")
        epd_display_virtual_japanese_and_arabic = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="japanese_and_arabic", image_folder="virtual-images-grey-japanese-and-arabic", emoji_path="emoji")
        epd_display_virtual_chinese = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="chinese_synonym", image_folder="virtual-images-grey-chinese", emoji_path="emoji")
    else:
        epd_display_virtual_kanji = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="kanji", image_folder="virtual-images-grey-kanji")
        epd_display_virtual_kanji_synonym = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="kanji_synonym", image_folder="virtual-images-grey-kanji-synonym")
        epd_display_virtual_japanese = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="japanese_synonym", image_folder="virtual-images-grey-japanese-synonym")
        epd_display_virtual_arabic = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="arabic_synonym", image_folder="virtual-images-grey-arabic-synonym")
        epd_display_virtual_japanese_and_arabic = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="japanese_and_arabic", image_folder="virtual-images-grey-japanese-and-arabic")
        epd_display_virtual_chinese = EPaperDisplay(epd_hardware, font_root, background_texture="grey", content_type="chinese_synonym", image_folder="virtual-images-grey-chinese")



    words_list = [
        # "impeccable"
        # "gratitude",
        # "appreciation"
        # "amalgamate"
    ]

    words_list = [
        # "incontrovertible", "benevolent", 
        # "peregrinate", "obstreperous", "hiraeth", "idyllic", "chimerical", 
        # "cacophony", 
        # "serendipity", "sacrosanct", 
        # "reticent", "do", "accolade", "jingoism", 
        # "facilitate", 
        # "infallible", "rescind", 
        # "trepidation", 
        # "raconteur", 
        # "xenophobia", 
        # "indubitable", 
        # "propensity", 
        # "gregarious", 
        # "magnate",
        # "perambulation", 
        # "wanderlust",
        # "apoplectic",
        # "xenial",
        # "metamorphosis",
        # "misanthrope",
        # "fairydom",
        # "subgyrus",
        # "rescind",
        # "raconteur",
        # "malaise",
        # "insouciance",
        # "inarticulately",
        # "opulence",
        # "mundane",
        # "prosaic",
        # "gratitude",
        # "appreciation",
        # "dry"，
        # "stem",
        # "run"
    ]


    chooser = OpenAiChooser(words_db, word_fetcher, words_list=words_list)
    # Set the openaichooser.enable_openai based on the command line argument
    chooser.enable_openai = enable_openai

    words_set = set(words_list)
    try:
        while True:
            item = chooser.choose()
            # item = chooser.choose()
            print("word: ", item)
            content_image = epd_display.create_content_layout(item)
            epd_display_virtual_kanji.create_content_layout(item)
            epd_display_virtual_kanji_synonym.create_content_layout(item)
            epd_display_virtual_japanese.create_content_layout(item)
            epd_display_virtual_arabic.create_content_layout(item)
            epd_display_virtual_japanese_and_arabic.create_content_layout(item)
            epd_display_virtual_chinese.create_content_layout(item)

            if not make_emoji:
                epd_hardware.display_image(content_image)
                time.sleep(300)  # Display each word for 5 minutes


            if len(words_list) > 0:
                try: 
                    words_set.discard(item["word"])
                except:
                    continue

                if len(words_set) == 0:
                    break

    except Exception as e:
        print("Exception: ", str(e))

        traceback.print_exc()

        logging.info(e)
    finally:
        epd_hardware.clear_and_sleep()

