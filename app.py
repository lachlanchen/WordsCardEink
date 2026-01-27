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

# Essential imports and path setup
import sys
import os
import argparse
import logging
import traceback
import threading
import json
import base64
from io import BytesIO
import time
import csv
import random
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import opencc
import tornado.ioloop
import tornado.web
from tornado.web import RequestHandler
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado import autoreload

from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling
from openai import OpenAI

# Load env overrides early for OpenAI config
from env_loader import load_env

load_env()

# Custom imports from the local library
pic_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pic')
lib_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
font_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'font')
if os.path.exists(lib_root):
    sys.path.append(lib_root)

from waveshare_epd import epd7in3f
from words_data import WordsDatabase, AdvancedWordFetcher, OpenAiChooser, EmojiWordChooser, split_word, split_word_with_color, count_syllables
from words_gpt import EPaperDisplay, EPaperHardware, EmojiWordChooser, OpenAiChooser, read_words_list  # Ensure these are imported correctly

from words_data import PhoneticRechecker


# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Define the port for the Tornado server
PORT = 8082

# Initialize global variables
current_word_item = None
lock = threading.Lock()

# Usage example for OpenAI GPT
client = OpenAI()

# Database path and initialization
db_path = 'words_phonetics.db'
words_db = WordsDatabase(db_path)
word_fetcher = AdvancedWordFetcher()
phonetic_checker = PhoneticRechecker()


# # Initialize logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the port for the Tornado server
# PORT = 8082

# # Initialize global variables
# current_word_item = None
# lock = threading.Lock()

# Function to convert PIL Image to base64 string
def get_image_as_base64_string(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

# async def periodic_task():
#     http_client = AsyncHTTPClient()
#     try:
#         request = HTTPRequest(url="http://localhost:{}/next_random_word".format(PORT), method="GET")
#         response = await http_client.fetch(request)
#         if response.code == 200:
#             logging.info("Successfully called next_random_word API")
#         else:
#             logging.error(f"Failed to call next_random_word API. Status code: {response.code}")
#     except Exception as e:
#         logging.error(f"Error in periodic_task: {str(e)}")


async def periodic_task():
    http_client = AsyncHTTPClient()
    try:
        # Increase timeout to 5 minutes for complex word processing
        request = HTTPRequest(
            url=f"http://localhost:{PORT}/next_random_word", 
            method="GET",
            request_timeout=300.0  # 5 minutes timeout
        )
        response = await http_client.fetch(request)
        if response.code == 200:
            logging.info("Successfully called next_random_word API")
        else:
            logging.error(f"Failed to call next_random_word API. Status code: {response.code}")
    except Exception as e:
        logging.warning(f"Timeout in periodic_task (expected for complex processing): {str(e)}")
    finally:
        http_client.close()


class DisplayWordHandler(RequestHandler):
    def post(self):
        global current_word_item
        try:
            data = json.loads(self.request.body)
            input_word = data.get('word', '')
            
            # Check if the input is a list or a single word
            if isinstance(input_word, list):
                # If it's a list, update the chooser's word list
                chooser.update_words_list(input_word)
                word_details = chooser.choose()
            elif isinstance(input_word, str):
                # If it's a single word, treat it as a new word list with one word
                chooser.update_words_list([input_word])
                word_details = chooser.choose()
            else:
                self.write({"status": "error", "message": "Invalid word format"})
                return

            with lock:
                current_word_item = word_details
                # Display the word on the E-ink display

                content_image = epd_display.create_content_layout(word_details)
                epd_hardware.display_image(content_image)

            # Respond with the generated image
            image_base64 = get_image_as_base64_string(content_image)
            self.write({"status": "success", "word": input_word, "image": image_base64})
        except Exception as e:
            logging.error(f"Error in DisplayWordHandler: {str(e)}")
            traceback.print_exc()
            self.write({"status": "error", "message": "Internal server error"})


class GetCurrentWordHandler(RequestHandler):
    def get(self):
        global current_word_item



        try:
            with lock:
                phonetic_checker.recheck_word_phonetics_with_paired_tuple([current_word_item["word"]], words_db)

                if current_word_item:
                    content_image = epd_display.create_content_layout(current_word_item)
                    image_base64 = get_image_as_base64_string(content_image)
                    self.write({"status": "success", "word": current_word_item, "image": image_base64})
                else:
                    self.write({"status": "error", "message": "No current word displayed"})
        except Exception as e:
            logging.error(f"Error in GetCurrentWordHandler: {str(e)}")
            traceback.print_exc()
            self.write({"status": "error", "message": "Internal server error"})


class GetCurrentWordPageHandler(RequestHandler):
    def get(self):
        global current_word_item

        try:
            with lock:
                phonetic_checker.recheck_word_phonetics_with_paired_tuple([current_word_item["word"]], words_db)

                if current_word_item:
                    content_image = epd_display.create_content_layout(current_word_item)
                    image_io = BytesIO()
                    content_image.save(image_io, format='PNG')
                    image_bytes = image_io.getvalue()

                    # Ensure the directory exists
                    image_directory = 'words_card_temp'
                    os.makedirs(image_directory, exist_ok=True)

                    # Save the image to the words_card_temp folder
                    image_path = os.path.join(image_directory, f"{current_word_item['word']}.png")
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)

                    self.set_header("Content-Type", "image/png")
                    self.write(image_bytes)
                else:
                    self.set_status(404)
                    self.write({"status": "error", "message": "No current word displayed"})
        except Exception as e:
            logging.error(f"Error in GetCurrentWordPageHandler: {str(e)}")
            traceback.print_exc()
            self.set_status(500)
            self.write({"status": "error", "message": "Internal server error"})

class NextRandomWordHandler(RequestHandler):
    def get(self):
        global current_word_item
        try:
            word_details = chooser.choose()  # Assuming chooser has choose_random_word method

            with lock:
                current_word_item = word_details

                print("word: ", current_word_item)
                
                # Display the word on the E-ink display
                content_image = epd_display.create_content_layout(word_details)
                epd_hardware.display_image(content_image)

            # Get the image of the word card
            image_base64 = get_image_as_base64_string(content_image)
            self.write({"status": "success", "word": word_details, "image": image_base64})
        except Exception as e:
            logging.error(f"Error in NextRandomWordHandler: {str(e)}")
            traceback.print_exc()
            self.write({"status": "error", "message": "Internal server error"})

class GetWordsCardHandler(RequestHandler):
    # def get(self):
    #     return self.post()

    def post(self):
        try:


            data = json.loads(self.request.body)
            input_word = data.get('word', '')
            background_texture = data.get('background_texture', 'grey')
            content_type = data.get('content_type', 'japanese_and_arabic')
            image_folder = data.get('image_folder', f"virtual-images-{background_texture}-{content_type}")
            is_emoji = data.get('is_emoji', False)
            emoji_path = "emoji" if is_emoji else None

            # Validate input
            if not input_word:
                self.write({"status": "error", "message": "Word is required"})
                return

            # Validate and set content_type, background_texture, and image_folder
            if content_type not in ["japanese_synonym", "kanji", "kanji_synonym", "arabic_synonym", "chinese_synonym", "simplified_chinese_synonym", "japanese_and_arabic", "film"]:
                self.write({"status": "error", "message": "Invalid content type"})
                return
            
            # Create EPaperDisplay instance based on provided parameters
            epd_display_temp = EPaperDisplay(epd_hardware, font_root, background_texture=background_texture, content_type=content_type, image_folder=image_folder, emoji_path=emoji_path)
            
            # Check if the input is a list or a single word, and update the chooser's word list accordingly
            if isinstance(input_word, list):
                input_word = input_word
                # If it's a list, update the chooser's word list
                # chooser = OpenAiChooser(words_db, word_fetcher, words_list=input_word)
                # phonetic_checker.recheck_word_phonetics_with_paired_tuple(input_word, words_db)
                # chooser.update_words_list(input_word)
            elif isinstance(input_word, str):
                input_word = [input_word]
                # If it's a single word, treat it as a new word list with one word
                # chooser.update_words_list([input_word])
                # chooser = OpenAiChooser(words_db, word_fetcher, words_list=[input_word])
                # phonetic_checker.recheck_word_phonetics_with_paired_tuple([input_word], words_db)
            else:
                self.write({"status": "error", "message": "Invalid word format"})
                return
            
            phonetic_checker.recheck_word_phonetics_with_paired_tuple(input_word, words_db)
            chooser_temp = OpenAiChooser(words_db, word_fetcher, words_list=input_word)
            # chooser.update_words_list(input_word)


            # Use chooser to get word details
            word_details = chooser_temp.choose()

            # Generate the image without displaying it on the e-ink display
            content_image = epd_display_temp.create_content_layout(word_details)
            image_base64 = get_image_as_base64_string(content_image)
            self.write({"status": "success", "word": input_word, "image": image_base64})
        except Exception as e:
            logging.error(f"Error in GetWordsCardHandler: {str(e)}")
            traceback.print_exc()
            self.write({"status": "error", "message": "Internal server error"})



def make_app():
    return tornado.web.Application([
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "words_card_temp"}),
        (r"/display_word", DisplayWordHandler),
        (r"/get_current_word", GetCurrentWordHandler),
        (r"/get_current_word_page", GetCurrentWordPageHandler),
        (r"/next_random_word", NextRandomWordHandler),
        (r"/get_words_card", GetWordsCardHandler),
    ])

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Run Eink Words GPT with options.')
    parser.add_argument('--enable_openai', action='store_true', help='Enable OpenAI features')
    parser.add_argument('--make_emoji', action='store_true', default=False, help='Enable emoji making feature')
    parser.add_argument('--ignore_list', action='store_true', default=False, help='Ignore the words list')
    parser.add_argument('--simplify', action='store_true', default=False, help='Simplify kanji and traditional Chinese')
    parser.add_argument('--use_csv', action='store_true', default=False, help='Use words from csv')
    parser.add_argument('--complete_csv', action='store_true', default=False, help='Use word details from csv')
    parser.add_argument('--filename', help='Filename of word details csv')
    args = parser.parse_args()

    # E-Ink display initialization
    epd_module = epd7in3f
    epd_hardware = EPaperHardware(epd_module)
    epd_display = EPaperDisplay(epd_hardware, font_root, content_type="japanese_and_arabic", image_folder="images-japanese-synonym")
    epd_display.pallete = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 165), (0, 255, 165)]
    # Setup based on arguments
    if args.complete_csv:
        chooser = EmojiWordChooser(csv_file_path=f"data/{args.filename}.csv")
    else:
        words_list = read_words_list("data/words_list.csv") if args.use_csv else []
        words_list = words_list if not args.ignore_list else []
        chooser = OpenAiChooser(words_db, word_fetcher, words_list=words_list)
        chooser.enable_openai = args.enable_openai  # Enable OpenAI based on the command line argument


    # Set the interval for updating the e-ink display (e.g., every 5 minutes)
    update_interval_ms = 5 * 60 * 1000  # 5 minutes in milliseconds

    # Execute the periodic_task immediately.
    tornado.ioloop.IOLoop.current().spawn_callback(periodic_task)

    # Create and start the periodic callback
    periodic_callback = tornado.ioloop.PeriodicCallback(
        lambda: tornado.ioloop.IOLoop.current().spawn_callback(periodic_task),
        update_interval_ms
    )
    periodic_callback.start()

    # Setup the Tornado server and start listening
    app = make_app()
    app.listen(PORT)
    print(f"Server is running on http://lazyingart:{PORT}")

    # # Enable autoreload
    # if tornado.options.options.debug:
    #     # Watch files and restart server on changes
    autoreload.start()
    tornado.ioloop.IOLoop.current().start()
