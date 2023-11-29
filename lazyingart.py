#!/usr/bin/python
# -*- coding:utf-8 -*-

# import os
# os.environ['GPIOZERO_PIN_FACTORY'] = os.environ.get('GPIOZERO_PIN_FACTORY', 'mock')
# import gpiozero
# from gpiozero.pins.mock import MockFactory
# gpiozero.Device.pin_factory = MockFactory()

# from gpiozero.pins.native import NativeFactory
# from gpiozero import LED

# factory = NativeFactory()
# led = LED(12, pin_factory=factory)

#

import sys
import os
picdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pic')
libdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

import logging
from waveshare_epd import epd7in3f
import time
from PIL import Image, ImageDraw, ImageFont
import traceback

logging.basicConfig(level=logging.DEBUG)

try:
    logging.info("epd7in3f Demo")

    epd = epd7in3f.EPD()
    logging.info("init and Clear")
    epd.init()
    epd.Clear()
    # Increase the font size to make it larger
    font_size = 60  # Adjust the font size as needed
    font = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), font_size)

    # Smaller image dimensions
    small_width, small_height = epd.width // 2, epd.height // 2

    # Drawing on the image
    logging.info("Drawing on the image...")
    small_image = Image.new('RGB', (small_width, small_height), (255,255,255))  # 255: clear the frame (white background)
    draw = ImageDraw.Draw(small_image)

    # Define the text and colors
    text = "lazying.art"
    # Colors: Black, Red, Green, Blue, Red, Yellow, Orange
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 165), (0, 255, 165)]

    # Draw each letter in a different color
    x = 10  # Starting position (x coordinate)
    y = 75  # Starting position (y coordinate)
    for i, char in enumerate(text):
        draw.text((x, y), char, font=font, fill=colors[i % len(colors)])
        # Update the x position for the next character
        x += font.getbbox(char)[2] + 10  # Adding a gap of 10 pixels between characters

    # Displaying the image
    # Scale the image up
    Himage = small_image.resize((epd.width, epd.height), Image.ANTIALIAS)
    epd.display(epd.getbuffer(Himage))
    time.sleep(12*3600)

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