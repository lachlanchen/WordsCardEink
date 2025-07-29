# /*****************************************************************************
# * | File        :	  epdconfig.py
# * | Author      :   Waveshare team
# * | Function    :   Hardware underlying interface
# * | Info        :
# *----------------
# * | This version:   V1.2
# * | Date        :   2022-10-29
# * | Info        :   
# ******************************************************************************
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import os
import logging
import sys
import time

from gpiozero import DigitalOutputDevice, DigitalInputDevice
import spidev

logger = logging.getLogger(__name__)


# class RaspberryPi:
#     # Pin definition
#     RST_PIN  = 17
#     DC_PIN   = 25
#     CS_PIN   = 8
#     BUSY_PIN = 24
#     PWR_PIN  = 18

#     def __init__(self):
#         import spidev
#         import RPi.GPIO

#         self.GPIO = RPi.GPIO
#         self.SPI = spidev.SpiDev()

#     def digital_write(self, pin, value):
#         self.GPIO.output(pin, value)

#     def digital_read(self, pin):
#         return self.GPIO.input(pin)

#     def delay_ms(self, delaytime):
#         time.sleep(delaytime / 1000.0)

#     def spi_writebyte(self, data):
#         self.SPI.writebytes(data)

#     def spi_writebyte2(self, data):
#         self.SPI.writebytes2(data)

#     def module_init(self):
#         self.GPIO.setmode(self.GPIO.BCM)
#         self.GPIO.setwarnings(False)
#         self.GPIO.setup(self.RST_PIN, self.GPIO.OUT)
#         self.GPIO.setup(self.DC_PIN, self.GPIO.OUT)
#         self.GPIO.setup(self.CS_PIN, self.GPIO.OUT)
#         self.GPIO.setup(self.PWR_PIN, self.GPIO.OUT)
#         self.GPIO.setup(self.BUSY_PIN, self.GPIO.IN)
        
#         self.GPIO.output(self.PWR_PIN, 1)

#         # SPI device, bus = 0, device = 0
#         self.SPI.open(0, 0)
#         self.SPI.max_speed_hz = 4000000
#         self.SPI.mode = 0b00
#         return 0

#     def module_exit(self):
#         logger.debug("spi end")
#         self.SPI.close()

#         logger.debug("close 5V, Module enters 0 power consumption ...")
#         self.GPIO.output(self.RST_PIN, 0)
#         self.GPIO.output(self.DC_PIN, 0)
#         self.GPIO.output(self.PWR_PIN, 0)

#         self.GPIO.cleanup([self.RST_PIN, self.DC_PIN, self.CS_PIN, self.BUSY_PIN, self.PWR_PIN])

class RaspberryPi:
    # Pin definition
    # Pin definition using BCM GPIO numbers
    RST_PIN  = 17   # BCM GPIO 17, Board Pin 11
    DC_PIN   = 25   # BCM GPIO 25, Board Pin 22
    CS_PIN   = 8    # BCM GPIO 8, Board Pin 24
    BUSY_PIN = 24   # BCM GPIO 24, Board Pin 18
    # PWR_PIN is not a GPIO pin but typically tied to a power pin, here we use the BCM GPIO number for consistency
    PWR_PIN  = 18   # BCM GPIO 18, assumed to be used for power control, verify this with your specific setup


    def __init__(self):
        self.RST = DigitalOutputDevice(self.RST_PIN)
        self.DC = DigitalOutputDevice(self.DC_PIN)
        # self.CS = DigitalOutputDevice(self.CS_PIN)
        self.PWR = DigitalOutputDevice(self.PWR_PIN)
        self.BUSY = DigitalInputDevice(self.BUSY_PIN)

        self.SPI = spidev.SpiDev()

        # print("self.RST: ", self.RST)
        # print("self.DC: ", self.DC)
        # print("self.PWR: ", self.PWR)
        # print("self.BUSY: ", self.BUSY)

    def digital_write(self, pin, value):

        # print("pin: ", pin)
        # print("value: ", value)
        pin.on() if value else pin.off()

    def digital_read(self, pin):
        return pin.value

    def delay_ms(self, delaytime):
        time.sleep(delaytime / 1000.0)

    def spi_writebyte(self, data):
        self.SPI.writebytes(data)

    def spi_writebyte2(self, data):
        self.SPI.writebytes2(data)

    def module_init(self):
        self.digital_write(self.PWR, 1)

        # SPI device, bus = 0, device = 0
        self.SPI.open(0, 0)
        self.SPI.max_speed_hz = 4000000
        self.SPI.mode = 0b00
        return 0

    def module_exit(self):
        

        self.SPI.close()
        self.digital_write(self.RST, 0)
        self.digital_write(self.DC, 0)
        self.digital_write(self.PWR, 0)


        # gpiozero automatically cleans up the GPIO pins on object deletion


class JetsonNano:
    # Pin definition
    RST_PIN  = 17
    DC_PIN   = 25
    CS_PIN   = 8
    BUSY_PIN = 24
    PWR_PIN  = 18

    def __init__(self):
        import ctypes
        find_dirs = [
            os.path.dirname(os.path.realpath(__file__)),
            '/usr/local/lib',
            '/usr/lib',
        ]
        self.SPI = None
        for find_dir in find_dirs:
            so_filename = os.path.join(find_dir, 'sysfs_software_spi.so')
            if os.path.exists(so_filename):
                self.SPI = ctypes.cdll.LoadLibrary(so_filename)
                break
        if self.SPI is None:
            raise RuntimeError('Cannot find sysfs_software_spi.so')

        import Jetson.GPIO
        self.GPIO = Jetson.GPIO

    def digital_write(self, pin, value):
        self.GPIO.output(pin, value)

    def digital_read(self, pin):
        return self.GPIO.input(self.BUSY_PIN)

    def delay_ms(self, delaytime):
        time.sleep(delaytime / 1000.0)

    def spi_writebyte(self, data):
        self.SPI.SYSFS_software_spi_transfer(data[0])

    def spi_writebyte2(self, data):
        for i in range(len(data)):
            self.SPI.SYSFS_software_spi_transfer(data[i])

    def module_init(self):
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setwarnings(False)
        self.GPIO.setup(self.RST_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.DC_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.CS_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.PWR_PIN, self.GPIO.OUT)
        self.GPIO.setup(self.BUSY_PIN, self.GPIO.IN)
        
        self.GPIO.output(self.PWR_PIN, 1)
        
        self.SPI.SYSFS_software_spi_begin()
        return 0

    def module_exit(self):
        logger.debug("spi end")
        self.SPI.SYSFS_software_spi_end()

        logger.debug("close 5V, Module enters 0 power consumption ...")
        self.GPIO.output(self.RST_PIN, 0)
        self.GPIO.output(self.DC_PIN, 0)
        self.GPIO.output(self.PWR_PIN, 0)

        self.GPIO.cleanup([self.RST_PIN, self.DC_PIN, self.CS_PIN, self.BUSY_PIN, self.PWR_PIN])


class SunriseX3:
    # Pin definition
    RST_PIN  = 17
    DC_PIN   = 25
    CS_PIN   = 8
    BUSY_PIN = 24
    PWR_PIN  = 18
    Flag     = 0

    def __init__(self):
        import spidev
        import Hobot.GPIO

        self.GPIO = Hobot.GPIO
        self.SPI = spidev.SpiDev()

    def digital_write(self, pin, value):
        self.GPIO.output(pin, value)

    def digital_read(self, pin):
        return self.GPIO.input(pin)

    def delay_ms(self, delaytime):
        time.sleep(delaytime / 1000.0)

    def spi_writebyte(self, data):
        self.SPI.writebytes(data)

    def spi_writebyte2(self, data):
        # for i in range(len(data)):
        #     self.SPI.writebytes([data[i]])
        self.SPI.xfer3(data)

    def module_init(self):
        if self.Flag == 0:
            self.Flag = 1
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(self.RST_PIN, self.GPIO.OUT)
            self.GPIO.setup(self.DC_PIN, self.GPIO.OUT)
            self.GPIO.setup(self.CS_PIN, self.GPIO.OUT)
            self.GPIO.setup(self.PWR_PIN, self.GPIO.OUT)
            self.GPIO.setup(self.BUSY_PIN, self.GPIO.IN)

            self.GPIO.output(self.PWR_PIN, 1)
        
            # SPI device, bus = 0, device = 0
            self.SPI.open(2, 0)
            self.SPI.max_speed_hz = 4000000
            self.SPI.mode = 0b00
            return 0
        else:
            return 0

    def module_exit(self):
        logger.debug("spi end")
        self.SPI.close()

        logger.debug("close 5V, Module enters 0 power consumption ...")
        self.Flag = 0
        self.GPIO.output(self.RST_PIN, 0)
        self.GPIO.output(self.DC_PIN, 0)
        self.GPIO.output(self.PWR_PIN, 0)

        self.GPIO.cleanup([self.RST_PIN, self.DC_PIN, self.CS_PIN, self.BUSY_PIN], self.PWR_PIN)


if os.path.exists('/sys/bus/platform/drivers/rpi-gpiomem'):
    implementation = RaspberryPi()
elif os.path.exists('/sys/bus/platform/drivers/gpio-x3'):
    implementation = SunriseX3()
else:
    implementation = JetsonNano()

for func in [x for x in dir(implementation) if not x.startswith('_')]:
    setattr(sys.modules[__name__], func, getattr(implementation, func))

### END OF FILE ###
