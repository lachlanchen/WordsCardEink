from gpiozero import DigitalOutputDevice

pin_number = 8

cs_pin = DigitalOutputDevice(pin_number, active_high=False)  # Configure as needed

# To select the device:
cs_pin.off()

# To deselect the device:
cs_pin.on()
