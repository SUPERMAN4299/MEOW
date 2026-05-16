sudo raspi-config

: ' 
Interface Options
I2C
Enable
'

sudo reboot 

sudo apt update
sudo apt install -y i2c-tools # installing I2C tools

# detecting i2c connected devices
i2cdetect -y 1