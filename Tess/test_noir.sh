# Blue ide face thernet/usb (in v4)

# while led - normal images
# while led off - ir images

sudo  raspi-config

: '
Interface Options
    ↓
Camera
    ↓
Enable
'

sudo reboot

# verifying 
libcamera-hello

# capturing first image
libcamera-still -o test.jpg

from picamera2 import Picamera2
import cv2
picam2 = Picamera2()
picam2.start()

: ' For making it constant
while True:
    frame = picam2.capture_array()
    cv2.imshow("Live Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'

frame = picam2.capture_array()
cv2.imwrite("test.jpg", frame)
print("Image captured and saved as test.jpg")

# live preview
libcamera-hello 

# opening the image
xdg-open test.jpg

# -  Test it using a remote (tv, ac)

# installing python lib
sudo apt install python3-picamera2 -y


