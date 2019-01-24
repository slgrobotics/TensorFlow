import cv2
import numpy as np
import moviepy.editor as mpy

from libDonkeyCar.datastore import Tub

path_video = 'C:\\temp\\06240011.MOV'
path_tub = 'logtub'

inputs = ["cam/image_array", "user/angle", "user/throttle", "user/mode"]  # ['user/speed', 'cam/image']
types = ["image_array", "float", "float", "str"]  # ['float', 'image']

tub = Tub(path=path_tub, inputs=inputs, types=types)


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(path_video)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        frame = cv2.resize(frame, (160, 120))
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        image = frame
        angle = 0.0
        throttle = 0.0
        mode = 'user'

        data = {'cam/image_array': image, 'user/angle': angle, 'user/throttle': throttle, 'user/mode': mode}

        tub.put_record(data)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
