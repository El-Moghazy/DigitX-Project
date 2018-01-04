# Import the modules
# for the implementation of this code i have used and modified the code in http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import imutils
import cv2

from keras.models import model_from_json# Import the modules
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import imutils
import cv2
import serial, time


from keras.models import model_from_json

from keras import backend as k

k.set_image_dim_ordering('th')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, 1)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# Load the classifier
clf = loaded_model

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    im = imutils.resize(frame, width=850)

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray,  50   , 255 , cv2.THRESH_BINARY_INV)

    cv2.imshow("Print", im_th)

    # Find contours in the image
    im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(ctrs) > 0:
        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        num = list()
        for rect in rects:
            # print ( rect[2] , rect[3] )
            # print()
            if (rect[2] > 10 and rect[3] > 47) and (rect[2] < 90 and rect[3] < 90):
                # Make the rectangular region around the digit
                leng = int(rect[3] * 1.6)
                pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
                # Resize the image
                if (roi.size > 0):
                    # print(len(roi[0]), len(roi[1]))
                    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = cv2.dilate(roi, (3, 3))
                    # Calculate the HOG features
                    img_re = np.array([roi], 'float64').reshape(1, 1, 28, 28)
                    nbr = clf.predict(img_re)
                    max = -1
                    maxi = 0
                    i = 0
                    for i in range(len(nbr[0])):
                        if ((nbr[0])[i] > max):
                            max = (nbr[0])[i]
                            maxi = i
                    if (maxi in [1, 2, 3, 4, 5, 6]):
                        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
                        num.append(maxi)
                        # print(maxi)
                        cv2.putText(im, str(maxi), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    cv2.imshow("Frame", im)
    key = cv2.waitKey(20)
