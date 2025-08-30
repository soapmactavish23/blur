import cv2
import numpy as np
import scipy
import scipy.ndimage
from scipy import signal
from scipy.signal.windows import gaussian

PATH_XML = 'haarcascade_frontalface_default.xml'

def generate_kernel(kernel_len=5, desvio_padrao=5):
    if kernel_len % 2 == 0:
        kernel_len += 1
    g1d = gaussian(kernel_len, std=desvio_padrao).reshape(-1, 1)
    return g1d @ g1d.T

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + PATH_XML)

kernel = generate_kernel(kernel_len=31, desvio_padrao=30)

kernel_tile = np.tile(kernel, (3, 1, 1))
kernel_sum = kernel.sum()
kernel = kernel / kernel_sum

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w] = scipy.ndimage.convolve(frame[y:y+h, x:x+w], np.atleast_3d(kernel), mode='nearest')
        # cv2.GaussianBlur(frame[y:y+h, x:x+w], (63, 63), sigmaX=20, sigmaY=20)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()