import cv2
import numpy as np
import scipy
import scipy.ndimage
from scipy import signal

PATH_XML = 'haarcascade_frontalface_default.xml'

def generate_kernel(kernel_len=5, desvio_padrao=5):
    generate_kernel1d = signal.gaussian(kernel_len, std=desvio_padrao).reshape(kernel_len, 1)
    generate_kernel2d = np.outer(generate_kernel1d, generate_kernel1d)

    return generate_kernel2d


video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascade + PATH_XML)

kernel = generate_kernel()

kernel_tile = np.tile(kernel, (3, 1, 1))
kernel_sum = kernel.sum()
kernel = kernel / kernel_sum

while True:
    ret, frame = video_capture.read()

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()