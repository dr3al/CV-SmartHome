from typing import Union

import cv2
import numpy as np

from datetime import datetime as dt
timeFormat = "%d-%m-%Y_%H-%M-%S.png"

EXPORT_PATH = 'D:/photo/'
FILE_NAME = f"{dt.now():{timeFormat}}"


def takeFrame(videoPath: int):
    cap = cv2.VideoCapture(videoPath)
    if cap.isOpened():
        ret, frame = cap.read()
    cap.release()
    return frame


def takePicture(frame: Union[int, np.ndarray] = 0):
    if type(frame) == np.ndarray:
        cv2.imwrite(EXPORT_PATH + FILE_NAME, frame)
        return EXPORT_PATH + FILE_NAME

    elif type(frame) == int:
        cv2.imwrite(EXPORT_PATH + FILE_NAME, takeFrame(frame))
        return EXPORT_PATH + FILE_NAME

    else:
        raise ValueError(f"False type of frame! \n Your input: ({frame})")
