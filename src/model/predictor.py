import os
import wget

import cv2
import numpy as np
from ultralytics import YOLO

import streamlit as st


class Predict:
    MODEL_PATH = "model/makeup.pt"

    def __init__(self):
        if not os.path.isfile(self.MODEL_PATH):
            wget.download(st.secrets["MODEL_URL"], out="model")

        self.model = YOLO(self.MODEL_PATH)

    def __call__(self, image, *args, **kwargs):
        image_bytes = np.asarray(bytearray(image), dtype=np.uint8)
        image_array = cv2.imdecode(image_bytes, 1)

        result = self.model.predict(image_array, conf=kwargs['conf'], iou=kwargs['iou'])

        return result[0].plot()


predict = Predict()
