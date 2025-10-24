# SPDX-License-Identifier: Apache-2.0

import cv2
import onnxruntime as ort
import argparse
import numpy as np
from pathlib import Path
from age_and_gender_detection.box_utils import predict
from pprint import pprint
import logging
import os
import sys

log_path = "/debug_preprocess.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("=== Container session started ===")


# scale current rectangle to box
def scale(box, image_width=None, image_height=None):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)

    x1 = box[0] - dx
    y1 = box[1] - dy
    x2 = box[2] + dx
    y2 = box[3] + dy

    if image_width is not None and image_height is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

    return [x1, y1, x2, y2]


# crop image
def cropImage(image, box):
    num = image[box[1] : box[3], box[0] : box[2]]
    return num


def get_images_from_dir(image_dir, image_file_extensions):
    image_dir = Path(image_dir)
    image_files = [f for f in image_dir.iterdir() if f.suffix in image_file_extensions]
    return image_files


class AgeGenderDetector:
    def __init__(
        self,
        face_detector_path="models/version-RFB-640.onnx",
        age_classifier_path="models/age_googlenet_dynamic.onnx",
        gender_classifier_path="models/gender_googlenet_dynamic.onnx",
    ):
        self.ageList = [
            "(0-2)",
            "(4-6)",
            "(8-12)",
            "(15-20)",
            "(25-32)",
            "(38-43)",
            "(48-53)",
            "(60-100)",
        ]
        session_options = ort.SessionOptions()
        self.runtime_providers = [
            "CUDAExecutionProvider" "CPUExecutionProvider",
        ]
        self.genderList = ["Male", "Female"]
        self.image_file_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        self.face_detector_path = face_detector_path
        self.age_classifier_path = age_classifier_path
        self.gender_classifier_path = gender_classifier_path

        self.face_detector = ort.InferenceSession(
            self.face_detector_path,
            sess_options=session_options,
            providers=self.runtime_providers,
        )
        self.age_classifier = ort.InferenceSession(
            self.age_classifier_path,
            sess_options=session_options,
            providers=self.runtime_providers,
        )
        self.gender_classifier = ort.InferenceSession(
            self.gender_classifier_path,
            sess_options=session_options,
            providers=self.runtime_providers,
        )

    def faceDetector(self, orig_image, threshold=0.7):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(
            orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold
        )
        return boxes, labels, probs

    def genderClassifier(self, images):
        processed = []
        for img in images:
            img = cv2.resize(img, (224, 224))
            img_mean = np.array([104, 117, 123])
            img = img - img_mean
            img = np.transpose(img, [2, 0, 1])  # [3,224,224]
            processed.append(img)
        images = np.stack(processed, axis=0).astype(np.float32)
        input_name = self.gender_classifier.get_inputs()[0].name
        preds = self.gender_classifier.run(None, {input_name: images})[0]

        pred_indices = preds.argmax(axis=1)
        return [self.genderList[i] for i in pred_indices]


    def ageClassifier(self, images):    
        processed = []
        for img in images:
            img = cv2.resize(img, (224, 224))
            img_mean = np.array([104, 117, 123])
            img = img - img_mean
            img = np.transpose(img, [2, 0, 1])
            processed.append(img)
        images = np.stack(processed, axis=0).astype(np.float32)

        input_name = self.age_classifier.get_inputs()[0].name
        preds = self.age_classifier.run(None, {input_name: images})[0]
        pred_indices = preds.argmax(axis=1)
        return [self.ageList[i] for i in pred_indices]

    '''
    def predict_age_and_gender(self, image_path):
        orig_image = cv2.imread(image_path)
        boxes, labels, probs = self.faceDetector(orig_image)
        preds = []
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :], orig_image.shape[1], orig_image.shape[0])
            cropped = cropImage(orig_image, box)
            gender = self.genderClassifier(cropped)
            age = self.ageClassifier(cropped)
            preds.append(
                {
                    "box": [int(e) for e in box],
                    "gender": gender,
                    "age": age,
                }
            )
        return preds
    '''
    def predict_age_and_gender(self, image_paths):

        images = [cv2.imread(p) for p in image_paths]
        preds = {}
        all_crops = []
        image_info = []
        for img_idx, orig_image in enumerate(images):
            boxes, labels, probs = self.faceDetector(orig_image)
            for i in range(boxes.shape[0]):
                box = scale(boxes[i, :], orig_image.shape[1], orig_image.shape[0])
                cropped = cropImage(orig_image, box)
                resized = cv2.resize(cropped, (224, 224))
                all_crops.append(resized)
                image_info.append((img_idx, box))  # keep track for mapping back
        
        batch = np.stack(all_crops)  # shape [N, C, H, W]
        gender_preds = self.genderClassifier(batch)
        age_preds = self.ageClassifier(batch)
        # map predictions back to each image
        for (img_idx, box), gender, age in zip(image_info, gender_preds, age_preds):
            image_path = str(image_paths[img_idx])
            preds.setdefault(image_path, []).append({
                "box": [int(e) for e in box],
                "gender": gender,
                "age": age,
            })

        return preds

    def predict_age_and_gender_on_dir(self, image_dir, batch_size=4):
        image_files = get_images_from_dir(image_dir, self.image_file_extensions)
        preds = {}

        # process in mini-batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_preds = self.predict_age_and_gender(batch_files)
            preds.update(batch_preds)

        return preds


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    args = parser.parse_args()

    detector = AgeGenderDetector()
    preds = detector.predict_age_and_gender_on_dir(args.image_dir)
    pprint(preds)  # Print the predictions for all images in the directory
