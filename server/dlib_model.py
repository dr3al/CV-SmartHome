import dlib

import numpy as np
import tensorflow as tf

from faiss import IndexFlatL2
import faiss as BigF
from config import CV_Config
from os import path
from PIL import Image
from database import Photos, Users, engine
from sqlmodel import Session, select


class Dlib_Model:
    def __init__(self, shape_weights, recognition_weights):
        self.shape_weights = shape_weights
        self.recognition_weights = recognition_weights
        self.settings = CV_Config()

        self.detectnet = dlib.get_frontal_face_detector()
        self.shapenet = dlib.shape_predictor(self.settings.face_recognition_path + "/" + self.shape_weights)
        self.facenet = dlib.face_recognition_model_v1(self.settings.face_recognition_path + "/" + self.recognition_weights)

        self.threshold = self.settings.threshold
        self.neighbours = self.settings.neighbours

        self.face_storage = IndexFlatL2(128)
        if path.isfile(self.settings.faiss_database):
            self.upload_storage()

        self.target_shape = (150, 150)
        self.box = dlib.rectangle(0, 0, 150, 150)

    @staticmethod
    def load_image(filename):
        image = Image.open(filename)
        image.load()
        image = np.asarray(image)

        return image

    def dump_storage(self):
        BigF.write_index(self.face_storage, self.settings.faiss_database)
        return True

    def upload_storage(self):
        self.face_storage = BigF.read_index(self.settings.faiss_database)
        return True

    def preprocess_image(self, filename):
        image = self.load_image(filename)
        image = tf.image.resize(image, self.target_shape)

        return image.numpy()

    def check_correct_size(self, filename):
        try:
            image = self.load_image(filename)

        except:
            return False

        else:
            if image.shape[0] != self.target_shape[0] or image.shape[1] != self.target_shape[1]:
                return False

            else:
                return True

    def add_to_storage(self, filename):
        image = self.load_image(filename)
        face_detects = self.detectnet(image, 1)

        if not face_detects:
            return None

        face = face_detects[0]
        landmarks = self.shapenet(image, face)

        embedding = self.facenet.compute_face_descriptor(image, landmarks)
        embedding = np.asarray(embedding)
        embedding = np.array([embedding]).astype(np.float32)

        print(embedding, embedding.shape)

        self.face_storage.add(embedding)

        _, index_ = self.face_storage.search(embedding, 1)
        return index_

    def recognize_vector(self, filename):
        image = self.load_image(filename)
        face_detects = self.detectnet(image, 1)

        if not face_detects:
            return None, None

        face = face_detects[0]
        landmarks = self.shapenet(image, face)

        embedding = self.facenet.compute_face_descriptor(image, landmarks)
        embedding = np.asarray(embedding)
        embedding = np.array([embedding]).astype(np.float32)

        dists, indexes = self.face_storage.search(embedding, self.neighbours)
        cur_id, cur_dist = indexes[0].tolist(), dists[0].tolist()

        s_cur_id, s_cur_dist = [], []

        sess = Session(engine)
        find_users = [select(Photos).where(Photos.faiss_id == x) for x in cur_id]
        find_users = [sess.exec(x).one() for x in find_users]
        cur_id = [x.user_id for x in find_users]
        sess.close()

        print(cur_id)

        for id_, dist in zip(cur_id, cur_dist):
            if dist >= self.settings.threshold:
                continue

            else:
                s_cur_id.append(id_)
                s_cur_dist.append(dist)

        defining = {}

        for id_, distance in zip(s_cur_id, s_cur_dist):
            if id_ not in defining:
                defining[id_] = {"count": 0, "distance": 10}

            if defining[id_]["distance"] > distance:
                defining[id_]["distance"] = distance

            defining[id_]["count"] += 1

        winners = sorted([(k, v) for k, v in defining.items()], key=lambda x: x[1]["count"], reverse=True)

        if not winners:
            return None, None

        else:
            winner_id, winner_distance = winners[0][0], winners[0][1]["distance"]
            return int(winner_id), round(float(winner_distance), 5)

    @staticmethod
    def dlib_numpy_rect_converter(main_data, convert_to, additional_data=None):
        if not isinstance(main_data, convert_to):
            if convert_to == np.ndarray:
                if isinstance(main_data, dlib.full_object_detection):
                    return np.array([(part.x, part.y) for part in main_data.parts()], dtype=np.float32)
                elif isinstance(main_data, dlib.rectangle):
                    return np.array((main_data.left(), main_data.top(), main_data.right(), main_data.bottom()), dtype=np.float32)
            elif convert_to == dlib.rectangle:
                return dlib.rectangle(*main_data)
            elif convert_to == dlib.full_object_detection:
                return dlib.full_object_detection(rect=main_data, parts=dlib.points([dlib.point(point) for point in additional_data]))
        else:
            return main_data
