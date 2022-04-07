import numpy as np

from layers import FaceResnet
import tensorflow as tf
from tensorflow.keras.applications import resnet
from faiss import IndexFlatL2
import faiss as BigF
from config import CV_Config
from os import path


class CV_Model(FaceResnet):
    def __init__(self, weights_name, compile=True):
        super().__init__(weights_name)

        self.settings = CV_Config()

        self.threshold = self.settings.threshold
        self.neighbours = self.settings.neighbours

        self.face_storage = IndexFlatL2(256)
        if path.isfile(self.settings.faiss_database):
            self.upload_storage()

        self.build_model()

    @staticmethod
    def load_image(filename):
        print(filename)
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.repeat(image, 3, axis=2)
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

        return image

    def preprocess_triplets(self, anchor, positive, negative):
        return (
            self.preprocess_image(anchor),
            self.preprocess_image(positive),
            self.preprocess_image(negative),
        )

    def check_correct_size(self, filename):
        image = self.load_image(filename)
        if image.numpy().shape[0] != self.target_shape[0] or image.numpy().shape[1] != self.target_shape[1]:
            return False

        else:
            return True

    def return_face_embed(self, face_arr) -> np.ndarray:
        face_embed = self.embedding(resnet.preprocess_input(
            np.array(face_arr).reshape([1, self.target_shape[0], self.target_shape[1], 3]))
        )

        # print(face_embed)
        return face_embed.numpy().astype(np.float32)

    def add_to_storage(self, filename):
        image = self.preprocess_image(filename)
        face_embed = self.return_face_embed(image)
        self.face_storage.add(face_embed)

        _, index_ = self.face_storage.search(face_embed, 1)
        return index_

    def recognize_vector(self, filename):
        image = self.preprocess_image(filename)
        face_embed = self.return_face_embed(image)

        dists, indexes = self.face_storage.search(face_embed, self.neighbours)
        cur_id, cur_dist = indexes[0].tolist(), dists[0].tolist()

        defining = {}

        for id_, distance in zip(cur_id, cur_dist):
            if id_ not in defining:
                defining[id_] = {"count": 0, "distance": 10}

            if defining[id_]["distance"] > distance:
                defining[id_]["distance"] = distance

            defining[id_]["count"] += 1

        winner = sorted([(k, v) for k, v in defining.items()], key=lambda x: x[1]["count"], reverse=True)[0]
        winner_id, winner_distance = winner[0], winner[1]["distance"]

        if winner_distance < self.settings.threshold:
            return int(winner_id), round(float(winner_distance), 5)

        else:
            return None, None

        # print(int(cur_id), round(float(cur_dist), 5))
        #
        # if cur_dist < self.threshold:
        #
        #     return int(cur_id), round(float(cur_dist), 5)
        #
        # else:
        #     return None, None

        # print(dists, indexes)
        # max_ = sorted([(i, indexes[0].tolist().count(i)) for i in set(indexes[0].tolist())], key=lambda x: x[1], reverse=True)[0][0]
        #
        # dist_id = dists[0][indexes[0].tolist().index(max_)]
        # print(dist_id)
        #
        # if dists[0][dist_id] < self.threshold:
        #     return max_, dist_id
        #
        # else:
        #     return None, None
