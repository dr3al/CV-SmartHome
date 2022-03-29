from layers import FaceResnet


class CV_Model(FaceResnet):
    def __init__(self, weights_name):
        super().__init__(weights_name)
