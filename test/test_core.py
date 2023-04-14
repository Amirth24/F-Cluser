from core import read_image, get_mtcnn, extract_faces

import core.classifier as classifier
import numpy as np
import mtcnn
import unittest



class TestFaceDetector(unittest.TestCase):
    """
        Series of tests for the data preprocessing steps (img_loading, face_extraction).
    """

    def test_load_img(self):
        """
            Test that check images are loaded properly.
        """
        imgs = [read_image(f'test_imgs/img{i}.jpg') for i in range(2, 6)]
        for img in imgs:
            self.assertIsInstance(img, np.ndarray)

    def test_get_model(self):
        """
            Test that checks the get_mtcnn function.
        """
        model = get_mtcnn()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, mtcnn.MTCNN)

    def test_extract_faces(self):
        """
            Test that checks correct no of face samples from the give images
        """
        imgs = [read_image(f'test_imgs/img{i}.jpg') for i in range(2, 6)]

        model = get_mtcnn()
        face_count = 0
        for img in imgs:
            faces = extract_faces(img, model)
            face_count += len(faces)

        self.assertEqual(face_count, 8)






class TestClassifier(unittest.TestCase):
    """
        The Series of test for the classifier.
    """   
    def test_load_classifier(self):
        mdl = classifier.Classifier()

        data = np.load('test_imgs/face_embeds.npy')

        mdl.fit(data)


        self.assertEqual(mdl.n_clusters, 4)

if __name__ == '__main__':
    unittest.main()
