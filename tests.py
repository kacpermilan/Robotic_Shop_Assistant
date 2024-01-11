import unittest
import configparser
import os
import torch
import cv2
import face_recognition
from database_module import DatabaseModule
from llm_module import LlmModule
from recognition_module import RecognitionModule
del torch

class TestProductRecognition(unittest.TestCase):
    recognition_module = None
    database_module = None
    local_llm = None

    @classmethod
    def setUpClass(cls):
        config = configparser.ConfigParser()
        config.read('config.ini')

        cls.llm_path = config.get("ROBOTIC_SHOP_ASSISTANT", "LOCAL_LLM_PATH")
        cls.layers_on_gpu = int(config.get("ROBOTIC_SHOP_ASSISTANT", "N_GPU_LAYERS"))

        cls.database_module = DatabaseModule("localhost", "robotic_shop_assistant",
                                             config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_USERNAME'),
                                             config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_PASSWORD'))

        cls.recognition_module = RecognitionModule(tolerance=0.575)
        cls.recognition_module.set_product_data_source(cls.database_module)

    def test_known_faces(self):
        """
        Test the currently stored known faces encodings using the images in a given test directory.
        """
        test_dir = "unknown_faces"
        self.recognition_module.load_known_faces("known_faces")

        print(f"Evaluating against {test_dir}...")
        for filename in os.listdir(test_dir):
            print(filename)
            image = face_recognition.load_image_file(os.path.join(test_dir, filename))
            faces = self.recognition_module.detect_faces(image)
            self.recognition_module.display_detected_objects(image, detected_faces=faces)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)

    def test_known_barcodes(self):
        """
        Test the barcode detection using the images in a given test directory.
        """
        test_dir = "barcode_images"
        self.database_module.refresh_data()

        print(f"Evaluating against {test_dir}...")
        for filename in os.listdir(test_dir):
            print(filename)
            image = cv2.imread(os.path.join(test_dir, filename))
            image = cv2.resize(image, (800, 600))

            products = self.recognition_module.detect_products(image)
            self.recognition_module.display_detected_objects(image, detected_barcodes=products)

            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)

    def test_local_llm(self):
        self.local_llm = LlmModule(llm_path=self.llm_path, layers_on_gpu=self.layers_on_gpu)
        output = self.local_llm.test_simple_completion()
        print(output['id'])
        print(output['object'])
        print(output['created'])
        print(output['choices'])
        print(output['usage'])
        usage = output['usage']
        self.assertGreater(int(usage['total_tokens']), 20)
        self.assertLessEqual(int(usage['total_tokens']), 60)


if __name__ == '__main__':
    unittest.main()
