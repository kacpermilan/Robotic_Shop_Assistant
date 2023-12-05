import unittest
import configparser
from recognition_module import RecognitionModule
from database_module import DatabaseModule


class TestProductRecognition(unittest.TestCase):
    recognition_module = None
    database_module = None

    @classmethod
    def setUpClass(cls):
        config = configparser.ConfigParser()
        config.read('config.ini')
        cls.database_module = DatabaseModule("localhost", "robotic_shop_assistant",
                                             config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_USERNAME'),
                                             config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_PASSWORD'))

        cls.recognition_module = RecognitionModule(tolerance=0.575)
        cls.recognition_module.set_product_data_source(cls.database_module)

    def test_known_faces(self):
        self.recognition_module.load_known_faces("known_faces")
        self.recognition_module.test_on_face_images("unknown_faces")

    def test_known_barcodes(self):
        self.database_module.refresh_data()
        self.recognition_module.test_on_barcode_images("barcodes")


if __name__ == '__main__':
    unittest.main()

