import unittest
import configparser
from database_module import DatabaseModule
from llm_module import LlmModule
from recognition_module import RecognitionModule


class TestProductRecognition(unittest.TestCase):
    recognition_module = None
    database_module = None
    local_llm = None

    @classmethod
    def setUpClass(cls):
        config = configparser.ConfigParser()
        config.read('config.ini')

        llm_path = config.get("ROBOTIC_SHOP_ASSISTANT", "LOCAL_LLM_PATH")
        layers_on_gpu = int(config.get("ROBOTIC_SHOP_ASSISTANT", "N_GPU_LAYERS"))
        cls.local_llm = LlmModule(llm_path=llm_path, layers_on_gpu=layers_on_gpu)

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

    def test_local_llm(self):
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
