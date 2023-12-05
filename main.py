import cv2
import configparser
from database_module import DatabaseModule
from recognition_module import RecognitionModule

config = configparser.ConfigParser()
config.read('config.ini')

video = cv2.VideoCapture(0)
database_module = DatabaseModule("localhost", "robotic_shop_assistant",
                                 config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_USERNAME'),
                                 config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_PASSWORD'))
recognition_module = RecognitionModule(tolerance=0.575)

database_module.refresh_data()

recognition_module.load_known_faces("known_faces")
recognition_module.set_product_data_source(database_module)
recognition_module.detect_on_camera(video)
