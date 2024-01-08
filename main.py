import cv2
import configparser
from database_module import DatabaseModule
from recognition_module import RecognitionModule

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize components
video = cv2.VideoCapture(0)
database_module = DatabaseModule("localhost", "robotic_shop_assistant",
                                 config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_USERNAME'),
                                 config.get('ROBOTIC_SHOP_ASSISTANT', 'DB_PASSWORD'))
recognition_module = RecognitionModule(tolerance=0.575)

# Prepare data
database_module.refresh_data()

recognition_module.load_known_faces("known_faces")  # Add force_rebuild=True if the faces repository has changed
recognition_module.set_product_data_source(database_module)

# Start operating
print("Detecting on camera...")
while True:
    ret, image = video.read()
    recognition_module.detect_on_image(image)
    cv2.imshow("Camera Video", image)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("r"):
        print("Refreshing data...")
        database_module.refresh_data()
    elif key_pressed == ord("q"):
        break

