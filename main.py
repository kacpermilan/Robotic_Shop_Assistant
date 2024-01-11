import cv2
import configparser
from communication_module import CommunicationModule
from database_module import DatabaseModule
from llm_module import LlmModule
from recognition_module import RecognitionModule
from visualization_module import VisualizationModule

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

camera_width = config.getfloat("ROBOTIC_SHOP_ASSISTANT", "CAMERA_WIDTH")
camera_height = config.getfloat("ROBOTIC_SHOP_ASSISTANT", "CAMERA_HEIGHT")
db_username = config.get("ROBOTIC_SHOP_ASSISTANT", "DB_USERNAME")
db_password = config.get("ROBOTIC_SHOP_ASSISTANT", "DB_PASSWORD")
use_local_llm = config.getboolean("ROBOTIC_SHOP_ASSISTANT", "USE_LOCAL_LLM")
llm_path = config.get("ROBOTIC_SHOP_ASSISTANT", "LOCAL_LLM_PATH")
layers_on_gpu = config.getint("ROBOTIC_SHOP_ASSISTANT", "N_GPU_LAYERS")
tts_model_name = config.get("ROBOTIC_SHOP_ASSISTANT", "TTS_MODEL_NAME")

# Initialize components
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

database = DatabaseModule("localhost", "robotic_shop_assistant", db_username, db_password)
llm = LlmModule(llm_path=llm_path, layers_on_gpu=layers_on_gpu)
recognition_module = RecognitionModule(tolerance=0.575)
gui = VisualizationModule()
interface = CommunicationModule(tts_model_name)
cart = []

# Prepare data
database.refresh_data()

recognition_module.load_known_faces("known_faces")  # Add force_rebuild=True if the faces repository has changed
recognition_module.set_product_data_source(database)

# Start operating
interface.say("Turning on...")
while True:
    ret, image = video.read()
    detected_people = recognition_module.detect_faces(image)
    detected_products = recognition_module.detect_products(image)
    gui.display_detected_objects(image, detected_people, detected_products)
    gui.display_gui(image, cart)
    cv2.imshow("Camera Video", image)

    # Control section
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        interface.say("Turning off...")
        break
    elif key_pressed == ord("r"):
        interface.say("Refreshing data...")
        database.refresh_data()
    elif key_pressed == ord("a"):
        interface.say("Adding product to the cart...")
        for decoded_barcode, product in detected_products:
            cart.append(product)
    elif key_pressed == ord("c"):
        interface.say("Clearing the cart...")
        cart.clear()
    elif key_pressed == ord("s"):
        interface.say("Toggling the shopping list...")
        gui.toogle_shopping_list_visibility()

video.release()
