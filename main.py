import cv2
import configparser
import queue
from control_module import ControlModule
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


# Prepare command mapping
def refresh_database():
    interface.say("Refreshing data...")
    database.refresh_data()


def add_to_cart():
    interface.say("Adding product to the cart...")
    for decoded_barcode, product in detected_products:
        cart.append(product)


def clear_cart():
    interface.say("Clearing the cart...")
    cart.clear()


def toggle_shopping_list():
    interface.say("Toggling the shopping list...")
    gui.toogle_shopping_list_visibility()


command_mapping = {
    "quit": lambda: None,
    "refresh": lambda: refresh_database(),
    "add": lambda: add_to_cart(),
    "clear": lambda: clear_cart(),
    "list": lambda: toggle_shopping_list(),
    "buy": lambda: interface.say(f"{VisualizationModule.calculate_total_cost(cart):.2f} [PLN]"),
    "voice": lambda: interface.hear(stt_queue),
}

# Initialize components
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

controller = ControlModule(command_mapping)
database = DatabaseModule("localhost", "robotic_shop_assistant", db_username, db_password)
llm = LlmModule(llm_path=llm_path, mapping=command_mapping, layers_on_gpu=layers_on_gpu)
recognition_module = RecognitionModule(tolerance=0.575)
gui = VisualizationModule()
interface = CommunicationModule(tts_model_name)
stt_queue = queue.Queue()
cart = []

# Prepare data
database.refresh_data()

recognition_module.load_known_faces("known_faces")  # Add force_rebuild=True if the faces repository has changed
recognition_module.set_product_data_source(database)

# Start operating
interface.say("Turning on...")
terminate_loop = False
while not terminate_loop:
    ret, image = video.read()
    detected_people = recognition_module.detect_faces(image)
    detected_products = recognition_module.detect_products(image)
    gui.display_detected_objects(image, detected_people, detected_products)
    gui.display_gui(image, cart)
    cv2.imshow("Camera Video", image)

    # Handle keyboard interface
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed != 255:
        terminate_loop = controller.handle_key_press(key_pressed)

    # Handle voice interface
    if not stt_queue.empty():
        stt_result = stt_queue.get_nowait()
        print(stt_result)
        llm_processed_result = llm.obtain_command_from_stt(stt_result)
        print(llm_processed_result)
        terminate_loop = controller.handle_stt_input(llm_processed_result)

interface.say("Turning off...")
video.release()
cv2.destroyAllWindows()
