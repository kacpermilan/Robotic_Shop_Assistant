from modules.control_module import ControlModule
from modules.database_module import DatabaseModule
from modules.gui_module import GUIModule
from modules.llm_module import LlmModule
from modules.shopping_module import ShoppingModule
from modules.settings_module import SettingsModule
from modules.recognition_module import RecognitionModule
from modules.voice_interface_module import VoiceInterfaceModule

# Prepare command mapping
command_mapping = {
    "quit_application": lambda: None,
    "refresh_data": lambda: voice_interface.say_and_execute("Refreshing data...", database.refresh_data),
    "add_product": lambda: voice_interface.say_and_execute("Adding products to the cart...", shopping_cart.add_products_to_cart, detected_products),
    "clear_cart": lambda: voice_interface.say_and_execute("Clearing the cart...", shopping_cart.clear_cart),
    "toggle_shopping_list": lambda: voice_interface.say_and_execute("Toggling the shopping list...", gui.toggle_shopping_list_visibility),
    "finalize_transaction": lambda: voice_interface.say_and_execute(f"{shopping_cart.products_total_cost:.2f} [PLN]", shopping_cart.finalize_transaction),
    "voice_interface": lambda: voice_interface.hear(),
}

# Initialize components
SETTINGS = SettingsModule("config.ini")
controller = ControlModule(command_mapping)
database = DatabaseModule("localhost", "robotic_shop_assistant", SETTINGS.db_username, SETTINGS.db_password)
gui = GUIModule(SETTINGS.camera_width, SETTINGS.camera_height)
llm = LlmModule(llm_path=SETTINGS.llm_path, available_functions=command_mapping, layers_on_gpu=SETTINGS.layers_on_gpu)
recognition_module = RecognitionModule(tolerance=0.575)
shopping_cart = ShoppingModule()
voice_interface = VoiceInterfaceModule(SETTINGS.tts_model_name, SETTINGS.stt_model_name)

# Prepare data
database.refresh_data()
recognition_module.load_known_faces("known_faces")  # Add force_rebuild=True if the faces repository has changed
recognition_module.set_product_data_source(database)

# Start operating
voice_interface.say("Turning on...")
terminate_loop = False
while not terminate_loop:
    image, key_pressed = gui.get_image_frame()
    detected_people = recognition_module.detect_faces(image)
    detected_products = recognition_module.detect_products(image)
    gui.display_detected_objects(image, detected_people, detected_products)
    gui.render_gui(image, shopping_cart.cart, shopping_cart.products_total_cost)

    # Handle keyboard interface
    if key_pressed != 255:
        terminate_loop = controller.handle_keyboard_input(key_pressed)

    # Handle voice interface
    if not voice_interface.stt_queue.empty():
        stt_result = voice_interface.stt_queue.get_nowait()
        print(stt_result)
        llm_processed_result = llm.obtain_command_from_stt(stt_result)
        print(llm_processed_result)
        terminate_loop = controller.handle_stt_input(llm_processed_result)

voice_interface.say("Turning off...")
