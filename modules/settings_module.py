import configparser


class SettingsModule:
    """
    A class to handle settings for a robotic shop assistant.

    Attributes:
    -----------
    camera_width : float
        The width of the camera used by the robotic assistant.
    camera_height : float
        The height of the camera used by the robotic assistant.
    db_username : str
        The username for the database connection.
    db_password : str
        The password for the database connection.
    use_local_llm : bool
        A flag to determine if a local large language model should be used.
    llm_path : str
        The file path for the local large language model.
    layers_on_gpu : int
        The number of layers of the model to be loaded on GPU.
    tts_model_name : str
        The name of the Text-to-Speech model.
    stt_model_name : str
        The name of the Speech-to-Text model.
    """

    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.camera_width = config.getfloat("ROBOTIC_SHOP_ASSISTANT", "CAMERA_WIDTH")
        self.camera_height = config.getfloat("ROBOTIC_SHOP_ASSISTANT", "CAMERA_HEIGHT")
        self.db_username = config.get("ROBOTIC_SHOP_ASSISTANT", "DB_USERNAME")
        self.db_password = config.get("ROBOTIC_SHOP_ASSISTANT", "DB_PASSWORD")
        self.use_local_llm = config.getboolean("ROBOTIC_SHOP_ASSISTANT", "USE_LOCAL_LLM")
        self.llm_path = config.get("ROBOTIC_SHOP_ASSISTANT", "LOCAL_LLM_PATH")
        self.layers_on_gpu = config.getint("ROBOTIC_SHOP_ASSISTANT", "N_GPU_LAYERS")
        self.tts_model_name = config.get("ROBOTIC_SHOP_ASSISTANT", "TTS_MODEL_NAME")
        self.stt_model_name = config.get("ROBOTIC_SHOP_ASSISTANT", "STT_MODEL_NAME")
