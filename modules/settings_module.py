import configparser


class SettingsModule:
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
