class ControlModule:
    def __init__(self, command_mapping):
        self.command_mapping = command_mapping
        self.key_to_command = {
            ord("q"): "quit_application",
            ord("r"): "refresh_data",
            ord("a"): "add_product",
            ord("c"): "clear_cart",
            ord("s"): "toggle_shopping_list",
            ord("b"): "finalize_transaction",
            ord("v"): "voice_interface",
        }

    def execute_command(self, command):
        if command == "quit_application":
            return True

        action = self.command_mapping.get(command)
        if action:
            action()

        return False

    def handle_keyboard_input(self, key_pressed):
        command = self.key_to_command.get(key_pressed)
        if command:
            return self.execute_command(command)

        return False

    def handle_stt_input(self, command):
        return self.execute_command(command)
