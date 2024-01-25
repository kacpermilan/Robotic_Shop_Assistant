class ControlModule:
    """
    A class to manage the execution of commands based on different inputs.

    Attributes:
    -----------
    command_mapping : dict
        A dictionary mapping command strings to their corresponding function objects.
    key_to_command : dict
        A mapping of keyboard key ordinals to command strings.

    Methods:
    --------
    execute_command(self, command)
        Executes the command associated with the given command string.

    handle_keyboard_input(self, key_pressed)
        Processes keyboard input and executes the associated command.

    handle_stt_input(self, command)
        Executes the command received from speech-to-text input.
    """
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
        """
        Executes the command associated with the provided command string.
        """
        if command == "quit_application":
            return True

        action = self.command_mapping.get(command)
        if action:
            action()

        return False

    def handle_keyboard_input(self, key_pressed):
        """
        Handles keyboard input and executes the corresponding command.
        """
        command = self.key_to_command.get(key_pressed)
        if command:
            return self.execute_command(command)

        return False

    def handle_stt_input(self, command):
        """
        Executes the command received from speech-to-text input.
        """
        return self.execute_command(command)
