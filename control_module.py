class ControlModule:
    def __init__(self, command_mapping):
        self.command_mapping = command_mapping
        self.key_to_command = {
            ord("q"): "quit",
            ord("r"): "refresh",
            ord("a"): "add",
            ord("c"): "clear",
            ord("s"): "list",
            ord("b"): "buy",
            ord("v"): "voice",
        }

    def execute_command(self, command):
        if command == "quit":
            return True

        action = self.command_mapping.get(command)
        if action:
            action()

        return False

    def handle_key_press(self, key_pressed):
        command = self.key_to_command.get(key_pressed)
        if command:
            return self.execute_command(command)

        return False

    def handle_stt_input(self, command):
        return self.execute_command(command)
