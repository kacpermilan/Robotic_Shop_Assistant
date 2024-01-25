from llama_cpp import Llama


class LlmModule:
    """
    A class to handle interactions with a local large language model (LLM).

    Attributes:
    -----------
    llm : Llama
        An instance of the Llama model for language processing.
    mapping_keys : str
        Comma-separated string of keys for available functions.
    max_tokens : int
        The maximum number of tokens to generate in each completion.
    stop : list
        A list of stop sequences for the language model.
    temperature : float
        Controls the randomness of the output (lower is more deterministic).
    frequency_penalty : float
        Decreases the model's likelihood to repeat the same line.
    presence_penalty : float
        Increases the model's likelihood to talk about new concepts.

    Methods:
    --------
    obtain_command_from_stt(self, stt_output)
        Processes speech-to-text output to obtain a relevant command from the LLM.

    test_simple_completion(self)
        Tests the LLM with a simple completion task.
    """
    def __init__(self, llm_path, available_functions=None, layers_on_gpu=0):
        self.llm = Llama(model_path=llm_path,
                         n_gpu_layers=layers_on_gpu,
                         use_mlock=True)

        if available_functions is not None:
            self.mapping_keys = ', '.join(available_functions.keys())

        self.max_tokens = 7
        self.stop = [".", ",", ";", "\n"]
        self.temperature = 0.15
        self.frequency_penalty = 0.5
        self.presence_penalty = 0.5

    def obtain_command_from_stt(self, stt_output):
        """
        Processes speech-to-text output to identify a relevant command.
        """
        llm_response = self.llm.create_completion(
            prompt="I am a command mapping machine. "
                   "I need to identify the most relevant command key for a given input. "
                   "I can only respond with a single command key from the following list or 'NO_MATCH' "
                   f"if no relevant command is found: {self.mapping_keys}.\n"
                   "Input: Turn on the shopping list\n"
                   "Output: toggle_list\n"
                   "Input: System turn off\n"
                   "Output: quit_appication\n"
                   "Input: What do you think?\n"
                   "Output: NO_MATCH\n"
                   f"Input: {stt_output}\n"
                   "Output: ",
            stop=self.stop,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty)

        response = llm_response['choices'][0]['text'].strip()
        return response

    def test_simple_completion(self):
        """
        Tests the LLM with a simple completion task.
        """
        return self.llm("Q: Name the planets in the solar system: A: 1. Mercury ",
                        max_tokens=40, stop=["Q:", "\n"], echo=True)
