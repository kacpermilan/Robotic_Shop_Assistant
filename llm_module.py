from llama_cpp import Llama


class LlmModule:
    llm = None

    def __init__(self, llm_path, layers_on_gpu=0):
        self.llm = Llama(model_path=llm_path,
                         n_gpu_layers=layers_on_gpu,
                         use_mlock=True)

    def test_simple_completion(self):
        return self.llm("Q: Name the planets in the solar system: A: 1. Mercury ",
                        max_tokens=40, stop=["Q:", "\n"], echo=True)
