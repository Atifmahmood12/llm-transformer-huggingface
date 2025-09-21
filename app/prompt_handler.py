from resources.profiler import Profiler
from models.gpt2_model import GPT2Model

class PromptHandler:
    def __init__(self):
        self.model = GPT2Model()
        self.profiler = Profiler()

    def handle_prompt(self, prompt, max_new_tokens=50, temperature=1.0, do_sample=True):
        self.profiler.start()
        output = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
        profile = self.profiler.stop()
        return {"output": output, "profile": profile}