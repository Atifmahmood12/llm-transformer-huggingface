from transformers import pipeline

class GPT2Model:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline("text-generation", model=model_name)

    def generate(self, prompt, max_new_tokens=50, temperature=1.0, do_sample=True):
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=1
        )
        return result[0]['generated_text']