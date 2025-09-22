from transformers import pipeline

class GPT2Model:
    """
    Wrapper around Hugging Face's text-generation pipeline for GPT-2.
    Supports advanced decoding parameters for more control and less repetition.
    """

    def __init__(self, model_name="gpt2"):
        # Initialize the text generation pipeline with the specified model
        self.generator = pipeline("text-generation", model=model_name)

    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        do_sample=False,
        repetition_penalty=1.0,
        top_k=50,
        top_p=0.95
    ):
        """
        Generate text using the underlying pipeline with provided decoding parameters.
        """
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1
        )
        return result[0]['generated_text']