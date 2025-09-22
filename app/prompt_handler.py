from resources.profiler import Profiler
from models.gpt2_model import GPT2Model

class PromptHandler:
    """
    Handles prompt generation requests and profiles resource usage for each request.
    - Uses an underlying GPT2Model for text generation.
    - Uses a Profiler to measure system resource consumption.
    """

    def __init__(self):
        # Initialize the text generation model and resource profiler
        self.model = GPT2Model()
        self.profiler = Profiler()

    def handle_prompt(
        self,
        prompt,
        max_new_tokens=250,
        temperature=1.0,
        do_sample=True,
        repetition_penalty=1.0,
        top_k=50,
        top_p=0.95
    ):
        """
        Handles a single prompt request.
        - Profiles system resources before and after generation.
        - Passes all relevant generation parameters to the model.
        - Returns both the generated text and the profiling information.
        """
        # Start profiling system resources before model inference
        self.profiler.start()

        # Generate text using the model with all provided decoding parameters
        output = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p
        )

        # Stop profiling and collect resource usage statistics
        profile = self.profiler.stop()

        # Return both the generated output and the profiling data
        return {"output": output, "profile": profile}