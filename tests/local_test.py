from app.prompt_handler import PromptHandler


if __name__ == "__main__":
    prompt = "Once upon a time,"
    handler = PromptHandler()
    result = handler.handle_prompt(prompt, max_new_tokens=50, temperature=0.9, do_sample=True)
    print("Generated Text:\n", result["output"])
    print("\nResource Profile:\n", result["profile"])