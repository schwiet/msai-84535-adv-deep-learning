from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": "You calculate unit conversion upon request. Be concise. Show a brief calculation, then give the final answer as <answer>NUMBER</answer> with a bare number and no units inside the tags",
            },
            {"role": "user", "content": "how many feet are in 3 meters?"},
            {
                "role": "assistant",
                "content": "1 m = 3.28 ft, so 3x3.28 = <answer>9.84</answer>",
            },
            {"role": "user", "content": "liters in 5 gallons"},
            {
                "role": "assistant",
                "content": "1 gal = 3.79 l, so 5x3.79 = <answer>18.95</answer>",
            },
            {"role": "user", "content": "can you translate 9 kg to pounds"},
            {
                "role": "assistant",
                "content": "1 pound = 0.454, so 9/0.454 = <answer>19.82</answer>",
            },
            {"role": "user", "content": "bytes in 2 megabytes"},
            {
                "role": "assistant",
                "content": "1 MB = 1000000 B, so 2x1000000 = <answer>2000000</answer>",
            },
            {"role": "user", "content": "years in 18 months"},
            {
                "role": "assistant",
                "content": "12 months = 1 year, so 18/12 = <answer>1.5</answer>",
            },
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
