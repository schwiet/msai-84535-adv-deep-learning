import json
import torch

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(
    output_json: str,
    oversample: int = 10,
    temperature: float = 0.6,
    limit: int | None = None,
    checkpoint: str | None = None,
):
    model = CoTModel(checkpoint) if checkpoint else CoTModel()
    model.model = model.model.to(torch.bfloat16)
    trainset = Dataset("train")

    # cap rows by limit, if provided
    rows = trainset.data[:limit]
    prompts = [model.format_prompt(x) for x, y in rows]

    # for each training row, generate <oversample> completions
    candidates = model.batched_generate(prompts, oversample, temperature)

    # for each training row, see if there was a correct completion,
    # if so keep first
    res = []
    for (question, answer), group in zip(rows, candidates):
        for completion in group:
            gen_answer = model.parse_answer(completion)
            if is_answer_valid(gen_answer, answer):
                res.append([question, answer, completion])
                break

    print(f"kept {len(res)} / {len(rows)}")
    # serialize list of triples to file
    with open(output_json, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
