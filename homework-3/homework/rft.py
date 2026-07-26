from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example(prompt: str, _: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair.
    """
    return {"question": prompt, "answer": answer}


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    from peft import get_peft_model
    from peft.tuners.lora import LoraConfig
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments
    from .data import Dataset
    from .sft import TokenizedDataset

    llm = BaseLLM()
    model = get_peft_model(
        llm.model,
        LoraConfig(
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
        ),
    )
    # prove to ourselves that we're training the adapter
    model.print_trainable_parameters()

    if llm.device != "cpu":
        llm.model.enable_input_require_grads()

    tokenized_data = TokenizedDataset(llm.tokenizer, Dataset("rft"), format_example)
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        learning_rate=3e-4,
    )
    trainer = Trainer(model, args, train_dataset=tokenized_data)
    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
