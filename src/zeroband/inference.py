from vllm import LLM, SamplingParams
from pydantic_config import BaseConfig, parse_argv

from zeroband.models import ModelName, name_to_hf_model

from datasets import load_dataset


class Config(BaseConfig):
    name_model: ModelName = "150M"
    dataset: str = "justus27/test-vcu"
    batch_size: int = 32
    max_samples: int | None = None


def fake_chat_template(messages):
    formatted_prompts = []

    for conversation in messages:
        prompt = ""
        for message in conversation:
            if message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        formatted_prompts.append(prompt.strip())

    return formatted_prompts


def main(config: Config):  # -> list[dict[str, Any]]:
    prompts = ["Write me a novel" for _ in range(5)]

    llm = LLM(model=name_to_hf_model[config.name_model])
    # tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100, presence_penalty=0.1, frequency_penalty=0.1)

    # Load dataset
    dataset = load_dataset(config.dataset, split="train")

    max_samples = config.max_samples or len(dataset)

    # Process batches
    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        # Get batch
        batch = dataset.select(range(i, min(i + config.batch_size, len(dataset))))

        # Prepare messages
        messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]

        # Get tokenized inputs
        prompts = fake_chat_template(messages)

        llm.generate(prompts, sampling_params)


if __name__ == "__main__":
    config = Config(**parse_argv())  # type: ignore

    main(config)
