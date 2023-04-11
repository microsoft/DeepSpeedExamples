import json
from enum import Enum
from typing import TypedDict, List, Optional, Tuple, Union

import refile
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from sentencepiece import SentencePieceProcessor


class AlpacaCollectionEnum(str, Enum):
    Alpaca = "alpaca"
    GPT4 = "gpt4"


class AlpacaRecord(TypedDict):
    instruction: str
    input: Optional[str]
    output: str


class ModelInputInstance(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]


class LLamaTokenizer:

    """Tokenizer for LLaMA."""

    def __init__(self, model_path: str) -> None:
        self.processor = SentencePieceProcessor(model_file=str(model_path))
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        pad_id: int = 0,
        device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        tokens = self.processor.encode_as_ids(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0 and len(tokens) >= max_length:
            tokens = tokens[:max_length]

        real_length = len(tokens)
        if pad and len(tokens) < max_length:
            pad_id = pad_id or self.pad_id
            tokens += [pad_id] * (max_length - len(tokens))

        if max_length > 0:
            assert len(tokens) == max_length, (len(tokens), max_length)

        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        if pad:
            return tokens_tensor, real_length
        else:
            return tokens_tensor

    def decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.processor.decode_ids(tokens)


def make_prompt(instruction: str, context: Optional[str] = None, response: Optional[str] = None) -> str:
    if context is not None:
        sys_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    else:
        sys_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    buf = [sys_prompt]
    buf.append(f"### Instruction:\n{instruction}")

    if context:
        buf.append(f"### Input:\n{context}")

    if response:
        buf.append(f"### Response:\n{response}\n")
    else:
        buf.append("### Response:\n")

    return "\n\n".join(buf)


class SimplePrompter:

    def __call__(self, record: AlpacaRecord) -> str:
        return make_prompt(record['instruction'], record.get('input'), record['output'])


class ToTensor:

    def __init__(self, tokenizer: LLamaTokenizer, block_size: int):
        self._tokenizer = tokenizer
        self._block_size = block_size

    def __call__(self, text):
        input_ids, no_pad_length = self._tokenizer.encode(
            text, bos=True, eos=True,
            max_length=self._block_size, pad=True,
            pad_id=self._tokenizer.eos_id, device=None,
        )
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:no_pad_length].fill_(1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class AlpacaJsonDataset(Dataset):

    def __init__(self, data_dir: str, input_jsons: str, transforms=[]):
        self._input_jsons = []
        self._records = []
        for input_json in input_jsons:
            self._input_jsons.append(refile.smart_path_join(data_dir, input_json))

        pbar = tqdm(self._input_jsons)
        for input_json in pbar:
            pbar.set_description(f"Loading {input_json.split('/')[-1]}")
            with refile.smart_open(input_json) as f:
                self._records.extend(json.load(f))

        self.transforms = transforms

    def __getitem__(self, index):
        record: AlpacaRecord = self._records[index]

        x = record
        for transform in self.transforms:
            x = transform(x)
        return x

    def __len__(self):
        return len(self._records)


def make_dataset(data_dir: str, collections: Tuple[AlpacaCollectionEnum], tokenizer: LLamaTokenizer, block_size: int):
    name2json = {
        AlpacaCollectionEnum.Alpaca: "alpaca_data_cleaned_archive.json",
        AlpacaCollectionEnum.GPT4: "alpaca_data_gpt4.json",
    }

    input_jsons = [name2json[c] for c in collections]

    return AlpacaJsonDataset(
        data_dir=data_dir,
        input_jsons=input_jsons,
        transforms=[
            SimplePrompter(),
            ToTensor(tokenizer, block_size=block_size)
        ]
    )


def _main():
    tokenizer = LLamaTokenizer("/data/huggingface/decapoda-research/llama-7b-hf/tokenizer.model")
    print(f"EOS: {tokenizer.eos_id}, BOS: {tokenizer.bos_id}, PAD: {tokenizer.pad_id}")

    tokens = tokenizer.encode("Power Human with AI.")
    print(len(tokens), tokenizer.decode(tokens))

    dataset = AlpacaJsonDataset(
        data_dir="/data/alpaca",
        input_jsons=["alpaca_data_cleaned_archive.json", "alpaca_data_gpt4.json"],
        transforms=[
            SimplePrompter(),
            ToTensor(tokenizer, block_size=256)
        ]
    )

    print(dataset[0])

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for idx, batch in enumerate(loader):
        if idx > 1:
            break
        print(batch)


if __name__ == "__main__":
    _main()
