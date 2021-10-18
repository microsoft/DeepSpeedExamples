from typing import Iterable, Dict, Any, Callable, Tuple, List, Union, Optional
import fire
import pathlib
import uuid
import datetime
import pytz
import json
import numpy as np
from functools import partial
import loguru
import sh

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.roberta import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaPreTrainedModel

logger = loguru.logger

######################################################################
############### Dataset Creation Related Functions ###################
######################################################################


TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

def collate_function(batch: List[Tuple[List[int], List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_length = max(
        len(token_ids)
        for token_ids, _ in batch 
    )
    padded_token_ids = [
        token_ids + [pad_token_id for _ in range(0, max_length - len(token_ids))]
        for token_ids, _ in batch
    ]
    padded_labels = [
        labels + [pad_token_id for _ in range(0, max_length - len(labels))]
        for _, labels in batch
    ]
    src_tokens = torch.LongTensor(padded_token_ids)
    tgt_tokens = torch.LongTensor(padded_labels)
    attention_mask = src_tokens.ne(pad_token_id).type_as(src_tokens)
    return {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "attention_mask": attention_mask
    }

def masking_function(text: str,
                     tokenizer: TokenizerType,
                     mask_prob: float,
                     random_replace_prob: float,
                     unmask_replace_prob: float,
                     max_length: int) -> Tuple[List[int], List[int]]:
    # Note: By default, encode does add the BOS and EOS token
    # Disabling that behaviour to make this more clear
    tokenized_ids = [tokenizer.bos_token_id] + \
        tokenizer.encode(text,
                         add_special_tokens=False,
                         truncation=True,
                         max_length=max_length - 2) + \
            [tokenizer.eos_token_id]
    seq_len = len(tokenized_ids)
    tokenized_ids = np.array(tokenized_ids)
    subword_mask = np.full(len(tokenized_ids), False)

    # Masking the BOS and EOS token leads to slightly worse performance
    low = 1
    high = len(subword_mask) - 1
    mask_choices = np.arange(low, high)
    num_subwords_to_mask = max(int((mask_prob * (high - low)) + np.random.rand()), 1)
    subword_mask[np.random.choice(mask_choices, num_subwords_to_mask, replace=False)] = True

    # Create the labels first
    labels = np.full(seq_len, tokenizer.pad_token_id)
    labels[subword_mask] = tokenized_ids[subword_mask]

    tokenized_ids[subword_mask] = tokenizer.mask_token_id

    # Now of the masked tokens, choose how many to replace with random and how many to unmask
    rand_or_unmask_prob = random_replace_prob + unmask_replace_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = subword_mask & (np.random.rand(len(tokenized_ids)) < rand_or_unmask_prob)
        if random_replace_prob == 0:
            unmask = rand_or_unmask
            rand_mask = None
        elif unmask_replace_prob == 0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = unmask_replace_prob / rand_or_unmask_prob
            decision = np.random.rand(len(tokenized_ids)) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
        if unmask is not None:
            tokenized_ids[unmask] = labels[unmask]
        if rand_mask is not None:
            weights = np.ones(tokenizer.vocab_size)
            weights[tokenizer.all_special_ids] = 0
            probs = weights / weights.sum()
            num_rand = rand_mask.sum()
            tokenized_ids[rand_mask] = np.random.choice(
                tokenizer.vocab_size,
                num_rand,
                p=probs
            )
    return tokenized_ids.tolist(), labels.tolist()

class WikiTextMLMDataset(Dataset):
    def __init__(self,
                 dataset: datasets.arrow_dataset.Dataset,
                 masking_function: Callable[[str], Tuple[List[int], List[int]]]) -> None:
        self.dataset = dataset
        self.masking_function = masking_function
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        tokens, labels = self.masking_function(self.dataset[idx]["text"])
        return (tokens, labels)

def create_data_iterator(mask_prob: float,
                         random_replace_prob: float,
                         unmask_replace_prob: float,
                         batch_size: int,
                         max_seq_length: int = 512,
                         tokenizer: str = "roberta-base") -> DataLoader:
    wikitext_dataset = datasets.load_dataset(
        "wikitext",
        "wikitext-2-v1",
        split="train"
    )
    wikitext_dataset = wikitext_dataset.filter(
        lambda record: record["text"] != ""
    ).map(
        lambda record: {"text": record["text"].rstrip("\n")}
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    masking_function_partial = partial(
        masking_function,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        max_length=max_seq_length
    )
    dataset = WikiTextMLMDataset(wikitext_dataset, masking_function_partial)
    collate_fn_partial = partial(
        collate_function,
        pad_token_id=tokenizer.pad_token_id
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_partial
    )
    return dataloader


######################################################################
############### Model Creation Related Functions #####################
######################################################################

class RobertaLMHeadWithMaskedPredict(RobertaLMHead):
    def __init__(self,
                 config,
                 embedding_weight: Optional[torch.Tensor] = None) -> None:
        super(RobertaLMHeadWithMaskedPredict, self).__init__(config)
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(  # pylint: disable=arguments-differ
            self,
            features: torch.Tensor,
            masked_token_indices: Optional[
                torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
        """The current ``Transformers'' library does not provide support
        for masked_token_indices. This function provides the support

        Args:
            masked_token_indices (torch.Tensor, optional):
            The indices of masked tokens for index select. Defaults to None.

        Returns:
            torch.Tensor: The output logits

        """
        if masked_token_indices is not None:
            features = torch.index_select(
                features.view(-1, features.shape[-1]), 0, masked_token_indices
            )
        return super().forward(features)

class RobertaMLMModel(RobertaPreTrainedModel):
    def __init__(self,
                 config: RobertaConfig,
                 encoder: RobertaModel) -> None:
        super().__init__(config)
        self.encoder = encoder
        self.lm_head = RobertaLMHeadWithMaskedPredict(
            config, self.encoder.embeddings.word_embeddings.weight
        )
        self.lm_head.apply(self._init_weights)
    
    def forward(self,
                src_tokens,
                attention_mask,
                tgt_tokens) -> torch.Tensor:
        sequence_output, *_ = self.encoder(input_ids=src_tokens,
                                           attention_mask=attention_mask, return_dict=False)
        
        pad_token_id = self.config.pad_token_id
        # (labels have also been padded with pad_token_id)
        # filter out all masked labels
        masked_token_indexes = torch.nonzero(
            (tgt_tokens != pad_token_id).view(-1)).view(-1)

        prediction_scores = self.lm_head(sequence_output,
                                         masked_token_indexes)

        target = torch.index_select(tgt_tokens.view(-1), 0,
                                    masked_token_indexes)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target)
        return masked_lm_loss

def create_model(num_layers: int,
                 num_heads: int,
                 ff_dim: int,
                 h_dim: int,
                 dropout: float) -> RobertaModel:
    roberta_config_dict = {
        "attention_probs_dropout_prob": dropout,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": dropout,
        "hidden_size": h_dim,
        "initializer_range": 0.02,
        "intermediate_size": ff_dim,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "model_type": "roberta",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265
    }
    roberta_config = RobertaConfig.from_dict(roberta_config_dict)
    roberta_encoder = RobertaModel(roberta_config)
    roberta_model = RobertaMLMModel(roberta_config, roberta_encoder)
    return roberta_model


######################################################################
########### Experiment Management Related Functions ##################
######################################################################

def create_experiment_dir(checkpoint_dir: pathlib.Path,
                          all_arguments: Dict[str, Any]):
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = "bert_pretrain.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        str(uuid.uuid4())
    )
    exp_dir = (checkpoint_dir / expname)
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
    with (exp_dir / "githash.log").open("w") as handle:
        handle.write(gitlog.stdout.decode("utf-8"))
    # And the git diff
    gitdiff = sh.git.diff(_fg=False, _tty_out=False)
    with (exp_dir / "gitdiff.log").open("w") as handle:
        handle.write(gitdiff.stdout.decode("utf-8"))
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir()
    return exp_dir

######################################################################
####################### Driver Functions #############################
######################################################################

def train(
    checkpoint_dir: str,
    # Dataset Parameters
    mask_prob: float = 0.15,
    random_replace_prob: float = 0.1,
    unmask_replace_prob: float = 0.1,
    max_seq_length: int = 512,
    tokenizer: str = "roberta-base",
    # Model Parameters
    num_layers: int = 6,
    num_heads: int = 8,
    ff_dim: int = 512,
    h_dim: int = 256,
    dropout: float = 0.1,
    # Training Parameters
    batch_size: int = 8,
    num_iterations: int = 10000,
    checkpoint_every: int = 1000,
    log_every: int = 10,
    device: int = -1
) -> None: 
    device = torch.device("cuda", device) \
        if (device > 0) and torch.cuda.is_available() \
            else torch.device("cpu")
    ################################
    ###### Create Datasets #########
    ################################
    logger.info("Creating Datasets")
    data_iterator = create_data_iterator(
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size
    )
    logger.info("Dataset Creation Done")
    ################################
    ###### Create Model ############
    ################################
    logger.info("Creating Model")
    model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout
    )
    model = model.to(device)
    logger.info("Model Creation Done")
    ################################
    ###### Create Exp. Dir #########
    ################################
    logger.info("Creating Experiment Directory")
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    all_arguments = {
        # Dataset Params
        "mask_prob": mask_prob,
        "random_replace_prob": random_replace_prob,
        "unmask_replace_prob": unmask_replace_prob,
        "max_seq_length": max_seq_length,
        "tokenizer": tokenizer,
        # Model Params
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "h_dim": h_dim,
        "dropout": dropout,
        # Training Params
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "checkpoint_every": checkpoint_every,
    }
    exp_dir = create_experiment_dir(checkpoint_dir, all_arguments)
    tb_dir = exp_dir / "tb_dir"
    assert tb_dir.exists()
    summary_writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"Experiment Directory created at {exp_dir}")
    ################################
    ###### Create Optimizer #######
    ################################
    logger.info("Creating Optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    logger.info("Optimizer Creation Done")
    ################################
    ####### The Training Loop ######
    ################################
    losses = []
    for step, batch in enumerate(data_iterator, start=1):
        optimizer.zero_grad()
        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to(device)
        # Forward pass
        loss = model(**batch)
        # Backward pass
        loss.backward()
        # Optimizer Step
        optimizer.step()
        losses.append(loss.item())
        if step % log_every == 0:
            logger.info("Loss: {0:.4f}".format(np.mean(losses)))
            summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
        if step % checkpoint_every == 0:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(obj=state_dict, f=str(exp_dir / f"checkpoint.iter_{step}.pt"))
        if step == num_iterations:
            break



if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "data": create_data_iterator,
        "model": create_model
    })