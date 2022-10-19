import torch, deepspeed
import json

from transformers import AutoConfig, AutoModel
import os

model_config = {
    "attention_probs_dropout_prob": 0.1,
    "bos_token_id": 0,
    "classifier_dropout": None,
    "eos_token_id": 2,
    "gradient_checkpointing": False,
    "has_spacev_lang_tokens": True,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 384,
    "initializer_range": 0.02,
    "intermediate_size": 1536,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 610,
    "model_type": "xlm-roberta",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "output_past": True,
    "pad_token_id": 1,
    "position_embedding_type": "absolute",
    "transformers_version": "4.11.0",
    "type_vocab_size": 1,
    "use_cache": True,
    "vocab_size": 250049
}

os.makedirs('/tmp/roberta_model_test/', exist_ok = True)
with open('/tmp/roberta_model_test/config.json', 'w+') as f:
    json.dump(model_config, f)

auto_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path="/tmp/roberta_model_test",
    params={
        "num_hidden_layers": 12,
        "hidden_size": 384,
    },
)
auto_model = AutoModel.from_config(config=auto_config)

sequence_length = 512

input_ids = [[i for i in range(sequence_length)]]
attention_mask = [[1 for _ in range(sequence_length)]]


input_ids = torch.Tensor(
    input_ids
).to(dtype=torch.long).to(torch.device("cuda:0"))

attention_mask = torch.Tensor(
    attention_mask
).to(dtype=torch.long).to(torch.device("cuda:0"))


auto_model.to(torch.device("cuda:0"))
auto_model.eval()

vallina_model_output = auto_model(input_ids=input_ids, attention_mask=attention_mask)

vallina_model_output_pooled = vallina_model_output["pooler_output"]
print(
    f"vallina_model_output_pooled is {vallina_model_output_pooled}."
)


deepspeed_infer_engine = deepspeed.init_inference(
    auto_model,
    mp_size=1,
    dtype=torch.float,
    replace_method="auto",
    replace_with_kernel_inject=True,
)

deepspeed_output = auto_model(input_ids=input_ids, attention_mask=attention_mask)

deepspeed_output_pooled = deepspeed_output["pooler_output"]
print(f"deepspeed_output_pooled is: {deepspeed_output_pooled}.")