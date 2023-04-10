# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from transformers import pipeline, set_seed
from transformers import (AutoConfig, OPTForCausalLM, AutoTokenizer)
import re

path = '/vc_data/users/zheweiyao/shared/ChatGPT_RLHF_Example/step1_supervised_finetuning/output_13b'

# '/vc_data/users/zheweiyao/shared/ChatGPT_RLHF_Example/step1_supervised_finetuning/output_13b'

# '/home/zheweiyao/ChatGPT_Pipeline/DeepSpeedExamples-internal/ChatGPT_RLHF_Example/step3_rlhf_finetuning/releasing_test/output_1.3b_Z2NoHE-debug-run1-401/actor'

tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token

model_config = AutoConfig.from_pretrained(path)
model = OPTForCausalLM.from_pretrained(path,
                                       from_tf=bool(".ckpt" in path),
                                       config=model_config).half()

model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))

generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     device='cuda:0')
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

set_seed(42)


def take_input():
    user_input = ""
    output = ""
    num_rounds = 0
    while True:
        num_rounds += 1
        # user_input = "Human: " + user_input + " Assistant: "
        tmp = input(
            "Enter input (type 'quit' to exit, 'clear' to clean memory): ")
        new_inputs = "Human: " + tmp + "\n Assistant: "
        user_input = user_input + " " + new_inputs
        if tmp == "quit":
            break
        if tmp == "clear":
            user_input = ""
            num_rounds = 0
            continue
        # print(f"User: {tmp}")
        gen = generator(user_input, max_new_tokens=128)
        output = str(gen[0]['generated_text'])
        # import pdb; pdb.set_trace()
        output = output.replace("<|endoftext|></s>", "")
        # output = output.replace("<|endoftext|></s>", ".")
        # all_positions = [m.start() for m in re.finditer('<|e', output)]
        # if len(all_positions) > 0:
        #     output = output[0:all_positions[0]]

        all_positions = [m.start() for m in re.finditer('Human: ', output)]
        place_of_second_q = -1
        if len(all_positions) > num_rounds:
            place_of_second_q = all_positions[num_rounds]
        # place_of_second_q = output.find("Human: ", 1+num_rounds)
        if place_of_second_q != -1:
            output = output[0:place_of_second_q]
        print(
            "-------------------------------- Round {} --------------------------------"
            .format(num_rounds))
        print(f"{output}")
        user_input = output + '\n\n'  #user_input + " " + output


take_input()

# what is an internet explorer? Assistant: Internet Explorer is a web browser developed by Microsoft. Human: who made it?
