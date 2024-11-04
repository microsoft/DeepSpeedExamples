# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self,
                 base_model,
                 tokenizer,
                 num_padding_at_beginning=0,
                 loss_to_fp32=False,
                 opt_loss_calc=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.optimized_loss_calc = opt_loss_calc
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.loss_to_fp32 = loss_to_fp32
        self.fallback_mask = None

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]
            if self.fallback_mask is None:
                fallback_mask = torch.zeros([seq_len]).bool()
                fallback_mask[seq_len - 1] = 1
                self.fallback_mask = fallback_mask.to(rejected_reward.device)
            if self.optimized_loss_calc:
                # the below code is an performance optimize implementation to the else statement flow
                # by using masks, we are achieving non dynamic shapes and infereable flow
                def get_last_before_padding(paddings, num_begin_padding):
                    # this function returns a mask, where there is a single 1, in the last non padded element
                    # for example, flat = 11000011 (see below) with num begin paddings 2
                    # 1. remove begin padding indication: 00001100
                    shifted = torch.roll(paddings, -num_begin_padding, 0)
                    # 2.  put the first 1's to start on the last non padded: 00000110
                    shifted = torch.roll(shifted, num_begin_padding - 1, 0)
                    # not_paddings will indicate where we don't have padding: 00111100
                    not_paddings = torch.logical_not(paddings)
                    # now we will get the last non padded value: 00000100
                    mask = not_paddings * shifted
                    return mask

                # target is to create a mask that has 1's from the first token that
                # chosen and rejected ids are different till the last non padded element (the longest among the two)
                # incase of identical rejected/chosen - the mask will contain the last element in each sequence

                # a mask for each sequence that will place 1's where we have padded elements
                chosen_padding_mask = (chosen_id == self.PAD_ID).bool()
                rejected_padding_mask = (rejected_id == self.PAD_ID).bool()

                # united_unpadding_mask will what are the unite between the unpadded elements
                # will indicate 1's where we have non padded tokens, in either of the inputs
                united_unpadding_mask = torch.logical_not(
                    torch.logical_and(chosen_padding_mask,
                                      rejected_padding_mask))

                # get a mask of all the different tokens
                divergence_mask = (chosen_id != rejected_id)
                divergence_mask = divergence_mask.cumsum(0).bool()

                # loss mask indicates the elements which should be taken into consideration after sigmoid calc
                # from the first divergence, till the last non padded token
                loss_mask = torch.logical_and(divergence_mask,
                                              united_unpadding_mask)
                loss_mask = torch.where(divergence_mask.sum().bool(),
                                        loss_mask, self.fallback_mask)

                # calc logsigmoid on all the input and mask the not interesting ones
                if self.loss_to_fp32:
                    chosen_reward = chosen_reward.float()
                    rejected_reward = rejected_reward.float()
                logsigmoid = torch.nn.functional.logsigmoid(
                    chosen_reward.float() -
                    rejected_reward.float()) * loss_mask
                #average according to the interesting number of elements
                num_elements_in_loss = loss_mask.sum().float()
                loss += -(logsigmoid.sum() / num_elements_in_loss)

                # log the c_ind / r_ind in chosen_mean_scores / rejected_mean_scores
                c_ind_mask = get_last_before_padding(
                    chosen_padding_mask, self.num_padding_at_beginning)
                c_ind_mask = torch.where(
                    chosen_padding_mask.sum() > self.num_padding_at_beginning,
                    c_ind_mask, self.fallback_mask)
                chosen_mean_score = (c_ind_mask.float() *
                                     chosen_reward.float()).sum()
                chosen_mean_scores.append(chosen_mean_score)

                r_ind_mask = get_last_before_padding(
                    rejected_padding_mask, self.num_padding_at_beginning)
                r_ind_mask = torch.where(
                    rejected_padding_mask.sum() >
                    self.num_padding_at_beginning, r_ind_mask,
                    self.fallback_mask)
                rejected_mean_score = (r_ind_mask.float() *
                                       rejected_reward.float()).sum()
                rejected_mean_scores.append(rejected_mean_score)
            else:
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                    c_inds
                ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
                check_divergence = (chosen_id != rejected_id).nonzero()
                if len(check_divergence) == 0:
                    end_ind = rejected_reward.size(-1)
                    divergence_ind = end_ind - 1
                    r_ind = c_ind
                else:
                    # Check if there is any padding otherwise take length of sequence
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = r_inds[self.num_padding_at_beginning].item(
                    ) if len(
                        r_inds) > self.num_padding_at_beginning else seq_len
                    end_ind = max(c_ind, r_ind)
                    divergence_ind = check_divergence[0]
                assert divergence_ind > 0
                c_truncated_reward = chosen_reward[divergence_ind:end_ind]
                r_truncated_reward = rejected_reward[divergence_ind:end_ind]
                if self.loss_to_fp32:
                    c_truncated_reward = c_truncated_reward.float()
                    r_truncated_reward = r_truncated_reward.float()
                loss += -torch.nn.functional.logsigmoid(
                    c_truncated_reward - r_truncated_reward).mean()

                chosen_mean_scores.append(
                    chosen_reward[c_ind - 1])  #use the end score for reference
                rejected_mean_scores.append(rejected_reward[r_ind - 1])

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
