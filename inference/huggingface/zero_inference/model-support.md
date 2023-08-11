# Supporting Hugging Face Models via KV-Cache Offloading to CPU

We apply `non-intrusive` changes to Hugging Face (HF) OPT and BLOOM models to enable KV cache CPU offloading. Similar to FlexGen, kv cache offloading is implemented on the client side.
To learn more about the exact code change, compare the differences (conditioned on the `kv_offload` flag) in the model files `modeling_opt.py` and `modeling_bloom.py`. The following steps are taken to enable KV cache CPU offloading in our implementation. There could be alternative designs/implementations in these steps which are optimal in different system setups.

We are detailing our current approach below. With the following five steps, KV cache offloading can be easily enabled through ZeRO-Inference v2.0 for any generative models in Hugging Face.

## 1. Specify KV cache offloading to HF model

KV cache offloading is set in the HF model config by `model.config.kv_offload = True` before the model runs inference. The flag is read and passed along in the HF model's forward functions to trigger the offloading behavior in the attention module.

## 2. Initialize an empty CPU tensor buffer to hold KV cache

The KV cache tensor has a size of
`2 * num_layers * batch_size * max_seq_len * hidden_size`, where `2` is for both K values and V values, `num_layers` is the number of transformer blocks, `batch_size` is the inference batch size, `max_seq_len` is the total length of the prompt and generated tokens, `hidden_size` is the model's hidden dimension.
If the K values and V values require different layout, two separate tensor buffers can be used to hold each. This is what's used in BLOOM models.

This empty tensor allocation can be done at the model initialization stage or the prompt processing stage in inference.
Although in theory initializing the empty tensor(s) at the model initialization shall improve the prompt stage throughput, our experiments show that allocation of the KV cache tensor in prompt processing actually leads to slightly better overall throughput in the tested hardware/software environments. Thus, we take this approach in the current implementation.

## 3. Transfer KV values to the CPU buffer in the prompt processing stage

Passing the corresponding tensor buffer slice to each transformer block at the model level, and then in the prompt stage transfer the generated KV values to the CPU buffer. We use `slice` to easily index into the right buffer location.

Since this GPU to CPU data transfer can happen as early as the KV values are projected, we put it in a different GPU stream and parallelize it with the following computation in the attention block.

## 4. Use the CPU KV cache in the decode stage

During the decode stage, after the KV values for the current token are projected, we transfer it to the CPU KV cache. The full KV values on CPU are then used for the following attention score computation.

## 5. Compute attention scores

This can be done either on CPU or GPU. If done on CPU (slow compute), the projected Q values (less data) need to be transferred to CPU; if done on GPU (fast compute), KV values (more data) need to be transferred to GPU.

As attention score computation is often memory bandwidth bound, with a decent CPU, the former approach delivers better performance. Thus, our current implementation computes attention scores on CPU.
Note that if computed on CPU, the attention output needs to be transferred back to GPU before the later output projection linear layer.

