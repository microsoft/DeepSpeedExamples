
# dependences
- [transformer-deploy](https://github.com/cli99/transformer-deploy/tree/ds

  in the repo dir, do `pip install .`

- latest tensorrt (8.4.3.1)

- onnx (1.12.0), onnxruntime

  do `pip install onnxruntime onnx==1.12.0`

and other packages required

# run

refer to `run.sh`.

`$BACKEND` below can be `trt` or `ds-inference`, or `ort`:

```sh
deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
    --model_type=gptneo \
    --model_name_or_path=EleutherAI/gpt-neo-2.7B \
    --sample_input sample_query.txt \
    --fp16 \
    --$BACKEND
```

note that running with `trt` can take a while to generate the tensorrt execution plan.