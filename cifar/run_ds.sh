#!/bin/bash

deepspeed.pt cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json
