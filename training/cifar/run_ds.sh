#!/bin/bash

deepspeed --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 cifar10_deepspeed.py --log-interval 250 --deepspeed --deepspeed_config ds_config.json $@
