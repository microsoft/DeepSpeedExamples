#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs.
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.

import threading
import time
import os
import numpy as np

import gpustat
import logging
import json
import itertools

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

exitFlag = 0
GPU_MEMORY_THRESHOLD = 20  # MB?
NUM_GPUS = 1
GPU_LIST = set([0, 1, 2, 3, 4, 5, 6, 7])
CURRENT_USED = set([])


def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            # if memory_used < GPU_MEMORY_THRESHOLD:
            if i not in CURRENT_USED and memory_used < GPU_MEMORY_THRESHOLD:
                return_list.append(i)
                if len(return_list) == NUM_GPUS:
                    return return_list

        logger.info("Waiting on GPUs")
        time.sleep(10)


class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device,
                                  bash_command)
            thread1.start()
            import time
            time.sleep(1)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        string = ''
        for i in self.cuda_device:
            if string:
                string += ','
            string += f'{i}'
            # add the gpu to using list
            CURRENT_USED.add(i)

        os.environ['CUDA_VISIBLE_DEVICES'] = string
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)
        # remove the gpu from using list
        for i in self.cuda_device:
            CURRENT_USED.remove(i)
        logger.info("Finishing " + self.name)


BASH_COMMAND_LIST = []

stage = 'one_stage'
index = 0
port_numebr = np.random.randint(10000, 100000000)
lr = 2e-5
bz = 32
budget = 'C'
init_method = 'CorrectsimpleQ'

layer_reduction = False

quantization = True
Group = 1  #768
task_name = 'mnli'
prune = False
for weight_bit in [1]:  #[1,2,8]:
    for task_name in ['mnli']:  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
        json_file = './config/ds_config.json'
        with open(f'{json_file}') as f:
            data_json = json.load(f)
        student_init_method = init_method
        if layer_reduction:
            data_json["compression_training"]["layer_reduction"][
                "enable"] = layer_reduction
            if student_init_method == 'skipBERT5-2-10':
                data_json["keep_number_layer"] = 5
                data_json["teacher_layer"] = [2, 4, 6, 8, 10]
            elif student_init_method == 'skipBERT4-1-[10':
                data_json["keep_number_layer"] = 4
                data_json["teacher_layer"] = [1, 4, 7, 10]
            elif student_init_method == 'skipBERT5-2-11':
                data_json["keep_number_layer"] = 4
                data_json["teacher_layer"] = [2, 5, 8, 11]
            if task_name == 'cola':
                data_json["train_micro_batch_size_per_gpu"] = 16
                data_json["train_batch_size"] = 16
                bz = 16
        else:
            data_json["compression_training"]["layer_reduction"][
                "enable"] = layer_reduction

        if quantization:
            data_json["compression_training"]["layer_reduction"][
                "keep_number_layer"] = 12
            data_json["compression_training"]["weight_quantization"][
                "shared_parameters"]["enabled"] = True
            data_json["compression_training"]["weight_quantization"][
                "shared_parameters"]["schedule_offset"] = 0
            data_json["compression_training"]["weight_quantization"][
                "shared_parameters"]["quantize_groups"] = Group
            data_json["compression_training"]["weight_quantization"][
                "shared_parameters"]["quantization_type"] = "symmetric"
            data_json["compression_training"]["weight_quantization"][
                "different_groups"]["wq1"]["params"][
                    "target_bits"] = weight_bit
            data_json["compression_training"]["weight_quantization"][
                "different_groups"]["wq1"]["params"]["start_bits"] = weight_bit
            data_json["compression_training"]["weight_quantization"][
                "different_groups"]["wq2"]["params"][
                    "target_bits"] = weight_bit
            data_json["compression_training"]["weight_quantization"][
                "different_groups"]["wq2"]["params"]["start_bits"] = weight_bit
            data_json["compression_training"]["activation_quantization"][
                "shared_parameters"]["enabled"] = True
            data_json["compression_training"]["activation_quantization"][
                "shared_parameters"]["quantization_type"] = "symmetric"
            data_json["compression_training"]["activation_quantization"][
                "shared_parameters"]["schedule_offset"] = 0

        if prune:
            data_json["compression_training"]["head_pruning"][
                "shared_parameters"]["enabled"] = True
            data_json["compression_training"]["row_pruning"][
                "shared_parameters"]["enabled"] = True
            #data_json["compression_training"]["sparse_pruning"]["shared_parameters"]["enabled"]=True
            print(
                'check ', data_json["compression_training"]["head_pruning"]
                ["shared_parameters"]["enabled"],
                data_json["compression_training"]["row_pruning"]
                ["shared_parameters"]["enabled"],
                data_json["compression_training"]["sparse_pruning"]
                ["shared_parameters"]["enabled"])
        data_json["fp16"]["enabled"] = False
        if not data_json["fp16"]["enabled"]:
            data_json["compression_training"]["weight_quantization"][
                "shared_parameters"]["quantize_weight_in_forward"] = True
        else:
            data_json["compression_training"]["weight_quantization"][
                "shared_parameters"]["quantize_weight_in_forward"] = False

        data_json["steps_per_print"] = 200
        new_json_path = f"config/my_{task_name}_{init_method}_{weight_bit}.json"
        if os.path.exists(new_json_path):
            os.remove(new_json_path)
        with open(new_json_path, 'w') as f:
            json.dump(data_json, f)

        if task_name not in ['mnli', 'qqp']:
            if task_name in ['cola', 'mrpc']:
                epoch = int(2 * 6)
            elif task_name == 'qnli':
                epoch = int(2 * 1.5)
            else:
                epoch = int(2 * 2)
        else:
            epoch = int(6 * 2)
            if budget == 'B':
                epoch = 9
            if budget == 'C':
                epoch = 18
            if budget == 'D':
                epoch = 36
        if task_name in ['mrpc']:
            epoch = 4

        if prune:
            output_path = f'/data/users/xwu/June7fp32QuantG{Group}_BertBase_{lr}_{init_method}_budget{budget}_prune'
            output_path = 'output/test_prune'
        else:
            output_path = f'/data/users/xwu/June7fp32QuantG{Group}_BertBase_{lr}_{init_method}_budget{budget}'
            output_path = 'output/test'
        teacher_model = f'/blob/users/xwu/compression/huggingface_models/bert-base-uncased-{task_name}/pytorch_model.bin'
        student_model = teacher_model

        OUTPUT_PATH = f"{output_path}/{task_name}/w_{weight_bit}_{stage}"
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH)
        time.sleep(10)

        comm = f'''python -m torch.distributed.launch --nproc_per_node=1 \
                --master_port {port_numebr+index} \
                run_glue.py \
                --seed 42  --distill_method {stage} \
                --model_name_or_path /blob/users/xwu/compression/huggingface_models/bert_base_uncased \
                --pretrained_dir_student  {student_model} \
                --pretrained_dir_teacher {teacher_model}\
                --task_name {task_name} --weight_bit {weight_bit} \
                --learning_rate {lr} \
                --pad_to_max_length \
                --per_device_train_batch_size 32 \
                --save_best_checkpoint --save_last_model --clean_last_model \
                --num_train_epochs {epoch} \
                --gradient_accumulation_steps 1 \
                --deepspeed_config  {new_json_path}  --deepspeed \
                --output_dir {OUTPUT_PATH}  | tee -a {OUTPUT_PATH}/train_log.txt'''

        BASH_COMMAND_LIST.append(comm)
        index += 1

dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST[::-1])
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")
