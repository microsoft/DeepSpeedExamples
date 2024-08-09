# Using DeepNVMe to implement simple file reads and writes of CPU/GPU tensors

The purpose of this folder is to provide example codes that illustrate how to use DeepNVMe for simple file operations of moving raw data bytes between persistent storage and CPU/GPU tensors. For each file operation, we provide an implementation using Python I/O functionality, and a DeepNVMe implementation using CPU bounce buffer and NVIDIA GPUDirect Storage (GDS) as appropriate. 

The following table is a mapping of file operations to the corresponding Python and DeepNVMe implementations. 


File Operation | Python | DeepNVMe (CPU bounce buffer) | DeepNVMe (GDS)
|---|---|---|---|
Load CPU tensor from file | py_load_cpu_tensor.py | bounce_buffer_load_cpu_tensor.py | - |
Load GPU tensor from file | py_load_gpu_tensor.py | bounce_buffer_load_gpu_tensor.py | gds_load_gpu_tensor.py |
Store CPU tensor to file | py_store_cpu_tensor.py | bounce_buffer_store_cpu_tensor.py | - |
Store GPU tensor to file | py_store_gpu_tensor.py | bounce_buffer_store_gpu_tensor.py | gds_store_gpu_tensor.py |  

The Python implemenations are the scripts with `py_` prefix. while the DeepNVMe implemenetations are those with`bounce_buffer_` and `gds_`prefixes. 


## Tensor Load Examples
The tensor load example scripts share a common command-line interface, which is illustrated below using `py_read_load_cpu_tensor.py`.
```bash
$ python py_load_cpu_tensor.py --help
usage: py_load_cpu_tensor.py [-h] --input_file INPUT_FILE [--loop LOOP] [--validate]

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        File on NVMe device that will read as input.
  --loop LOOP           The number of times to repeat the operation (default 3).
  --validate            Run validation step that compares tensor value against Python file read
```
Before running these example scripts ensure that the input file exists on an NVMe device. The `--validate` option is relevant to only the DeepNVme implementations. This option provides minimal correctness checking by comparing against a tensor loaded using Python. We also provide a bash script `run_load_tensor.sh`, which runs all the example tensor load scripts.


## Tensor Store Examples
The tensor store examples share a command-line interface, which is illustrated below uisng `py_store_cpu_tensor.py`
```bash
$ python py_store_cpu_tensor.py --help
usage: py_store_cpu_tensor.py [-h] --nvme_folder NVME_FOLDER [--mb_size MB_SIZE] [--loop LOOP] [--validate]

options:
  -h, --help            show this help message and exit
  --nvme_folder NVME_FOLDER
                        NVMe folder for file write.
  --mb_size MB_SIZE     Size of tensor to save in MB (default 1024).
  --loop LOOP           The number of times to repeat the operation (default 3).
  --validate            Run validation step that compares tensor value against Python file read

```
Before running these examples ensure that the output folder exists on an NVMe device and that you have write permissions. The `--validate` option is relevant to only the DeepNVMe implementations. This option provides minimal correcness checkping by comparing the output file against that created using Python. We also provide a bash script `run_store_tensor.sh`, which runs all the example tensor store scripts.  


## Performance Advisory
Although this folder is primarily meant to help with integrating DeepNVMe into your Deep Learning applications, the example scripts also print out performance numbers of read and write throughputs. So, we expect you will observe some performance advantage of DeepNVMe compared to Python. However, do note that it is very likely that better performance can be realized by tuning DeepNVMe for your system. Such tuning efforts will generate more optimal values for configuring the DeepNVMe handles. 

For reference, DeepNVMe configuration using hardcoded constants for bounce buffer implementations is as follows:

```python
    aio_handle = AsyncIOBuilder().load().aio_handle(1024**2, 128, True, True, 1)
```

The corresponding DeepNVMe configuration for GDS implementations is as follows:

```python
    gds_handle = GDSBuilder().load().gds_handle(1024**2, 128, True, True, 1)
```

Despite the above caveat, it seems that some performance numbers would be useful here. So, below are the results obtained for 1GB data transfers using the unmodified scripts, i,e. with untuned DeepNVMe configurations. The experiments were conducted on a Lambda RTX A6000 workstation with a single [CS3040 NVMe 2TB SDD](https://www.pny.com/CS3040-M2-NVMe-SSD?sku=M280CS3040-2TB-RB) that has peak sequential read and write throughputs of 5600 MB/s and 4300 MB/s respectively. The software stack included Ubuntu 22.04.4 LTS, Pytorch 2.4, and CUDA 12.1. 

The performance results of the tensor load examples are presented in the table below and show ~2.5X speedup for DeepNVMe. 

Tensor load script | GB/sec (1GB file read)|
|---|---|
py_load_cpu_tensor.py | 1.9 |
py_load_gpu_tensor.py | 1.6 | 
bounce_buffer_load_cpu_tensor | 4.9 | 
bounce_buffer_load_gpu_tensor | 4.1 | 


The performance results of the tensor store examples are presented in the table below and show 4.6X--5.8X  speedup for DeepNVMe. 

Tensor store script | GB/sec (1GB file write)|
|---|---|
py_store_cpu_tensor.py | 0.8 |
py_store_gpu_tensor.py | 0.6 | 
bounce_buffer_store_cpu_tensor | 3.7 | 
bounce_buffer_store_gpu_tensor | 3.5 | 


# Conclusion
We hope these example scripts help you to easily and quicly integrate DeepNVMe into your applications. 
