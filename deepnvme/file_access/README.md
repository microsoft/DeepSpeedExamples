# Using DeepNVMe to implement simple file operations  CPU/GPU tensors

This folder contains examples illustrating how to use DeepNVMe to implement simple file operations for moving data between persistent storage and CPU/GPU tensors. For each file operation, we provide an implementation using Python I/O functionality, and a DeepNVMe implementation using CPU bounce buffer and NVIDIA GPUDirect Storage (GDS) as appropriate. 

The following table is a mapping of file operations to the corresponding Python and DeepNVMe implementations. 


File Operation | Python | DeepNVMe (CPU bounce buffer) | DeepNVMe (GDS)
|---|---|---|---|
Load CPU tensor from file | py_load_cpu_tensor.py | bounce_buffer_load_cpu_tensor.py | - |
Load GPU tensor from file | py_load_gpu_tensor.py | bounce_buffer_load_gpu_tensor.py | gds_load_gpu_tensor.py |
Store CPU tensor to file | py_store_cpu_tensor.py | bounce_buffer_store_cpu_tensor.py | - |
Store GPU tensor to file | py_store_gpu_tensor.py | bounce_buffer_store_gpu_tensor.py | gds_store_gpu_tensor.py |  

