# Benchmarking Scripts for DeepSpeed-FastGen

## Run the Benchmark

The benchmarking scripts use DeepSpeed-FastGen in the persistent mode.
You can launch the server by the following command. 

```bash
python server.py [options] start
```

`-h` option shows all options.
You can also stop the server by the following command.

```bash
python server.py stop
```

After you launch the server, you can run the client by the following command.
`-h` option shows all options.

```bash
python run_benchmark_client.py [options]
```

`run_all.sh` sweeps different model sizes and number of clients.
`run_all_vllm.sh` runs the same benchmark for VLLM.
These script saves the log in the directory named `logs.[BENCHMARK_PARAMETERS]`.


## Analyze the Benchmark Results

We used these scripts to plot the results in our blog.
Set the root directory of log directories to `--log_dir`.

- `plot_th_lat.py`: Plot throughput and latency for different model sizes and number of clients
- `plot_effective_throughput.py`: Plot effective throughput
- `plot_latency_percentile.py`: Plot P50/P90/P95 latency


