

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.multiprocessing as mp
import os
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.multiprocessing import Std
from typing import Any, Dict, Optional, Tuple
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.metrics.api import prof
import signal
from datetime import timedelta
# from train import trainer
# In[29]:
import torch.distributed as dist
# import deepspeed
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.multiprocessing import PContext, start_processes
import shutil
from torch.distributed.elastic.utils.logging import get_logger

log = get_logger()
class FunctionElasticAgent(LocalElasticAgent):
    def __init__(
        self,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        rdzv_run_id = spec.rdzv_handler.get_run_id()
        self._log_dir = self._make_log_dir(log_dir, rdzv_run_id)


    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None:
        print("Shut PIDS:",[proc.is_alive() for proc in self._pcontext._pc.processes  ])
        # if self._pcontext:
        #     self._pcontext.close(death_sig)

        to_drop = []
        for i, proc in enumerate(self._pcontext._pc.processes):
            if(proc.is_alive()==False):
                to_drop.append(proc)
        for proc in to_drop:
            self._pcontext._pc.processes.remove(proc)

        
        if self._pcontext:
            # pids = self._pcontext.pids()
            for proc in self._pcontext._pc.processes:
                print("Sending USR kill")
                os.kill(proc.pid,signal.SIGUSR1)
            
        #     self._pcontext._pc.processes
            

    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        if self._pcontext!= None:
            return self._pcontext.pids()
            print("Start PIDS:",[proc.is_alive() for proc in self._pcontext._pc.processes  ])
            pids = [proc.pid for proc in self._pcontext._pc.processes  ]
            to_remove = []
            for w in worker_group.workers:
                if w.id not in pids:
                    to_remove.append(w)
            
            for w in to_remove:
                worker_group.workers.remove(w)
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"

        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        # scaling events do not count towards restarts (gets same attempt #)
        # remove existing log dir if this restart is due to a scaling event
        attempt_log_dir = os.path.join(self._log_dir, f"attempt_{restart_count}")
        shutil.rmtree(attempt_log_dir, ignore_errors=True)
        os.makedirs(attempt_log_dir)

        assert spec.entrypoint is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
        )
        print("PIDS:",[proc.is_alive() for proc in self._pcontext._pc.processes  ])

        return self._pcontext.pids()

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        pids = [proc.pid for proc in self._pcontext._pc.processes  ]
        to_remove = []
        for w in worker_group.workers:
            if w.id not in pids:
                to_remove.append(w)
            
        for w in to_remove:
            print("Removing worker:",w.id)
            worker_group.workers.remove(w)


        role = worker_group.spec.role
        worker_pids = {w.id for w in worker_group.workers}
        assert self._pcontext is not None
        pc_pids = set(self._pcontext.pids().values())
        if worker_pids != pc_pids:
            log.error(
                f"[{role}] worker pids do not match process_context pids."
                f" Expected: {worker_pids}, actual: {pc_pids}"
            )
            return RunResult(state=WorkerState.UNKNOWN)

        result = [int(not proc.is_alive()) for proc in self._pcontext._pc.processes  ]
        # result = self._pcontext.wait(0)
        if sum(result) > 0:
            # if result.is_failed():
            if sum(result) > 0:
                # map local rank failure to global rank
                worker_failures = {}
                for local_rank, failure in enumerate(result):
                    worker = worker_group.workers[local_rank]
                    worker_failures[worker.global_rank] = failure
                return RunResult(
                    state=WorkerState.FAILED,
                    failures=worker_failures,
                )
            else:
                # copy ret_val_queue into a map with a global ranks
                workers_ret_vals = {}
                for local_rank, ret_val in result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,
                )
        else:
            return RunResult(state=WorkerState.HEALTHY)



def trainer(trainloader,model_rg):
    def usr_signal(signalNumber, frame):
        global model 
        print("Recieved usr signal:",rank)
        dist.destroy_process_group()
        print("Group Destroyed:",rank)
        dist.init_process_group(
            backend='nccl', init_method="env://", timeout=timedelta(seconds=20), world_size = 7
            )
        model = DistributedDataParallel(model_rg, device_ids=[device_id])
        
    def SIGABRT_signal(signalNumber, frame):
        print("Recieved abort")

    signal.signal(signal.SIGUSR1, usr_signal)
    # signal.signal(signal.SIGABRT, SIGABRT_signal)


    device_id = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend='nccl', init_method="env://", timeout=timedelta(seconds=20) 
    )
    rank = dist.get_rank()


    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100

    print("Init dev:",device_id)
    
    model_rg.to(device_id)
    model = DistributedDataParallel(model_rg, device_ids=[device_id])
    # log_dist("Creating DeepSpeed engine", ranks=[0], level=logging.INFO)
    ds_config = {
        "train_micro_batch_size_per_gpu": 128,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "fp16": {
            "enabled": True
        }
    }

    # model, _, _, _ = deepspeed.initialize(model=model,
    #                                       model_parameters=model.parameters(),
    #                                       config=ds_config,
    #                                       )
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Construct data_loader, optimizer, etc.

    for epoch in range(epochs):
        # try:
        for data, labels in trainloader:
            data = data.to(device_id)
            labels = labels.to(device_id)
            optimizer.zero_grad()
            loss = criterion(model(data), labels)
            print("Before backward:",device_id)
            loss.backward()
            optimizer.step()
            print("dev:",device_id, loss.item())
        # except Run:
        #     print("My runtime error")
        #     pass

# In[38]:

if __name__ == "__main__":
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    model = torchvision.models.resnet18()

    rdzv_configs: Dict[str, str] = {'timeout': 100}

    start_method="spawn"
    shared_queue= mp.get_context(start_method).Queue()
    rdzv_parameters = RendezvousParameters(
            backend='c10d',
            endpoint="worker-0:29401",
            run_id='123456789',
            min_nodes=1,
            max_nodes=2,
            **rdzv_configs
        )

    spec = WorkerSpec(
            role='trainer',
            local_world_size=8,
            entrypoint=trainer,
            args=(trainloader,model),
            rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
            max_restarts=100,
            monitor_interval=50,
            redirects=Std.from_str("0"),
            tee=Std.from_str("0"),
            master_addr='worker-0',
            master_port='51000',
        )
    # spec = WorkerSpec(
    #         role='trainer',
    #         local_world_size=8,
    #         entrypoint="/opt/conda/bin/python",
    #         args=("trainer.py",),
    #         rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
    #         max_restarts=100,
    #         monitor_interval=50,
    #         redirects=Std.from_str("0"),
    #         tee=Std.from_str("0"),
    #         master_addr='worker-0',
    #         master_port='51000',
    #     )
    agent = FunctionElasticAgent(
            spec, start_method
        )
    agent.run()


# In[ ]:





# In[23]:


    


# In[27]:





# 
