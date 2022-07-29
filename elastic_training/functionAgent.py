

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
# from train import trainer
# In[29]:
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
        mp_bin = True

        if(mp_bin==False):
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
        else:
            print("Shut PIDS:",[sh.proc.poll() for lr, sh in self._pcontext.subprocess_handlers.items()  ])
            # if self._pcontext:
            #     self._pcontext.close(death_sig)

            to_drop = []
            for lr, sh in self._pcontext.subprocess_handlers.items():
                if(sh.proc.poll()!=None):
                    to_drop.append(lr)
            for lr in to_drop:
                del self._pcontext.subprocess_handlers[lr]

            
            if self._pcontext:
                # pids = self._pcontext.pids()
                for lr, sh in self._pcontext.subprocess_handlers.items():
                    print("Sending USR kill")
                    os.kill(sh.proc.pid,signal.SIGUSR1)
            
        #     self._pcontext._pc.processes
            

    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        mp_bin = True
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
        # print("PIDS:",[proc.is_alive() for proc in self._pcontext._pc.processes  ])

        return self._pcontext.pids()

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        mp_bin = True

        if (mp_bin):
            pids = [sh.proc.pid for lr,sh in self._pcontext.subprocess_handlers.items()  ]
            to_remove = []
            for w in worker_group.workers:
                if w.id not in pids:
                    to_remove.append(w)
                
            for w in to_remove:
                print("Removing worker:",w.id)
                worker_group.workers.remove(w)
        else:
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

        if (mp_bin):
            result = [int(not sh.proc.poll()==None) for lr, sh in self._pcontext.subprocess_handlers.items()  ]
        else:
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



