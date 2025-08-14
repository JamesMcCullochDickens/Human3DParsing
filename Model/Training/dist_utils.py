import os
import datetime
import subprocess
import signal

import torch.multiprocessing as mp
import torch.distributed as dist

import utils.os_utils as os_utils


def spawn_processes(fn, args):
    mp.spawn(fn, args=(*args,), nprocs=args[-1], join=True)


def cleanup():
    dist.destroy_process_group()


def kill_process_on_port(port: int):
    """Kill the process using the given port."""
    try:
        # Use lsof to find processes using the port
        output = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        pids = output.decode().strip().split('\n')
        for pid in pids:
            if pid.isdigit():
                os.kill(int(pid), signal.SIGKILL)
                print(f"Killed process {pid} using port {port}")
    except subprocess.CalledProcessError:
        # No process is using the port
        pass


def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    port = int(os.environ['MASTER_PORT'])
    try:
        if rank == 0:
            kill_process_on_port(port)
    except:
        print(f"Failed to kill all processes at port: {port}, continuing anyway...")
    if os_utils.is_linux():
        os.environ['NCCL_AVOID_RECORD_STREAMS'] = '0'
    dist.init_process_group("nccl" if os_utils.is_linux()
                            else "gloo", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=7200))


def shutdown_dataloader(dl):
    if dl is not None:
        try:
            _ = iter(dl)
            if dl._iterator is not None:
                dl._iterator._shutdown_workers()
                if hasattr(dl._iterator, "_workers"):
                    for w in dl._iterator._workers:
                        if w.is_alive():
                            w.terminate()
        except Exception as e:
            print(f"Warning during dataloader shutdown: {e}")