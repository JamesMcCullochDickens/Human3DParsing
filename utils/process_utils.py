import os
import signal


def kill_child_processes(parent_pid=os.getpid()):
    pgid = os.getpgid(parent_pid)
    os.killpg(pgid, signal.SIGKILL)