import multiprocessing as mp
import numpy as np
from functools import partial
from tqdm import tqdm

def run_mp_str_list(func, l, kwargs, num_processes:int=8):
    splits = np.array_split(l, num_processes)
    splits = [s.tolist() for s in splits]
    procs = []
    func_p = partial(func, **kwargs)
    for split in splits:
        proc = mp.Process(target=func_p, args=(split,))
        proc.start()
    for p in procs:
        p.join()

