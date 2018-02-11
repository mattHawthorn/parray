#coding:utf-8
from logging import getLogger
from os import cpu_count
log = getLogger()

def n_processes(n):
    cores = cpu_count()
    if n > cores:
        log.warning("Warning: more processes requested ({proc}) than "
                    "cores available ({cores}); using {cores} instead".format(proc=n, cores=cores))
        procs = cores
    elif n < 0:
        procs = cores + 1 + n
        if procs < 0:
            log.warning("{n} fewer than available cores requested, but only {cores} cores "
                        "are available; using 1 process instead".format(n=n+1, cores=cores))
            procs = 1
    elif isinstance(n, int) and n > 0:
        procs = n
    else:
        raise ValueError("Number of requested processes must be an int, greater than or less than 0")

    return procs

