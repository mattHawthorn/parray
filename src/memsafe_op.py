#coding:utf-8
from typing import Optional
from multiprocessing import Pool
from .types_ import *
from .utils import n_processes


class MemsafeArrayOp:
    def __init__(self,
                 slicers: Tuple[IndexGenerator, ...],
                 joiner: IterJoiner,
                 chunk_op: ArrayOp,
                 gather: Gatherer,
                 constraints: Optional[Tuple[ArrayConstraint, ...]]=None,
                 n_proc: int = -2):
        self.arity = len(slicers)
        self.n_processes = n_processes(n_proc)
        self.slicers = slicers
        self.joiner = joiner
        self.chunk_op = chunk_op
        self.gather = gather
        self.constraints = constraints

    def __call__(self, *arrays) -> R:
        if self.constraints:
            assert all(c(*arrays) for c in self.constraints), "Arrays do not meet specified constraints"

        slice_iters = [g(a) for g, a in zip(self.slicers, arrays)]
        indices = self.joiner(*slice_iters)
        ix_slices = ((ixs, tuple(a[i] for a, i in zip(arrays, ixs))) for ixs in indices)

        with Pool(self.n_processes) as pool:
            outputs = pool.starmap(self._op, ix_slices)
            result = self.gather(arrays, outputs)

        return result

    def _op(self, indices: Tuple[Indexer, ...], arrays: Tuple[Array, ...]) \
            -> Tuple[Tuple[Indexer, ...], Array]:
        return indices, self.chunk_op(*arrays)
