#coding:utf-8
from .types_ import *
from numpy import ndarray, empty


class ArrayGatherer(Gatherer):
    def __init__(self, initializer: ArrayOp, index_op: IndexOp):
        """
        Callable which can be passed to the 'gather' arg of memsafe_op.MemSafeArrayOp.
        Take the input arrays and an iterable of ((input_index, ...), worker_output_array) tuples
        and return a result.
        :param shape_op: function taking the shapes of the input arrays and returning the shape of
         the output array. E.g., for matrix product this would be lambda (a,b), (c,d): (a,d)
        :param index_op: function taking the indexers used to index the inputs and returning the
         indexer to use to assign the output array into the result.
        """
        self.initializer = initializer
        self.index_op = index_op

    def __call__(self, arrays: Tuple[Array, ...], ix_array_iter: ArrayChunks):
        result = self.initializer(*arrays)

