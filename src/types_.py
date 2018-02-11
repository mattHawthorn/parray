#coding:utf-8

from typing import Callable, Iterable, Iterator, Tuple, Union, Generic, TypeVar
from numpy import ndarray
from scipy.sparse.base import spmatrix

T = TypeVar("T")
# Alias for return types
R = TypeVar("R", contravariant=True)
newaxis = None
NoneType = type(None)

class SparseMatrix(Generic[T], spmatrix):
    pass

class DenseArray(Generic[T], ndarray):
    pass

Array = Union[SparseMatrix, DenseArray]
Shape = Tuple[int, ...]

# all the types that can serve as a single index in an array[ix1, ix2, ...] call
Slicer = Union[int, slice, NoneType, Iterable[int], DenseArray[int]]
Indexer = Union[Slicer, Tuple[Slicer, ...]]
# those indexing types which can be used in an array[ix] = value call
# SetIndexer = Indexer

IndexIter = Iterable[Indexer]
# a function used to generate the indices of array chunks to be passed to the workers
IndexGenerator = Callable[[Array], IndexIter]

# constraints to be checked on input arrays.
# example: for matrix product, the second dim of the first array must equal the
# first dim of the second array. Other examples include type compatibility checks
ArrayConstraint = Callable[[Tuple[Array, ...]], bool]

# something that takes a tuple of iterables of certain types and returns an
# iterable of tuples of those types.
# examples are zip, itertools.product, itertools.combinations
IterJoiner = Callable[[Tuple[Iterable[T], ...]], Iterator[Tuple[T, ...]]]

# the job to be done by a worker
ArrayOp = Callable[[Tuple[Array, ...]], Array]

# take the locations of chunks in the input array and produce the location to be placed
# in an output array
IndexOp = Callable[[Tuple[Indexer, ...]], Indexer]

# take the shapes of the input arrays and return the shape of the output array
ShapeOp = Callable[[Tuple[Shape, ...]], Shape]

# an iterable of results as returned from the workers acting on slices of the inputs
ArrayChunks = Iterable[Tuple[Tuple[Indexer], Array]]

# the function that gathers the iterable of worker results, taking the input arrays as
# the first argument as well for possible size inference
Gatherer = Callable[[Tuple[Array, ...], ArrayChunks], R]
