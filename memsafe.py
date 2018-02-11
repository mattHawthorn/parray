#coding:utf-8

### Say you have an operation that takes some arrays, produces a possibly bigger array, and then applies a reduce
### function on slices of that array to ultimately return a smaller array.
### A quintessential example is matrix multiplication.  Say A, B are matrices and we want C = A*B. (other examples: distance matrices, np.outer())
### If A is a row vector and B is a column vector, the output is a singleton. But the opposite scenario produces
### a product of shape len(A) x len(B), potentially quite larger than either A or B.  Suppose you wanted this matrix
### but only to apply some reduce operation on the rows, like max or sum.  You wouldn't have to inflate the whole
### product in memory; you could iterate over slices of A or B, filling up appropriate slices of C with the reduced
### values as you go.  Moreover, you could hand off those jobs to separate processes and save time!

from psutil import cpu_count, virtual_memory as vm
import sharedmem
import numpy as np


class MemSafeBinaryArrayOp:
    def __init__(self, binary_op, reduce_op,
                 intermediate_shape_func, reduce_shape_func, 
                 intermediate_dtype_func, reduce_dtype_func,
                 max_proportional_mem_usage=0.5, max_mem_usage=None, 
                 n_jobs=-1):
        self.intermediate_shape_func = intermediate_shape_func
        self.reduce_shape_func = reduce_shape_func
        self.intermediate_dtype_func = intermediate_dtype_func
        self.reduce_dtype_func = reduce_dtype_func
        self.binary_op = binary_op
        self.reduce_op = reduce_op
        self.max_proportional_mem_usage = max_proportional_mem_usage
        self.max_mem_usage = max_mem_usage
        
        cpus = cpu_count()
        if -(cpus - 1) < n_jobs < 0:
            n_jobs = cpus + n_jobs
        elif n_jobs > cpus or n_jobs <= -cpus or n_jobs == 0:
            raise ValueError("n_jobs must be either a positive int <= num_cpus or a negative int > -num_cpus")
        
        self.n_jobs = n_jobs
        
    def get_chunk_size(self, shape, dtype):
        # find the chunk size such that self.n_jobs*intermediate size
        # is less than max_available_mem_usage
        available = self.max_proportional_mem_usage * vm().available
        if self.max_mem_usage is not None:
            available = min(self.max_mem_usage, available)
        
        # solve chunk_size*slice_size*n_jobs = available
        mem_per_slice = self.get_mem_usage(shape[1:], dtype)
        return int(available/(self.n_jobs*mem_per_slice))
        
    def get_mem_usage(self, shape, dtype):
        return np.prod(shape)*dtype.itemsize
        
    def __call__(self, left, right):
        inter_shape = self.intermediate_shape_func(left.shape, right.shape)
        inter_dtype = self.intermediate_dtype_func(left.dtype, right.dtype)
        out_shape = self.reduce_shape_func(inter_shape)
        out_dtype = self.reduce_dtype_func(inter_dtype)
        
        # vstack results from processes, or share result and use pool.critical?
        # a vstack seems most foolproof, though it uses more mem.
        with sharedmem.MapReduce(np=self.n_jobs) as pool:
            m_left = sharedmem.full_like(left,left)
            m_right = sharedmem.full_like(right, right)
            result = sharedmem.empty(out_shape, out_dtype)
        
            chunksize = self.get_chunk_size(inter_shape, inter_dtype)
            #print("chunksize {}".format(chunksize))
            n = m_left.shape[0]
            
            def op(i):
                ix = slice(i, min(i+chunksize, n))
                res = self.reduce_op(self.binary_op(m_left[ix], m_right))
                return ix, res
            
            def insert(ix_slice, res):
                result[ix_slice] = res
                
            pool.map(op, range(0, m_left.shape[0], chunksize), reduce=insert)
        
        return np.array(result)
        
        
# an example: multiply and take the argmax (useful e.g. for spherical k-means)
def prod_shape(l_shape, r_shape):
    return (l_shape[0], r_shape[1])

def horiz_argmax(a):
    return np.array(a.argmax(axis=1))

def horiz_max(a):
    return np.array(a.max(axis=1))

def reduce_shape(shape):
    return shape[:-1]

def same_dtype(l_dt,r_dt):
    return l_dt

def identity(x):
    return x

def int_dtype(dt):
    return np.int32
    
product_argmax = MemSafeBinaryArrayOp(binary_op=np.dot,
                                           reduce_op=horiz_argmax,
                                           intermediate_shape_func=prod_shape,
                                           reduce_shape_func=reduce_shape,
                                           intermediate_dtype_func=same_dtype,
                                           reduce_dtype_func=identity
                                           )


def main():
    from time import time
    import sys
    
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 10000
    d = 50
    dtype = np.int64
    
    a = np.arange(n*d, dtype=dtype).reshape(n,d)
    print("instantiated array A with shape {}\n".format(a.shape))
    
    n_rows_per_proc = 500
    n_proc = cpu_count()

    safe_prod_argmax = MemSafeBinaryArrayOp(binary_op=np.dot,
                                           reduce_op=horiz_argmax,
                                           intermediate_shape_func=prod_shape,
                                           reduce_shape_func=reduce_shape,
                                           intermediate_dtype_func=same_dtype,
                                           reduce_dtype_func=identity,
                                           max_mem_usage = a.dtype.itemsize*n*n_rows_per_proc*n_proc
                                           )
    
    print("doing argmax-multiply on A and A.T, usual way")
    tic = time()
    r1 = horiz_argmax(np.dot(a,a.T))
    toc = time()
    print("runtime was {} ms".format((toc - tic)*1000.0))
    print("intermediate result has memory size {} Mb\n".format(np.prod(prod_shape(a.shape,a.T.shape))*a.dtype.itemsize/1024**2))
    
    print("doing argmax-multiply on A and A.T, parallel memory-safe way, using {} cores".format(safe_prod_argmax.n_jobs))
    tic = time()
    r2 = safe_prod_argmax(a, a.T)
    toc = time()
    print("runtime was {} ms".format((toc - tic)*1000.0))
    print("intermediate results of all processes used at most {} Mb simultaneously\n".format(safe_prod_argmax.max_mem_usage/1024**2))
    
    assert all(r1 == r2)
    print("results are in agreement")

if __name__=="__main__":
    main()

