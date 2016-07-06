
# def init(shared_arr_):
#     global shared_arr
#     shared_arr = shared_arr_ # must be inhereted, not passed as an argument

from multiprocessing import sharedctypes
from multiprocess import Pool
import ctypes

global_vars = {}

class A(object):
    def __init__(self, size):
        self.size = size
        # self.x = np.random.random((size,))
        global global_vars
        global_vars['volume_f'] = np.random.random((size,))

    def func2(self, t):
        print t
        return len(self.x)

    def func(self, t):
        print t
        return len(global_vars['volume_f'])

    def func_parallel(self, processes=4):
        # global shared_arr
        # shared_arr = sharedctypes.RawArray(ctypes.c_double, self.size)
        # arr = np.frombuffer(shared_arr)
        # arr[:] = self.x
        # arr_orig = arr.copy()

        # p = Pool(processes=4, initializer=init, initargs=(shared_arr,))
        p = Pool(processes=processes)
        res = p.map(self.func, range(processes))
        p.close()
        p.join()
        print res

    # def func_parallel(self):
    #     pool = Pool(processes=10)
    #     results = pool.map(self.func, range(16))
    #     return results
