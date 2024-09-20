import pyfftw
import time
import numpy as np
import os
import torch
import scipy
import gc


def test1():
    print(os.cpu_count())
    print(pyfftw.interfaces.cache.enable())
    flags = ('FFTW_MEASURE','FFTW_DESTROY_INPUT')
    npix = 256
    tidk = np.zeros((npix, npix, npix,3))
    tidk_al = pyfftw.empty_aligned(tidk.shape, dtype='float64')
    tidk_al[:] = tidk

    tstart = time.time()
    print('start torch')
    tidk_tensor = torch.from_numpy(tidk).to(torch.complex64)
    tid = torch.fft.irfftn(tidk_tensor)
    print(time.time()-tstart)
    del tidk_tensor
    gc.collect()

    tstart = time.time()
    print('start pyfftw n=8')
    pyfftw.config.NUM_THREADS = 8
    tid = pyfftw.builders.irfftn(tidk_al, axes=(0,1,2), threads=8)
    #tid = scipy.fft.irfftn(tidk, workers=8)
    print(time.time()-tstart)
    
    print('start pyfftw n=1')
    pyfftw.config.NUM_THREADS = 1
    tstart = time.time()
    tid = pyfftw.builders.irfftn(tidk_al, axes=(0,1,2), threads=1)
    #tid = scipy.fft.irfftn(tidk, workers=1)
    print(time.time()-tstart)

    print('start scipy n=8')
    tstart = time.time()
    tid = scipy.fft.irfftn(tidk, workers=8)
    print(time.time()-tstart)

    print('start scipy n=1')
    tstart = time.time()
    tid = scipy.fft.irfftn(tidk, workers=1)
    print(time.time()-tstart)

def test2():
    # suggested by chatgpt
    n = 1024
    a = pyfftw.empty_aligned(n * n, dtype='complex128')
    b = pyfftw.empty_aligned(n * n, dtype='complex128')
    a[:] = np.random.randn(n * n) + 1j * np.random.randn(n * n)

    # Set up FFTW plan with different numbers of threads
    for num_threads in [8,1]:
        print(num_threads)
        fft_object = pyfftw.FFTW(a, b, threads=num_threads)
        start_time = time.time()
        fft_object()
        print(f"Time taken with {num_threads} threads: {time.time() - start_time:.4f} seconds")


if __name__ == '__main__':
    test1()
    #test2()
