import pyfftw
import time
import numpy as np
import torch


def test1():
    print(pyfftw.interfaces.cache.enable())
    flags = ('FFTW_MEASURE','FFTW_DESTROY_INPUT')
    npix = 512
    tidk = np.zeros((npix, npix, npix,3))
    tidk_al = pyfftw.empty_aligned(tidk.shape, dtype='float64')
    tidk_al[:] = tidk
    tstart = time.time()
    print('start')
    pyfftw.config.NUM_THREADS = 8
    #tid = torch.fft.irfftn(tidk, axes=(0,1,2))
    tid = pyfftw.builders.irfftn(tidk_al, axes=(0,1,2), threads=8)
    print(time.time()-tstart)
    pyfftw.config.NUM_THREADS = 1
    print('start')
    tstart = time.time()

    tid = pyfftw.builders.irfftn(tidk_al, axes=(0,1,2), threads=1)
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