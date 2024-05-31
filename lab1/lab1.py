import matplotlib.pyplot as plt
import numpy as np
import math

def coordinates(N):
    X = np.linspace(0, 2*np.pi, N)
    Y = np.cos(3*X) + np.sin(2*X)
    return X,Y

def fft(a, option):
    N = len(a)
    if N == 1: return a

    even = fft(a[0::2], option)
    odd = fft(a[1::2], option)
    Wn = complex(math.cos(2 * math.pi / N), option * math.sin(2 * math.pi / N))
    W = complex(1) 

    y = [0 for _ in range(N)]
    for j in range (N // 2):
        y[j] = even[j] + W * odd[j]
        y[j + N // 2] = even[j] - W * odd[j]
        W*= Wn
    return y
  
N = 8
x, y = coordinates (N) 
xx, yy = coordinates(1000)

fft_y = fft(y,-1)

result_fft = [x / N for x in fft_y]

amplitude = np.abs(result_fft)
phase = np.angle(result_fft)

ifft_y = fft (result_fft, 1)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(x, y, marker='o')
plt.plot(xx, yy)
plt.title('Signal y=cos(3x)+sin(2x)')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.subplot(2, 2, 2)
plt.stem(amplitude)
plt.title('FFT - Amplitude')
plt.xlabel('k')
plt.ylabel('|Ck|')
plt.subplot(2, 2, 3)
plt.stem(phase)
plt.title('FFT - Phase')
plt.xlabel('k')
plt.ylabel('<Ck')
plt.subplot(2, 2, 4)
plt.plot(x, np.real(ifft_y), label='New Signal')
plt.plot(xx, yy, label='Original Signal')
plt.title('Inversed FFT')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()
plt.tight_layout()
plt.show()
