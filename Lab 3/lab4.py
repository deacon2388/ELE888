import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft
from scipy import signal

def design_hp_filter(N):
    k = np.arange(N)
    omega_k = 2 * np.pi * k / N
    
    H_mag = np.zeros(N)
    passband = (omega_k >= 2 * np.pi / 3) & (omega_k <= 4 * np.pi / 3)
    H_mag[passband] = 1
    
    alpha = (N - 1) / 2
    H_ideal = H_mag * np.exp(-1j * omega_k * alpha)
    
    h = np.real(ifft(H_ideal))
    
    w, H_freqz = signal.freqz(h, 1, worN=np.arange(0, 2*np.pi, 2*np.pi/1001))
    
    return h, w, H_freqz

h_35, w_35, H_35 = design_hp_filter(35)

h_71, w_71, H_71 = design_hp_filter(71)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].stem(np.arange(35), h_35, basefmt=" ", markerfmt='b.', linefmt='b-')
axs[0, 0].set_title('Impulse Response h[n] (35 points)')
axs[0, 0].grid(True)

axs[0, 1].plot(w_35, np.abs(H_35), 'b-')
axs[0, 1].set_title('Magnitude |H(Omega)| (35 points)')
axs[0, 1].set_xlim(0, 2*np.pi)
axs[0, 1].grid(True)

axs[1, 0].stem(np.arange(71), h_71, basefmt=" ", markerfmt='r.', linefmt='r-')
axs[1, 0].set_title('Impulse Response h[n] (71 points)')
axs[1, 0].grid(True)

axs[1, 1].plot(w_71, np.abs(H_71), 'r-')
axs[1, 1].set_title('Magnitude |H(Omega)| (71 points)')
axs[1, 1].set_xlim(0, 2*np.pi)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()