import numpy as np
import matplotlib.pyplot as plt

# Define a helper function to calculate and shift the FFT for plotting
def compute_and_shift_dft(x, N):
    # Compute the N-point DFT
    X = np.fft.fft(x, n=N)
    # Shift the zero-frequency component to the center
    X_shifted = np.fft.fftshift(X)
    # Generate frequency axis from -0.5 to 0.5
    freqs = np.fft.fftshift(np.fft.fftfreq(N))
    return freqs, np.abs(X_shifted)



N1 = 10
n_10 = np.arange(N1)


x1_10 = np.exp(1j * 2 * np.pi * n_10 * 0.1) + np.exp(1j * 2 * np.pi * n_10 * 0.33)
x2_10 = 2.5 * np.cos(2 * np.pi * n_10 * 0.1)


f_10, X1_10_mag = compute_and_shift_dft(x1_10, N1)
f_10, X2_10_mag = compute_and_shift_dft(x2_10, N1)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.stem(f_10, X1_10_mag)
plt.title("DFT of x1[n] (10 samples)")
plt.xlabel("Frequency (f_r)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(f_10, X2_10_mag)
plt.title("DFT of x2[n] (10 samples)")
plt.xlabel("Frequency (f_r)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================

N2 = 500


f_500_from_10, X1_500_mag = compute_and_shift_dft(x1_10, N2)
f_500_from_10, X2_500_mag = compute_and_shift_dft(x2_10, N2)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(f_500_from_10, X1_500_mag)
plt.title(" DFT of x1[n] (10 samples, zero-padded to 500)")
plt.xlabel("Frequency (f_r)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f_500_from_10, X2_500_mag)
plt.title("DFT of x2[n] (10 samples, zero-padded to 500)")
plt.xlabel("Frequency (f_r)")
plt.grid(True)
plt.tight_layout()
plt.show()


N3 = 100
n_100 = np.arange(N3)

x1_100 = np.exp(1j * 2 * np.pi * n_100 * 0.1) + np.exp(1j * 2 * np.pi * n_100 * 0.33)
x2_100 = 2.5 * np.cos(2 * np.pi * n_100 * 0.1)


f_100, X1_100_mag = compute_and_shift_dft(x1_100, N3)
f_100, X2_100_mag = compute_and_shift_dft(x2_100, N3)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.stem(f_100, X1_100_mag)
plt.title("DFT of x1[n] (100 samples)")
plt.xlabel("Frequency (f_r)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(f_100, X2_100_mag)
plt.title("DFT of x2[n] (100 samples)")
plt.xlabel("Frequency (f_r)")
plt.grid(True)
plt.tight_layout()
plt.show()


N4 = 500
f_500_from_100, X1_500_pad100_mag = compute_and_shift_dft(x1_100, N4)
f_500_from_100, X2_500_pad100_mag = compute_and_shift_dft(x2_100, N4)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(f_500_from_100, X1_500_pad100_mag) 
plt.title("DFT of x1[n] (100 samples, zero-padded to 500)")
plt.xlabel("Frequency (f_r)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f_500_from_100, X2_500_pad100_mag)
plt.title("DFT of x2[n] (100 samples, zero-padded to 500)")
plt.xlabel("Frequency (f_r)")
plt.grid(True)
plt.tight_layout()
plt.show()