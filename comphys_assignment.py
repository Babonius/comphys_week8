from pydub import AudioSegment
import numpy as np
import scipy.fft

#exercise 1
audio = AudioSegment.from_file("note.mp3")
audio = audio.set_channels(1)
samples = np.array(audio.get_array_of_samples())
sample_rate = audio.frame_rate

window = np.hanning(len(samples))
spectrum = np.abs(scipy.fft.fft(window * samples))
frequencies = scipy.fft.fftfreq(len(spectrum), 1 / sample_rate)

positive_frequencies = frequencies[:len(frequencies) // 2]
positive_spectrum = spectrum[:len(spectrum) // 2]

dominant_frequency = positive_frequencies[np.argmax(positive_spectrum)]
print(f"Dominant frequency: {dominant_frequency:.2f} Hz")

#exrcise 2
def f(x, y):
    return y**2 - x**3
x0 = 0
y0 = 1
h = 0.1

f_val = f(x0, y0)
f_prime_val = -3 * x0**2 + 2 * y0 * (y0**2 - x0**3)
y_taylor = y0 + h * f_val + (h**2 / 2) * f_prime_val
print("Taylor Order 2 -> y(0.1) ≈", y_taylor)

k1 = f(x0, y0)
k2 = f(x0 + h/2, y0 + h * k1 / 2)
k3 = f(x0 + h/2, y0 + h * k2 / 2)
k4 = f(x0 + h, y0 + h * k3)
y_rk4 = y0 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
print("Runge-Kutta Order 4 -> y(0.1) ≈", y_rk4)

