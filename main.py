import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


duration = 10
sampling_rate = 1000
heart_rate_bpm = 70

low_cutoff = 0.5
high_cutoff = 40
noise_amplitude = 0.05


# Geracao do ECG sintetico
ecg_clean = nk.ecg_simulate(
    duration=duration,
    sampling_rate=sampling_rate,
    heart_rate=heart_rate_bpm,
    method="ecgsyn"
)


# Vetor de tempo e adicao de ruido
n_samples = len(ecg_clean)
time = np.arange(n_samples) / sampling_rate

noise = np.random.normal(0, noise_amplitude, n_samples)
ecg_noisy = ecg_clean + noise


# Salvamento dos dados em CSV
df = pd.DataFrame({
    "tempo": time,
    "ECG_limpo": ecg_clean,
    "ECG_com_ruido": ecg_noisy
})

df.to_csv("simulated_ecg.csv", index=False)
print("Dados salvos em 'simulated_ecg.csv'")


# Graficos no dominio do tempo
plt.figure(figsize=(12, 5))
plt.plot(time, ecg_clean, label="ECG limpo", linewidth=1.5)
plt.plot(time, ecg_noisy, label="ECG com ruido", linewidth=1, alpha=0.8)
plt.title("ECG sintetico no dominio do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# FFT do sinal com ruido
fft_noisy = np.fft.fft(ecg_noisy)
frequencies = np.fft.fftfreq(n_samples, d=1 / sampling_rate)
magnitude = np.abs(fft_noisy) / n_samples

positive_frequencies = frequencies >= 0
frequency_limit = frequencies <= 100
plot_range = positive_frequencies & frequency_limit

plt.figure(figsize=(12, 5))
plt.plot(frequencies[plot_range], magnitude[plot_range])
plt.title("Espectro de magnitude do ECG com ruido")
plt.xlabel("Frequencia (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()


# Filtragem passa-faixa no dominio da frequencia
bandpass_mask = (np.abs(frequencies) >= low_cutoff) & (np.abs(frequencies) <= high_cutoff)
fft_filtered = fft_noisy.copy()
fft_filtered[~bandpass_mask] = 0

ecg_filtered = np.fft.ifft(fft_filtered).real


# Comparacao entre sinal com ruido e sinal filtrado
plt.figure(figsize=(12, 5))
plt.plot(time, ecg_noisy, label="ECG com ruido", linewidth=1, alpha=0.7)
plt.plot(time, ecg_filtered, label="ECG filtrado", linewidth=1.5)
plt.title("Comparacao entre ECG com ruido e ECG filtrado")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Parametros finais
nyquist_frequency = sampling_rate / 2
heart_rate_hz = heart_rate_bpm / 60
fft_resolution = sampling_rate / n_samples

print("\nParametros utilizados:")
print(f"Duracao do sinal: {duration} s")
print(f"Frequencia de amostragem: {sampling_rate} Hz")
print(f"Frequencia de Nyquist: {nyquist_frequency:.2f} Hz")
print(f"Numero de amostras: {n_samples}")
print(f"Frequencia cardiaca: {heart_rate_bpm} bpm ({heart_rate_hz:.2f} Hz)")
print(f"Resolucao da FFT: {fft_resolution:.2f} Hz")
