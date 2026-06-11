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

rng = np.random.default_rng(42)
noise = rng.normal(0, noise_amplitude, n_samples)
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
plot_frequencies = frequencies[plot_range]
plot_magnitude = magnitude[plot_range]

# Identificacao dos principais picos da FFT para marcacao visual no grafico
peak_search_range = plot_frequencies >= low_cutoff
search_frequencies = plot_frequencies[peak_search_range]
search_magnitude = plot_magnitude[peak_search_range]

local_maxima = np.where(
    (search_magnitude[1:-1] > search_magnitude[:-2])
    & (search_magnitude[1:-1] > search_magnitude[2:])
)[0] + 1

n_peaks_to_mark = 8
most_relevant_peaks = local_maxima[
    np.argsort(search_magnitude[local_maxima])[-n_peaks_to_mark:]
]
most_relevant_peaks = most_relevant_peaks[np.argsort(search_frequencies[most_relevant_peaks])]
peak_frequencies = search_frequencies[most_relevant_peaks]
peak_magnitudes = search_magnitude[most_relevant_peaks]

plt.figure(figsize=(12, 5))
plt.plot(plot_frequencies, plot_magnitude, label="Magnitude da FFT")
plt.scatter(
    peak_frequencies,
    peak_magnitudes,
    color="red",
    s=45,
    zorder=3,
    label="Picos identificados"
)

for peak_frequency, peak_magnitude in zip(peak_frequencies, peak_magnitudes):
    plt.annotate(
        f"{peak_frequency:.1f} Hz",
        xy=(peak_frequency, peak_magnitude),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="red"
    )

plt.title("Espectro de magnitude do ECG com ruido com picos marcados")
plt.xlabel("Frequencia (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fft_picos_marcados.png", dpi=300)
plt.show()


# Filtragem passa-faixa no dominio da frequencia
bandpass_mask = (np.abs(frequencies) >= low_cutoff) & (np.abs(frequencies) <= high_cutoff)
fft_filtered = fft_noisy.copy()
fft_filtered[~bandpass_mask] = 0
filtered_magnitude = np.abs(fft_filtered) / n_samples
plot_filtered_magnitude = filtered_magnitude[plot_range]

ecg_filtered = np.fft.ifft(fft_filtered).real


# Comparacao entre os espectros antes e depois da filtragem
comparison_frequency_limit = 50
comparison_range = positive_frequencies & (frequencies <= comparison_frequency_limit)
comparison_frequencies = frequencies[comparison_range]
comparison_magnitude = magnitude[comparison_range]
comparison_filtered_magnitude = filtered_magnitude[comparison_range]
comparison_y_limit = max(comparison_magnitude.max(), comparison_filtered_magnitude.max()) * 1.05

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
spectra = [
    (axes[0], comparison_magnitude, "ECG com ruido", "tab:blue"),
    (axes[1], comparison_filtered_magnitude, "ECG filtrado", "tab:orange")
]

for axis, spectrum_magnitude, label, color in spectra:
    axis.plot(
        comparison_frequencies,
        spectrum_magnitude,
        label=label,
        linewidth=1.3,
        color=color
    )
    axis.axvspan(
        low_cutoff,
        high_cutoff,
        color="green",
        alpha=0.12,
        label="Faixa preservada pelo filtro"
    )
    axis.axvline(low_cutoff, color="green", linestyle="--", linewidth=1)
    axis.axvline(high_cutoff, color="green", linestyle="--", linewidth=1)
    axis.set_ylabel("Magnitude")
    axis.set_ylim(0, comparison_y_limit)
    axis.legend()
    axis.grid(True)

fig.suptitle("Comparacao dos espectros antes e depois da filtragem")
axes[-1].set_xlabel("Frequencia (Hz)")
axes[-1].set_xlim(0, comparison_frequency_limit)
fig.tight_layout()
plt.savefig("fft_comparacao_filtragem.png", dpi=300)
plt.show()


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
