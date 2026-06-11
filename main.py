import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, find_peaks
from scipy.signal import freqz
from numpy.fft import fft, fftfreq
from scipy.signal.windows import hann

# ─────────────────────────────────────────────
# PARÂMETROS GLOBAIS
# ─────────────────────────────────────────────
fs       = 500          # frequência de amostragem (Hz) — critério Nyquist: 2 x 150 Hz
duracao  = 10           # duração do sinal (s)
f_card   = 1.2          # frequência cardíaca (~72 bpm)
N        = fs * duracao # número de amostras
t        = np.linspace(0, duracao, N, endpoint=False)


# ─────────────────────────────────────────────
# ETAPA 1: GERAÇÃO DO ECG SINTÉTICO
# ─────────────────────────────────────────────
def gaussiana(t, amplitude, centro, sigma):
    return amplitude * np.exp(-((t - centro)**2) / (2 * sigma**2))

def batimento(t):
    """Um único ciclo cardíaco como soma de gaussianas (ondas P, Q, R, S, T)."""
    return (
        gaussiana(t,  0.25, -0.20, 0.025) +   # onda P
        gaussiana(t, -0.10, -0.05, 0.010) +   # onda Q
        gaussiana(t,  1.60,  0.00, 0.010) +   # pico R
        gaussiana(t, -0.25,  0.05, 0.010) +   # onda S
        gaussiana(t,  0.35,  0.20, 0.035)     # onda T
    )

T_card = 1.0 / f_card
ecg_limpo = np.zeros(N)
for i, ti in enumerate(t):
    tc = (ti % T_card) - T_card / 2
    ecg_limpo[i] = batimento(tc)


# ─────────────────────────────────────────────
# ETAPA 5: ADIÇÃO DE RUÍDO CONTROLADO (SNR)
# ─────────────────────────────────────────────
def ruido_por_snr(sinal, snr_db):
    P = np.mean(sinal**2)
    P_ruido = P / (10**(snr_db / 10))
    return np.random.normal(0, np.sqrt(P_ruido), len(sinal))

snr_alvo = 10  # dB

ruido_baseline = 0.05 * np.sin(2 * np.pi * 0.3 * t)       # deriva de linha de base
ruido_rede     = 0.08 * np.sin(2 * np.pi * 60.0 * t)       # interferência 60 Hz
ruido_emg      = ruido_por_snr(ecg_limpo, snr_alvo + 10)   # ruído muscular gaussiano

ecg_ruidoso = ecg_limpo + ruido_baseline + ruido_rede + ruido_emg

SNR_antes = 10 * np.log10(np.mean(ecg_limpo**2) / np.mean((ecg_ruidoso - ecg_limpo)**2))
print(f"SNR antes da filtragem: {SNR_antes:.2f} dB")


# ─────────────────────────────────────────────
# ETAPA 6: FILTRAGEM
# ─────────────────────────────────────────────
# Filtro notch — rejeita 60 Hz
b_notch, a_notch = iirnotch(w0=60, Q=30, fs=fs)
sos_notch = tf2sos(b_notch, a_notch)
ecg_sem_rede = sosfiltfilt(sos_notch, ecg_ruidoso)

# Filtro passa-banda — mantém 0.5 a 40 Hz (Butterworth ordem 4)
sos_bp = butter(N=4, Wn=[0.5, 40], btype='bandpass', fs=fs, output='sos')
ecg_filtrado = sosfiltfilt(sos_bp, ecg_sem_rede)


# ─────────────────────────────────────────────
# ETAPA 3: FFT / ESPECTRO
# ─────────────────────────────────────────────
def espectro(sinal, fs, janela=True):
    N = len(sinal)
    w = hann(N) if janela else np.ones(N)
    X = fft(sinal * w)
    freqs = fftfreq(N, d=1/fs)[:N//2]
    mag   = (2 / N) * np.abs(X[:N//2])
    return freqs, mag

freqs, mag_limpo    = espectro(ecg_limpo,   fs)
_,     mag_ruidoso  = espectro(ecg_ruidoso, fs)
_,     mag_filtrado = espectro(ecg_filtrado, fs)


# ─────────────────────────────────────────────
# ETAPA 4: IDENTIFICAÇÃO DE PICOS
# ─────────────────────────────────────────────
picos, _ = find_peaks(mag_limpo, prominence=0.005, distance=10)
print("\nPicos espectrais do ECG limpo:")
for p in picos[:10]:
    print(f"  f = {freqs[p]:.2f} Hz  |  |X| = {mag_limpo[p]:.4f}")


# ─────────────────────────────────────────────
# ETAPA 7: MÉTRICAS DE COMPARAÇÃO
# ─────────────────────────────────────────────
def metricas(original, processado, label=""):
    mse = np.mean((original - processado)**2)
    prd = 100 * np.sqrt(np.sum((original - processado)**2) / np.sum(original**2))
    snr = 10 * np.log10(np.sum(original**2) / np.sum((original - processado)**2))
    print(f"\n{label}")
    print(f"  MSE   = {mse:.6f}")
    print(f"  PRD   = {prd:.2f} %")
    print(f"  SNR   = {snr:.2f} dB")

metricas(ecg_limpo, ecg_ruidoso,  "ECG ruidoso vs. original")
metricas(ecg_limpo, ecg_filtrado, "ECG filtrado vs. original")


# ─────────────────────────────────────────────
# ETAPA 8: GRÁFICOS
# ─────────────────────────────────────────────
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Análise de ECG Sintético — Sinais e Sistemas', fontsize=14)

# Domínio do tempo
for ax, sig, titulo in zip(
    [axs[0,0], axs[1,0]],
    [ecg_limpo, ecg_ruidoso],
    ['ECG limpo', f'ECG com ruído (SNR ≈ {SNR_antes:.1f} dB)']
):
    ax.plot(t[:fs*3], sig[:fs*3], linewidth=0.8)
    ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Amplitude (mV)')
    ax.set_title(titulo); ax.grid(alpha=0.3)

axs[2,0].plot(t[:fs*3], ecg_limpo[:fs*3],    label='Original', linewidth=1.2)
axs[2,0].plot(t[:fs*3], ecg_filtrado[:fs*3], label='Filtrado',  linewidth=0.8, linestyle='--')
axs[2,0].set_xlabel('Tempo (s)'); axs[2,0].set_ylabel('Amplitude (mV)')
axs[2,0].set_title('Comparação: original vs. filtrado')
axs[2,0].legend(); axs[2,0].grid(alpha=0.3)

# Domínio da frequência
xlim = 150
for ax, mag, titulo in zip(
    [axs[0,1], axs[1,1], axs[2,1]],
    [mag_limpo, mag_ruidoso, mag_filtrado],
    ['Espectro — ECG limpo', 'Espectro — ECG com ruído', 'Espectro — ECG filtrado']
):
    ax.plot(freqs, mag, linewidth=0.8)
    ax.set_xlim([0, xlim])
    ax.set_xlabel('Frequência (Hz)'); ax.set_ylabel('|X(f)|')
    ax.set_title(titulo); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_analise_completa.png', dpi=150)
plt.show()

print("\nAnálise concluída. Gráfico salvo em 'ecg_analise_completa.png'.")