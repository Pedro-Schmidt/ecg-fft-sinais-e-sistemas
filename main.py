import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt


# 1. Generate a synthetic ECG signal (10 seconds, 500Hz, 70 BPM)
# The "ecgsyn" method provides a more realistic waveform
ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=1000, heart_rate=70, method="ecgsyn")

# 2. Convert the signal to a Pandas DataFrame for easy saving
df = pd.DataFrame({"ECG": ecg_signal})

# 3. Save the data to a CSV file
df.to_csv("simulated_ecg.csv", index=False)

print("ECG signal generated and saved to 'simulated_ecg.csv'")

nk.signal_plot(ecg_signal, sampling_rate=1000)
plt.show()
