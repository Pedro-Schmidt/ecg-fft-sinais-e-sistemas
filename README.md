![Static Badge](https://img.shields.io/badge/Python-V3.12-blue)

# Analise de ECG com Fast Fourier Transform

Trabalho de Sinais e Sistemas sobre analise de um sinal ECG sintetico usando a Transformada de Fourier.

## O que o codigo faz

O script `main.py` gera um sinal ECG sintetico de 10 segundos, com frequencia de amostragem de 1000 Hz e frequencia cardiaca de 70 bpm.

Depois, um ruido aleatorio simples e adicionado ao sinal para simular uma situacao mais proxima de uma medicao real. Os dados sao salvos no arquivo `simulated_ecg.csv`, contendo o tempo, o ECG limpo e o ECG com ruido.

Em seguida, o codigo calcula a FFT do ECG com ruido, mostra o espectro de magnitude ate 100 Hz e aplica um filtro passa-faixa simples entre 0,5 Hz e 40 Hz no dominio da frequencia. Por fim, o sinal filtrado e reconstruido com a transformada inversa e comparado graficamente com o sinal ruidoso.

## Dependencias

- neurokit2
- numpy
- pandas
- matplotlib
