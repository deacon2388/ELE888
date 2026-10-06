import csv
import numpy as np
from scipy.io.wavfile import write

# --- 1. Read the CSV File ---
csv_filename = "teleplot_2026-5-15_21-55.csv"  # Replace with your actual CSV file name
y_values = []

print(f"Reading data from {csv_filename}...")

with open(csv_filename, 'r') as file:
    reader = csv.reader(file)
    
    # Optional: Skip the header row if your CSV has column names (like "Time", "Value")
    next(reader, None) 
    
    for row in reader:
        try:
            # Assuming the data you want to hear is in the SECOND column (index 1)
            # If it's just a single column of numbers, change this to row[0]
            val = float(row[1]) 
            y_values.append(val)
        except (ValueError, IndexError):
            # Skips any empty rows or rows that don't contain numbers
            continue

y_data = np.array(y_values)

# --- 2. Setup Audio Parameters ---
sample_rate = 16000         # Standard CD-quality audio (samples per second)
duration_per_point = 0.05   # How long each CSV row plays (in seconds). 0.05s = fast sweep.
min_freq = 200              # Lowest pitch (Hz) for the smallest CSV number
max_freq = 800              # Highest pitch (Hz) for the largest CSV number

# --- 3. Map the CSV Data to Frequencies ---
print(f"Processing {len(y_data)} data points...")
# Normalize the data between 0.0 and 1.0, then scale to our frequency range
y_normalized = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
frequencies = min_freq + (y_normalized * (max_freq - min_freq))

# --- 4. Generate the Audio Waves ---
print("Generating audio waveforms...")
audio_wave = []
for freq in frequencies:
    # Generate a short sine wave for each specific frequency
    t = np.linspace(0, duration_per_point, int(sample_rate * duration_per_point), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    audio_wave.extend(wave)

# Convert the raw floating-point waves into 16-bit integers (the standard for WAV files)
audio_data = np.int16(np.array(audio_wave) * 32767)

# --- 5. Save the WAV File ---
wav_filename = "csv_output.wav"
write(wav_filename, sample_rate, audio_data)
print(f"Success! Saved '{wav_filename}'.")