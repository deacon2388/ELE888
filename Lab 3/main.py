import serial
import time
import numpy as np
from scipy.io import wavfile

# --- Configuration ---
COM_PORT = 'COM5'          # Updated to COM5
BAUD_RATE = 921600         # Must match the ESP32 code
SAMPLE_RATE = 16000
RECORD_SECONDS = 5         # How many seconds of audio do you want to capture?
OUTPUT_FILE = "analog_dma_recording.wav"

def record_audio():
    print(f"Connecting to {COM_PORT} at {BAUD_RATE} baud...")

    try:
        # timeout=1 ensures the script doesn't hang forever if the connection drops
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Error opening port! Ensure the Arduino IDE/VS Code Serial Monitor is strictly CLOSED.")
        print(f"Details: {e}")
        return

    print("ESP32 Connected. Waiting 2 seconds for the board to reboot...")
    time.sleep(2) 

    # TRASH the bootloader text that was just sent over the line
    ser.reset_input_buffer() 

    total_bytes_to_read = SAMPLE_RATE * 2 * RECORD_SECONDS
    print(f"\n[ RECORDING ] Speak into the mic for {RECORD_SECONDS} seconds...")

    # Read the data dynamically so we don't freeze the system
    raw_bytes = bytearray()
    start_time = time.time()
    
    # Give it a tiny bit of buffer time (RECORD_SECONDS + 2) in case the USB lags
    while len(raw_bytes) < total_bytes_to_read and (time.time() - start_time) < (RECORD_SECONDS + 2):
        chunk = ser.read(total_bytes_to_read - len(raw_bytes))
        if chunk:
            raw_bytes.extend(chunk)

    ser.close()

    print(f"\n[ PROCESSING ] Captured {len(raw_bytes)} bytes. Formatting audio...")

    if len(raw_bytes) < 100:
        print("Error: Barely any data was received. Is the ESP32 actually running the C++ code?")
        return

    # Byte Alignment Fix: 16-bit audio requires an even number of bytes. 
    # If the USB dropped a single byte, this prevents numpy from crashing.
    if len(raw_bytes) % 2 != 0:
        raw_bytes = raw_bytes[:-1]

    # 1. Convert bytes to 16-bit integers
    raw_data = np.frombuffer(raw_bytes, dtype=np.uint16)

    # 2. Remove the DC Offset (Center the waveform at 0)
    centered_data = np.float32(raw_data) - 2048.0

    # 3. Scale the volume
    scaled_data = centered_data * 16.0
    scaled_data = np.clip(scaled_data, -32768, 32767)

    # 4. Convert back to standard 16-bit audio format
    audio_data = np.int16(scaled_data)

    # 5. Save the WAV file
    wavfile.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)

    print(f"Success! Perfect audio saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    record_audio()