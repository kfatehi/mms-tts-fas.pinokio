from transformers import VitsModel, AutoTokenizer
import torch
import sounddevice as sd
import scipy.io.wavfile

# Load the model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-fas")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")

# Function to play audio
def play_audio(audio, sample_rate):
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

# Function to save audio to a file
def save_audio_to_file(audio, sample_rate, filename):
    scipy.io.wavfile.write(filename, rate=sample_rate, data=audio)

# Loop to continuously accept input
try:
    counter = 0
    while True:
        # Accept user input
        text = input("Enter text (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break

        try:
            # Tokenize and generate waveform
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
                audio = output.squeeze().numpy()
        except RuntimeError:
            continue

        # Get the sample rate from the model configuration
        sample_rate = model.config.sampling_rate

        # Play the audio
        play_audio(audio, sample_rate)

        # Save the audio to a file
        filename = f"output_{counter}.wav"
        save_audio_to_file(audio, sample_rate, filename)
        print(f"Audio saved to {filename}")
        counter += 1

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
