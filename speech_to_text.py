import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset, Audio, Dataset
import sounddevice as sd
from pydub import AudioSegment
import librosa
import numpy as np
from datasets import Dataset

# Function to load an audio file
def load_audio_file(file_path):
    return librosa.load(file_path, sr=None)


def segment_audio(file_path, segment_length_seconds, target_sr=16000):
    # Load the audio file
    audio_data, original_sr = librosa.load(file_path, sr=None)
    # Resample the audio to the target sample rate
    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    # Calculate the segment length in samples
    segment_length_samples = segment_length_seconds * target_sr
    # Determine the number of segments
    num_segments = int(len(audio_data) // segment_length_samples)
    # Extract segments
    segments = [audio_data[i * segment_length_samples : (i + 1) * segment_length_samples] for i in range(num_segments)]
    return segments, target_sr


segment_length_seconds = 1 * 60
# List to hold all segments
all_segments = []
# File paths
file_paths = ["data/audio/glt1014526399.mp3"]
# Iterate over file paths, load and segment audio, and append to all_segments
for file_path in file_paths:
    segments, sr = segment_audio(file_path, segment_length_seconds)
    all_segments.extend(segments)
# Create the dataset
audio_dataset = Dataset.from_dict({"audio": all_segments})

#sd.play(audio_dataset[0]["audio"], samplerate=16000)  # You might need to adjust the samplerate depending on your audio data
#sd.wait()

example_snippet = np.array(audio_dataset[0]["audio"])
example_snippet = example_snippet.astype(np.float32)
audio_segment = AudioSegment(
    example_snippet.tobytes(),
    frame_rate=sr,  # Adjust according to your data
    sample_width=4,
    channels=1,
)
# Export as MP3
audio_segment.export("data/audio/example_ile.mp3", format="mp3")


model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


inputs = processor(audio_dataset[0]["audio"], sampling_rate=sr, return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)



#dataset = load_dataset("audiofolder", data_files=["data/audio/glt1024839827.mp3"]).cast_column("audio", Audio())
#audio_dataset = Dataset.from_dict({"audio": ["data/audio/glt1014526399.mp3"]}).cast_column("audio", Audio(decode=False))
#ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
#dataset_all = load_dataset("audiofolder", data_dir="data/audio")

