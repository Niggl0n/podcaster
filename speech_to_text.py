import os
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import load_dataset, Audio, Dataset
import sounddevice as sd
from pydub import AudioSegment
import librosa
import numpy as np
from datasets import Dataset

# Function to load an audio file
def load_audio_file(file_path):
    return librosa.load(file_path, sr=None)

def save_to_mp3(example_snippet, export_path, sr):
    example_snippet = np.array(example_snippet)
    example_snippet = example_snippet.astype(np.float32)
    audio_segment = AudioSegment(
        example_snippet.tobytes(),
        frame_rate=sr,  # Adjust according to your data
        sample_width=4,
        channels=1,
    )
    audio_segment.export(export_path, format="mp3")


def segment_audio(file_path, segment_length_seconds, target_sr=16000):
    audio_data, original_sr = librosa.load(file_path, sr=None)
    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    segment_length_samples = segment_length_seconds * target_sr
    num_segments = int(len(audio_data) // segment_length_samples)
    segments = [audio_data[i * segment_length_samples: (i + 1) * segment_length_samples] for i in range(num_segments)]
    # todo: include overlap of 1 second
    return segments, target_sr


def transcribe_s2t_small(audio_snippet, sr):
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
    inputs = processor(audio_snippet, sampling_rate=sr, return_tensors="pt")
    generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription


def transcribe_whisper(audio_snippet, sr, model_str="whisper-base"):
    model = WhisperForConditionalGeneration.from_pretrained("openai/"+model_str)
    model.config.forced_decoder_ids = None
    processor = WhisperProcessor.from_pretrained("openai/"+model_str)
    inputs = processor(audio_snippet, sampling_rate=sr, return_tensors="pt")
    generated_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription

def create_audio_dataset(file_path, segment_length_seconds=30):
    # create audio dataset based on list of audio segments
    all_segments = []
    segments, sr = segment_audio(file_path, segment_length_seconds)
    all_segments.extend(segments)
    audio_dataset = Dataset.from_dict({"audio": all_segments})
    return audio_dataset


def create_podcast_transcription(audio_dataset, sr=16000, break_after_n_snippets=np.inf):
    text_snippets = []
    for i, snippet in enumerate(audio_dataset):
        print(i)
        audio_snippet = snippet["audio"]
        # transcription_s2t = transcribe_s2t_small(audio_dataset[0]["audio"], sr)
        transcription_whisper = transcribe_whisper(audio_snippet, sr, model_str="whisper-base")
        text_snippets.append(transcription_whisper[0])
        if i >= break_after_n_snippets:
            break
    podcast_transcript = ' '.join(text_snippets)
    return podcast_transcript


file_paths = ["data/audio/glt1014526399.mp3"]
segment_length_seconds = 1 * 30
sr = 16000
for file_path in file_paths:
    print(file_path)
    filename = os.path.basename(file_path).split('.')[0]
    audio_dataset = create_audio_dataset(file_path, segment_length_seconds=segment_length_seconds)
    podcast_transcript = create_podcast_transcription(audio_dataset, sr=sr, break_after_n_snippets=2)
    write_file_path = f"data/transcriptions/{filename}.txt"
    print(f"Save Transcript to: {write_file_path}")
    with open(write_file_path, 'w') as file:
        file.write(podcast_transcript)


# play audio
# for i in range(5):
#     sd.play(audio_dataset[i]["audio"], samplerate=sr)  # You might need to adjust the samplerate depending on your audio data
#     sd.wait()




print("Finished")