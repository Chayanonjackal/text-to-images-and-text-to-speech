import argparse
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import random
import string
import soundfile as sf
import os

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the processor, model, and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load the embeddings dataset for speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Speaker ids from the embeddings dataset
speakers = {
    'awb': 0,     # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799   # US female (default)
}

def save_text_to_speech(text, speaker=None, sequence=1):
    # Preprocess text
    inputs = processor(text=text, return_tensors="pt").to(device)
    if speaker is not None:
        # Load xvector containing speaker's voice characteristics from a dataset
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
    else:
        # Random vector, meaning a random voice
        speaker_embeddings = torch.randn((1, 512)).to(device)
    
    # Generate speech with the models
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Create output directory if it doesn't exist
    output_dir = "./audios"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create an output filename
    if speaker is not None:
        output_filename = os.path.join(output_dir, f"{speaker}-part{sequence}.mp3")
    else:
        random_str = ''.join(random.sample(string.ascii_letters+string.digits, k=5))
        output_filename = os.path.join(output_dir, f"{random_str}-part{sequence}.mp3")
    
    # Save the generated speech to a file with 16KHz sampling rate
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    
    return output_filename

def split_text(text, max_length=580):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def main(text, speaker='slt'):
    # Check if the specified speaker is valid
    if speaker is not None and speaker not in speakers:
        print(f"Invalid speaker. Available speakers are: {', '.join(speakers.keys())}")
        return
    
    # Split the text into chunks
    text_chunks = split_text(text)
    
    # Generate speech and save to files for each chunk
    for i, chunk in enumerate(text_chunks):
        output_file = save_text_to_speech(chunk, speaker=speakers.get(speaker), sequence=i+1)
        print(f"Generated speech saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Text-to-Speech using SpeechT5")
    parser.add_argument('--text', type=str, required=True, help="Text to convert to speech or path to a text file")
    parser.add_argument('--speaker', type=str, default='slt', help="Speaker ID (optional, default='slt' for US female)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read text from file if the provided text is a file path
    if os.path.isfile(args.text):
        with open(args.text, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        text = args.text
    
    # Run the main function with the provided arguments
    main(text, args.speaker)
