import argparse
import torch
from diffusers import StableDiffusionPipeline
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def read_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt

def split_text(prompt, max_length=500):
    return [prompt[i:i + max_length] for i in range(0, len(prompt), max_length)]

def generate_image(prompt, output_path="generated_image.png", model_path="CompVis/stable-diffusion-v1-4"):
    # Determine if we are using a safetensors file based on file extension
    use_safetensors = model_path.endswith(".safetensors")

    if use_safetensors:
        # If model_path is a safetensors file, we assume the directory contains the necessary configuration files
        model_dir = os.path.dirname(model_path)
        pipe = StableDiffusionPipeline.from_single_file(
            model_dir,
            # "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
            # "https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_6.2_BakedVae_pruned.safetensors"
            # torch_dtype=torch.float16,
            # use_safetensors=use_safetensors
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        )
    
    pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")
    print(f"Device using cuda: {torch.cuda.is_available()}")
    
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Generated image saved to {output_path}")

def main(prompt, output_path, model_path):
    output_dir = "./images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(prompt):
        prompt = read_prompt_from_file(prompt)
    
    prompt_chunks = split_text(prompt)
    
    for i, chunk in enumerate(prompt_chunks):
        chunk_output_path = os.path.join(output_dir, output_path.replace(".png", f"_{i+1}.png"))
        generate_image(chunk, chunk_output_path, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt using Stable Diffusion")
    parser.add_argument('--text', type=str, required=True, help="Text prompt or path to a text file")
    parser.add_argument('--output', type=str, default="generated_image.png", help="Output image file path")
    parser.add_argument('--model', type=str, default="runwayml/stable-diffusion-v1-5", help="Path to the model directory or safetensors file")
    
    args = parser.parse_args()
    main(args.text, args.output, args.model)
