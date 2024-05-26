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
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
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
    parser.add_argument('--model', type=str, default="CompVis/stable-diffusion-v1-4", help="Path to the model directory")
    
    args = parser.parse_args()
    main(args.text, args.output, args.model)
