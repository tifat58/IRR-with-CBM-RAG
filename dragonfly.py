import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from huggingface_hub import login
from dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
from dragonfly.models.processing_dragonfly import DragonflyProcessor
from pipeline.train.train_utils import random_seed
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


def format_text(text, system_prompt=""):
    instruction = f"{system_prompt} {text}" if system_prompt else text
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n" f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt


torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

pretrained_model_name_or_path = "togethercomputer/Llama-3-8B-Dragonfly-Med-v1"
image_dir = "xray_images/"


# parameters
device = "cuda:1"
seed = 42
temperature = 0


def main():
    random_seed(seed)
    login(token=HUGGING_FACE_TOKEN)
    print(f"Loading pretrained model from {pretrained_model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = clip_processor.image_processor
    processor = DragonflyProcessor(image_processor=image_processor, tokenizer=tokenizer, image_encoding_style="llava-hd")

    model = DragonflyForCausalLM.from_pretrained(pretrained_model_name_or_path)
    model = model.to(torch.bfloat16)
    model = model.to(device)

    responses = []

    df = pd.read_csv('instances.csv')

    for index, row in df.iterrows():
        print(f"Processing image {index + 1} of {len(df)}")
        image_path = image_dir + row['Image']

        image = Image.open(image_path)
        image = image.convert("RGB")
        images = [image]

        question = f"""
            Generate a detailed medical report by analysing the image and using the information below.
            Disease: None
            The disease was detected based on the analysis of chest x-ray images. The concepts and their corresponding contributions to the classification are as follows:
            {row['Contributions']}
        """

        text_prompt = format_text(question)

        inputs = processor(text=[text_prompt], images=images, max_length=2048, return_tensors="pt", is_generate=True)
        inputs = inputs.to(device)

        with torch.inference_mode():
            generation_output = model.generate(**inputs, max_new_tokens=1024, eos_token_id=tokenizer.encode("<|eot_id|>"), do_sample=temperature > 0, temperature=temperature, use_cache=True)

        generation_text = processor.batch_decode(generation_output, skip_special_tokens=False)
        response = generation_text[0].replace("<|reserved_special_token_0|>", "").replace("<|reserved_special_token_1|>", "")
        responses.append(response)
    
    df['GroundTruth'] = responses
    df.to_csv('gt.csv', index=False)


if __name__ == "__main__":
    main()
