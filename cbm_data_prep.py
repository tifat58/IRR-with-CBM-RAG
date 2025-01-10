from concepts import concepts
from modeling_chexagent import CheXagentVisionEmbeddings
from configuration_chexagent import CheXagentVisionConfig
from PIL import Image
from torchvision import transforms
from mistralai.client import MistralClient
import os
import torch
import pickle
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=MISTRAL_API_KEY)

embeddings_batch_response = client.embeddings(
    model="mistral-embed",
    input= concepts,
)

t = []
for concept_embedding in embeddings_batch_response.data:
    t.append(torch.tensor(concept_embedding.embedding))

splits = ['Train', 'Test', 'Val']
classes = ['Non-COVID', 'COVID-19', 'Normal']

output_dirs = {
    'Train': 'similarities/train',
    'Test': 'similarities/test',
    'Val': 'similarities/val'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
])

# Vision Model
config = CheXagentVisionConfig(
    hidden_size=1024,
    intermediate_size=6144,
    num_hidden_layers=39,
    num_attention_heads=16,
    image_size=224,
    patch_size=14,
    hidden_act="gelu",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
    initializer_range=1e-10,
    qkv_bias=True
)

vision_model = CheXagentVisionEmbeddings(config)

count = 0
for split in splits:
    for cls in classes:
        dir_path = f'Lung Segmentation Data/Lung Segmentation Data/{split}/{cls}/images'
        label = cls.lower().replace('-', '_')
        for file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, file)
            count += 1
            I = Image.open(img_path).convert("RGB")
            I = transform(I).unsqueeze(0)
            V = vision_model(I)
            V = V.squeeze(0)
            e = []
            for concept in t: 
                s = torch.nn.functional.cosine_similarity(V, concept, dim=1)
                e.append(max(s))
            output_path = os.path.join(output_dirs[split], f'{count}.obj')
            data = {"image_path": img_path, "label": label, "e": e}
            with open(output_path, 'wb') as datafile:
                pickle.dump(data, datafile)