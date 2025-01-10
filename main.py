from cbm_inference_app import get_image_concept_similarity_vector
from multi_agent_rag import generate_report as generate_report_multi
from single_agent_rag import generate_report as generate_report_single
from gpt4 import generate_report as generate_report_gpt4
from config import W_F, num_classes, class_mapping, num_concepts
from concepts import pneumonia_concepts, covid19_concepts, normal_concepts, concepts
import torch 
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description="Select an agent type.")

parser.add_argument(
    "--agent",
    choices=["multi", "single", "gpt4"],
    default="multi",
    help="Choose the agent type: 'multi', 'single', or 'gpt4'. Default is 'multi'.",
)

args = parser.parse_args()

img_path = 'test.jpg'

# Get the image concept similarity vector
r = get_image_concept_similarity_vector(img_path)

# Predict the class of the image
predicted_class = torch.argmax(torch.nn.functional.softmax(torch.matmul(r, W_F.T), dim=0)).item()

# Concept Contribution
contribution = W_F * r
contribution = contribution.detach().numpy()
conc, cont = shuffle(concepts, contribution[predicted_class])
paired_lists = list(zip(conc, cont))
sorted_paired_lists = sorted(paired_lists, key=lambda x: abs(x[1]), reverse=True)
sorted_concepts, sorted_contributions = zip(*sorted_paired_lists)
sorted_concepts = list(sorted_concepts)
sorted_contributions = list(sorted_contributions)

# Generate the report
if args.agent == "multi":
    report, logs = generate_report_multi(class_mapping, predicted_class, sorted_concepts, sorted_contributions)
elif args.agent == "single":
    report = generate_report_single(class_mapping, predicted_class, sorted_concepts, sorted_contributions)
elif args.agent == "gpt4":
    report = generate_report_gpt4(class_mapping, predicted_class, sorted_concepts, sorted_contributions)
