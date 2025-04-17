
# Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG

This is the official repository for the paper titled "Towards Interpretable Radiology Report Generation via Concept Bottlenecks using a Multi-Agentic RAG" accepted in the 47th European Conference for Information Retrieval (ECIR) 2025 

Publication Link: https://link.springer.com/chapter/10.1007/978-3-031-88714-7_18

Preprint: https://arxiv.org/abs/2412.16086

Contact: hasan.alam@dfki.de

The clinical documentation used for RAG can be downloaded from [here](https://drive.google.com/file/d/1w8aGQQoa6UKqDO260A-5-bMUuxYZ0F5D/view?usp=sharing).

## Overview

Deep learning models have advanced medical image analysis, particularly in Chest X-ray (CXR) interpretation. However, interpretability challenges hinder their clinical adoption. This study proposes a novel framework that integrates:

1. **Concept Bottleneck Models (CBMs):** Enhancing interpretability by modeling relationships between visual features and clinical concepts.
2. **Multi-Agent Retrieval-Augmented Generation (RAG):** Providing robust and clinically relevant radiology report generation.

Our framework bridges the gap between high-performing AI and the explainability required for clinical use.


## Key Features

1. **Interpretable Classification:**
   - Utilizes CBMs for CXR image classification by explicitly associating predictions with human-interpretable clinical concepts.
   - Enables concept-level interventions to correct misclassified samples.

2. **Multi-Agentic Report Generation:**
   - Employs a multi-agent RAG system with specialized agents for disease-specific retrieval and explanation.
   - Generates clinically relevant and consistent radiology reports.

## Results

### Classification Performance
| Model                     | Accuracy | Interpretability |
|---------------------------|----------|------------------|
| CLIP                      | 0.47     | No               |
| Bio-VIL                   | 0.78     | No               |
| Label-free CBM            | 0.72     | Yes              |
| Robust CBM                | 0.78     | Yes              |
| **Ours**                  | **0.81** | Yes              |

### Report Generation Evaluation
Reports were evaluated using LLM-based metrics for Semantic Similarity, Accuracy, Clinical Usefulness, and Consistency.

#### Evaluation of Report Generation Approaches using LLM as Judge

| **Model**            | **Semantic Similarity (GPT4)** | **Semantic Similarity (Single Agent)** | **Semantic Similarity (Multi-Agent)** | **Accuracy (GPT4)** | **Accuracy (Single Agent)** | **Accuracy (Multi-Agent)** | **Correctness (GPT4)** | **Correctness (Single Agent)** | **Correctness (Multi-Agent)** | **Clinical Usefulness (GPT4)** | **Clinical Usefulness (Single Agent)** | **Clinical Usefulness (Multi-Agent)** | **Consistency (GPT4)** | **Consistency (Single Agent)** | **Consistency (Multi-Agent)** |
|-----------------------|--------------------------------|-----------------------------------------|---------------------------------------|---------------------|----------------------------|----------------------------|------------------------|-------------------------------|-------------------------------|--------------------------------|----------------------------------------|----------------------------------------|-------------------------|--------------------------------|--------------------------------|
| **Llama 3.1 8B**      | 0.84                          | 0.82                                    | 0.84                                 | 0.91                | 0.87                       | 0.91                       | 0.92                   | 0.88                          | 0.91                          | 0.89                           | 0.85                                   | 0.84                                   | 0.93                    | 0.85                           | 0.89                           |
| **Mistral 7B**        | 0.79                          | 0.88                                    | 0.89                                 | 0.84                | 0.88                       | 0.94                       | 0.85                   | 0.85                          | 0.95                          | 0.88                           | 0.92                                   | 0.96                                   | 0.86                    | 0.88                           | 0.96                           |
| **Gemma 2 9B**        | 0.77                          | 0.79                                    | 0.85                                 | 0.80                | 0.80                       | 0.82                       | 0.81                   | 0.83                          | 0.87                          | 0.69                           | 0.67                                   | 0.78                                   | 0.76                    | 0.77                           | 0.83                           |
| **LLaVA 9B**          | 0.78                          | 0.80                                    | 0.80                                 | 0.83                | 0.87                       | 0.89                       | 0.86                   | 0.86                          | 0.91                          | 0.78                           | 0.82                                   | 0.86                                   | 0.80                    | 0.83                           | 0.89                           |
| **GPT 3.5 Turbo**     | 0.79                          | 0.75                                    | 0.82                                 | 0.84                | 0.78                       | 0.86                       | 0.86                   | 0.79                          | 0.88                          | 0.81                           | 0.75                                   | 0.86                                   | 0.84                    | 0.76                           | 0.88                           |
| **Average**           | 0.79                          | 0.80                                    | 0.84                                 | 0.84                | 0.84                       | 0.88                       | 0.86                   | 0.84                          | 0.90                          | 0.81                           | 0.80                                   | 0.86                                   | 0.84                    | 0.81                           | 0.89                           |


#### Clustering Evaluation for Report Generation Approaches

| **Metric**           | **GPT4** | **Single Agent** | **Multi-Agent** |
|-----------------------|----------|------------------|-----------------|
| **Silhouette**        | 0.37     | 0.41             | 0.27            |
| **Davies-Bouldin**    | 1.11     | 0.96             | 1.44            |
| **Calinski-Harabasz** | 69.94    | 93.99            | 44.78           |
| **Dunn**              | 0.54     | 0.73             | 0.36            |


---

## Citation
If you use this code or results in your research, please cite:

```
@inproceedings{10.1007/978-3-031-88714-7_18,
author = {Alam, Hasan Md Tusfiqur and Srivastav, Devansh and Kadir, Md Abdul and Sonntag, Daniel},
title = {Towards Interpretable Radiology Report Generation via&nbsp;Concept Bottlenecks Using a&nbsp;Multi-agentic RAG},
year = {2025},
isbn = {978-3-031-88713-0},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-88714-7_18},
doi = {10.1007/978-3-031-88714-7_18},
abstract = {Deep learning has advanced medical image classification, but interpretability challenges hinder its clinical adoption. This study enhances interpretability in Chest X-ray (CXR) classification by using concept bottleneck models (CBMs) and a multi-agent Retrieval-Augmented Generation (RAG) system for report generation. By modeling relationships between visual features and clinical concepts, we create interpretable concept vectors that guide a multi-agent RAG system to generate radiology reports, enhancing clinical relevance, explainability, and transparency. Evaluation of the generated reports using an LLM-as-a-judge confirmed the interpretability and clinical utility of our model’s outputs. On the COVID-QU dataset, our model achieved 81\% classification accuracy and demonstrated robust report generation performance, with five key metrics ranging between 84\% and 90\%. This interpretable multi-agent framework bridges the gap between high-performance AI and the explainability required for reliable AI-driven CXR analysis in clinical settings. Our code will be released at},
booktitle = {Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part III},
pages = {201–209},
numpages = {9},
keywords = {Interpretable Radiology Report Generation, Concept Bottleneck Models, Multi-Agent RAG, Explainable AI, LLMs, VLMs},
location = {Lucca, Italy}
}
```

## License
This repository is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements
- **[CheXAgent](https://huggingface.co/StanfordAIMI/CheXagent-8b)**: Used for extracting CXR image embeddings.
- **[DragonflyMED](https://huggingface.co/togethercomputer/Llama-3-8B-Dragonfly-Med-v1)**: Used for generating ground-truth readings.

- This work is funded by the Federal Ministry of Education, Science, Research and Technology (BMBF), Germany, under grant number 01IW23002 (No-IDLE).
