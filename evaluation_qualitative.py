from llama_index.llms.ollama import Ollama
from llama_index.packs.mixture_of_agents import MixtureOfAgentsPack
import nest_asyncio
import pandas as pd
nest_asyncio.apply()

mixture_of_agents_pack = MixtureOfAgentsPack(
    llm=Ollama(model="medllama2", request_timeout=1200.0),  # Aggregator
    reference_llms=[
        Ollama(model="llama3.1", request_timeout=1200.0),
        Ollama(model="mistral", request_timeout=1200.0),
    ],  # Proposers
    num_layers=3,
    temperature=0.1
)

df = pd.read_csv('reports.csv')

feedbacks = []
for idx, row in df.iterrows():
    prompt = f"""
        Provide detailed qualitative feedback on the following report compared to the ground truth reading. Evaluate the report based on the following criteria: 
        1. **Semantic Similarity**: How closely does the content of the report align with the ground truth in meaning and context?
        2. **Accuracy**: Is the information in the report factually correct and precise?
        3. **Correctness**: Does the report adhere to the correct terminology and standard practices?
        4. **Clinical Usefulness**: How useful is the report for clinical decision-making or practical application?
        5. **Consistency**: Is the report internally consistent, and does it maintain consistency with the ground truth?

        Please provide detailed feedback on each of these criteria in separate paragraphs, focusing on both strengths and areas for improvement. Do not include any numerical scores. 

        Ground Truth: {row['GroundTruth']}
        Report: {row['Report']}
    """
    resp = mixture_of_agents_pack.run(prompt)
    feedbacks.append(resp)
df['Feedback'] = feedbacks

df.to_csv('reports_with_feedback.csv', index=False)

