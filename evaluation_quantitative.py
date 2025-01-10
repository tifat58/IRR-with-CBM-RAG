import pandas as pd
import ollama
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

llm = OpenAI(api_key=OPENAI_API_KEY)

df = pd.read_csv('reports.csv')

report = df['Report'].values.tolist()
gt = df['GroundTruth'].values.tolist()

llama_res = []
mistral_res = []
llava_res = []
gemma_res = []
gpt_res = []

for i in range(len(gt)):
    prompt = f"""
                Output a number between 0 and 1 describing the semantic similiarity, accuracy, correctness, clinical usefulness, and consistency 
                between the following report and the ground truth reading: please output only the number for each of the five metrics formatted in json without any explaination or other text.
                Ground Truth: {gt[i]}
                Report: {report[i]}
            """
    res = ollama.generate(model='llama3.1', prompt=prompt)
    llama_res.append(res['response'])
    res = ollama.generate(model='mistral', prompt=prompt)
    mistral_res.append(res['response'])
    res = ollama.generate(model='llava', prompt=prompt)
    llava_res.append(res['response'])
    res = ollama.generate(model='gemma2', prompt=prompt)
    gemma_res.append(res['response'])
    res = llm.invoke(prompt)
    gpt_res.append(res)