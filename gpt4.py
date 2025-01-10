from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4')
  
def generate_report(class_mapping, predicted_class, concepts, contribution):
    prompt = f"""
        Generate a detailed medical report based on the below information and knowledge from the documents.

        Disease: {class_mapping[predicted_class]}

        The disease was detected based on the analysis of chest x-ray images. The concepts and their corresponding contributions to the classification are as follows:

        Concepts: {concepts}
        Contributions: {contribution}

        The higher values of contributions indicate the importance of the concepts in the classification of the disease. Negative values of contribution means the concept does not contribute for the classification.
        The disease has been detected based on the presence of these concepts.

        For the report, focus on the concepts that have the highest contributions to the classification of the disease. Discuss the top 10 contributing concepts and their relevance to the disease, and where are they found.
        Also discuss the relevance of these concepts in the diagnosis and treatment of the disease.

        The findings based on the concepts should be explained in detail in medical terminology.

        In the report, also provide a brief summary of the disease, including its symptoms, causes, medical diagnosis, treatment options, and prognosis. Make sure to write the report in proper clinical and medical terms and then explain all medical terms in a way that a person with no medical background can understand. Also, include information about any relevant clinical trials or research studies, and any potential risks or complications associated with the disease.

        Finally, provide a section on prevention and self-care measures that a person with the disease can take to improve their health and well-being.
    """
    
    report = llm.invoke(prompt)
    return report