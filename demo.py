#from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.2", temparature=0.7)
#Test the LLM with a simple prompt
response = llm.invoke("What is today's date?")
print(response) 