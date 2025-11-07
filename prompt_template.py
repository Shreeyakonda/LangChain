from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2")
template = "You are an expert in {topic}. Answer the following question: {question}"
prompt = PromptTemplate( input_variables=[ "topic", "question" ], template=template )

chain = prompt | llm
response = chain.invoke({ "topic": "English", "question": "Give me a story" })
print(response)