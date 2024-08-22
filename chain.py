from langchain_community.vectorstores import Chroma

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fastapi.responses import HTMLResponse, JSONResponse

import base64



#create load vector store
def load_vector_store(directory, embedding_model):
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=directory
    )
    return vectorstore

def response(model_name, vectorstore , prompt_template, question,age, learining_rate,communication_format,tone_style ,chat_history ):
    relevant_docs = vectorstore.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])

    qa_chain = LLMChain(llm=model_name,
                    prompt=PromptTemplate.from_template(prompt_template))

    result = qa_chain.run({'context': context, 'question': question,  'age': age, 'learning_rate': learining_rate, 'communication_format': communication_format, 'tone_style': tone_style, 'chat_history': chat_history})

    

    return result


def summarize_chat(model_name, prompt_template, question, answer):
    qa_chain = LLMChain(llm=model_name,
                    prompt=PromptTemplate.from_template(prompt_template))

    result = qa_chain.run({'human_question': question, 'ai_answer': answer,})

    return result
    

    
    