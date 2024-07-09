import os
from fastapi import FastAPI,HTTPException, Depends, status,Request , Form
from pydantic import BaseModel
from typing import Annotated
import models
from database import engine, SessionLocal
import logging
from twilio.rest import Client
from urllib.parse import parse_qs

from sqlalchemy.orm import Session
from sqlalchemy import asc , desc

from dotenv import load_dotenv
load_dotenv()

from twillio import send_message
from chain import response
from chain import load_vector_store
from chain import summarize_chat

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI



app = FastAPI()


# google api key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# vector data base path 
vector_database_path = 'chroma_db/'



# twilio api keys
TWILIO_ACCOUNT_SID=os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN=os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER=os.getenv("TWILIO_NUMBER")

account_sid = TWILIO_ACCOUNT_SID
auth_token = TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
twilio_number = TWILIO_NUMBER 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


models.Base.metadata.create_all(bind=engine)


#  Pydantic models

class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str
    phone_number: str
    learning_rate: str = "Active"
    age:  int = 10
    communication_format: str = "Textbook"
    tone_style: str = "Neutral"
    

class QueryBase(BaseModel):
    question: str = "q1"
    answer: str = "a1"
    user_id: int ="1"
    chache_chat_summary: str = "nllkllll"





# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




# load gemini pro model
def load_model(model_name):
  if model_name=="gemini-pro":
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key= GOOGLE_API_KEY)
  else:
    llm=ChatGoogleGenerativeAI(model="gemini-pro-vision" , google_api_key= GOOGLE_API_KEY)

  return llm


text_model = load_model("gemini-pro")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# load vector store
vectorstore = load_vector_store(directory=vector_database_path, embedding_model=embedding_model)

#prompt template
prompt_template = """
    You are an AI tutor. Adjust your response based on the following student profile, the chat history and the context.
    If the student want to change the tone style or communication format, you should adjust your response accordingly.
    Answer like a converation between a student and a tutor.If student say hi, or any greeting, you should respond accordingly.
    You can use the context to provide the answer to the question. If you dont have the answer in the context, you can give I dont know.
    Generate answers without exceed the curriculum content. Dont tell like in the context you provided.
    Dont use images in the answer and limit the answer to 1500 maximum characters.
    [profile]
    Age: {age}
    Learning rate: {learning_rate}
    Communication Format: {communication_format}
    Tone Style: {tone_style}
    previous chat history: {chat_history}

    [Context]
    Curriculum: RoboticGen Academy, Notes Content: {context},

    [student question]
    {question}

    [tutor response]
    """



history_summarize_prompt_template = """You are an assistant tasked with summarizing text for retrieval.
Summarize the student question and tutor answer in a concise manner.It should be a brief summary of the conversation.

student question: {human_question}
tutor answer: {ai_answer}

Summary:
"""





# load dependency
db_dependency = Annotated[Session, Depends(get_db)]


@app.post("/users/" , status_code=status.HTTP_201_CREATED)
async def create_user(user: UserBase, db:db_dependency):
    db_user  = models.User(**user.model_dump()) #data validation
    db.add(db_user)
    db.commit() 


@app.get("/users/{user_id}" , status_code=status.HTTP_200_OK)
async def read_user(user_id: int, db: db_dependency):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/phone_number/" , status_code=status.HTTP_200_OK)
async def read_user(phone_number: str, db: db_dependency):
    user = db.query(models.User).filter(models.User.phone_number == phone_number).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/queries/" , status_code=status.HTTP_201_CREATED)
async def create_query(query: QueryBase, db: db_dependency):
    db_query = models.Query(**query.model_dump())
    db.add(db_query)
    db.commit()



@app.get("/queries/" , status_code=status.HTTP_200_OK)
async def read_queries(db: db_dependency):
    queries = db.query(models.Query).order_by(models.Query.id.desc()).offset(0).limit(20).all()


    return queries
    


@app.get("/queries/{query_id}" , status_code=status.HTTP_200_OK)
async def read_query(query_id: int, db: db_dependency):
    query = db.query(models.Query).filter(models.Query.id == query_id).first()
    if query is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return query

@app.get("/queries/user_id/" , status_code=status.HTTP_200_OK)
async def read_query(user_id: int, db: db_dependency):
    query = db.query(models.Query).filter(models.Query.user_id == user_id)
    if query is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return query


@app.post("/query/llm/" , status_code=status.HTTP_200_OK)
async def query_llm(phone_number: str, query: str, db: db_dependency):
    user = db.query(models.User).filter(models.User.phone_number == phone_number).first()
    quiries = db.query(models.Query).filter(models.Query.user_id == user.id).order_by(models.Query.id.desc()).offset(0).limit(20).all()
    chat_history = ""
    for q in quiries:
        chat_history += q.chache_chat_summary + ","
    print(chat_history)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    chat_response = response(text_model, vectorstore, prompt_template, query, user.age, user.learning_rate, chat_history)
    summary = summarize_chat(text_model, history_summarize_prompt_template, query, chat_response)
    db_query = models.Query(question=query, answer=chat_response, user_id=user.id, chache_chat_summary=summary)
    db.add(db_query)
    db.commit()
    return chat_response



#wahtsapp message paths

async def check_user_exist(phone_number:str, db: db_dependency):

    user = db.query(models.User).filter(models.User.phone_number == phone_number).first()
    if user is None:
        return False
    return True








@app.post("/")
async def reply(question: Request,db: db_dependency):
    phone_number = parse_qs(await question.body())[b'WaId'][0].decode('utf-8')
    message_body = parse_qs(await question.body())[b'Body'][0].decode('utf-8')
    try:
        user = db.query(models.User).filter(models.User.phone_number == phone_number).first()
        
        if user is not None:
            quiries = db.query(models.Query).filter(models.Query.user_id == user.id).order_by(models.Query.id.desc()).offset(0).limit(20).all()
            chat_history = ""
            for q in quiries:
                chat_history += q.chache_chat_summary + ","
            print(chat_history)
            chat_response = response(text_model, vectorstore, prompt_template, message_body, user.age, user.learning_rate, user.communication_format, user.tone_style, chat_history)
            send_message(phone_number, chat_response)
            summary = summarize_chat(text_model, history_summarize_prompt_template, message_body, chat_response)
            db_query = models.Query(question=message_body, answer=chat_response, user_id=user.id, chache_chat_summary=summary)
            db.add(db_query)
            db.commit()



        else:
            chat_response = "Hello, I am a chatbot. You have not signed up yet."
            send_message(phone_number, chat_response)
    except:
        send_message(phone_number, "wait")
  
    # try:

    #     chat_response = "Hello, I am a chatbot. I am still learning. Please wait for a moment."
    #     send_message("+94722086410", chat_response)
    # except:
    #     send_message("+94722086410", "wait")








