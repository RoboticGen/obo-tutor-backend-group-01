import os
from fastapi import FastAPI,HTTPException, Depends, status,Request , Form
from typing import Annotated, Optional
from pydantic import ValidationError
import models
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dto import UserBase, UserLogin, Chatbox, Message, UserValidate, TokenData # ---
from database import engine, SessionLocal
from passlib.context import CryptContext
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

from datetime import datetime, timedelta
import jwt

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

SECRET_KEY =os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes






logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


models.Base.metadata.create_all(bind=engine)


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)



# Define the OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# JWT token generation and decoding
def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now()  + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_jwt_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")




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






# ======================== API ENDPOINTS ========================


# --TO DO--
# password hashing and match databases hashed password
# login
@app.post("/login" , status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin, db: db_dependency):
    user_db = db.query(models.User).filter(models.User.email == user.email).first()
    
    if user_db is None or not pwd_context.verify(user.password, user_db.password):
        raise HTTPException(status_code=404, detail="Invalid Credentials")
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token({"sub": user_db.email}, expires_delta=access_token_expires)
    
    return {
        "user_details": user_db, 
        "access_token": access_token, 
        "token_type": "bearer"
    }
    




# --TO DO--
# JWT token generation


# sign in
@app.post("/signup", status_code=status.HTTP_200_OK)
async def signin_user(user: UserBase, db: db_dependency):
    # Custom validation checks
    try:
        #  Pydantic's validation to catch any issues
        valid_user = UserValidate(
            email=user.email, 
            password=user.password, 
            phone_number=user.phone_number)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e.errors()[0]['msg'])
        )
    
    user.password = hash_password(user.password)
    db_user = models.User(**user.model_dump()) 

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    user_id = db_user.id
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not created")
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token({"sub": user.email}, expires_delta=access_token_expires)
    
    return {
        "user_id": user_id, 
        "access_token": access_token, 
        "token_type": "bearer"
    }
    



# create chatbox
@app.post("/chatbox", status_code=status.HTTP_200_OK)
async def create_chatbox(chatbox: Chatbox, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    print(payload)
    db_chatbox = models.Chatbox(**chatbox.model_dump())  

    db.add(db_chatbox)
    db.commit()
    db.refresh(db_chatbox)
    
    chatbox_id = db_chatbox.id
    if chatbox_id is None:
        raise HTTPException(status_code=404, detail="Chatbox not created")
    return chatbox_id  



# create message
@app.post("/chatbox/message", status_code=status.HTTP_200_OK)
async def create_message(message: Message, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_message = models.User(**message.model_dump())  

    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    message_id = db_message.id
    if message_id is None:
        raise HTTPException(status_code=404, detail="Message not created")
    return message_id  



# delete chatbox
@app.delete("/chatbox/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    
    if db_chatbox is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")

    db.delete(db_chatbox)
    db.commit()
    
    return {"detail": "Message deleted successfully"}



# delete user
@app.delete("/user/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(user_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.delete(db_user)
    db.commit()
    
    return {"detail": "User deleted successfully"}



# get all messages by user id
@app.get("/message/{user_id}" , status_code=status.HTTP_200_OK)
async def read_message(user_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_messages = db.query(models.Message).filter(models.Message.user_id == user_id)
    if db_messages is None:
        raise HTTPException(status_code=404, detail="Messages not found")
    return db_messages



# from fastapi.testclient import TestClient
# from your_app import app  # Import your FastAPI app instance
# from .schemas import UserBase

# client = TestClient(app)

# def call_create_user(user_data: dict):
#     response = client.post("/users/", json=user_data)
#     if response.status_code == 201:
#         return response.json()
#     else:
#         return {"error": response.text}
