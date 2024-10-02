import os
from fastapi import FastAPI,HTTPException, Depends, status,Request , Form
from typing import Annotated, Optional
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import models
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dto import UserBase, UserLogin, Chatbox, Message, UserValidate, TokenData , ChatboxUpdateRequest # ---
from database import engine, SessionLocal
from passlib.context import CryptContext
import logging
from twilio.rest import Client
from urllib.parse import parse_qs

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

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
from fastapi.middleware.cors import CORSMiddleware
import jwt
from fastapi.middleware.cors import CORSMiddleware
import markdown2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # This allows all headers
)


# google api key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# vector data base path 
vector_database_path = os.getenv("VECTOR_DATABASE_PATH")


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

# def load_model():
#     llm = ChatOpenAI(model="gpt-4o")
#     return llm

# text_model = load_model("gemini-pro")
text_model =  ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=1500,
)



embedding_model =  OpenAIEmbeddings(model="text-embedding-3-small")

# load vector store
vectorstore = load_vector_store(directory=vector_database_path, embedding_model=embedding_model)

#prompt template
prompt_template = """
    You are an AI tutor. Adjust your response based on the following student profile, the chat history and the context.
    If the student want to change the tone style or communication format, you should adjust your response accordingly.
    Answer like a converation between a student and a teacher.If student say hi, or any greeting, you should respond accordingly.
    Strickly follow the curriculum content. Dont exceed the curriculum content.
    You can use the context to provide the answer to the question. If you dont have the answer in the context, you can give I dont know.
    Dont use images in the answer and Limit the answer to 1500 maximum characters.

    If the student ask for a website link or a youtube video link, you should provide the link to the student.

    If the student ask question from the chat history, you should provide the answer but dont give any answer outside the curriculum content.

    [profile]
    Age: {age} years
    Learning rate: {learning_rate}
    Communication Format: {communication_format}
    Tone Style: {tone_style}
    previous chat history: {chat_history}
    
    [Context]
    Curriculum: RoboticGen Academy, Notes Content: {context},

    [student question]
    {question}

    If you have no context, Tell the student that you dont know the answer and Dont give any references.
    If you have context,you should provide more sources like website links , youtube video links for the student to refer to.
    underline if you give any links.
   

    [tutor response]

    """

whatsapp_prompt_template = """
    You are an AI tutor. Adjust your response based on the following student profile, the chat history and the context.
    If the student want to change the tone style or communication format, you should adjust your response accordingly.
    Answer like a converation between a student and a tutor.If student say hi, or any greeting, you should respond accordingly.
    Strickly follow the curriculum content. Dont exceed the curriculum content.
    You can use the context to provide the answer to the question. If you dont have the answer in the context, you should give I dont know.
    Dont use images in the answer and Limit the answer to 800 maximum characters.

    If the student ask for a website link or a youtube video link, you should provide the link to the student only for roboticGen Accademy's curriculum.

    If the student ask question from the chat history, you should provide the answer but dont give any answer outside the curriculum content.


    [profile]
    Age: {age}
    Learning rate: {learning_rate}
    Communication Format: {communication_format}
    Tone Style: {tone_style}
    previous chat history: {chat_history}
    
    [Context]
    RoboticGen Academy's Curriculum Topics: Programming and Algorithms, Electronics and  Embedded Systems 
    Notes Content: {context},



    [student question]
    {question}

    If you have no context or out of the roboticGen academy's curriculum, Tell the student that you dont know the answer and Dont give any references.
    If you have context or out of the roboticGen academy's curriculum,you should provide more sources like website links , youtube video links for the student to refer to.
 


    [tutor response]


    """



history_summarize_prompt_template = """You are an assistant tasked with summarizing text for retrieval.
Summarize the student question and tutor answer in a concise manner.It should be a brief summary of the conversation.
But include the main points of the conversation. Add the student question in the summary.

student question: {human_question}
tutor answer: {ai_answer}

Summary:
"""





# load dependency
db_dependency = Annotated[Session, Depends(get_db)]




#============convert base 64 to image==================

# import base64
# from PIL import Image

# def convert_base64_to_image(base64_string):
#     base64_string = base64_string.split(",")[1]
#     imgdata = base64.b64decode(base64_string)
#     return imgdata











# ===================Whatsapp endpoints===========================




# ask question and get answer from the chatbot in the WHATSAPP 
@app.post("/api/whatsapp")
async def reply(question: Request,db: db_dependency):
    phone_number = parse_qs(await question.body())[b'WaId'][0].decode('utf-8')
    message_body = parse_qs(await question.body())[b'Body'][0].decode('utf-8')

    print(phone_number)
    print(message_body)

    local_phone_number = "0" + phone_number[2:]  

    print(local_phone_number)  

    try:
        user = db.query(models.User).filter(models.User.phone_number == local_phone_number).first()

        print("user", user)
        
        if user is not None:
            quiries = db.query(models.WhatsappSummary).filter(models.WhatsappSummary.user_id == user.id).order_by(models.WhatsappSummary.created_at.desc()).offset(0).limit(20).all()
            print("queries", quiries)
            chat_history = ""
            for q in quiries:
                chat_history += q.summary + ","
            print(chat_history)
            chat_response = response(text_model, vectorstore, whatsapp_prompt_template, message_body, user.age, user.learning_rate, user.communication_format, user.tone_style, chat_history)
            print("chat_response", chat_response)
            print("chat_response",type(chat_response) )
            send_message(phone_number, chat_response.get('result'))
            summary = summarize_chat(text_model, history_summarize_prompt_template, message_body, chat_response.get('result'))
            db_query = models.WhatsappSummary(summary=summary,user_id=user.id, phone_number=local_phone_number) 
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
@app.post("/api/login" , status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin, db: db_dependency):
    user_db = db.query(models.User).filter(models.User.email == user.email).first()
    
    if user_db is None or not pwd_context.verify(user.password, user_db.password):
        raise HTTPException(status_code=404, detail="Invalid Credentials")
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_jwt_token({"sub": user_db.id}, expires_delta=access_token_expires)
    
    return {
        "user_details": user_db, 
        "access_token": access_token, 
        "token_type": "bearer"
    }
    




# --TO DO--
# JWT token generation


# sign up
@app.post("/api/signup", status_code=status.HTTP_200_OK)
async def signup_user(user: UserBase, db: db_dependency):
   

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
    
    # Check if user already exists
    user_db = db.query(models.User).filter(models.User.email == user.email).first()
    if user_db is not None:
        raise HTTPException(status_code=411, detail="User already exists")
    
    
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
    access_token = create_jwt_token({"sub": user_id}, expires_delta=access_token_expires)
    
    return {
        "user_id": user_id, 
        "access_token": access_token, 
        "token_type": "bearer"
    }



#get user by user id
@app.get("/api/user", status_code=status.HTTP_200_OK)
async def get_user(db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return db_user

#update user tone_style , communication_format , learning_rate  by user id
@app.put("/api/user", status_code=status.HTTP_200_OK)
async def update_user(user_update: UserBase, db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        db_user.tone_style = user_update.tone_style
        db_user.communication_format = user_update.communication_format
        db_user.learning_rate = user_update.learning_rate
    
        db.commit()
        db.flush()
        db.refresh(db_user)
        
        return db_user
    



# create chatbox
@app.post("/api/chatbox", status_code=status.HTTP_200_OK)
async def create_chatbox(chatbox: Chatbox, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    print(user_id)
    
    #create chatbox
    newChatBox = Chatbox(chat_name=chatbox.chat_name, user_id=user_id)
    
    db_chatbox = models.Chatbox(**newChatBox.model_dump())  

    db.add(db_chatbox)
    db.commit()
    db.refresh(db_chatbox)
    
    chatbox_id = db_chatbox.id
    if chatbox_id is None:
        raise HTTPException(status_code=404, detail="Chatbox not created")
    return db_chatbox


# ask question and get answer from the chatbot in the WHATSAPP
# create message
@app.post("/api/chatbox/message", status_code=status.HTTP_200_OK)
async def create_message(message: Message, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")

    newMessage = Message(message=message.message, message_type=message.message_type, chatbox_id=message.chatbox_id, user_id=user_id)


    db_message = models.Message(**newMessage.model_dump())  

    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    message_id = db_message.id
    user_id = db_message.user_id
    chatbox_id = db_message.chatbox_id

    user = db.query(models.User).filter(models.User.id == user_id).first()
    print(user)

    # get chat summaries for the chat history accoriding to the user id & chatbox id
    chat_summaries = db.query(models.Summary).filter(
        models.Summary.user_id == user_id,
        models.Summary.chatbox_id == chatbox_id
    ).order_by(models.Summary.created_at.desc()).offset(0).limit(20).all()

    print(chat_summaries)

    chat_history = ""
    for c in chat_summaries:
        chat_history += c.summary + ","
    print(chat_history)

    try:
        chat_response = response(text_model, vectorstore, prompt_template, message.message, user.age, user.learning_rate, user.communication_format, user.tone_style, chat_history)
        summary = summarize_chat(text_model, history_summarize_prompt_template, message.message, chat_response.get('result'))
        db_query = models.Summary(summary=summary, user_id=user_id, chatbox_id=chatbox_id)
        db.add(db_query)
        db.commit()
    except:
        chat_response = {'result': "Sorry. At this moment, I am unable to give the answer. Please Try again later", 'relevant_images':[]}
        summary = "User question: " + message.message + " AI answer: " + chat_response.get('result')
        db_query = models.Summary(summary=summary, user_id=user_id, chatbox_id=chatbox_id)
        db.add(db_query)
        db.commit()

    related_images = ''
    for img in chat_response.get('relevant_images'):
        related_images += img + ","
    print(related_images)

    # add chat response to db
    db_message = models.Message(message=chat_response.get('result'), message_type="gpt", chatbox_id=chatbox_id, user_id=user_id , related_images= related_images)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)

    return JSONResponse({"message": "Message created successfully", "result": chat_response.get('result') , "relevant_images": related_images} )





# delete chatbox
@app.delete("/api/chatbox/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    
    if db_chatbox is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")

    db.delete(db_chatbox)
    db.commit()
    
    return {"detail": "Message deleted successfully"}



# delete user
@app.delete("/api/user", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message( db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")

    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.delete(db_user)
    db.commit()
    
    return {"detail": "User deleted successfully"}


# irrelavant
# get all messages by user id
@app.get("/api/messages/{chat_id}" , status_code=status.HTTP_200_OK)
async def read_message(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_messages = db.query(models.Message).filter(
        models.Message.chatbox_id == chat_id,
        models.Message.user_id == user_id
    ).all()
    if db_messages is None:
        raise HTTPException(status_code=404, detail="Messages not found")
    return db_messages



# get all chatboxes by user id
@app.get("/api/chatbox/user" , status_code=status.HTTP_200_OK)
async def get_chatboxes( db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")

    db_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.user_id == user_id).order_by(models.Chatbox.created_at.desc()).all()
    if db_chatboxes is None:
        raise HTTPException(status_code=404, detail="Chatboxes not found")
    return db_chatboxes

# get all chatboxes by chat id
@app.get("/api/chatbox/{chat_id}" , status_code=status.HTTP_200_OK)
async def get_chatboxes(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    db_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    if db_chatboxes is None:
        raise HTTPException(status_code=404, detail="Chatboxes not found")
    return db_chatboxes

#delete chatbox by chat id and user id
@app.delete("/api/chatbox/{chat_id}" , status_code=status.HTTP_200_OK)
async def delete_chatbox(chat_id: int, db: db_dependency, token: str = Depends(oauth2_scheme)):
        
        payload = decode_jwt_token(token)
        user_id = payload.get("sub")


        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or missing user ID")



        print(user_id , chat_id)
        db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id ,models.Chatbox.user_id == user_id ).first()
        
        if db_chatbox is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")
        
        db.delete(db_chatbox)
        db.commit()
        
        # return all chatboxes by user id
        all_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.user_id == user_id).all()
        
        return all_chatboxes


#update chatboxname by chat id
@app.put("/api/chatbox/{chat_id}" , status_code=status.HTTP_200_OK)
async def update_chatbox(chat_id: int, chatbox_update: ChatboxUpdateRequest, db: db_dependency, token: str = Depends(oauth2_scheme)):
    
    payload = decode_jwt_token(token)
    user_id = payload.get("sub")
    db_chatbox = db.query(models.Chatbox).filter(models.Chatbox.id == chat_id).first()
    
    if db_chatbox is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbox not found")
    
    db_chatbox.chat_name = chatbox_update.chat_name
    db_chatbox.user_id = user_id

    db.commit()
    db.flush()
    db.refresh(db_chatbox)

    # return all chatboxes by user id
    all_chatboxes = db.query(models.Chatbox).filter(models.Chatbox.user_id == db_chatbox.user_id).all()
    
    return all_chatboxes




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
