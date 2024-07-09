from sqlalchemy import Boolean, Column, Integer, String, Text
from database import Base




class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(50), unique=True)
    password = Column(String(50))
    phone_number = Column(String(50))
    learning_rate = Column(String(50) , default="Active")
    age = Column(Integer, default=10)
    communication_format = Column(String(50), default="Textbook")
    tone_style = Column(String(50) , default="Neutral")
    chache_chat_summary = Column(Text , default="" )
    

class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    # title = Column(String(50))
    question = Column(Text)
    answer = Column(Text)

    user_id = Column(Integer)

    
    

