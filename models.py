from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from zoneinfo import ZoneInfo
from database import Base
# from enum import Enum

from dto import Role




#    database models

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(50), unique=True)
    password = Column(String(255))
    phone_number = Column(String(50))
    learning_rate = Column(String(50) , default="Active")
    role = Column(Enum(Role))
    age = Column(Integer, default=10)
    communication_format = Column(String(50), default="Textbook")
    tone_style = Column(String(50) , default="Neutral")
    
    chatbox = relationship('Chatbox', back_populates='user')
    message = relationship('Message', back_populates='user')
  
  
class Chatbox(Base):
    __tablename__ = "chatbox"

    id = Column(Integer, primary_key=True, index=True)
    chat_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(ZoneInfo('UTC'))) 
    user_id = Column(Integer, ForeignKey('user.id'))
    
    user = relationship('User', back_populates='chatbox')
    message = relationship('Message', back_populates='chatbox')
    
    
class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(ZoneInfo('UTC')))
    message_type = Column(String(50), default="text")
    chatbox_id = Column(Integer, ForeignKey('chatbox.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id'))
    
    chatbox = relationship('Chatbox', back_populates='message')
    user = relationship('User', back_populates='message')





    

