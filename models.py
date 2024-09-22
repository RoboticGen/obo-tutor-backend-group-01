from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime , timezone
from zoneinfo import ZoneInfo
from database import Base
# from enum import Enum

from dto import Role
from dto import UserRole




#    database models

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(50), unique=True)
    password = Column(String(255))
    phone_number = Column(String(50), unique=True)
    learning_rate = Column(String(50) , default="Active")
    age = Column(Integer, default=10)
    role = Column(Enum(UserRole), default="Student")
    communication_format = Column(String(50), default="Textbook")
    tone_style = Column(String(50) , default="Neutral")
    
    chatbox = relationship('Chatbox', back_populates='user')
    message = relationship('Message', back_populates='user')
    summary = relationship('Summary', back_populates='user')
    whatsapp_summary = relationship('WhatsappSummary', back_populates='user')
  
  
class Chatbox(Base):
    __tablename__ = "chatbox"

    id = Column(Integer, primary_key=True, index=True)
    chat_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc)) 
    user_id = Column(Integer, ForeignKey('user.id'))
    
    user = relationship('User', back_populates='chatbox')
    message = relationship('Message', back_populates='chatbox' , cascade="all, delete-orphan")
    summary = relationship('Summary', back_populates='chatbox' ,cascade="all, delete-orphan")
    
    
class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    message_type = Column(Enum(Role), default="text")
    chatbox_id = Column(Integer, ForeignKey('chatbox.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id'))
    #list of images
    related_images = Column(Text, nullable=True)
    
    chatbox = relationship('Chatbox', back_populates='message')
    user = relationship('User', back_populates='message')


class Summary(Base):
    __tablename__ = "summary"

    id = Column(Integer, primary_key=True, index=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc)) 
    chatbox_id = Column(Integer, ForeignKey('chatbox.id'))
    user_id = Column(Integer, ForeignKey('user.id'))
    
    user = relationship('User', back_populates='summary')
    chatbox = relationship('Chatbox', back_populates='summary')

class WhatsappSummary(Base):
    __tablename__ = "whatsapp_summary"

    id = Column(Integer, primary_key=True, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey('user.id'))
    phone_number =  Column(String(50))

    user = relationship('User', back_populates='whatsapp_summary')



    

