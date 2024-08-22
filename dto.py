from pydantic import BaseModel, EmailStr, Field, field_validator
from enum import Enum
from datetime import datetime
import re




#   Pydantic BaseModels for


class UserBase(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str
    phone_number: str
    learning_rate: str = "Active"
    role: str = "Student"
    age:  int
    communication_format: str = "Textbook"
    tone_style: str = "Neutral"
    

class Chatbox(BaseModel):
    chat_name: str
    created_at: datetime 
    user_id: int
    
    
class Message(BaseModel):
    Message: str
    created_at: datetime 
    message_type: str
    chatbox_id: int
    user_id: int


class UserLogin(BaseModel):
    email: str
    password: str
    
    
class TokenData(BaseModel):
    email: str
    
    
class UserValidate(BaseModel):
    email: EmailStr
    phone_number: str = Field(
        pattern=r'^\+?\d{10,11}$'
    )
    password: str = Field(..., example="P@ssw0rd")

    @field_validator('password')
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', value):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', value):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', value):
            raise ValueError('Password must contain at least one digit')
        return value
    
    class Config:
        schema_extra = {
            "phone number": {
                "format 1": "+94 xx xxx xxxx",
                "format 2": "  0 xx xxx xxxx"
            }
        }
    
    
class Role(Enum):
    STUDENT = "student"
    TEACHER = "teacher"