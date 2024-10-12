from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker 
from main import app, get_db
from database import Base
import models

TEST_DATABASE_URL = "sqlite:///:memory:"

engine  =  create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


client = TestClient(app)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
    
app.dependency_overrides[get_db] = override_get_db



def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Server is running"
    
def test_signup():
    setup()
    input_data = {
        "first_name": "Pasindu",
        "last_name": "Sankalpa",
        "email": "abc@gmail.com",
        "password": "123abcABC",
        "phone_number": "0702225222",
        "role": "Student",
        "age": 20,
        "communication_rating": 5,
        "leadership_rating": 9,
        "behaviour_rating": 7,
        "responsiveness_rating": 4,
        "difficult_concepts": "string",
        "understood_concepts": "string",
        "activity_summary": "string",
        "tone_style": "string"
    }
    response = client.post("/api/signup", json=input_data)

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    
    teardown()
    
    
def test_whatsapp():
    setup()

    # Create a user in the test database
    db = TestingSessionLocal()
    user = models.User(
        first_name="Pasindu",
        last_name="Sankalpa",
        email="abc@gmail.com",
        password="123abcABC",
        phone_number="0702225222",
        role="Student",
        age=20,
        communication_rating=5,
        leadership_rating=9,
        behaviour_rating=7,
        responsiveness_rating=4,
        difficult_concepts="string",
        understood_concepts="string",
        activity_summary="string",
        tone_style="string",
    )
    db.add(user)
    db.commit()

    
     # Simulate a WhatsApp message via POST request
    waid = "+94702225222"
    body = "Hello chatbot!" 

    # Post the WhatsApp message
    response = client.post(
        "/api/whatsapp",
        data={"WaId": waid, "Body": body},
    )

    # Assert that the response is successful
    assert response.status_code == 200
    teardown()


def setup():
    Base.metadata.create_all(bind=engine)
    
def teardown():
    Base.metadata.drop_all(bind=engine)
    
