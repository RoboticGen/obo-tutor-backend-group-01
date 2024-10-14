from fastapi.testclient import TestClient
from passlib.context import CryptContext
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker 
from main import app, decode_jwt_token, get_db
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

# Mock the decode_jwt_token to return a fake user ID
def mock_decode_jwt_token(token: str):
    return {"sub": 1}  # Return the user ID 1

# Override the JWT decoding function
app.dependency_overrides[decode_jwt_token] = mock_decode_jwt_token


# Mock the get_db to return a sqllite memory database
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
    
app.dependency_overrides[get_db] = override_get_db

def add_fake_user_to_db():
    # Password hashing context (should match the one in your main code)
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 
    
    # Create a user in the test database
    db = TestingSessionLocal()
    user = models.User(
        first_name="Kavindu",
        last_name="Senarathna",
        email="abc@gmail.com",
        password=pwd_context.hash("123abcABC"),
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


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Server is running"
    print("\nTest 1 : root api work successfully", end="")
    
def test_signup():
    setup()
    input_data = {
        "first_name": "Kavindu",
        "last_name": "Senarathna",
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
    respond = response.json()
    assert respond["user_id"] == 1
    
    teardown()
    print("\nTest 2 : signup succesfully", end="")
    
    
    
def test_whatsapp():
    setup()
    
    add_fake_user_to_db()
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
    print("\nTest 3 : sending whatsapp message successfully", end="")
    
    
    
def test_login_successful():
    setup()
    
    add_fake_user_to_db()
    # Test input: valid user credentials
    user_data = {
        "email": "abc@gmail.com",
        "password": "123abcABC"
    }
    
    # Send POST request to the login endpoint
    response = client.post("/api/login", json=user_data)
    
    # Assertions for a successful login
    assert response.status_code == 200
    respond = response.json()
    assert "access_token" in respond
    assert response.json()["user_details"]["email"] == "abc@gmail.com"
        
    teardown()
    print("\nTest 4 : login succesfully", end="")
    
    
def test_login_failure():
    setup()
    
    add_fake_user_to_db()
    # Test input: valid user credentials
    user_data = {
        "email": "abc@gmail.com",
        "password": "wrong_password"
    }
    
    # Send POST request to the login endpoint
    response = client.post("/api/login", json=user_data)
    
    # Assertions for a failure login
    assert response.status_code == 404
    assert response.json()["detail"] == "Invalid Credentials"
        
    teardown()
    print("\nTest 5 : login failure", end="")
    
# Test for successful user retrieval
# def test_get_user_success():
#     setup()
#     add_fake_user_to_db()
#     # Simulated token (it doesn't need to be valid since we're mocking the decode function)
#     token = "fake_token"

#     response = client.get("/api/user", headers={"Authorization": f"Bearer {token}"})
    
#     print(f"Request headers: {response.request.headers}")
#     # Assertions for a successful user retrieval
#     assert response.status_code == 200
#     assert response.json()["email"] == "testuser@example.com"
#     assert response.json()["id"] == 1
    
#     teardown()


def setup():
    Base.metadata.create_all(bind=engine)
    
def teardown():
    Base.metadata.drop_all(bind=engine)
    
