
To check out the model follow this link: http://2.obotutor.roboticgenacademy.com/

main.py is the backend part. Run only model.py 

To run it.
1. First make environment typing following commands on terminal

python -m venv env
env/Scripts/activate

3.Create a .env file inside the directory you're working with and add GOOGLE_API_KEY ="Your_API_key" inside it

4. Then add dependencies

pip install -r requirements.txt

5.  run server

uvicorn main:app --reload

6. do to the swagger window


type on browser

http://127.0.0.1:8000/docs

github link - https://github.com/Obo-Tutor/obotutor-whatsapp-backend-fastapi-version1.git








rag_for_pdf.py is used to create db . Dont run it, because it will change our vector db
