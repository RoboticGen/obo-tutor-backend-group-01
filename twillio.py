

import os
import logging
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()


TWILIO_ACCOUNT_SID=os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN=os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER=os.getenv("TWILIO_NUMBER")

account_sid = TWILIO_ACCOUNT_SID
auth_token = TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)
twilio_number = TWILIO_NUMBER 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def send_message(to_number, body_text, imgae_url=None):
    try:
        message = client.messages.create(
            from_=f"whatsapp:{twilio_number}",
            body=body_text,
            to=f"whatsapp:{to_number}",
            media_url=[imgae_url] if imgae_url else None
            )
        logger.info(f"Message sent to {to_number}: {message.body}")
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")


