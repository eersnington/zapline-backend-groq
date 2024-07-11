from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from typing import List, Literal
from groq import Groq
import os
import logging

load_dotenv()

# logging.basicConfig(level=logging.INFO)
logging.getLogger("groq").setLevel(logging.INFO)

class Settings(BaseModel):
    debug: bool = Field(default=False, alias="DEBUG")
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")

    @classmethod
    def from_env(cls):
        debug = os.getenv("DEBUG", "False").lower() == "true"
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set in the environment")
        
        logging.info(f"Debug mode: {debug}")
        logging.info(f"GROQ_API_KEY: {'*' * len(groq_api_key)}")  # Log masked API key
        
        return cls(DEBUG=debug, GROQ_API_KEY=groq_api_key)

# Load environment variables and handle possible errors
try:
    settings = Settings.from_env()
except ValidationError as e:
    logging.error("Configuration validation error: %s", e)
    raise
except ValueError as e:
    logging.error(str(e))
    raise

class Message(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str

class ChatCompletionResponse(BaseModel):
    content: str

DEV_MODE = settings.debug

if DEV_MODE:
    logging.info("Running in development mode...")
else:
    logging.info("Running in production mode...")

def get_model() -> str:
    return "llama3-8b-8192" if DEV_MODE else "llama3-70b-8192"

model = get_model()



logging.info(f"Initializing Groq client with model: {model}")

class GroqClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = self.initialize_client()

    def initialize_client(self):
        """Initialize the Groq client with the API key."""
        return Groq(api_key=self.api_key)

    def chat(self, messages: List[Message]) -> ChatCompletionResponse:
        """
        Chat Completion with Groq Client

        Args:
        - messages (list[Message]): List of messages to generate completion.

        Returns:
        - ChatCompletionResponse: The generated completion.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[msg.dict() for msg in messages],
            model=self.model,
            max_tokens=128,
        )

        return ChatCompletionResponse(content=chat_completion.choices[0].message.content)

if __name__ == "__main__":
    # Initialize the Groq client with the correct model and API key
    client = GroqClient(api_key=settings.groq_api_key, model=model)

    # Example usage
    messages = [Message(content="Explain the importance of low latency LLMs.")]
    chat_response = client.chat(messages=messages)
    print(chat_response.content)