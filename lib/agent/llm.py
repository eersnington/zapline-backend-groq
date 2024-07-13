from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from typing import Dict, List, Literal
from groq import Groq
import os
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
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
        logging.info(f"GROQ_API_KEY: {groq_api_key}")  # Log masked API key
        
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
    role: Literal["user", "assistant", "system"]
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

class LLMModel:
    def __init__(self):
        self.client = GroqClient(api_key=settings.groq_api_key, model="llama3-70b-8192")

    def generate_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate text completion using the Llama3 70b model.

        Args:
        - messages (List[Dict[str, str]]): List of message dictionaries.

        Returns:
        - str: The generated completion.
        """
        message_objects = [Message(role=msg['role'], content=msg['content']) for msg in messages]
        chat_response = self.client.chat(messages=message_objects)
        return chat_response.content
    
class ClassifierModel:
    def __init__(self):
        self.client = GroqClient(api_key=settings.groq_api_key, model="llama3-8b-8192")

    def classify(self, message: str) -> str:
        """
        Classify the call type of a message.

        Args:
        - message (str): The message to classify.

        Returns:
        - str: The classification output.
        """
        classification_prompt = (
            "Classify the following message into one of these intents: "
            "Order Status, Returns, Refund, Cancellation, Sales, Transfer, Product Info, General\n\n"
            f"Message: {message}\n\n"
            "Your response should just be the intent."
        )        
        classification_response = self.client.chat([Message(role="user", content=classification_prompt)])
        return classification_response.content
    
class LLMChat:
    def __init__(self, store_name: str, store_details: str):
        self.llm_model = LLMModel()
        self.classifier_model = ClassifierModel()
        self.chat_history: List[Dict[str, str]] = []
        self.store_name = store_name
        self.store_details = store_details
        self.customer_info = {
            "order_id": "123456",
            "order_status": "Shipped",
            "order_date": "2022-01-01",
            "items_ordered": "Smartphone, Laptop, Smartwatch"
        }

    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def get_system_prompt(self) -> str:
        return (
            f"You are an AI assistant for {self.store_name}, an e-commerce store, acting as a seasoned phone support representative. "
            f"Here are some details about the store: {self.store_details}\n"
            f"Customer's order details: "
            f"Order ID: {self.customer_info.get('order_id', 'N/A')}, "
            f"Order Status: {self.customer_info.get('order_status', 'N/A')}, "
            f"Order Date: {self.customer_info.get('order_date', 'N/A')}, "
            f"Items Ordered: {self.customer_info.get('items_ordered', 'N/A')}\n"
            "Return/Refund/Cancellation Policy: Returns accepted within 30 days of purchase. "
            "Is Refundable: Yes, Is Returnable: Yes, Is Cancellable: Yes for unshipped orders\n"
            "Guidelines:\n"
            "1. Address the customer's specific intent and question directly.\n"
            "2. Only ask for necessary details not already provided in the customer information.\n"
            "3. For refund or cancellation requests:\n"
            "   a. Try to convince the customer to get a replacement instead of a discount.\n"
            "   b. If it makes sense, you can offer an incentive like 20$ discount with replacment.\n"
            "   c. Present it as an option to the customer, ask them which do they prefer.\n"
            "4. If the call intent is sales or transfer, inform the customer that the call will be transferred immediately.\n"
            "5. Keep responses concise and relevant to the user's query.\n"
            "6. If unsure about any information, admit it rather than making assumptions.\n"
            "7. Maintain a professional, friendly, and helpful tone throughout the conversation.\n"
            "8. Do not be talkative. Try to keep your responses short. Don't act humanley like apologise or say I understand or say thank you.\n"
            "9. End the conversation by thanking the customer for contacting the store.\n"
            "Remember, you don't process any requests directly. Just follow the guidelines above, and inform the customer that the appropriate team will handle their request after you have handled it."
        )
    def classify_intent(self, message: str) -> str:
        return self.classifier_model.classify(message)

    def generate_response(self, user_message: str) -> str:
        # Classify the intent
        intent = self.classify_intent(user_message)
        
        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "system", "content": f"The current user intent is: {intent}"},
            *self.chat_history,
            {"role": "user", "content": user_message}
        ]
        
        # Generate the response
        response = self.llm_model.generate_text(messages)
        
        # Update chat history
        self.add_message("user", user_message)
        self.add_message("assistant", response)
        
        return response

    def reset_chat(self):
        self.chat_history.clear()

# Example usage:
if __name__ == "__main__":
    store_name = "TechGadgets Inc."
    store_details = "We sell high-quality electronics and gadgets. Our most popular categories are smartphones, laptops, and smart home devices."
    
    chat = LLMChat(store_name, store_details)
    
    while True:
        user_message = input("User: ")
        if user_message.lower() == "exit":
            break
        
        response = chat.generate_response(user_message)
        print(f"Assistant: {response}")