import asyncio
from typing import List, Optional
from datetime import datetime
import shopify
from pydantic import BaseModel

from lib.agent.llm import GroqClient
from lib.db import track_metrics

class Order(BaseModel):
    id: str
    order_number: str
    items: List[str]
    status: str
    date: datetime

class CallChatSession:
    def __init__(self, app_token: str, myshopify: str, bot_name: str, brand_name: str):
        self.app_token = app_token
        self.myshopify = myshopify
        self.bot_name = bot_name
        self.brand_name = brand_name
        self.groq_client = GroqClient(api_key=settings.groq_api_key, model=get_model())
        self.shopify_client = self._init_shopify_client()
        self.order: Optional[Order] = None
        self.call_intent: Optional[str] = None

    def _init_shopify_client(self):
        session = shopify.Session(self.myshopify, "2024-01", self.app_token)
        shopify.ShopifyResource.activate_session(session)
        return shopify

    async def start(self, sid: str, customer_phone_no: str) -> str:
        try:
            orders = self.shopify_client.Order.find()
            recent_order = next((order for order in orders if order.customer and order.customer.phone == customer_phone_no), None)

            if recent_order is None:
                return "You seem to be a new customer based on my records. How can I help you today?"

            self.order = Order(
                id=recent_order.id,
                order_number=recent_order.order_number,
                items=[item.title for item in recent_order.line_items],
                status=recent_order.fulfillment_status or "unfulfilled",
                date=datetime.strptime(recent_order.created_at.split("T")[0], "%Y-%m-%d")
            )

            return f"You've a recent order of {', '.join(self.order.items)}. Do you want to know the order status, start a return, or anything else?"
        except Exception as e:
            print(f"Error starting session: {e}")
            return "I'm having trouble accessing your order information. How else can I assist you today?"

    async def get_response(self, message: str) -> str:
        try:
            self.call_intent = await self._classify_intent(message)
            prompt = self._create_prompt(message)
            response = await self.groq_client.chat([{"role": "user", "content": prompt}])
            return response.content
        except Exception as e:
            print(f"Error getting response: {e}")
            return "I'm sorry, I'm having trouble understanding. Could you please rephrase your question?"

    async def _classify_intent(self, message: str) -> str:
        # Implement intent classification using Groq
        classification_prompt = f"Classify the following message into one of these intents: Order Status, Returns, Refund, Cancellation, Sales, Transfer, General Inquiry\n\nMessage: {message}"
        response = await self.groq_client.chat([{"role": "user", "content": classification_prompt}])
        return response.content.strip()

    def _create_prompt(self, message: str) -> str:
        order_status = f"Order status: {self.order.status}" if self.order else "No recent order found"
        return f"""
        Bot Name: {self.bot_name}
        Brand Name: {self.brand_name}
        {order_status}
        Order Date: {self.order.date.strftime('%B %d') if self.order else 'N/A'}
        Order Items: {', '.join(self.order.items) if self.order else 'N/A'}
        Call Intent: {self.call_intent}

        User Message: {message}

        Please provide a helpful response based on the above information.
        """

    async def track_call(self, user_id: str) -> str:
        try:
            call_type = "transferred" if self.call_intent in ["Sales", "Transfer"] else "automated"
            await track_metrics(user_id, call_type=call_type, call_intent=self.call_intent or "Other")
            return "Call status updated successfully."
        except Exception as e:
            print(f"Error tracking call: {e}")
            return f"Error occurred while tracking call: {str(e)}"