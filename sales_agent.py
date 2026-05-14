import asyncio
import os
import logging
import time
from typing import Any, Literal, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agentscope.agent import AgentBase, ReActAgent, UserAgent
from agentscope.formatter import (
    GeminiChatFormatter,
    GeminiMultiAgentFormatter,
)
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import GeminiChatModel
from agentscope.pipeline import MsgHub
from agentscope.tool import Toolkit, ToolResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SalesAgentSystem")

# --- Models ---

class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    category: Literal["technical", "order", "complaint", "general"] = Field(
        description="Issue category: technical, order, complaint, or general inquiries",
    )
    confidence: float = Field(
        description="Classification confidence, between 0 and 1",
        ge=0,
        le=1,
    )
    summary: str = Field(description="Brief summary of the issue")
    priority: Literal["low", "medium", "high"] = Field(description="Issue priority")

class ResolutionReport(BaseModel):
    """Structured output for resolution reports."""
    resolved: bool = Field(description="Whether the issue is resolved")
    solution: str = Field(description="The solution provided")
    follow_up: str = Field(description="Follow-up action items")

# --- Tools ---

def query_order(order_id: str) -> ToolResponse:
    """Query order status.
    Args:
        order_id (str): The order ID.
    """
    orders = {
        "12345": {"status": "shipped", "eta": "2024-01-20"},
        "67890": {"status": "processing", "eta": "2024-01-22"},
    }
    order = orders.get(order_id, {"status": "not_found"})
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Order {order_id}: {order}")],
    )

# --- Hooks ---

async def human_review_post_reply_hook(
    self: AgentBase,
    kwargs: dict[str, Any],
    output: Msg,
) -> Optional[Msg]:
    """Post-reply hook for manual review."""
    print("\n" + "=" * 50)
    print("[Manual Review] Agent response:")
    print(f" {output.get_text_content()}")
    print("=" * 50)
    
    human = UserAgent(name="Reviewer")
    review_msg = await human(
        Msg(
            "System",
            "Please review the response. Type 'ok' to approve, or enter feedback:",
            "user",
        ),
    )
    review_text = review_msg.get_text_content().strip()
    
    if review_text.lower() == "ok":
        logger.info("Review approved.")
        return None
    
    logger.warning(f"Review rejected. Feedback: {review_text}")
    original_msg = kwargs.get("msg")
    if original_msg is not None:
        self.clear_instance_hooks("post_reply")
        revised_msg = Msg(
            original_msg.name,
            f"{original_msg.get_text_content()}\n\n[Review Feedback]: {review_text}",
            original_msg.role,
        )
        revised_output = await self.reply(revised_msg)
        self.register_instance_hook(
            hook_type="post_reply",
            hook_name="human_review",
            hook=human_review_post_reply_hook,
        )
        return revised_output
    return None

# --- Main System ---

class CustomerSupportSystem:
    """Multi-agent customer support system using AgentScope and Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", enable_human_review: bool = False) -> None:
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found in environment.")
            raise ValueError("GOOGLE_API_KEY is required.")
            
        self.model_name = model_name
        self.enable_human_review = enable_human_review
        
        # Initialize common model instance
        self.shared_model = self._init_model(stream=True)
        self.router_model = self._init_model(stream=False)
        
        # Create agents
        self.router = self._create_router()
        self.tech_agent = self._create_specialist("TechSupport", "technical support")
        self.order_agent = self._create_specialist("OrderSupport", "order services", [query_order])
        self.complaint_agent = self._create_specialist("ComplaintHandler", "complaint handling")
        self.supervisor = self._create_supervisor()
        
        if self.enable_human_review:
            self.supervisor.register_instance_hook(
                hook_type="post_reply",
                hook_name="human_review",
                hook=human_review_post_reply_hook,
            )

    def _init_model(self, stream: bool) -> GeminiChatModel:
        """Initialize a Gemini model instance."""
        return GeminiChatModel(
            model_name=self.model_name,
            api_key=self.api_key,
            # stream=stream, # GeminiChatModel might not support stream parameter in constructor for some versions
        )

    def _create_router(self) -> ReActAgent:
        return ReActAgent(
            name="Router",
            sys_prompt="You are an intelligent routing system that classifies customer issues.",
            model=self.router_model,
            formatter=GeminiChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )

    def _create_specialist(self, name: str, specialty: str, tools: list = None) -> ReActAgent:
        toolkit = Toolkit()
        if tools:
            for tool in tools:
                toolkit.register_tool_function(tool)
        
        return ReActAgent(
            name=name,
            sys_prompt=f"You are a {specialty} specialist. Handle issues professionally.",
            model=self.shared_model,
            formatter=GeminiMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=toolkit,
        )

    def _create_supervisor(self) -> ReActAgent:
        return ReActAgent(
            name="Supervisor",
            sys_prompt="You are a supervisor monitoring quality and summarizing results.",
            model=self.shared_model,
            formatter=GeminiMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )

    async def handle_customer(self, customer_id: str, issue: str, max_retries: int = 3) -> str:
        """Process a customer issue with automatic retries for server errors."""
        logger.info(f"Processing issue for {customer_id}: {issue}")
        
        last_error = ""
        for attempt in range(max_retries):
            try:
                # 1. Routing
                route_response = await self.router(
                    Msg("System", f"Analyze this issue: {issue}", "user"),
                    structured_model=RouteDecision,
                )
                decision = route_response.metadata
                category = decision.get("category", "general")
                logger.info(f"Routed to category: {category} (Attempt {attempt + 1})")
                
                # 2. Select Specialist
                specialist_map = {
                    "technical": self.tech_agent,
                    "order": self.order_agent,
                    "complaint": self.complaint_agent,
                    "general": self.tech_agent,
                }
                specialist = specialist_map.get(category, self.tech_agent)
                
                # 3. Collaborative Handling
                async with MsgHub(participants=[specialist, self.supervisor]):
                    await specialist(Msg("System", f"Handle this issue: {issue}", "user"))
                    
                    final_response = await self.supervisor(
                        Msg("System", "Review result and provide final response.", "user"),
                        structured_model=ResolutionReport,
                    )
                    
                return final_response.get_text_content()
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {customer_id}: {last_error}")
                if "503" in last_error or "429" in last_error or "400" in last_error:
                    wait_time = (attempt + 1) * 15
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    break # Non-retryable error
                    
        return f"I apologize, but we encountered a persistent error: {last_error}"

async def main() -> None:
    # Use gemini-1.5-flash as default, can be overridden by env
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    system = CustomerSupportSystem(model_name=model_name, enable_human_review=False)
    
    customer_issues = [
        ("C001", "Your app keeps crashing. I can't use it at all!"),
        ("C002", "Has my order 12345 shipped? When will it arrive?"),
        ("C003", "I strongly protest! The product quality is terrible. I demand a refund!"),
    ]
    
    for i, (customer_id, issue) in enumerate(customer_issues):
        if i > 0:
            logger.info("Waiting 20 seconds to respect free tier rate limits...")
            time.sleep(20)
        
        print(f"\n--- Handling {customer_id} ---")
        response = await system.handle_customer(customer_id, issue)
        print(f"\n[Final Response]\n{response}")
        print("-" * 30)

if __name__ == "__main__":
    load_dotenv()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user.")
    except Exception as e:
        logger.critical(f"System failed: {e}")
