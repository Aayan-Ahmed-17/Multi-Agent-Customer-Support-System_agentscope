import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Literal
from pydantic import BaseModel, Field
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

# --- Part 2: Structured Output Models ---

class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    category: Literal["technical", "order", "complaint", "general"] = Field(
        description=(
            "Issue category: technical (technical issues), "
            "order (order issues), complaint (complaints), "
            "general (general inquiries)"
        ),
    )
    confidence: float = Field(
        description="Classification confidence, between 0 and 1",
        ge=0,
        le=1,
    )
    summary: str = Field(
        description="Brief summary of the issue",
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="Issue priority",
    )

class ResolutionReport(BaseModel):
    """Structured output for resolution reports."""
    resolved: bool = Field(description="Whether the issue is resolved")
    solution: str = Field(description="The solution provided")
    follow_up: str = Field(description="Follow-up action items")

# --- Part 3.2: Tools and Workers ---

def query_order(order_id: str) -> ToolResponse:
    """Query order status.
    Args:
        order_id (``str``): The order ID.
    """
    # Simulate order data
    orders = {
        "12345": {"status": "shipped", "eta": "2024-01-20"},
        "67890": {"status": "processing", "eta": "2024-01-22"},
    }
    order = orders.get(order_id, {"status": "not_found"})
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Order {order_id}: {order}")],
    )

# --- Part 4: Human-in-the-Loop Hook ---

async def human_review_post_reply_hook(
    self: AgentBase,
    kwargs: dict[str, Any],
    output: Msg,
) -> Msg | None:
    """Post-reply hook: perform manual review after the agent replies."""
    print("\n" + "=" * 50)
    print("[Manual Review] Agent response:")
    print(f" {output.get_text_content()}")
    print("=" * 50)
    
    # Use UserAgent to get human input
    human = UserAgent(name="Reviewer")
    review_msg = await human(
        Msg(
            "System",
            "Please review the above response. Type 'ok' to approve, "
            "or enter your revision feedback:",
            "user",
        ),
    )
    review_text = review_msg.get_text_content().strip()
    
    if review_text.lower() == "ok":
        print("[Review Approved] Response confirmed.")
        return None
    
    # Review rejected: append feedback to the original message and regenerate
    print(f"[Review Rejected] Feedback: {review_text}")
    original_msg = kwargs.get("msg")
    if original_msg is not None:
        # Temporarily remove the hook to avoid infinite loops
        self.clear_instance_hooks("post_reply")
        revised_msg = Msg(
            original_msg.name,
            f"{original_msg.get_text_content()}\n\n"
            f"[Review Feedback] Please revise based on the following "
            f"feedback: {review_text}",
            original_msg.role,
        )
        revised_output = await self.reply(revised_msg)
        # Re-register the hook
        self.register_instance_hook(
            hook_type="post_reply",
            hook_name="human_review",
            hook=human_review_post_reply_hook,
        )
        return revised_output
    return None

# --- Part 5: Complete Customer Support System ---

class CustomerSupportSystem:
    """Multi-agent customer support system."""
    
    def __init__(self, enable_human_review: bool = False) -> None:
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            print("WARNING: GOOGLE_API_KEY not found in environment.")
            
        self.model = GeminiChatModel(
            model_name="gemini-1.5-flash",
            api_key=self.api_key,
        )
        self.enable_human_review = enable_human_review
        
        # Create the agents
        self.router = self._create_router()
        self.tech_agent = self._create_specialist(
            "technical support",
            "TechSupport",
        )
        self.order_agent = self._create_specialist(
            "order services",
            "OrderSupport",
        )
        self.complaint_agent = self._create_specialist(
            "complaint handling",
            "ComplaintHandler",
        )
        self.supervisor = self._create_supervisor()
        
        # If manual review is enabled, register the hook for the supervisor
        if self.enable_human_review:
            self.supervisor.register_instance_hook(
                hook_type="post_reply",
                hook_name="human_review",
                hook=human_review_post_reply_hook,
            )

    def _create_router(self) -> ReActAgent:
        """Create the router agent."""
        return ReActAgent(
            name="Router",
            sys_prompt="You are an intelligent routing system that analyzes "
                       "customer issues and classifies them.",
            model=GeminiChatModel(
                model_name="gemini-1.5-flash",
                api_key=self.api_key,
            ),
            formatter=GeminiChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )

    def _create_specialist(
        self,
        specialty: str,
        name: str,
    ) -> ReActAgent:
        """Create a specialized customer support agent."""
        toolkit = Toolkit()
        if "order" in specialty:
            toolkit.register_tool_function(query_order)
        return ReActAgent(
            name=name,
            sys_prompt=f"You are a {specialty} specialist, professionally "
                       f"handling related issues.",
            model=self.model,
            formatter=GeminiMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=toolkit,
        )

    def _create_supervisor(self) -> ReActAgent:
        """Create the supervisor agent."""
        return ReActAgent(
            name="Supervisor",
            sys_prompt="You are a customer service supervisor, responsible "
                       "for monitoring service quality and summarizing results.",
            model=self.model,
            formatter=GeminiMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )

    async def handle_customer(
        self,
        customer_id: str,
        issue: str,
    ) -> str:
        """Main workflow for handling customer issues."""
        print(f"\n{'=' * 60}")
        print(f"[New Customer Issue] {customer_id}: {issue}")
        print("=" * 60)
        
        # Step 1: Route classification
        route_response = await self.router(
            Msg("System", f"Analyze this customer issue: {issue}", "user"),
            structured_model=RouteDecision,
        )
        decision = route_response.metadata
        category = decision.get("category", "general")
        priority = decision.get("priority", "medium")
        print(f"\n[Routing Decision] Category: {category}, "
              f"Priority: {priority}")
        
        # Step 2: Assign to a specialist agent
        specialist_map = {
            "technical": self.tech_agent,
            "order": self.order_agent,
            "complaint": self.complaint_agent,
            "general": self.tech_agent,
        }
        specialist = specialist_map.get(category, self.tech_agent)
        
        # Step 3: Multi-agent collaborative handling (MsgHub)
        async with MsgHub(participants=[specialist, self.supervisor]):
            # Specialist agent handles the issue
            await specialist(
                Msg(
                    "System",
                    f"Please handle this customer issue: {issue}",
                    "user",
                ),
            )
            
            # Supervisor reviews and summarizes
            final_response = await self.supervisor(
                Msg(
                    "System",
                    "Please review the handling result and provide "
                    "the final response.",
                    "user",
                ),
                structured_model=ResolutionReport,
            )
            return final_response.get_text_content()

async def main() -> None:
    """Run the complete multi-agent customer support system."""
    # Set enable_human_review=True to enable manual review
    system = CustomerSupportSystem(enable_human_review=False)
    
    # Simulate multiple customer issues
    customer_issues = [
        ("C001", "Your app keeps crashing. I can't use it at all!"),
        ("C002", "Has my order 12345 shipped? When will it arrive?"),
        (
            "C003",
            "I strongly protest! The product quality is terrible. "
            "I demand a refund!",
        ),
    ]
    
    for customer_id, issue in customer_issues:
        try:
            response = await system.handle_customer(customer_id, issue)
            print(f"\n[Final Response]\n{response}")
        except Exception as e:
            print(f"Error handling customer {customer_id}: {e}")
        print("\n" + "-" * 60)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
