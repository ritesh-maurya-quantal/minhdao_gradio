import os
import json
import asyncio
import textwrap
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import logging

from langchain_openai import ChatOpenAI
# LangChain / LangGraph / MCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────────────────────────────── Configuration ─────────────────────────
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

_load_env()

# Use environment variables for better security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

NOTION_MCP_TOKEN = os.getenv("NOTION_MCP_TOKEN")
NOTION_VERSION = "2022-06-28"
CACHE_FILE = Path("workspace_cache.pkl")
CACHE_DURATION = timedelta(minutes=30)  # Reduced cache duration for fresher data

# Connection pooling and optimization
notion_cfg = {
    "notion": {
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "transport": "stdio",
        "env": {
            "OPENAPI_MCP_HEADERS": json.dumps({
                "Authorization": f"Bearer {NOTION_MCP_TOKEN}",
                "Notion-Version": NOTION_VERSION,
            })
        },
    }
}

# ─────────────── Enhanced Workspace Knowledge (with async caching) ───────────────
@dataclass
class FieldConstraints:
    required: bool = False
    options: Optional[List[Dict[str, Any]]] = None

@dataclass
class PropertyDetails:
    name: str
    type: str
    constraints: FieldConstraints = field(default_factory=FieldConstraints)

@dataclass
class EntitySchema:
    id: str
    name: str
    properties: Dict[str, PropertyDetails] = field(default_factory=dict)

@dataclass
class WorkspaceEntity:
    id: str
    name: str

class UniversalWorkspaceKnowledge:
    def __init__(self):
        self.entities: Dict[str, WorkspaceEntity] = {}
        self.schemas: Dict[str, EntitySchema] = {}
        self.users: Dict[str, str] = {}
        self.last_updated: datetime = datetime.now()
        self._context_cache: Optional[str] = None

    def add_entity(self, eid: str, name: str):
        self.entities[eid] = WorkspaceEntity(eid, name)
        self._context_cache = None  # Invalidate cache

    def add_schema(self, db_id: str, props: Dict[str, PropertyDetails]):
        self.schemas[db_id] = EntitySchema(db_id, self.entities[db_id].name, props)
        self._context_cache = None  # Invalidate cache

    def add_user(self, uid: str, name: str):
        self.users[name] = uid
        self._context_cache = None  # Invalidate cache

    def is_cache_valid(self) -> bool:
        return (datetime.now() - self.last_updated) < CACHE_DURATION

    async def save_to_cache(self):
        """Async save workspace knowledge to cache file"""
        try:
            async with aiofiles.open(CACHE_FILE, 'wb') as f:
                await f.write(pickle.dumps(self))
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    @classmethod
    async def load_from_cache(cls) -> Optional['UniversalWorkspaceKnowledge']:
        """Async load workspace knowledge from cache if valid"""
        try:
            if CACHE_FILE.exists():
                async with aiofiles.open(CACHE_FILE, 'rb') as f:
                    data = await f.read()
                    cached = pickle.loads(data)
                    if cached.is_cache_valid():
                        logger.info("✅ Using cached workspace schema")
                        return cached
                    else:
                        logger.info("⏰ Cache expired, will refresh")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
        return None

    def context(self) -> str:
        """Cached context generation"""
        if self._context_cache is not None:
            return self._context_cache
        
        lines = ["=== WORKSPACE SCHEMA ==="]
        for eid, ent in self.entities.items():
            lines.append(f"\n*{ent.name}* (DB ID: {eid})")
            schema = self.schemas.get(eid)
            if schema:
                for pname, pdet in schema.properties.items():
                    req = " (required)" if pdet.constraints.required else ""
                    opts = ""
                    if pdet.constraints.options:
                        opts = " | opts: " + ", ".join(o["name"] for o in pdet.constraints.options)
                    lines.append(f"  • {pname}: {pdet.type}{req}{opts}")
        if self.users:
            lines.append("\n=== USERS ===")
            for name, uid in self.users.items():
                lines.append(f"  • {name} → {uid}")
        
        self._context_cache = "\n".join(lines)
        return self._context_cache

# ────────────────────── Ultra-Optimized Discovery with Batching ──────────────────────
async def discover_structure_optimized(node: ToolNode, know: UniversalWorkspaceKnowledge):
    """Ultra-optimized discovery with concurrent processing"""
    
    # Step 1: Discover databases and users simultaneously
    logger.info("🔍 Discovering databases and users in parallel...")
    
    parallel_tasks = [
        {
            "name": "API-post-search",
            "args": {"filter": {"property": "object", "value": "database"}},
            "id": "db_search"
        },
        {
            "name": "API-get-users",
            "args": {},
            "id": "users"
        }
    ]
    
    resp = await node.ainvoke({
        "messages": [AIMessage(content="", tool_calls=parallel_tasks)]
    })
    
    db_ids = []
    
    # Process responses concurrently
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            
            if m.tool_call_id == "db_search":
                for item in data.get("results", []):
                    if item["object"] == "database":
                        eid = item["id"]
                        name = "".join(t["plain_text"] for t in item["title"])
                        know.add_entity(eid, name)
                        db_ids.append(eid)
            
            elif m.tool_call_id == "users":
                for u in data.get("results", []):
                    know.add_user(u["id"], u.get("name", "Unknown"))

    # Step 2: Batch schema discovery with controlled concurrency
    if db_ids:
        logger.info(f"📊 Fetching schemas for {len(db_ids)} databases...")
        
        # Process schemas in smaller batches to avoid overwhelming the API
        BATCH_SIZE = 5
        for i in range(0, len(db_ids), BATCH_SIZE):
            batch = db_ids[i:i + BATCH_SIZE]
            schema_tasks = [
                {
                    "name": "API-retrieve-a-database",
                    "args": {"database_id": db},
                    "id": f"schema_{db}"
                }
                for db in batch
            ]
            
            batch_resp = await node.ainvoke({
                "messages": [AIMessage(content="", tool_calls=schema_tasks)]
            })
            
            # Process batch responses
            for m in batch_resp["messages"]:
                if isinstance(m, ToolMessage) and m.tool_call_id.startswith("schema_"):
                    data = json.loads(m.content)
                    db_id = m.tool_call_id.replace("schema_", "")
                    props = {}
                    for pname, pdat in data["properties"].items():
                        typ = pdat["type"]
                        req = pname.lower() in ("name", "title")
                        opts = None
                        if typ in ("select", "status", "multi_select"):
                            opts = pdat.get(typ, {}).get("options")
                        props[pname] = PropertyDetails(pname, typ, FieldConstraints(req, opts))
                    know.add_schema(db_id, props)

    know.last_updated = datetime.now()
    await know.save_to_cache()  # Async cache save
    return know

# ────────────────── Streamlined System Prompt ──────────────────
OPTIMIZED_GUARDRAILS = """
## MCP Usage Rules (CRITICAL)
- Always use `"parent": {"database_id": "<DB_ID>"}`
- Property formats:
  • status → {"status": {"name": "..."}}
  • select → {"select": {"name": "..."}}
  • date → {"date": {"start": "YYYY-MM-DD"}}
  • people → {"people": ["<USER_ID>", ...]}

## Missing Fields Protocol
1. Check required fields from schema
2. If missing: Ask "The '<FieldName>' field is required. Please provide it."
3. Validate & proceed

## Response Guidelines
- Be concise but complete
- Use exact DB IDs from schema
- Validate all data before submission
"""

def make_optimized_system_prompt(know: UniversalWorkspaceKnowledge):
    return (
        "You are a high-performance MCP-driven Notion assistant.\n"
        "PRIORITY: Speed and accuracy. Be concise but thorough.\n"
        + OPTIMIZED_GUARDRAILS
        + "\n" + know.context()
    )

# ──────────────────── High-Performance Agent & Workflow ────────────────────
class OptimizedState(MessagesState):
    context: Optional[str] = None  # Store context to avoid regeneration

def smart_message_trimming(messages: List, max_messages: int = 8) -> List:
    """Intelligent message trimming that preserves tool call/response pairs"""
    if len(messages) <= max_messages:
        return messages
    
    # Keep system messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    
    if len(non_system) <= max_messages - len(system_msgs):
        return system_msgs + non_system
    
    # Smart trimming that preserves tool call/response pairs
    keep_messages = []
    i = len(non_system) - 1
    
    while i >= 0 and len(keep_messages) < (max_messages - len(system_msgs)):
        current_msg = non_system[i]
        
        # If this is a tool message, find its corresponding tool call
        if isinstance(current_msg, ToolMessage):
            # Find the AI message with tool calls that this responds to
            for j in range(i-1, -1, -1):
                if isinstance(non_system[j], AIMessage) and getattr(non_system[j], 'tool_calls', None):
                    # Check if this tool call matches
                    for tool_call in non_system[j].tool_calls:
                        if tool_call['id'] == current_msg.tool_call_id:
                            # Add both messages as a pair
                            keep_messages.insert(0, non_system[j])
                            keep_messages.insert(1, current_msg)
                            i = j - 1
                            break
                    break
            else:
                # No matching tool call found, skip this tool message
                i -= 1
        else:
            # Add regular messages
            keep_messages.insert(0, current_msg)
            i -= 1
    
    return system_msgs + keep_messages

async def build_optimized_app():
    logger.info("🚀 Initializing optimized MCP client...")
    
    # Use connection pooling and timeout optimization
    client = MultiServerMCPClient(notion_cfg)
    tools = await asyncio.wait_for(client.get_tools(), timeout=15)
    
    # Optimized model configuration with streaming
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use gpt-4o for better speed/quality balance
        api_key=OPENAI_API_KEY,
        temperature=0.1,  # Lower for more consistent responses
        max_tokens=2000,  # Reasonable limit
        timeout=120,
        streaming=True,  # Enable streaming for perceived speed
        max_retries=2,   # Reduce retries for faster failure
    ).bind_tools(tools)
    
    node = ToolNode(tools)

    # Try async cache loading first
    know = await UniversalWorkspaceKnowledge.load_from_cache()
    if know is None:
        logger.info("🔄 Refreshing workspace schema...")
        know = UniversalWorkspaceKnowledge()
        await discover_structure_optimized(node, know)
    
    logger.info(f"✅ Workspace ready! Found {len(know.entities)} databases, {len(know.users)} users")

    # Pre-compile optimized system prompt
    system_prompt = make_optimized_system_prompt(know)
    sys_message = SystemMessage(content=system_prompt)

    # Custom agent function with proper message handling
    async def agent_fn(state: OptimizedState):
        # Get all messages and ensure proper pairing
        all_messages = state["messages"]
        
        # Filter and prepare messages for the LLM
        prepared_messages = []
        
        # Always include system message first
        prepared_messages.append(sys_message)
        
        # Add conversation messages, ensuring tool call/response pairs
        conversation_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
        
        # Keep only the most recent meaningful conversation
        if len(conversation_msgs) > 10:  # Keep more messages but clean
            conversation_msgs = conversation_msgs[-10:]
        
        # Validate message sequence
        valid_messages = []
        for i, msg in enumerate(conversation_msgs):
            if isinstance(msg, ToolMessage):
                # Only include tool messages that have a preceding tool call
                if valid_messages and isinstance(valid_messages[-1], AIMessage):
                    if hasattr(valid_messages[-1], 'tool_calls') and valid_messages[-1].tool_calls:
                        # Check if tool call ID matches
                        tool_call_ids = [tc['id'] for tc in valid_messages[-1].tool_calls]
                        if msg.tool_call_id in tool_call_ids:
                            valid_messages.append(msg)
                # Skip orphaned tool messages
            else:
                valid_messages.append(msg)
        
        prepared_messages.extend(valid_messages)
        
        try:
            # Use asyncio.create_task for better concurrency
            task = asyncio.create_task(llm.ainvoke(prepared_messages))
            resp = await asyncio.wait_for(task, timeout=20)
            
            # Return with properly managed messages
            return {"messages": all_messages + [resp]}
            
        except asyncio.TimeoutError:
            error_msg = AIMessage(content="⏰ Request timed out. Please try a simpler query.")
            return {"messages": all_messages + [error_msg]}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            # Check if it's a tool message validation error
            if "tool" in str(e).lower() and "role" in str(e).lower():
                logger.info("🔧 Cleaning message history due to tool message error")
                # Start fresh but keep the latest human message
                latest_human = None
                for msg in reversed(all_messages):
                    if isinstance(msg, HumanMessage):
                        latest_human = msg
                        break
                
                if latest_human:
                    clean_messages = [latest_human]
                    task = asyncio.create_task(llm.ainvoke([sys_message] + clean_messages))
                    try:
                        resp = await asyncio.wait_for(task, timeout=20)
                        return {"messages": clean_messages + [resp]}
                    except Exception as e2:
                        logger.error(f"Retry failed: {e2}")
            
            error_msg = AIMessage(content=f"❌ Error: {str(e)}")
            return {"messages": all_messages + [error_msg]}

    def router(state: OptimizedState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    async def optimized_tool_node(state: OptimizedState):
        """Optimized tool node with proper error handling"""
        try:
            # Use asyncio.wait_for with a reasonable timeout
            result = await asyncio.wait_for(node.ainvoke(state), timeout=25)
            
            # Don't trim messages here to preserve tool call/response pairs
            return result
            
        except asyncio.TimeoutError:
            logger.warning("Tool execution timed out")
            
            # Find the last AI message with tool calls to respond to
            last_ai_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    last_ai_msg = msg
                    break
            
            if last_ai_msg and last_ai_msg.tool_calls:
                # Create timeout responses for each tool call
                timeout_responses = []
                for tool_call in last_ai_msg.tool_calls:
                    timeout_responses.append(ToolMessage(
                        content="⏰ Tool execution timed out. Please try a simpler operation.", 
                        tool_call_id=tool_call['id']
                    ))
                return {"messages": state["messages"] + timeout_responses}
            else:
                # Fallback error message
                error_msg = AIMessage(content="⏰ Tool execution timed out. Please try a simpler operation.")
                return {"messages": state["messages"] + [error_msg]}
                
        except Exception as e:
            logger.error(f"Tool error: {e}")
            
            # Find the last AI message with tool calls to respond to
            last_ai_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    last_ai_msg = msg
                    break
            
            if last_ai_msg and last_ai_msg.tool_calls:
                # Create error responses for each tool call
                error_responses = []
                for tool_call in last_ai_msg.tool_calls:
                    error_responses.append(ToolMessage(
                        content=f"❌ Tool error: {str(e)}", 
                        tool_call_id=tool_call['id']
                    ))
                return {"messages": state["messages"] + error_responses}
            else:
                # Fallback error message
                error_msg = AIMessage(content=f"❌ Tool error: {str(e)}")
                return {"messages": state["messages"] + [error_msg]}

    # Build the workflow with custom agents (more reliable than create_react_agent)
    wf = StateGraph(OptimizedState)
    wf.add_node("agent", agent_fn)
    wf.add_node("tools", optimized_tool_node)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    wf.add_edge("tools", "agent")
    
    return wf.compile(checkpointer=MemorySaver())

# ──────────────────────────── Optimized Main with Streaming ────────────────────────────
async def main():
    try:
        app = await build_optimized_app()
        logger.info("✨ High-performance MCP-driven Notion assistant ready! Type 'quit' to exit.\n")
        
        while True:
            try:
                q = input("\n💬 You: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                
                print("🤖 Assistant:", end=" ", flush=True)
                
                # Use a clean thread ID for each session to avoid message history issues
                thread_id = f"mcp_session_{int(asyncio.get_event_loop().time())}"
                config = {"configurable": {"thread_id": thread_id}}
                
                # Use asyncio.create_task for better performance
                start_time = asyncio.get_event_loop().time()
                
                task = asyncio.create_task(
                    app.ainvoke({"messages": [HumanMessage(content=q)]}, config=config)
                )
                
                result = await asyncio.wait_for(task, timeout=40)  # Slightly longer timeout
                
                elapsed = asyncio.get_event_loop().time() - start_time
                out = result["messages"][-1].content
                
                print(f"{out}")
                logger.info(f"⚡ Response time: {elapsed:.2f}s")
                
            except asyncio.TimeoutError:
                print("⏰ Request timed out. Please try a simpler query.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                print(f"❌ Error: {e}")
                
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    # Use uvloop for better async performance on Unix systems
    # try:
    #     import uvloop
    #     uvloop.install()
    #     logger.info("🚀 Using uvloop for enhanced performance")
    # except ImportError:
    #     logger.info("📝 Using default asyncio event loop")
    
    asyncio.run(main())