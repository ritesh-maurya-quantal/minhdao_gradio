import os
import json
import asyncio
import textwrap
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
# LangChain / LangGraph / MCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ───────────────────────────────── Configuration ─────────────────────────
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

_load_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

NOTION_MCP_TOKEN = os.getenv("NOTION_MCP_TOKEN")
NOTION_VERSION = "2022-06-28"
CACHE_FILE = Path("workspace_cache.pkl")
CACHE_DURATION = timedelta(hours=1)  # Cache expires after 1 hour

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

# ─────────────── Universal Workspace Knowledge (with caching) ───────────────
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

    def add_entity(self, eid: str, name: str):
        self.entities[eid] = WorkspaceEntity(eid, name)

    def add_schema(self, db_id: str, props: Dict[str, PropertyDetails]):
        self.schemas[db_id] = EntitySchema(db_id, self.entities[db_id].name, props)

    def add_user(self, uid: str, name: str):
        self.users[name] = uid

    def is_cache_valid(self) -> bool:
        return (datetime.now() - self.last_updated) < CACHE_DURATION

    def save_to_cache(self):
        """Save workspace knowledge to cache file"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    @classmethod
    def load_from_cache(cls) -> Optional['UniversalWorkspaceKnowledge']:
        """Load workspace knowledge from cache if valid"""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'rb') as f:
                    cached = pickle.load(f)
                    if cached.is_cache_valid():
                        print("✅ Using cached workspace schema")
                        return cached
                    else:
                        print("⏰ Cache expired, will refresh")
        except Exception as e:
            print(f"Failed to load cache: {e}")
        return None

    def context(self) -> str:
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
        return "\n".join(lines)

# ────────────────────── Optimized Discovery ──────────────────────
async def discover_structure_parallel(node: ToolNode, know: UniversalWorkspaceKnowledge):
    """Parallel discovery with batching"""
    
    # 1) Get databases first
    print("🔍 Discovering databases...")
    resp = await node.ainvoke({
        "messages": [AIMessage(content="", tool_calls=[{
            "name": "API-post-search",
            "args": {"filter": {"property": "object", "value": "database"}},
            "id": "db_search"
        }])]
    })
    
    db_ids = []
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            for item in data.get("results", []):
                if item["object"] == "database":
                    eid = item["id"]
                    name = "".join(t["plain_text"] for t in item["title"])
                    know.add_entity(eid, name)
                    db_ids.append(eid)

    # 2) Batch schema discovery and user discovery in parallel
    print(f"📊 Fetching schemas for {len(db_ids)} databases and users in parallel...")
    
    # Create tasks for all schema fetches
    schema_tasks = []
    for db in db_ids:
        schema_tasks.append({
            "name": "API-retrieve-a-database",
            "args": {"database_id": db},
            "id": f"schema_{db}"
        })
    
    # Add user fetch task
    user_task = {
        "name": "API-get-users",
        "args": {},
        "id": "users"
    }
    
    # Execute all in one batch
    all_tasks = schema_tasks + [user_task]
    resp = await node.ainvoke({
        "messages": [AIMessage(content="", tool_calls=all_tasks)]
    })
    
    # Process responses
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            
            # Handle schema responses
            if m.tool_call_id.startswith("schema_"):
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
            
            # Handle user response
            elif m.tool_call_id == "users":
                for u in data.get("results", []):
                    know.add_user(u["id"], u.get("name", "Unknown"))

    know.last_updated = datetime.now()
    know.save_to_cache()  # Cache the results
    return know

# ────────────────── Optimized System Prompt ──────────────────
GUARDRAILS = """
## MCP Usage Rules
- Always use `"parent": {"database_id": "<DB_ID>"}`
- For each property:
  - status → {"status": {"name": "..."}}
  - select → {"select": {"name": "..."}}
  - date   → {"date": {"start": "YYYY-MM-DD"}}
  - people → {"people": ["<USER_ID>", ...]}
"""

MISSING_FIELD_GUIDE = """
## When creating a new record:
1. Identify required fields from schema
2. If missing, ask: "The '<FieldName>' field is required. Would you like to provide it now?"
3. Validate against type & options
4. Repeat until complete or user skips
"""

def make_system_prompt(know: UniversalWorkspaceKnowledge):
    return (
        "You are an interactive MCP‑driven Notion assistant.\n"
        "You know the full workspace schema and user list.\n"
        + GUARDRAILS
        + MISSING_FIELD_GUIDE
        + "\n" + know.context()
    )

# ──────────────────── Optimized Agent & Workflow ────────────────────
class State(MessagesState):
    pass

def trim_messages(messages: List, max_messages: int = 6) -> List:
    """Keep only the last max_messages (reduced from 10 to 6)"""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]

async def build_app():
    print("🚀 Initializing MCP client...")
    client = MultiServerMCPClient(notion_cfg)
    tools = await client.get_tools()
    
    # Use faster model configuration
    # llm = ChatAnthropic(
    #     model="claude-sonnet-4-20250514",
    #     api_key=CLAUDE_API_KEY, 
    #     temperature=0.1,
    #     max_tokens=1000,  # Limit response length for faster processing
    #     timeout=30  # Set timeout
    # ).bind_tools(tools)
    llm = ChatOpenAI(
        model="gpt-5",
        api_key=OPENAI_API_KEY, 
        temperature=1
        # max_tokens=1000,  # Limit response length for faster processing
        # timeout=30  # Set timeout
    ).bind_tools(tools)
    
    node = ToolNode(tools)

    # Try to load from cache first
    know = UniversalWorkspaceKnowledge.load_from_cache()
    if know is None:
        print("🔄 Refreshing workspace schema...")
        know = UniversalWorkspaceKnowledge()
        await discover_structure_parallel(node, know)
    
    print(f"✅ Workspace ready! Found {len(know.entities)} databases, {len(know.users)} users")

    # Pre-compile system prompt to avoid regeneration
    system_prompt = make_system_prompt(know)
    sys_message = SystemMessage(content=system_prompt)

    async def agent_fn(state: State):
        hist = trim_messages(state["messages"], max_messages=4)  # Even more aggressive trimming
        try:
            resp = await asyncio.wait_for(
                llm.ainvoke([sys_message] + hist), 
                timeout=25  # Timeout for LLM calls
            )
            return {"messages": trim_messages(hist + [resp], max_messages=6)}
        except asyncio.TimeoutError:
            error_msg = AIMessage(content="⏰ Request timed out. Please try a simpler query.")
            return {"messages": trim_messages(hist + [error_msg], max_messages=6)}

    def router(state: State) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    async def tool_node_wrapper(state: State):
        """Optimized tool node wrapper with timeout"""
        try:
            result = await asyncio.wait_for(node.ainvoke(state), timeout=20)
            result["messages"] = trim_messages(result["messages"], max_messages=6)
            return result
        except asyncio.TimeoutError:
            error_msg = ToolMessage(
                content="⏰ Tool execution timed out", 
                tool_call_id="timeout"
            )
            return {"messages": trim_messages(state["messages"] + [error_msg], max_messages=6)}

    wf = StateGraph(State)
    wf.add_node("agent", agent_fn)
    wf.add_node("tools", tool_node_wrapper)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    wf.add_edge("tools", "agent")
    
    return wf.compile(checkpointer=MemorySaver())

# ──────────────────────────── Optimized Main ────────────────────────────
async def main():
    try:
        app = await build_app()
        print("✨ Ready for MCP‑driven Notion interaction. Type 'quit' to exit.\n")
        config = {"configurable": {"thread_id": "mcp_session"}}

        while True:
            try:
                q = input("You: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                
                print("🤖 Assistant:", end=" ", flush=True)
                
                # Add timeout for the entire conversation turn
                result = await asyncio.wait_for(
                    app.ainvoke({"messages": [HumanMessage(content=q)]}, config=config),
                    timeout=45  # 45 second total timeout
                )
                
                out = result["messages"][-1].content
                print(out, "\n")
                
            except asyncio.TimeoutError:
                print("⏰ Request timed out. Please try again with a simpler query.\n")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
                
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    asyncio.run(main())