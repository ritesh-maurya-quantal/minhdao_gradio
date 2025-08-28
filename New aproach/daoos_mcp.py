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
import gradio as gr  # Add this import

from langchain_openai import ChatOpenAI
# LangChain / LangGraph / MCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
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

# Use environment variables ONLY (do not hardcode secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
NOTION_MCP_TOKEN = os.getenv("NOTION_MCP_TOKEN")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set in environment.")
if not NOTION_MCP_TOKEN:
    logger.warning("NOTION_MCP_TOKEN is not set in environment.")

NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")
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
                "Authorization": f"Bearer {NOTION_MCP_TOKEN}" if NOTION_MCP_TOKEN else "",
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
        name = self.entities.get(db_id).name if db_id in self.entities else db_id
        self.schemas[db_id] = EntitySchema(db_id, name, props)
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
                        opts = " | opts: " + ", ".join(o.get("name", "") for o in pdet.constraints.options)
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

    logger.info("🔍 Discovering databases and users in parallel...")

    parallel_tasks = [
        {
            "name": "API-post-search",
            "args": {"filter": {"property": "object", "value": "database"}},
            "id": "db_search",
        },
        {
            "name": "API-get-users",
            "args": {},
            "id": "users",
        },
    ]

    resp = await node.ainvoke({
        "messages": [AIMessage(content="", tool_calls=parallel_tasks)]
    })

    db_ids: List[str] = []

    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            try:
                data = json.loads(m.content)
            except Exception:
                logger.warning("Non-JSON tool response during discovery; skipping.")
                continue

            if m.tool_call_id == "db_search":
                for item in data.get("results", []):
                    if item.get("object") == "database":
                        eid = item["id"]
                        title = item.get("title") or []
                        name = "".join(t.get("plain_text", "") for t in title) or eid
                        know.add_entity(eid, name)
                        db_ids.append(eid)

            elif m.tool_call_id == "users":
                for u in data.get("results", []):
                    know.add_user(u.get("id", ""), u.get("name", "Unknown"))

    # Step 2: Batch schema discovery with controlled concurrency
    if db_ids:
        logger.info(f"📊 Fetching schemas for {len(db_ids)} databases...")
        BATCH_SIZE = 5
        for i in range(0, len(db_ids), BATCH_SIZE):
            batch = db_ids[i:i + BATCH_SIZE]
            schema_tasks = [
                {
                    "name": "API-retrieve-a-database",
                    "args": {"database_id": db},
                    "id": f"schema_{db}",
                }
                for db in batch
            ]

            batch_resp = await node.ainvoke({
                "messages": [AIMessage(content="", tool_calls=schema_tasks)]
            })

            for m in batch_resp["messages"]:
                if isinstance(m, ToolMessage) and m.tool_call_id.startswith("schema_"):
                    try:
                        data = json.loads(m.content)
                    except Exception:
                        logger.warning("Non-JSON schema response; skipping.")
                        continue

                    db_id = m.tool_call_id.replace("schema_", "")
                    props: Dict[str, PropertyDetails] = {}
                    for pname, pdat in data.get("properties", {}).items():
                        typ = pdat.get("type", "unknown")
                        req = pname.lower() in ("name", "title")
                        opts = None
                        if typ in ("select", "status", "multi_select"):
                            opts = (pdat.get(typ) or {}).get("options")
                        props[pname] = PropertyDetails(pname, typ, FieldConstraints(req, opts))
                    know.add_schema(db_id, props)

    know.last_updated = datetime.now()
    await know.save_to_cache()
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

# ─────────────────── Bundle-aware history stitching helpers ───────────────────
def _bundle_messages(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """
    Convert a flat list into bundles:
    - Normal message -> its own bundle [msg]
    - Assistant-with-tool_calls -> [assistant, tool_msg1, tool_msg2, ...] if all tool responses follow contiguously.
      If any tool response missing, the entire assistant-with-tools bundle is DROPPED (keeps history valid).
    """
    bundles: List[List[BaseMessage]] = []
    i = 0
    n = len(messages)

    while i < n:
        m = messages[i]

        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_ids = [tc.get("id") for tc in (m.tool_calls or []) if isinstance(tc, dict)]
            idset = set(tool_ids)
            j = i + 1
            collected_tool_msgs: List[ToolMessage] = []

            while j < n and isinstance(messages[j], ToolMessage) and messages[j].tool_call_id in idset:
                collected_tool_msgs.append(messages[j])
                idset.remove(messages[j].tool_call_id)
                j += 1

            if idset:
                logger.warning("Dropped assistant tool-call bundle due to missing tool replies; keeping history valid.")
                i = j
                continue
            else:
                bundles.append([m] + collected_tool_msgs)
                i = j
                continue

        bundles.append([m])
        i += 1

    return bundles


def _flatten(bundles: List[List[BaseMessage]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for b in bundles:
        out.extend(b)
    return out


def _trim_bundles_from_start(bundles: List[List[BaseMessage]], max_messages: int) -> List[List[BaseMessage]]:
    """
    Trim from the start while keeping bundles intact (never split an assistant-with-tools + its tool replies).
    max_messages counts individual messages, not bundles.
    """
    total = sum(len(b) for b in bundles)
    if total <= max_messages:
        return bundles

    trimmed = list(bundles)
    while trimmed and sum(len(b) for b in trimmed) > max_messages:
        trimmed.pop(0)
    return trimmed

# ──────────────────── High-Performance Agent & Workflow ────────────────────
class OptimizedState(MessagesState):
    context: Optional[str] = None  # Store context to avoid regeneration

async def build_optimized_app():
    logger.info("🚀 Initializing optimized MCP client...")

    # MCP client + tools
    client = MultiServerMCPClient(notion_cfg)
    tools = await asyncio.wait_for(client.get_tools(), timeout=30)

    # Optimized model configuration with streaming
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        # temperature=0.1,
        max_tokens=2000,
        timeout=120,
        # streaming=True,
        max_retries=2,
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

    # Custom agent function with strict, safe history
    async def agent_fn(state: OptimizedState):
        all_messages = state["messages"]

        # ---- Build a safe, strictly valid history for the LLM ----
        # Exclude any prior system messages; we insert our current system prompt at the top.
        conversation_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]

        # Bundle to ensure no orphaned tool-calls sneak in
        bundles = _bundle_messages(conversation_msgs)

        # Keep recent context but NEVER split bundles
        bundles = _trim_bundles_from_start(bundles, max_messages=16)  # tune as needed

        valid_messages = _flatten(bundles)

        # Final payload sent to LLM must start with the current system prompt
        prepared_messages: List[BaseMessage] = [sys_message] + valid_messages

        try:
            task = asyncio.create_task(llm.ainvoke(prepared_messages))
            resp = await asyncio.wait_for(task, timeout=30)
            return {"messages": all_messages + [resp]}
        except asyncio.TimeoutError:
            error_msg = AIMessage(content="⏰ Request timed out. Please try a simpler query.")
            return {"messages": all_messages + [error_msg]}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            # Clean reset fallback: keep only latest human message (if any)
            latest_human = None
            for msg in reversed(all_messages):
                if isinstance(msg, HumanMessage):
                    latest_human = msg
                    break

            if latest_human:
                try:
                    resp = await asyncio.wait_for(
                        llm.ainvoke([sys_message, latest_human]),
                        timeout=20
                    )
                    return {"messages": [latest_human, resp]}
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
            # IMPORTANT: pass the entire state (it contains the 'messages' key),
            # ToolNode expects {"messages":[...]} in the state.
            result = await asyncio.wait_for(node.ainvoke(state), timeout=40)
            return result
        except asyncio.TimeoutError:
            logger.warning("Tool execution timed out")

            # Find the last AI message with tool calls to respond to
            last_ai_msg: Optional[AIMessage] = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    last_ai_msg = msg
                    break

            if last_ai_msg and last_ai_msg.tool_calls:
                timeout_responses: List[ToolMessage] = []
                for tool_call in last_ai_msg.tool_calls:
                    timeout_responses.append(
                        ToolMessage(
                            content="⏰ Tool execution timed out. Please try a simpler operation.",
                            tool_call_id=tool_call.get("id"),
                        )
                    )
                return {"messages": state["messages"] + timeout_responses}
            else:
                error_msg = AIMessage(content="⏰ Tool execution timed out. Please try a simpler operation.")
                return {"messages": state["messages"] + [error_msg]}
        except Exception as e:
            logger.error(f"Tool error: {e}")

            last_ai_msg: Optional[AIMessage] = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    last_ai_msg = msg
                    break

            if last_ai_msg and last_ai_msg.tool_calls:
                error_responses: List[ToolMessage] = []
                for tool_call in last_ai_msg.tool_calls:
                    error_responses.append(
                        ToolMessage(
                            content=f"❌ Tool error: {str(e)}",
                            tool_call_id=tool_call.get("id"),
                        )
                    )
                return {"messages": state["messages"] + error_responses}
            else:
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

                # Use a clean thread ID per turn to avoid accidental long contexts
                thread_id = f"mcp_session_{int(asyncio.get_event_loop().time())}"
                config = {"configurable": {"thread_id": thread_id}}

                start_time = asyncio.get_event_loop().time()
                task = asyncio.create_task(
                    app.ainvoke({"messages": [HumanMessage(content=q)]}, config=config)
                )
                result = await asyncio.wait_for(task, timeout=60)
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

# Minimal async chat handler for Gradio with per-session history
async def gradio_chat(user_input, history):
    if history is None:
        history = []
    # Build messages from history
    messages = []
    for msg in history:
        # msg is a dict with 'role' and 'content'
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))

    # Build the app/workflow once and cache it
    if not hasattr(gradio_chat, "app"):
        gradio_chat.app = await build_optimized_app()
    app = gradio_chat.app

    thread_id = f"gradio_session_{len(history)}_{int(asyncio.get_event_loop().time())}"
    config = {"configurable": {"thread_id": thread_id}}

    import time
    start_time = time.perf_counter()
    try:
        result = await app.ainvoke({"messages": messages}, config=config)
        bot_reply = result["messages"][-1].content
    except Exception as e:
        bot_reply = f"❌ Error: {e}"
    elapsed = time.perf_counter() - start_time

    # Append as OpenAI-style dicts for gr.Chatbot(type='messages')
    history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": f"{bot_reply}\n\n⏱️ Response time: {elapsed:.2f}s"},
    ]
    return history, history

def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# Notion MCP AI Assistant\nAsk anything about your Notion workspace.")
        chatbot = gr.Chatbot(type='messages')  # Use OpenAI-style messages
        state = gr.State([])  # Per-session chat history

        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Type your message and press Enter")

        txt.submit(
            gradio_chat,
            inputs=[txt, state],
            outputs=[chatbot, state],
            queue=True,
            api_name="chat"
        )
    demo.launch(share=False)

# If this is the main module, start the Gradio interface
if __name__ == "__main__":
    launch_gradio()