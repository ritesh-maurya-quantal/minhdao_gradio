import os
import json
import uuid
import asyncio
from typing import List, Any
from datetime import datetime
from dataclasses import dataclass, field

import gradio as gr
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
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

# Prefer env vars; you can temporarily hard‑code for dev
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
NOTION_MCP_TOKEN = os.getenv("NOTION_MCP_TOKEN")
NOTION_VERSION   = "2022-06-28"

MODEL_PROVIDER    = "anthropic"
ANTHROPIC_MODEL   = "claude-3-5-sonnet-20241022"

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

# ─────────────── Workspace Knowledge (for system prompt) ───────────────
@dataclass
class FieldConstraints:
    required: bool = False
    options: list[dict[str, Any]] | None = None

@dataclass
class PropertyDetails:
    name: str
    type: str
    constraints: FieldConstraints = field(default_factory=FieldConstraints)

@dataclass
class EntitySchema:
    id: str
    name: str
    properties: dict[str, PropertyDetails] = field(default_factory=dict)

@dataclass
class WorkspaceEntity:
    id: str
    name: str

class UniversalWorkspaceKnowledge:
    def __init__(self):
        self.entities: dict[str, WorkspaceEntity] = {}
        self.schemas: dict[str, EntitySchema] = {}
        self.users: dict[str, str] = {}       # name → id
        self.last_updated: datetime = datetime.now()

    def add_entity(self, eid: str, name: str):
        self.entities[eid] = WorkspaceEntity(eid, name)

    def add_schema(self, db_id: str, props: dict[str, PropertyDetails]):
        self.schemas[db_id] = EntitySchema(db_id, self.entities[db_id].name, props)

    def add_user(self, uid: str, name: str):
        self.users[name] = uid

    def context(self) -> str:
        lines = ["### Workspace Schema (Discovered)"]
        for eid, ent in self.entities.items():
            lines.append(f"\n**{ent.name}** (DB ID: `{eid}`)")
            schema = self.schemas.get(eid)
            if schema:
                for pname, pdet in schema.properties.items():
                    req = " _(required)_" if pdet.constraints.required else ""
                    opts = ""
                    if pdet.constraints.options:
                        opts = " — options: " + ", ".join(o.get("name", "") for o in pdet.constraints.options)
                    lines.append(f"- `{pname}`: `{pdet.type}`{req}{opts}")
        if self.users:
            lines.append("\n### Users")
            for name, uid in self.users.items():
                lines.append(f"- {name} → `{uid}`")
        lines.append(f"\n_Last updated: {self.last_updated.isoformat(timespec='seconds')}_")
        return "\n".join(lines)

# ────────────────────── Discovery via MCP ──────────────────────
async def discover_structure(node: ToolNode, know: UniversalWorkspaceKnowledge):
    # 1) Databases
    resp = await node.ainvoke({
        "messages":[AIMessage(content="", tool_calls=[{
            "name":"API-post-search",
            "args":{"filter":{"property":"object","value":"database"}},
            "id":"db_search"
        }])]}
    )
    db_ids = []
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            for item in data.get("results", []):
                if item.get("object") == "database":
                    eid = item["id"]
                    title = item.get("title", [])
                    name = "".join(t.get("plain_text", "") for t in title) or "Untitled"
                    db_ids.append(eid)
                    # store entity
                    know.add_entity(eid, name)

    # 2) Schemas
    for db in db_ids:
        resp = await node.ainvoke({
            "messages":[AIMessage(content="", tool_calls=[{
                "name":"API-retrieve-a-database",
                "args":{"database_id":db},
                "id":"schema"
            }])]}
        )
        for m in resp["messages"]:
            if isinstance(m, ToolMessage):
                data = json.loads(m.content)
                props = {}
                for pname, pdat in data.get("properties", {}).items():
                    typ = pdat.get("type", "unknown")
                    req = pname.lower() in ("name","title")
                    opts = None
                    if typ in ("select","status","multi_select"):
                        opts = pdat.get(typ,{}).get("options")
                    props[pname] = PropertyDetails(pname, typ, FieldConstraints(req, opts))
                know.add_schema(db, props)

    # 3) Users
    resp = await node.ainvoke({
        "messages":[AIMessage(content="", tool_calls=[{
            "name":"API-get-users","args":{},"id":"users"
        }])]}
    )
    for m in resp["messages"]:
        if isinstance(m, ToolMessage):
            data = json.loads(m.content)
            for u in data.get("results",[]):
                know.add_user(u["id"], u.get("name","Unknown"))

    know.last_updated = datetime.now()
    return know

# ────────────────── System Prompt & Guardrails ──────────────────
GUARDRAILS = """
## MCP Usage Rules
- Always use `"parent": {"database_id": "<DB_ID>"}` for page creation.
- Property mappings:
  - status → {"status": {"name": "..."}}
  - select → {"select": {"name": "..."}}
  - date   → {"date": {"start": "YYYY-MM-DD"}}
  - people → {"people": [{"id": "<USER_ID>"} , ...]}
"""

MISSING_FIELD_GUIDE = """
## When creating a new record:
1. Identify required fields from the schema.
2. If required fields are missing, ask:
   "The '<FieldName>' field is required. Would you like to provide it now?"
3. Validate the reply by type & allowed options.
4. Repeat until required fields are handled or the user opts to skip.
"""

def make_system_prompt(know: UniversalWorkspaceKnowledge):
    return (
        "You are an interactive MCP‑driven Notion assistant.\n"
        "Decide per turn:\n"
        " - Call a Notion tool only if new/up-to-date data is required.\n"
        " - If the answer is in prior conversation/tool results, reply directly.\n"
        + GUARDRAILS
        + MISSING_FIELD_GUIDE
        + "\n\n" + know.context()
    )

# ──────────────────── Agent & Workflow ────────────────────
class State(MessagesState):
    """State wrapper (same as MessagesState, but lets us extend later)."""
    pass

def trim_messages(messages: List, max_messages: int = 10) -> List:
    return messages if len(messages) <= max_messages else messages[-max_messages:]

async def build_app_and_knowledge():
    # Start Notion MCP tools
    client = MultiServerMCPClient(notion_cfg)
    tools = await client.get_tools()

    # Choose model provider
    llm = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=CLAUDE_API_KEY, temperature=0.1).bind_tools(tools)

    tool_node = ToolNode(tools)

    # Discover workspace structure (once, at startup)
    know = UniversalWorkspaceKnowledge()
    await discover_structure(tool_node, know)

    async def agent_fn(state: State):
        sys = SystemMessage(content=make_system_prompt(know))
        hist = trim_messages(state["messages"])
        resp = await llm.ainvoke([sys] + hist)
        return {"messages": trim_messages(hist + [resp])}

    def router(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    async def tool_node_wrapper(state: State):
        result = await tool_node.ainvoke(state)
        result["messages"] = trim_messages(result["messages"])
        return result

    wf = StateGraph(State)
    wf.add_node("agent", agent_fn)
    wf.add_node("tools", tool_node_wrapper)
    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", router, {"tools":"tools", END:END})
    wf.add_edge("tools", "agent")

    app = wf.compile(checkpointer=MemorySaver())
    return app

# ─────────────────────────────── Gradio Chat UI ───────────────────────────────
APP = asyncio.run(build_app_and_knowledge())

async def respond(message, ui_history, thread_id):
    """Chat turn handler: runs the graph and updates history."""
    if ui_history is None:
        ui_history = []
    if not message:
        return "", ui_history, ui_history

    # Append user message (for the Chatbot + State)
    ui_history.append({"role": "user", "content": message})

    # Run graph; MemorySaver loads previous state via thread_id
    result = await APP.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract the latest AI message content
    ai_text = "No response."
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            ai_text = m.content
            break

    ui_history.append({"role": "assistant", "content": ai_text})
    # Return: clear textbox, update Chatbot, update ui_history State
    return "", ui_history, ui_history

def reset_chat():
    """Reset chat history and start a fresh thread (new memory)."""
    return [], str(uuid.uuid4()), gr.update(value=[])

with gr.Blocks(css="""
#chatbot_interface {background:#f8f9fb; padding:16px; border-radius:12px;}
""") as demo:
    gr.Markdown("## 🔗 Notion MCP Agent — Agentic RAG (Chat)")

    chatbot = gr.Chatbot(
        label="Chat",
        elem_id="chatbot_interface",
        type="messages",
        height=520
    )
    msg = gr.Textbox(
        label="Your message",
        lines=2,
        placeholder="Ask something about your Notion data…"
    )
    send_btn = gr.Button("Send", variant="primary")
    reset_btn = gr.Button("Reset Conversation")

    # Internal states
    ui_history = gr.State([])                       # list of {"role","content"}
    thread_id  = gr.State(str(uuid.uuid4()))        # MemorySaver thread

    # Wire events — note we also output ui_history State so it persists between turns
    send_btn.click(
        respond,
        inputs=[msg, ui_history, thread_id],
        outputs=[msg, chatbot, ui_history],
    )
    msg.submit(
        respond,
        inputs=[msg, ui_history, thread_id],
        outputs=[msg, chatbot, ui_history],
    )

    # Reset: new state + clear Chatbot
    reset_btn.click(
        reset_chat,
        outputs=[ui_history, thread_id, chatbot],
    )

if __name__ == "__main__":
    # Set server_name="0.0.0.0" for LAN/mobile access if needed
    demo.queue().launch()
