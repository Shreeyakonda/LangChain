from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

@tool
def get_current_date() -> str:
    """Return the current data as a string in YYYY-MM-DD format."""
    from datetime import date
    return date.today().isoformat()

llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools([get_current_date])

user_msg = HumanMessage(content="What is today's date?")
ai = llm_with_tools.invoke([user_msg])
msgs = [user_msg, ai]

for call in getattr(ai, "tool_calls", []) or []:
    tool_name = call["name"]
    args = call.get("args", {})

    if tool_name == "get_current_date":
        result = get_current_date.invoke(args)
        msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final = llm_with_tools.invoke(msgs)
print(final.content)