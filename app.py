import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from agent_tools import tools

load_dotenv()

tools = {
    "get_weather": tools.get_weather,
    "get_news": tools.get_news
}

@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        temperature=0.2)
    return (ChatHuggingFace(llm=llm)
            .bind_tools(list(tools.values())))


llm = get_llm()
st.title("🌆 City Intelligence System")
messages = st.session_state.setdefault("messages", [])

SYSTEM_PROMPT = SystemMessage(
    content="Never use tables in your answers. Present news as a simple bulleted list "
            "or short paragraphs, and keep the weather in plain sentences."
)

# Show history (user messages and final AI answers only)
for m in messages:
    if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls):
        st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").markdown(m.content)

if user_input := st.chat_input("Ask about weather or news in a city..."):
    st.chat_message("user").markdown(user_input)
    messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        while True:
            result = llm.invoke([SYSTEM_PROMPT, *messages])
            messages.append(result)

            if not result.tool_calls:
                st.markdown(result.content)
                break

            for call in result.tool_calls:
                output = tools[call["name"]].invoke(call["args"])
                messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))