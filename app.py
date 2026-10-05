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

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Never use tables in your answers. "

        "For weather, create a separate 'Weather' section. "
        "Present the weather in short, plain sentences. "
        "Include important details such as temperature, conditions, humidity, wind, "
        "and any relevant weather alerts. "

        "For news, create a separate 'Latest News' section. "
        "Present news as bullet points. "
        "Each news item should be detailed and written as a separate paragraph. "
        "Include enough context to explain what happened and why it is important. "

        "Always keep weather and news visually separate. "
        "Do not mix weather information with news items. "
        "Use clear headings so the user can easily distinguish between the Weather "
        "and Latest News sections."
    )
)


@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", temperature=0.2)
    return ChatHuggingFace(llm=llm).bind_tools(list(tools.values()))


llm = get_llm()
st.title("🌆 City Intelligence System")
messages = st.session_state.setdefault("messages", [])

# Show history (user messages and final AI answers only)
for m in messages:
    if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls):
        st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").markdown(m.content)

# Chat already closed: show a notice and hide the input box
if st.session_state.get("closed"):
    st.info("Chat closed. Refresh the page to start a new chat.")
    st.stop()

if user_input := st.chat_input("Ask about weather or news in a city..."):
    # Exit / quit closes the chat
    if user_input.strip().lower() in ["exit", "quit"]:
        st.session_state.closed = True
        st.rerun()

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