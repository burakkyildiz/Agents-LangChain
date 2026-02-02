from dotenv import load_dotenv
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
#from langgraph.prebuilt import create_react_agents
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

model = ChatOllama(model='llama3.1:8b')
memory = SqliteSaver.from_conn_string(":memory:")
search = TavilySearch(max_results=2)

tools = [search]

model_with_tools = model.bind_tools(tools)

agent_executor = create_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

if __name__ == '__main__':
    while True:
        user_input = input(">")
        for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
        ):
            print(chunk, end='')
