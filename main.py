from dotenv import load_dotenv
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
#from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOllama(model = 'llama3.1:8b')
memory = InMemorySaver()
search = TavilySearch(max_results=2)
tools = [search]

agent_executor = create_agent(model, tools, checkpointer=memory)

config = {'configurable': {'thread_id' : 'abc0102'}}

system_prompt = SystemMessage(
    content="""
        You are a helpful AI assistant.
        
        Rules:
        - If the user greets you (e.g. "hi", "hello"), respond politely.
        - Do NOT use any tools for greetings or casual conversation.
        - Use tools ONLY if the user asks for factual, real-time, or external information.
"""
)

if __name__ == '__main__':

        while True:
            user_input = input("> ")
            for chunk in agent_executor.stream(
                    {'messages': [system_prompt, HumanMessage(content = user_input)]},
                    config
            ):
                print(chunk)
                print('---')
