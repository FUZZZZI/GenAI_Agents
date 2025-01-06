import os, sys
from dotenv import load_dotenv
load_dotenv()
from phi.agent import Agent

from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

from phi.model.groq import Groq

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['PHI_API_KEY'] = os.getenv('PHI_API_KEY')

# Web search agent
web_search_agent = Agent(
    name='Web Search Agent',
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"), 
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Finance agent
finance_agent = Agent(
    name='Finance AI Agent',
    # role="Get financial information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"), 
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True, stock_fundamentals=True,),
        ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "use table to display the data"],
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVIDIA Corp", stream=True)
