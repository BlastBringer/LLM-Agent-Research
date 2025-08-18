from langchain.agents import initialize_agent, AgentType 
from tools import calculator
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

tools = [calculator]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

agent = initialize_agent(
    tools=tools,
    llm = llm,
    agent =AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
