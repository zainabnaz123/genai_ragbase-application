# # Importing necessary components from Langchain
# from urllib import request
# from langchain.agents import AgentExecutor, create_tool_calling_agent

# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitter import RecursiveCharacterTextSplitter
# from langchain_tools_retriever import create_retriever_tool
# from langchain_memory_chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_tools import tool

# # Importing other modules
# import os

# from dotenv import load_dotenv 

# # Load environment variables
# load_dotenv()

# # Initialize the LLM with GoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
#                              google_api_key=os.getenv("GOOGLE_API_KEY"))

# # Tavily API search setup
# search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

# # Loading documents from the web
# loader = WebBaseLoader("https://www.techloset.com/")
# docs = loader.load()

# # Split documents into chunks for processing
# documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# # Create vector embeddings from the documents
# vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# # Set up a retriever to fetch results
# retriever = vector.as_retriever()

# # Create a tool for the retriever
# retriever_tool = create_retriever_tool(
#     retriever, 
#     "techloset_search", 
#     "Search for information about Techloset. For any questions about Techloset Solutions, you must use this tool!"
# )

# # FDA Search Tool for querying FDA data
# @tool
# def fda_search_tool(query: str) -> str:
#     """Tool for querying FDA's open API data."""
#     base_url = "https://api.fda.gov/drug/event.json"
#     params = {
#         'api_key': os.getenv("FDA_API_KEY"),  # FDA API Key from the .env file
#         'search': query,
#         'limit': 10  # Limit the number of results
#     }
#     response = request.get(base_url, params=params)
    
#     if response.status_code == 200:
#         return str(response.json())  # Return the JSON data as a string
#     else:
#         return f"Error: {response.status_code} - Could not fetch FDA data."

# # Adding tools
# tools = [search, retriever_tool, fda_search_tool]

# # Pulling the prompt
# prompt = hub.pull("hwchase17/openai-functions-agent")

# # Creating the tool-calling agent
# agent = create_tool_calling_agent(llm, tools, prompt)

# # Creating agent executor
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Setting up chat history
# message_history = ChatMessageHistory()

# # Wrapping agent with message history
# agent_with_chat_history = RunnableWithMessageHistory(
#     agent_executor,
#     lambda session_id: message_history,  # Simulate session ID for history
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )

# # Main loop to invoke the agent
# while True:
#     user_input = input("How can I help you today? : ")
#     agent_with_chat_history.invoke(
#         {"input": user_input},
#         config={"configurable": {"session_id": "test123"}}
#     )
