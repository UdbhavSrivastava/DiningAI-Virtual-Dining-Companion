import gradio as gr
from huggingface_hub import InferenceClient
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
import gradio as gr

print("Current Working Directory:", os.getcwd())
with open('./openai_api_key.txt') as f:
    openai_api_key = f.read()

embedding_llm = OpenAIEmbeddings(openai_api_key=openai_api_key)

docs_base = Path("./docs")
files_all = docs_base.rglob("*")
excludes = [".DS_Store", ".ipynb_checkpoints"]
files = []
for f in files_all:
    to_exclude = False
    for ex in excludes:
        to_exclude = to_exclude or f.match(ex)
    if not to_exclude and f.is_file():
        files.append(f)

documents = []
for file in files:
    #print(file)
    loader = TextLoader(str(file.absolute()))
    documents += loader.load()

db = Chroma.from_documents(documents, embedding_llm, persist_directory="./chroma",
                          collection_name="olivegrove")
db.persist()

query = "opening hours"
result = db.similarity_search_with_relevance_scores(query)
print(len(result))

db.max_marginal_relevance_search(query, k=2)

retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': 2, 'fetch_k': 5})

retriever.invoke(query)


retriever_tool = create_retriever_tool(
    db.as_retriever(search_type='mmr'),
    name = "Olivegrove_search",
    description = """Search for information about Olive Grove store, including general information, 
    Food menus, dishes fact sheet.""",
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

system_message = """You are a helpful assistant working at a Greek Restaurant.
You can use a given chat history and given tools to respond to a user.
Give nutritional information and summary if asked about a particular item from the menu.
Your character is a polite and friendly female.
You answer concisely, without introduction or appending.
If you do not know the answer, just say 'I am sorry but I cannot help you with this. A store person will be there at your kiosk shortly. Thank you!'
When you calculate the price of an ordered item, you should think step-by-step.
Add 10 percent tax on the final bill.
Ask for the payment mode when user ask to generate a bill.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_message)),
        MessagesPlaceholder(
            variable_name="chat_history", optional=True
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the human input will injected
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),  # Where the memory will be stored.
    ]
)

tools = [retriever_tool, ]

agent = create_openai_functions_agent(llm, tools, prompt,)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

def wrapper_chat_history(chat_history, memory):
    chat_history = []
    for m in memory.chat_memory.messages:
        chat_history.append(m.content)
    return chat_history

def converse(message, chat_history):
    response = agent_executor.invoke({"input": message})
    chat_history = wrapper_chat_history(chat_history, memory)
    return response['output']

theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="pink",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="md",
).set(
    body_background_fill='*background_fill_secondary',
    body_background_fill_dark='*checkbox_background_color_focus',
    body_text_color='*neutral_950',
    background_fill_primary_dark='*table_even_background_fill'
)

# Create the ChatInterface and apply the theme
demo = gr.ChatInterface(fn=converse, title="Olive Grove's Greek Restaurant Chatbot", theme=theme)

demo.launch(share=False)
