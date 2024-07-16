import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv, dotenv_values
import os

# Setting up the API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def read_data_from_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = file.read()
    
    document = Document(page_content=data)

    # Splitting Data into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents_chunks = text_splitter.split_documents([document])

    # create a vectorestore from the chunks
    vector_store = Chroma.from_documents(documents_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevent to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    return create_retrieval_chain(retriever=retriever_chain, combine_docs_chain=stuff_documents_chain)
    
# Getting Response
def get_response(user_input):
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain=retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

# App Config
st.set_page_config(page_title="LevoSupportAgent", page_icon="🤖")
st.title("Chat With Levo.ai Support Agent To resolve your queries!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Your Customer Support Agent form Levo.ai! How can I help You?"),
    ]
if "vector_store" not in st.session_state:
    filename = 'data.txt'
    st.session_state.vector_store = read_data_from_file(file_path=filename)

# Handling User input
user_query = st.chat_input("Type your query here...")
if user_query is not None and user_query != "":
    response_for_user = get_response(user_input=user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response_for_user))

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)