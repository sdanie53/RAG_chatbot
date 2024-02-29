import os
from pathlib import Path
import chainlit as cl
from chainlit.server import app
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from embed import load_config, create_embeddings, initialize_index, load_files
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import time


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def memory_setup():
    try:
        Path("memory").mkdir(parents=True, exist_ok=True)
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=message_history,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        return memory
    except Exception as e:
        raise Exception(f"An error occurred in memory_setup: {str(e)}")


def init_pinecone(index: Pinecone, embed_model: OpenAIEmbeddings):
    try:
        vectordb = PineconeVectorStore(index, embed_model)
        return vectordb
    except Exception as e:
        raise Exception(f"An error occurred in fetch_db: {str(e)}")

       
def conversation_chain(config: dict, index: Pinecone, embed_model: OpenAIEmbeddings):
    try:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        memory = memory_setup()
        vectordb = init_pinecone(index, embed_model)
        #retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        model = ChatOpenAI(
            callback_manager=callback_manager,
            api_key=OPENAI_API_KEY,
            temperature=config["openai"]["temperature"],
            model=config["openai"]["chat_model"],
            max_tokens=config["openai"]["max_tokens"],
        )


        # Add chat prompt template to the chain
        chain = ConversationalRetrievalChain.from_llm(
            chain_type="stuff",
            llm=model,
            memory=memory,
            retriever=vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        )


        return chain
    except Exception as e:
        raise Exception(f"An error occurred in conversation_chain: {str(e)}")


# @cl.on_chat_start
# async def on_chat_start():
#     "Send a chat start message to the chat and load the model config."
#     try:
#         config = load_config()
#         cl.user_session.set("config", config)
#         chain = conversation_chain(config)
#         cl.user_session.set("chain", chain)

#         await cl.Message(content=config["introduction"], disable_feedback=True).send()
#     except Exception as e:
#         await cl.Message(content=f"An error occurred: {str(e)}").send()

# @cl.on_message
# async def on_message(message: cl.Message):
#     """Handle the incoming message and update the chainlit model."""
#     try:
#         chain = cl.user_session.get("chain")
#         config = cl.user_session.get("config")

#         res = await cl.make_async(chain)(
#             message.content,
#             callbacks=[cl.LangchainCallbackHandler()]
#             )
#         await cl.Message(content=res["answer"]).send()
#     except Exception as e:
#         await cl.Message(content=f"An error occurred: {str(e)}").send()
        
if __name__ == "__main__":
    config = load_config()
    embed_model = OpenAIEmbeddings(
        model=config["openai"]["embeddings_model"],
        openai_api_key=OPENAI_API_KEY
    )
    index = initialize_index(config["pc"]["index_name"], config, PINECONE_API_KEY)
    chain = conversation_chain(config, index, embed_model)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        res = chain.invoke(user_input)
        print(f"Bot: {res['answer']}")
