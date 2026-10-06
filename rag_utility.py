import os                         # to access the path of the directories when we want it to access the files and everything
from dotenv import load_dotenv    # load .env to load the environment variables with this API key

from langchain_community.document_loaders import UnstructuredPDFLoader     # to read the file that the user is uploading
from langchain_text_splitters import RecursiveCharacterTextSplitter          # used to split our document text into smaller chunks
from langchain_huggingface import HuggingFaceEmbeddings                    # embedding model to convert text chunks into vector embeddings
from langchain_chroma import Chroma # 
from langchain_ollama import ChatOllama
# from langchain.chains import RetrievalQA    #(out of date, legacy release)   # chain for retrieval augmented generation where it combines the retrieved chunks and the user query and send it to the LM and finally gives you the response
from langchain_classic.chains import RetrievalQA 

#load_dotenv()

#directory where PDFs going to be saved
working_dir = os.path.dirname(os.path.abspath((__file__))) #absolute path of current dir

#load the embedding model
embedding = HuggingFaceEmbeddings()

#load the llm model from ollama
llm = ChatOllama(
    model = "llama3:latest",
    temperature = 0
)

#document ingestion
def process_document_to_chroma_db(file_name):
    #load the PDF 
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    #split text into chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    #store the chunks in Chroma BD
    vectordb = Chroma.from_documents(
        documents = texts,
        embedding = embedding,
        persist_directory = f"{working_dir}/doc_vectorstore"
    )
    return 0

#answer user questions
def answer_question(user_question):
    #load persistent chroma DB
    vectordb = Chroma(
            persist_directory = f"{working_dir}/doc_vectorstore",
            embedding_function = embedding,
    )

    #create retriever for document search
    retriever =vectordb.as_retriever()

    #create RetrievalQA chain to answer user que using llm
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer


# 1. python -m venv .venv
# 2. .\.venv\Scripts\Activate.ps1
# 3. python -m streamllit run .\app.py