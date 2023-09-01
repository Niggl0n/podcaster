__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import logging
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

#loader = WebBaseLoader("https://www.theringer.com/nba/2023/7/26/23808374/jaylen-brown-biggest-contract-in-nba-history")
#index = VectorstoreIndexCreator().from_loaders([loader])
#index.query("What team does Jaylen Brown play for? ")



loader = WebBaseLoader("https://www.theringer.com/nba/2023/7/26/23808374/jaylen-brown-biggest-contract-in-nba-history")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# using vector similartiy to retrieve documents
question = "What team does Jaylen Brown play for?"
docs = vectorstore.similarity_search(question)
len(docs)

# using svm to retrieve documents
svm_retriever = SVMRetriever.from_documents(all_splits, OpenAIEmbeddings()) #all_splits,OpenAIEmbeddings()
docs_svm=svm_retriever.get_relevant_documents(question)
len(docs_svm)


logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
#
# retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
#                                                   llm=ChatOpenAI(temperature=0))
# unique_docs = retriever_from_llm.get_relevant_documents(query=question)
# len(unique_docs)



# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
# result = qa_chain({"query": question})
# print(result["result"])

from langchain.llms import GPT4All
model4all = GPT4All(model="models/orca-mini-7b.ggmlv3.q4_0.bin")  # (model="./models/gpt4all-model.bin", n_threads=8)
qa_chain = RetrievalQA.from_chain_type(model4all,retriever=vectorstore.as_retriever())
results_4all = qa_chain({"query": question})

print("dsada")
