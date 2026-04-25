from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain.schema import Document

import re


class DocumentProcessor:
    def __init__(self, file_path, start_page):
        self.file_path = file_path
        self.start_page = start_page
        
    def load_documents(self):
        # Placeholder for document loading logic
        print(f"Loading document from: {self.file_path}")
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()

        #Filter docs to avoid starting pages which contain titles and whole index which is not needed
        filtered_docs = [doc for doc in documents if doc.metadata['page'] >= self.start_page]
        return filtered_docs

    def clean_text(self, documents):
        cleaned_docs = []
        for doc in documents:
            text = doc.page_content        
            # Placeholder for text cleaning logic
            text = re.sub(r"Page \d+ of \d+", "", text) 
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n+", " ", text)
            cleaned_docs.append(Document (page_content = text.strip() 
                                    , metadata = doc.metadata))

        return cleaned_docs

    def split_docs(self, documents , chunk_size, chunk_overlap):
        # Split the text into chunks of a specified size
        splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ".", " "], chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        chunks = splitter.split_docs(documents)
        return chunks


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        self.vector_store = None

    def create_store(self, chunks):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

    def retrieve(self, query, top_k, fetch_k):
        if not self.vector_store:
            print("Vector Store not created or is empty")
            return None
        else:
            docs = self.vector_store.max_marginal_relevance_search(query, k = top_k, fetch_k = fetch_k)

            #Deduplicate returned docs from vector DB
            seen = set()
            unique_docs = []
            
            for doc in docs:
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    unique_docs.append(doc)
            

            return unique_docs


class RAGPipeline:
    def __init__(self, file_path):
        self.preprocessor = DocumentProcessor(file_path, start_page = 10)
        self.vector_store_manager = VectorStoreManager()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        loaded_docs = self.preprocessor.load_documents()
        cleaned_docs = self.preprocessor.clean_text(loaded_docs)
        chunks = self.preprocessor.split_docs(cleaned_docs, chunk_size=500, chunk_overlap=150)

        self.vector_store_manager.create_store(chunks)

    def rerank(self, query, docs):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        scored_docs= list(zip(docs, scores))
        scored_docs.sort(key = lambda x : x[1], reverse = True)

        return [doc for doc, score in scored_docs[:3]]

    def retrieve_and_rerank(self, query, top_k = 6 , fetch_k = 12):
        retrieved_docs = self.vector_store_manager.retrieve(query, top_k, fetch_k)
        if retrieved_docs:
            reranked_docs = self.rerank(query, retrieved_docs)
            return reranked_docs
        else:
            print("No documents retrieved.")
            return None
