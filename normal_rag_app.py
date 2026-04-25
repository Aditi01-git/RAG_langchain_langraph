from fastapi import FastAPI
from rag_pipeline import RAGPipeline
from transformers import pipeline

app = FastAPI()

rag = RAGPipeline(file_path = "NetBackup105_AdminGuide_OpenStack.pdf")

llm = pipeline("text-generation", model= "TinyLlama/TinyLlama-1.1B-Chat-v1.0", tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

def generate_response(query, docs):
    # take top 3 docs only
    context = "/n".join([doc.page_content for doc in docs[:3]])

    # Chat-style prompt for better response generation
    prompt = f"""You are a strict extraction-based assistant.

                ONLY extract steps from the context.
                DO NOT add any new steps.
                DO NOT give empty steps in the answer.
                DO NOT explain.
                DO NOT guess.

                If steps are incomplete, return only available steps.
                If not found, say: I don't know.

                <|user|>
                Context:
                {context}

                Question:
                {query}

                <|assistant|>
                """
    # Generate response using the LLM
    response = llm(prompt, max_new_tokens = 60 , do_sample=False, return_full_text = False, pad_token_id=llm.tokenizer.eos_token_id)[0]["generated_text"]

    #Cleanup the response
    stop_tokens = ["<|assistant|>", "<|user|>", "```", "##", "Table of Contents"]

    for token in stop_tokens:
        if token in response:
            response = response.split(token)[0]

    response = response.strip()

    # fallback safety
    if len(response) < 5:
        return "I don't know"
    
    return response

@app.get("/")
def home():
    return {"status": "RAG API running"}


@app.get("/query")
def query(q: str):
    # Retrieve docs from Retriever
    docs = rag.retrieve_and_rerank(q)

    #Generate response from llm
    response = generate_response(q, docs)

    return {"query": q, "response": response}
