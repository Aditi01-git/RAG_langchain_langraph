from rag_pipeline import RAGPipeline
from graph_builder import build_graph
from transformers import pipeline

rag = RAGPipeline(file_path="NetBackup105_AdminGuide_OpenStack.pdf")

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)

graph = build_graph(rag, llm)

result = graph.invoke({
    "query": "How to install NetBackup on OpenStack?",
    "retries": 0
})

print(result.get("answer"))

results =evaluate(eval_data=eval_data, rag=rag, k=3)

avg_recall = sum(results['recall'])/ len(results["recall"])
print(f"Recall@k: {avg_recall:.3f}")


avg_mrr = sum(results['mrr'])/ len(results["mrr"])
print(f"MRR: {avg_mrr:.3f}")

#Recall@k: 0.643
#MRR: 0.548