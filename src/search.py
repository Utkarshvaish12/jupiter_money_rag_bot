import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content
    
    def hybrid_search(self, query, top_k=5):
        vector_results = self.vectorstore.query(query, top_k=top_k)

        keyword_results = []
        for meta in self.vectorstore.metadata:
            text = meta.get("text", "")
            if query.lower() in text.lower():
                keyword_results.append({
                    "text": text,
                    "metadata": meta,
                    "score": 0.20
                })

        combined = vector_results + keyword_results
        return combined[:top_k]

    def rerank(self, query, results):
        query_lower = query.lower()

        def score(doc):
            text = doc.get("text", "") or doc["metadata"].get("text", "")
            base = doc.get("score", 0.0)
            keyword_match = query_lower in text.lower()
            return base + (0.15 if keyword_match else 0)

        for doc in results:
            doc["score"] = score(doc)

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def search_and_answer(self, query, top_k=5):
        results = self.hybrid_search(query, top_k)
        results = self.rerank(query, results)

        top_result = results[0]
        confidence = top_result.get("score", 0.0)
        #if confidence < 0.1:  # tune later
        #    return "I’m not fully sure — can you provide more details?"

        context_block = "\n\n".join([
        f"Source: {r.get('metadata', {}).get('source', 'Unknown')} | "
        f"Text: {r.get('text') or r.get('page_content') or r.get('chunk') or ''}"
        for r in results
        ])

        prompt = f"""
            Answer the query using only the provided context.
            Cite sources inside the answer as (source).

            Query: {query}

            Context:
            {context_block}

            Answer:
            """
        response = self.llm.invoke(prompt)
        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "I'm having trouble with the OTP/Activation step"
    summary = rag_search.search_and_answer(query, top_k=3)
    print("Summary:", summary)
