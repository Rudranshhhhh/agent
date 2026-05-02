"""
RAG Retriever — queries ChromaDB with optional company filtering and returns
formatted context strings for the LLM.
"""

from langchain_chroma import Chroma

from config import TOP_K


class CorpusRetriever:
    """Thin wrapper around ChromaDB for filtered similarity search."""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query : str
            The search query (typically issue + subject text).
        company : str, optional
            If provided (and not "None" / empty), restricts results to
            documents from that company's corpus.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict has keys: content, company, category, source, title.
        """
        search_kwargs: dict = {"k": top_k}

        if company and company.strip() and company.strip() != "None":
            search_kwargs["filter"] = {"company": company.strip()}

        results = self.vectorstore.similarity_search(query, **search_kwargs)

        return [
            {
                "content": doc.page_content,
                "company": doc.metadata.get("company", "Unknown"),
                "category": doc.metadata.get("category", "general"),
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
            }
            for doc in results
        ]

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieval results into a single string suitable for the LLM
        prompt context window.
        """
        if not results:
            return "(No relevant documentation found.)"

        sections: list[str] = []
        for i, r in enumerate(results, 1):
            header = f"[Doc {i}] {r['company']} / {r['category']} — {r['title']}"
            sections.append(f"{header}\n{r['content']}")

        return "\n\n---\n\n".join(sections)
