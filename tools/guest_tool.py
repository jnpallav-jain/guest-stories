import collections
import re

import datasets
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from smolagents import Tool

DATASET = "m-ric/english_historical_quotes"
MAX_QUOTES_PER_GUEST = 3
TOP_K = 3


def _categories(raw) -> list[str]:
    """The category column is a list on most rows but arrives as its string
    repr on some, so accept either rather than trusting one shape."""
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    if isinstance(raw, str):
        return [c.strip(" '\"[]") for c in raw.split(",") if c.strip(" '\"[]")]
    return []


def build_docs() -> list[Document]:
    """One document per guest: their name, themes and a few of their quotes."""
    dataset = datasets.load_dataset(DATASET, split="train")

    guests: dict[str, dict] = collections.defaultdict(
        lambda: {"quotes": [], "themes": collections.Counter()}
    )
    for row in dataset:
        name = (row["author"] or "").strip()
        quote = (row["quote"] or "").strip()
        if not name or not quote:
            continue
        guests[name]["quotes"].append(quote)
        guests[name]["themes"].update(_categories(row["category"]))

    docs = []
    for name, data in guests.items():
        themes = ", ".join(t for t, _ in data["themes"].most_common(5))
        quotes = "\n".join(f'  - "{q}"' for q in data["quotes"][:MAX_QUOTES_PER_GUEST])
        docs.append(
            Document(
                page_content=(
                    f"Name: {name}\n"
                    f"Known for: {themes or 'unknown'}\n"
                    f"Quotes on record ({len(data['quotes'])} total):\n{quotes}"
                ),
                metadata={"name": name, "quote_count": len(data["quotes"])},
            )
        )
    return docs


def preprocess(text: str) -> list[str]:
    """Tokenize for BM25: lowercase, and split on anything not alphanumeric.

    BM25Retriever's default preprocess_func is plain `text.split()`, which is
    case-sensitive and keeps punctuation attached, so "albert einstein" scores
    0.00 against Einstein's own record. The same function must be used for
    indexing and querying.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


docs = build_docs()


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "Looks up a gala guest by name or by what they are known for, and returns "
        "their themes and a few of their recorded quotes."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "A guest's name, or a theme such as 'science' or 'freedom'."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.docs = docs
        self.retriever = BM25Retriever.from_documents(docs, preprocess_func=preprocess)
        self.retriever.k = TOP_K

    def forward(self, query: str):
        # Score explicitly rather than taking the top k blindly: BM25 always
        # ranks the whole corpus, so without a threshold an absent guest still
        # returns three confident-looking strangers.
        scores = self.retriever.vectorizer.get_scores(preprocess(query))
        ranked = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)
        hits = [self.docs[i] for i, score in ranked[:TOP_K] if score > 0]

        if not hits:
            return f"No guest on the list matches '{query}'."
        return "\n\n".join(doc.page_content for doc in hits)


# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)

if __name__ == "__main__":
    print(f"{len(docs):,} guests indexed\n")
    print(guest_info_tool.forward("Ada Lovelace"))
