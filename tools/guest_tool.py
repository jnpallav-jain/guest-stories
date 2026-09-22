import re

import datasets
from langchain_core.documents import Document
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

def preprocess(text: str) -> list[str]:
    """Tokenize for BM25: lowercase, and split on anything that is not alphanumeric .

    BM25Retriever's default preprocess_func is plain `text.split()`, which is
    case-sensitive and keeps punctuation attached. That makes "nikola tesla" score
    0.00 against Tesla's own record, and "Tesla." a different term from "Tesla".
    The same function must be used for indexing and querying.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.retriever = BM25Retriever.from_documents(docs, preprocess_func=preprocess)

    def forward(self, query: str):
        results = self.retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."

# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)