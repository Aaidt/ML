import os
from typing_extensions import Doc
import chromadb
from chromadb.api.types import Document
import dotenv
from typing import List
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dataset import DOCUMENTS

dotenv.load_dotenv()
key = os.environ["OPENROUTER_API_KEY"]
if key is None:
    raise ValueError("API key not found")

oai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
)
chroma = chromadb.Client()

try:
    chroma.delete_collection("naive_rag")
except Exception as e:
    print(f"Error while connecting to chromadb: {e}")
    pass

collection = chroma.create_collection("naive_rag", metadata={"hnsw:space": "cosine"})


def chunk_recursive(text: str, max_size: int = 200) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_size:
            chunks.append(para)
        else:
            sentences = para.replace(". ", ".\n").split("\n")
            current, current_len = [], 0
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > max_size and current:
                    chunks.append(" ".join(current))
                    current, current_len = [sent], sent_len
                else:
                    current.append(sent)
                    current_len += sent_len
            if current:
                chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) > 10]


all_chunks = []
chunk_meta = []

for doc in DOCUMENTS:
    chunks = chunk_recursive(doc["content"])
    all_chunks.append(" ".join(chunks))
    chunk_meta.append({"title": doc["title"], "source": doc["source"]})


def get_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    cleaned_text = [t.replace("\n", " ") for t in texts]
    response = oai.embeddings.create(input=cleaned_text, model=model)
    return [d.embedding for d in response.data]


embs = get_embeddings(all_chunks)
collection.add(
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    embeddings=embs,
    documents=all_chunks,
    metadatas=chunk_meta,
)


# def reciprocal_rank_fusion(
#     semantic: List[List[float]], keyword: List[List[float]], k: int = 5
# ) -> Dict:
#     return {}


def contextual_retrieval(chunk: str, full_doc: str, title: str) -> str:
    """Prepend LLM-generated context to a chunk before embedding."""
    resp = oai.chat.completions.create(
        model="openrouter/free",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": f"""<document title="{title}">
                {full_doc}
                </document>

                Here is a chunk from that document:
                <chunk>
                {chunk}
                </chunk>

                Write a SHORT (2-3 sentence) context that situates this chunk within the document.
                Include: which document, what section/topic, key entities or time periods.
                This will be prepended to the chunk for search.

                Context:""",
            }
        ],
    )
    ctx = resp.choices[0].message.content
    return f"{ctx}\n\n{chunk}"


chunk = all_chunks[0]
doc = DOCUMENTS[0]["content"]
title = DOCUMENTS[0]["title"]

print(contextual_retrieval(chunk, doc, title))
