from typing import Dict, List
import re
import os
import chromadb
import dotenv
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


def recursve_chunking(text: str, max_size: int = 200) -> List[str]:
    chunks = []

    paragraphs = text.split("\n\n")

    for para in paragraphs:
        if len(para.split()) < max_size:
            chunks.append(para)
        else:
            sentences = re.split(r"[.!?]\s+", para)
            current, current_len = [], 0

            for sent in sentences:
                if current_len + len(sent) >= max_size and current:
                    chunks.append(" ".join(current))
                    current, current_len = [sent], len(sent)
                else:
                    current.append(" ".join(sent))
                    current_len += len(sent)

            if current:
                chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) > 10]


all_chunks = []
chunk_meta = []

for doc in DOCUMENTS:
    chunks = recursve_chunking(doc["content"])
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


def bm25(text: str, k: int = 5):
    tokens = re.findall(r"\s", text.lower())
    return tokens[:k]


def reciprocal_rank_fusion(
    semantic: List[List[float]], keyword: List[List[float]], k: int = 5
) -> Dict:
    return {}
