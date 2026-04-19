import os
import re
import chromadb
import dotenv
from typing import Dict, List, Sequence, Tuple
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dataset import DOCUMENTS
from sentence_transformers import CrossEncoder

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


def get_embedding(text: str) -> List[float]:
    return get_embeddings([text])[0]


embs = get_embeddings(all_chunks)
collection.add(
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    embeddings=embs,
    documents=all_chunks,
    metadatas=chunk_meta,
)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text)


def situate_context(chunk: str, full_doc: str, title: str) -> str:
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


print("Contextualizing chunks... (LLM call per chunk, patience)")
ctx_chunks = []
for i, (chunk, meta) in enumerate(zip(all_chunks, chunk_meta)):
    doc = next(d for d in DOCUMENTS if d["title"] == meta["title"])
    ctx_chunks.append(situate_context(chunk, doc["content"], doc["title"]))
    if (i + 1) % 5 == 0:
        print(f"  {i + 1}/{len(all_chunks)} done")

print(f"\n✅ Contextualized {len(ctx_chunks)} chunks")


def reciprocal_rank_fusion(
    semantic: List[Tuple[int, float]], keyword: Sequence[Tuple[int, float]], k: int = 60
) -> List[Tuple[int, float]]:
    scores = {}

    for rank, (idx, _) in enumerate(semantic):
        scores[idx] += 1 / (k + rank + 1)

    for rank, (idx, _) in enumerate(keyword):
        scores[idx] += 1 / (k + rank + 1)

    return sorted(scores, key=lambda x: x[1], reverse=True)


ctx_bm25 = BM25Okapi([tokenize(c) for c in ctx_chunks])


def ctx_hybrid_search(user_query: str, k: int = 10) -> List[Dict]:
    sem = collection.query(query_embeddings=[get_embedding(user_query)], n_results=20)
    sem_ranked = [
        (int(id.split("_")[1]), dist)
        for id, dist in zip(sem["ids"][0], sem["distances"][0])
    ]

    scores = ctx_bm25.get_scores(tokenize(user_query))
    bm25_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    fused = reciprocal_rank_fusion(sem_ranked, bm25_ranked)

    return [
        {
            "chunk": ctx_chunks[idx],
            "original": all_chunks[idx],
            "meta": chunk_meta[idx],
            "score": sc,
        }
        for idx, sc in fused[:k]
    ]


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("✅ Loaded cross-encoder reranker")


def rerank(question: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
    """Rerank results with cross-encoder."""
    pairs = [(question, r["chunk"]) for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def full_rag(question: str, verbose: bool = True) -> str | None:
    """
    The full pipeline:
    Contextual chunks → Hybrid search (top 10) → Rerank (top 3) → Generate
    """
    results = ctx_hybrid_search(question, k=10)
    top = rerank(question, results, top_k=3)

    if verbose:
        print(f"\n🔍 Query: '{question}'")
        print("\nTop 3 after reranking:")
        for i, r in enumerate(top):
            print(f"  [{i + 1}] rerank={r['rerank_score']:.3f} | {r['meta']['title']}")
            print(f"      {r['chunk'][:100]}...")

    context = "\n".join(f"[Source: {r['meta']['title']}]\n{r['chunk']}" for r in top)

    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": f"""Answer based ONLY on the context below.
                If the context doesn't have the answer, say \"I don't have enough information.\"
                Cite your sources.

                Context:
                {context}

                Question: {question}

                Answer:""",
            }
        ],
    )

    answer = resp.choices[0].message.content
    if verbose:
        print(f"\n💬 Answer:\n{answer}")
    return answer
