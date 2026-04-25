import os
import requests
import asyncio
from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
try:
    if not OPENROUTER_API_KEY or not LLAMA_CLOUD_API_KEY:
        raise ValueError("API keys are missing")
except ValueError as e:
    print("Error: ", e)

ROOT_MODEL = "openrouter/free" 
# SUB_MODEL  = "google/gemini-3-flash-preview" 

def llm_call(prompt:str, system:bool, model:str=ROOT_MODEL, max_tokens:int=4000) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": model, "messages": msgs, "max_tokens": max_tokens}
    )
    data = r.json()
    if "choices" not in data:
        raise Exception(f"API error: {data}")
    return data["choices"][0]["message"]["content"]

async def parse_document(file:str):
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

    file_obj = await client.files.create(file=file, purpose="parse")

    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="cost_effective",
        version="latest",
        expand=["markdown_full", "text_full"],
    )
    # print("\nFull text:")
    # print(result.text_full)

    print("Full markdown:", result.markdown_full)


# asyncio.run(parse_document("./Recursive Language Model.pdf"))