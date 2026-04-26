import os
import subprocess
import requests
from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
# import asyncio

load_dotenv()

RLM_SYSTEM_PROMPT = """
You are a Recursive Language Model(RLM). Your job is to find the
right answer to the query by exploring and interacting with the context using code.
You will NOT see the full context, you will only see what you ask for and it will be available in the
`context` variable.

Available tools:
- `repl()`: Use this to run and view the output of your code
- `context`: The full input text
- `llm_call()`: Use this to call the sub-llm to analyze a chunk
The sub-llms answer is returned as text and will not enter the context.
- `final(answer)`: Call this when you have the final answer

Strategy:
1. First, check the size: `print(len(context))`
2. Peek at the structure: `print(context[:500])`
3. Use code to search, filter, count, or slice the data
4. For complex subtasks, use `llm_call()` to delegate to a sub-LLM
5. When done, call `FINAL(your_answer)`

Rules:
- Write ONLY Python code. No markdown, no explanation.
- Your code block must be wrapped in ```python ... ```
- Use print() to see results — you only see what you print.
- Variables persist between steps (like Jupyter cells).
- Be systematic. Explore first, then solve.
"""


class RLM_agent:
    def __init__(self, model: str = "openrouter/free"):
        self.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
        self.LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.model = model
        self.context = ""
        self.prompt = RLM_SYSTEM_PROMPT

        if not self.OPENROUTER_API_KEY or not self.LLAMA_CLOUD_API_KEY:
            raise ValueError("API keys are missing")

    def llm_call(self, prompt: str, system: bool, max_tokens: int = 4000) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.OPENROUTER_API_KEY}"},
            json={"model": self.model, "messages": msgs, "max_tokens": max_tokens},
        )
        data = r.json()
        if "choices" not in data:
            raise Exception(f"API error: {data}")
        return data["choices"][0]["message"]["content"]

    async def _parse_document(self, file: str):
        client = AsyncLlamaCloud(api_key=self.LLAMA_CLOUD_API_KEY)

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

    def _repl(self, code: str, timeout: int = 5) -> str:
        try:
            result = subprocess.run(
                ["python3", "-c", code], capture_output=True, text=True, timeout=timeout
            )
            return result.stdout.strip() if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout} seconds"

    def _final(self, answer: str):
        print("Response: ", answer)

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0]
        return text

    def agent(self, query: str, max_steps: int = 10):
        self.context = self.context or ""

        # history = [
        #     {"role": "system", "content": self.prompt},
        #     {"role": "user", "content": query},
        # ]
