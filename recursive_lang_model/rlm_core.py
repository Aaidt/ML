import os
import re
import io
import json
import requests
from contextlib import redirect_stdout, redirect_stderr

ROOT_MODEL = "google/gemini-3-flash-preview"
SUB_MODEL = "google/gemini-3-flash-preview"


def load_api_key():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return os.environ.get("OPENROUTER_API_KEY", "")


API_KEY = load_api_key()


def llm_call(prompt, system="", model=ROOT_MODEL, max_tokens=4000):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "messages": msgs, "max_tokens": max_tokens},
    )
    data = r.json()
    if "choices" not in data:
        raise Exception(f"API error: {data}")
    return data["choices"][0]["message"]["content"]


RLM_SYSTEM_PROMPT = """You are an RLM (Recursive Language Model) agent.

You have access to a Python REPL environment. The user's data is stored
in a variable called `context` — it may be very long (millions of characters).
You CANNOT see the context directly. You must write Python code to explore it.

Available tools:
- `context` — the full input text (Python string variable)
- `print()` — use this to see output from your code
- `llm_query(query, sub_context)` — call a sub-LLM to analyze a chunk.
  The sub-LLM's response is returned as a string. It does NOT enter your context.
- `FINAL(answer)` — call this when you have the final answer.
- Standard Python: `re`, `json`, `len`, `sum`, etc.

Strategy:
1. First, check the size: `print(len(context))`
2. Peek at the structure: `print(context[:500])`
3. Use code to search, filter, count, or slice the data
4. For complex subtasks, use `llm_query()` to delegate to a sub-LLM
5. When done, call `FINAL(your_answer)`

Rules:
- Write ONLY Python code. No markdown, no explanation.
- Your code block must be wrapped in ```python ... ```
- Use print() to see results — you only see what you print.
- Variables persist between steps (like Jupyter cells).
- Be systematic. Explore first, then solve."""


class RLMRepl:
    def __init__(self, context: str, max_output_chars: int = 5000):
        self.final_answer = None
        self.max_output_chars = max_output_chars
        self.sub_call_count = 0
        self.namespace = {
            "context": context,
            "FINAL": self._final,
            "llm_query": self._llm_query,
            "re": re,
            "json": json,
            "len": len,
            "print": print,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "sum": sum,
            "sorted": sorted,
            "min": min,
            "max": max,
            "abs": abs,
            "set": set,
            "tuple": tuple,
            "zip": zip,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
            "True": True,
            "False": False,
            "None": None,
        }

    def _final(self, answer):
        self.final_answer = str(answer)
        print(f"[FINAL ANSWER SUBMITTED: {answer}]")

    def _llm_query(self, query: str, sub_context: str = "") -> str:
        self.sub_call_count += 1
        print(f"  [Sub-LLM call #{self.sub_call_count}: '{query[:80]}...'")
        prompt = query
        if sub_context:
            prompt = f"{query}\n\nContext:\n{sub_context}"
        result = llm_call(prompt, model=SUB_MODEL, max_tokens=2000)
        print(f"   Sub-LLM returned: '{result[:100]}...']")
        return result

    def execute(self, code: str) -> str:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self.namespace)
            output = stdout_capture.getvalue()
        except Exception as e:
            output = f"ERROR: {type(e).__name__}: {e}"
        if len(output) > self.max_output_chars:
            output = (
                output[: self.max_output_chars]
                + f"\n... [truncated to {self.max_output_chars} chars]"
            )
        return output


def extract_code(response: str) -> str:
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    lines = response.strip().split("\n")
    code_lines = []
    started = False
    for line in lines:
        if not started and line.startswith(
            (
                "import ",
                "from ",
                "print(",
                "#",
                "for ",
                "if ",
                "def ",
                "context",
                "result",
                "count",
                "data",
                "lines",
                "FINAL",
                "with ",
                "try",
                "matches",
                "output",
                "pattern",
            )
        ):
            started = True
        if started:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else response.strip()


def read_context(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    parts = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                rel = os.path.relpath(fpath, path)
                parts.append(f"--- FILE: {rel} ---\n{content}")
            except Exception:
                pass
    return "\n\n".join(parts)


def get_context_info(path: str) -> dict:
    path = os.path.expanduser(path)
    info = {
        "path": path,
        "files": 0,
        "size": 0,
        "lines": 0,
        "is_dir": os.path.isdir(path),
    }
    if os.path.isfile(path):
        info["files"] = 1
        info["size"] = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            info["lines"] = sum(1 for _ in f)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    info["size"] += os.path.getsize(fpath)
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        info["lines"] += sum(1 for _ in f)
                    info["files"] += 1
                except Exception:
                    pass
    return info


def run_rlm(query: str, context: str, max_iterations: int = 10, callbacks=None):
    repl = RLMRepl(context)
    history = []
    user_msg = f"""Task: {query}

The data is in the `context` variable ({len(context)} characters long).
Write Python code to explore and solve this. Start by checking the structure."""
    history.append({"role": "user", "content": user_msg})

    for i in range(max_iterations):
        if callbacks and callbacks.get("on_iteration"):
            callbacks["on_iteration"](i + 1, max_iterations)

        response_data = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": ROOT_MODEL,
                "messages": [{"role": "system", "content": RLM_SYSTEM_PROMPT}]
                + history,
                "max_tokens": 2000,
            },
        ).json()
        assistant_msg = response_data["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": assistant_msg})

        code = extract_code(assistant_msg)
        if callbacks and callbacks.get("on_code"):
            callbacks["on_code"](code)

        output = repl.execute(code)
        if callbacks and callbacks.get("on_output"):
            callbacks["on_output"](output)

        if repl.final_answer is not None:
            if callbacks and callbacks.get("on_final"):
                callbacks["on_final"](repl.final_answer, i + 1, repl.sub_call_count)
            return {
                "answer": repl.final_answer,
                "iterations": i + 1,
                "sub_calls": repl.sub_call_count,
                "history": history,
            }

        history.append(
            {
                "role": "user",
                "content": f"REPL output:\n```\n{output}\n```\nContinue. Write more code or call FINAL(answer) when done.",
            }
        )

    if callbacks and callbacks.get("on_timeout"):
        callbacks["on_timeout"](max_iterations)
    return {
        "answer": None,
        "iterations": max_iterations,
        "sub_calls": repl.sub_call_count,
        "history": history,
    }
