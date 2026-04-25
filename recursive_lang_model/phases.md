## What is already implemented in your file

From the notebook you shared , the full RLM pipeline is **mostly complete**. Break it down from first principles:

### 1. LLM Interface Layer

* `llm_call(...)`

  * Wraps API call to OpenRouter
  * Handles:

    * system + user messages
    * model selection (root vs sub)
    * token limits
* Used by:

  * root agent loop
  * sub-LLM recursion (`llm_query`)

---

### 2. Problem Setup (for testing RLM)

* Synthetic dataset generators:

  * `generate_dataset()` → structured counting task
  * `make_reviews()` → semantic reasoning task
* Ground truth computation for evaluation

Purpose:

* Create **controlled tasks** to compare vanilla LLM vs RLM

---

### 3. REPL Environment (Core of RLM)

Class: `RLMRepl`

This is the **execution engine**.

#### State

* `context` → stored outside LLM prompt
* `namespace` → execution scope
* `final_answer`
* `sub_call_count`

#### Tools exposed to model

* `context` (data access)
* `print()` (feedback loop)
* `FINAL()` (termination)
* `llm_query()` (recursion)

#### Key methods

* `_final(answer)`

  * stores answer
* `_llm_query(query, sub_context)`

  * recursive LLM call
* `execute(code)`

  * runs Python via `exec`
  * captures stdout/stderr
  * truncates output

---

### 4. Prompting Layer

* `RLM_SYSTEM_PROMPT`

  * defines:

    * available tools
    * rules (only Python)
    * strategy (explore → solve)
    * interaction protocol

---

### 5. Code Extraction Layer

* `extract_code(response)`

  * Parses LLM output
  * Handles:

    * ```python blocks
      ```
    * fallback heuristics

---

### 6. Agent Loop (Orchestrator)

Function: `run_rlm(...)`

Implements the full loop:

1. Send query (no context)
2. LLM generates code
3. Extract code
4. Execute in REPL
5. Return output to LLM
6. Repeat until `FINAL()`

Tracks:

* iterations
* sub-LLM calls
* conversation history

---

### 7. Evaluation / Experiments

* Vanilla LLM baseline
* RLM execution
* Scaling test
* Semantic task (uses recursion)

---

## What YOU need to implement (Checklist)

If your goal is to build RLM **from scratch**, here is the minimal dependency graph:

---

### Phase 1 — Minimal Working RLM

#### Core primitives

* [ ] Basic `llm_call()` (can mock initially)
* [ ] REPL executor (`exec` + stdout capture)
* [ ] `context` stored outside prompt
* [ ] `FINAL()` mechanism
* [ ] Iterative loop (LLM ↔ REPL)

If this works → you already have a basic RLM

---

### Phase 2 — Make it usable

* [ ] Code extraction (`extract_code`)
* [ ] Error handling in execution
* [ ] Output truncation
* [ ] Persistent namespace across iterations

---

### Phase 3 — Add recursion (true RLM)

* [ ] `llm_query()` function
* [ ] Separate sub-model (optional but ideal)
* [ ] Sub-call tracking

This is what makes it **recursive**, not just tool-using

---

### Phase 4 — Prompt design

* [ ] System prompt defining:

  * tools
  * rules
  * workflow
* [ ] Instruction format enforcement (Python-only)

---

### Phase 5 — Control & safety

* [ ] Max iterations limit
* [ ] Timeout / execution guard (important in real systems)
* [ ] Sandbox REPL (right now it's unsafe)

---

### Phase 6 — Evaluation layer

* [ ] Benchmark tasks (counting, grouping, reasoning)
* [ ] Ground truth comparison
* [ ] Logging iterations + outputs

---

## What is OPTIONAL (but important for real systems)

These are **not implemented fully** but you should consider:

* [ ] Secure sandbox (instead of raw `exec`)
* [ ] Memory pruning / history compression
* [ ] Tool abstraction (instead of hardcoding namespace)
* [ ] Structured outputs (JSON instead of print parsing)
* [ ] Retry logic for bad code
* [ ] Cost tracking (tokens per iteration)

---

## Mental Model (first principles)

You are building:

```
LLM = planner
REPL = executor
context = external memory
llm_query = recursive delegation
```

Instead of:

```
LLM does thinking + data processing
```

You shift to:

```
LLM writes program → program processes data
```

---

## Minimal build order (do this)

1. REPL with `exec` + print capture
2. Simple loop (no recursion)
3. Add `FINAL()`
4. Add code extraction
5. Add `llm_query()`
6. Improve prompt

---

If needed, next step can be:

* stripping this notebook down to a **clean 150-line minimal RLM implementation** or
* helping you redesign it in C++/Rust-style architecture for performance.
