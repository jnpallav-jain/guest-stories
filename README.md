# Guest Stories

Concierge, a conversational agent for a gala, built on
[smolagents](https://github.com/huggingface/smolagents). It looks up who is on
the guest list, checks whether the weather will ruin the fireworks, and searches
the web for anything else a host needs to know.

I built this to get hands-on with agent tooling — writing tools, wiring
retrieval, and keeping conversation memory bounded. The notes below are what
actually broke along the way, which turned out to be the more interesting half.

Real output, not a mock-up:

```
You: Tell me about our guest Dr. Nikola Tesla.

🎩 Concierge: Our guest, Dr. Nikola Tesla, is an old friend from university days. He
has recently patented a new wireless energy transmission system and would be
delighted to discuss it with you. Just remember he's passionate about pigeons, so
that might make for good small talk. Born in 1856 in what is now Croatia, Tesla
was a pioneering inventor, electrical engineer, mechanical engineer, and
physicist...
```

That answer combines the private guest record with web search — six steps,
23,675 cumulative input tokens.

## Tools

| Tool | What it does |
|---|---|
| `guest_info_retriever` | BM25 over the invitee list |
| `weather_info` | Current conditions from OpenWeatherMap |
| `hub_stats` | Most-downloaded model for an author on the Hugging Face Hub |
| `web_search` | DuckDuckGo, via `ddgs` |
| `visit_webpage` | Reads a page as markdown (smolagents built-in) |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
python app.py
```

`.env` needs an `HF_TOKEN` and an `OPENWEATHERMAP_API_KEY`. A new OpenWeatherMap
key takes **up to two hours to activate** and returns HTTP 401 until it does,
which is indistinguishable from a wrong key — `weather_info` says so explicitly
in its error message rather than leaving you to guess.

## Conversation memory

`app.py` runs a loop rather than a single query, because `CodeAgent.run()` wipes
memory by default:

```python
response = concierge.run(query, reset=False)
```

Memory in smolagents is `agent.memory.steps`, a list of `TaskStep` /
`ActionStep` / `PlanningStep` objects. There is no separate store — before every
LLM call, `write_memory_to_messages()` replays the whole list into the prompt.
Keep the list and the agent remembers.

The catch is that replay happens on *every step*, not every turn, so an
observation is re-sent for the rest of the run. `trim_memory()` keeps the last 5
tasks, cutting only on `TaskStep` boundaries — slicing at an arbitrary index
would strand an `ActionStep` whose originating question is gone, leaving the
model observations with nothing to attach them to.

## Notes from the build

Version drift is the theme. `smolagents` is pinned to 1.18.0 and nothing else
is, so every surrounding library resolved to a release newer than the API it
expects.

**`list_models(direction=-1)` no longer exists.** `huggingface_hub` 1.x dropped
the parameter. `sort="downloads"` alone already returns descending order —
verified, not assumed: `facebook/contriever` 8.07M, `facebook/opt-125m` 6.77M,
`facebook/dinov2-small` 3.08M.

**`duckduckgo_search` is retired** in favour of `ddgs`, and warns on every
instantiation. You cannot simply swap the package, because smolagents 1.18.0
hardcodes `from duckduckgo_search import DDGS` inside `DuckDuckGoSearchTool`.
`tools/search_tool.py` is the same tool rebuilt on `ddgs` instead.

**`add_base_tools=True` silently overwrites your tools.** `_setup_tools` builds
`{tool.name: tool}` from your list, then calls `.update()` with the base tools —
so base tools win on a name collision. `DuckDuckGoSearchTool.name` is
`"web_search"`, exactly like ours. The flag would have quietly reinstated the
deprecated tool with no error at all. Base tools are added by hand here instead.

**BM25 was case-sensitive.** `BM25Retriever`'s default `preprocess_func` is
plain `text.split()` — no lowercasing, no punctuation stripping. Measured
against the guest corpus:

| Query | Default tokenizer | Fixed tokenizer |
|---|---|---|
| `Dr. Nikola Tesla` | 2.00 | 2.25 |
| `nikola tesla` | **0.00** | **1.58** |
| `Barack Obama` (absent) | 0.00 | 0.00 |

A lowercase query scored zero against Tesla's own record. This was invisible
because the corpus holds 3 guests and the tool returned `results[:3]` — the
entire dataset, every time, so the right guest was always in the output no
matter what BM25 thought. The retriever wasn't retrieving; the LLM was doing the
filtering. Worth stating plainly: **the code was passing for the wrong reason.**

The fix is a shared tokenizer for indexing and querying:

```python
def preprocess(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
```

A score threshold would be the natural next step — absent guests score exactly
0.00, so the separation is clean — but it is deliberately *not* implemented.
Adding one before fixing the tokenizer would have converted a query that works
today into "no matching guest found". At three documents, retrieval here is
mostly theatre anyway; the corpus would fit in the system prompt.

## Token cost

The `Input tokens` figure smolagents prints is **cumulative across every LLM
call in a run**, not the size of one prompt (`total_input_token_count +=` in
`monitoring.py`). A run reporting 62,911 input tokens was about 8–10 steps, not
one enormous request.

Because memory is replayed per step, cost grows quadratically in steps, so what
matters is how much each observation adds. Measured with the
`Qwen/Qwen2.5-Coder-32B-Instruct` tokenizer (the default `InferenceClientModel`
model), via `tokenizers`:

| Component | Tokens |
|---|---|
| System prompt, all 6 tool schemas included | 2,301 |
| `web_search` at `max_results=10` | ~912 (mean of 3 queries: 1034 / 1012 / 690) |
| `web_search` at `max_results=5` | ~437 (mean of 3 queries: 395 / 458 / 458) |
| `visit_webpage` at the 40,000-char default | ~10,000 (estimated at 4 chars/token) |

Search counts vary run to run because the results do. Two caps follow from
this: `max_results=5`, and `VisitWebpageTool(max_output_length=8000)`.

Note that `trim_memory()` only runs *between* turns. Within a single
`concierge.run()`, memory grows unchecked — `max_steps` is the ceiling there.

## Answer quality is a separate problem from tool correctness

`hub_stats` returned `"The most downloaded model by Qwen is Qwen/Qwen3-0.6B with
24,836,019 downloads."` The agent then called
`final_answer("Qwen/Qwen3-0.6B")`, discarding the number it had just fetched.
Every tool call succeeded, nothing raised, and the answer was still worse than
the data behind it.

`CodeAgent` in 1.18.0 has no `instructions` parameter, and `system_prompt` is a
read-only property. It is rebuilt from `prompt_templates` on every run, so the
supported fix is to append there:

```python
concierge.prompt_templates["system_prompt"] += "\n\nWhen you call final_answer, ..."
```

That is a nudge to a small hosted model, not a guarantee, and it demonstrably
does not always take. Asked about fireworks after the instruction was added, the
agent still answered "clear skies, a moderate temperature, and low humidity and
wind speed" — having just received `13.6°C, humidity 70%, wind 1.54 m/s`. It
kept the shape of the facts and dropped every number.

The deterministic version is to return `{"model_id": ..., "downloads": ...}` from
the tool, so the figure is a value the agent must handle rather than prose it can
paraphrase away. Prompting moves the odds; structure decides the outcome.

## Environment

Verified on Python 3.12.12 with `smolagents==1.18.0`, `huggingface-hub==1.32.0`,
`ddgs==9.16.0`, `datasets==5.0.1`, `langchain-community==0.4.2`,
`rank-bm25==0.2.2`, `markdownify==1.2.3`.

`langchain-community` emits a sunset warning on import; it is still used only
for `BM25Retriever`.
