from smolagents import CodeAgent, InferenceClientModel, VisitWebpageTool
from smolagents.memory import TaskStep

from tools.guest_tool import guest_info_tool
from tools.hub_stats_tool import hub_stats_tool
from tools.search_tool import search_tool
from tools.weather_tool import weather_info_tool

# How many past exchanges to keep in the prompt. Older turns are dropped so the
# context (and the bill) stops growing with the conversation.
MAX_REMEMBERED_TURNS = 5

# Initialize the Hugging Face model
model = InferenceClientModel()
# Create Alfred, our gala agent, with the guest info tool
# add_base_tools=True is deliberately not used: it pulls in smolagents'
# DuckDuckGoSearchTool, which needs the retired `duckduckgo_search` package, and
# it would overwrite our own web_search tool (base tools win on name collision).
# VisitWebpageTool is the one base tool worth having, so it is added by hand.
alfred = CodeAgent(
    # Observations are replayed into the prompt on every later step, so an
    # oversized one is paid for repeatedly. VisitWebpageTool defaults to a
    # 40,000-char cap (~10k tokens); 8,000 is plenty for a readable page.
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool, VisitWebpageTool(max_output_length=8000)],
    model=model,
    planning_interval=3,
)

# The model tends to pass a bare identifier to final_answer ("Qwen/Qwen3-0.6B")
# and drop the numbers its tools just fetched. system_prompt is a read-only
# property rebuilt from prompt_templates on every run, so append the guidance
# there rather than assigning to it.
alfred.prompt_templates["system_prompt"] += (
    "\n\nWhen you call final_answer, give a complete sentence that carries the "
    "supporting facts your tools returned - names, counts, figures, units and "
    "dates - not just the bare identifier you looked up. Never make the user ask "
    "a follow-up for a number you already have."
)


def trim_memory(agent, max_turns=MAX_REMEMBERED_TURNS):
    """Drop everything before the last `max_turns` tasks, on task boundaries.

    Trimming mid-task would leave an ActionStep whose TaskStep is gone, so cut
    only at the TaskStep indices.
    """
    task_starts = [i for i, step in enumerate(agent.memory.steps) if isinstance(step, TaskStep)]
    if len(task_starts) > max_turns:
        agent.memory.steps = agent.memory.steps[task_starts[-max_turns]:]


if __name__ == "__main__":
    print("🎩 Alfred is at your service. Type 'exit' to leave the gala.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        # reset=False is what carries the conversation: memory.steps survives,
        # so this run sees every earlier task, action and observation.
        response = alfred.run(query, reset=False)

        print("🎩 Alfred's Response:")
        print(response)
        print()

        trim_memory(alfred)

    print("\n🎩 Very good. Enjoy the evening.")
