from src.utils.states import GenerateAnalystsState
from src.utils.nodes import create_analysts, human_feedback, dummy
from src.utils.edges import should_continue
from langgraph.graph import START, END, StateGraph


# building our graph
builder = StateGraph(GenerateAnalystsState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("dummy", dummy)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", "dummy"])
builder.add_edge("dummy", END)

# The `human_feedback` node calls interrupt() itself so it can surface the
# generated analysts for review. Do not also pass interrupt_before here: a
# static interrupt fires first and swallows that payload, leaving the user
# staring at an empty pause with nothing to approve.
graph = builder.compile()
