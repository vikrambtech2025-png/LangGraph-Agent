"""Tests for the deterministic routing / report-assembly logic.

Everything exercised here is LLM-free by design: `should_continue` and
`route_messages` live in `src.utils.edges` (which never imports the model), and
`finalize_report` is pure string manipulation. No network calls are made.
"""

from langchain.messages import AIMessage, HumanMessage

from src.utils.edges import route_messages, should_continue
from src.utils.nodes import finalize_report


def _turn(question: str, answer: str) -> list:
    """One question/answer exchange, tagged the way generate_* tags them."""
    return [
        AIMessage(content=question),      # analyst asks (name=None)
        AIMessage(content=answer, name="expert"),  # expert replies
    ]


# --------------------------------------------------------------------------
# should_continue
# --------------------------------------------------------------------------

def test_should_continue_ends_when_no_feedback():
    assert should_continue({"human_analyst_feedback": None}) == "dummy"


def test_should_continue_ends_when_key_absent():
    assert should_continue({}) == "dummy"


def test_should_continue_regenerates_on_real_feedback():
    feedback = "add an analyst who covers security"
    assert should_continue({"human_analyst_feedback": feedback}) == "create_analysts"


def test_should_continue_ignores_empty_string_feedback():
    # human_feedback normalises "" to None, but the router should not loop on it.
    assert should_continue({"human_analyst_feedback": ""}) == "dummy"


# --------------------------------------------------------------------------
# route_messages
# --------------------------------------------------------------------------

def test_route_messages_ends_at_turn_limit():
    messages = [HumanMessage(content="So you said you were writing an article?")]
    messages += _turn("q1", "a1")
    messages += _turn("q2", "a2")

    # Two expert answers == max_num_turns, so the interview is over.
    assert route_messages({"messages": messages, "max_num_turns": 2}) == "save_interview"


def test_route_messages_continues_below_turn_limit():
    messages = [HumanMessage(content="So you said you were writing an article?")]
    messages += _turn("q1", "a1")

    assert route_messages({"messages": messages, "max_num_turns": 3}) == "ask_question"


def test_route_messages_ends_on_analyst_sign_off():
    messages = [HumanMessage(content="So you said you were writing an article?")]
    messages += _turn("q1", "a1")
    # generate_question is told to close with this exact phrase.
    messages += _turn("Anything else? Thank you so much for your help!", "a2")

    assert route_messages({"messages": messages, "max_num_turns": 5}) == "save_interview"


def test_route_messages_defaults_to_two_turns():
    # max_num_turns is NotRequired, so the fallback must hold at 2.
    messages = [HumanMessage(content="So you said you were writing an article?")]
    messages += _turn("q1", "a1")
    messages += _turn("q2", "a2")

    assert route_messages({"messages": messages}) == "save_interview"


def test_route_messages_does_not_count_analyst_questions_as_expert_answers():
    # Only messages named "expert" count toward the limit.
    messages = [HumanMessage(content="So you said you were writing an article?")]
    messages += [AIMessage(content=f"q{i}") for i in range(4)]

    assert route_messages({"messages": messages, "max_num_turns": 1}) == "ask_question"


# --------------------------------------------------------------------------
# finalize_report
# --------------------------------------------------------------------------

def _finalize(content, introduction="INTRO", conclusion="CONCL"):
    return finalize_report(
        {"content": content, "introduction": introduction, "conclusion": conclusion}
    )["final_report"]


def test_finalize_report_assembles_all_three_parts():
    out = _finalize("## Insights\n\nBody text.")
    assert out.startswith("INTRO")
    assert "Body text." in out
    assert out.endswith("CONCL")


def test_finalize_report_strips_insights_header_without_eating_the_body():
    # str.strip("## Insights") removes a *character set* and would eat leading
    # characters. Verify the body survives intact.
    out = _finalize("## Insights\n\ninsights are contested")
    assert "insights are contested" in out
    assert "hts" not in out.replace("insights are contested", "")


def test_finalize_report_appends_sources_after_conclusion():
    out = _finalize("## Insights\n\nBody.\n\n## Sources\n[1] a.com\n[2] b.com")
    assert out.endswith("[1] a.com\n[2] b.com")
    assert out.index("CONCL") < out.index("## Sources")


def test_finalize_report_relocates_repeated_sources_header():
    # A model that emits the header twice must not strand sources mid-body.
    out = _finalize("## Insights\n\nBody.\n\n## Sources\n[1] a.com\n\n## Sources\n[2] b.com")
    body = out.split("---")[1]
    assert "a.com" not in body, "sources were left inline in the body"
    assert out.count("## Sources") == 1


def test_finalize_report_handles_unnewline_delimited_header():
    out = _finalize("## Insights\n\nBody.\n## Sources\n[1] a.com")
    body = out.split("---")[1]
    assert "a.com" not in body
    assert out.endswith("[1] a.com")


def test_finalize_report_omits_sources_when_absent():
    out = _finalize("## Insights\n\nBody only.")
    assert "## Sources" not in out
    assert out.endswith("CONCL")
