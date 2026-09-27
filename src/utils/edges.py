from src.utils.states import GenerateAnalystsState, InterviewState, ResearchGraphState
from langgraph.graph import END
from dotenv import load_dotenv
from typing import Literal
from langchain.messages import AIMessage, HumanMessage
from langgraph.types import Send

load_dotenv()

# edges 
def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", "dummy"]:
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', None)

    # The human_feedback node normalises approval replies ("perfect",
    # "continue", "yes", "") to None, so only genuine feedback lands here.
    if human_analyst_feedback:
        return "create_analysts"
    
    # Otherwise end
    return "dummy"

def route_messages(state: InterviewState, name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        # Interview depth is configurable; InterviewState marks it NotRequired so
        # the subgraph tolerates it being absent, but we always forward it.
        max_num_turns = state.get("max_num_turns", 2)
        return [Send("conduct_interview", {"analyst": analyst,
                                           "max_num_turns": max_num_turns,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]
