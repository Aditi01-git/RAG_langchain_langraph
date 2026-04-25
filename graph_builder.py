from langgraph.graph import StateGraph
from graph_nodes import get_nodes

class GraphState(dict):
    pass

def build_graph(rag):
    retrieve, evaluate, generate = get_nodes(rag,llm)

    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("evaluate", evaluate)    
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    #Routing function
    def route(state):
        return state['status']

    graph.add_conditional_edges("evaluate", 
                                 route,
                                 {
                                    "retry" : "retrieve",
                                    "ok": "__end__"
                                 }
                                )

    return graph.compile()