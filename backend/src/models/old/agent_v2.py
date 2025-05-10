from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langchain_experimental.utilities import PythonREPL
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict): # TODO CAN WE DOCUMENT WHAT EACH VARIABLE IS
    """
    A class representing the state of the agent and the data stored.
    """
    users_question: str
    conversation_history: List[Dict[str, str]] # TODO THIS IS NEVER SET????
    sub_tasks: List[str]
    current_subtask: str
    retrieval_history: List[Dict[str, Any]]
    sub_task_solutions: List[str]
    next_node: str
    chain_of_thought: str
    final_answer: str
    enhanced_context: str # TODO CURRENTLY THIS IS DOING NOTHING SINCE CONVERSATION HISTORY IS EMPTY
    validator_feedback: str
    validation_flag: str
    confidence_score: str
    confidence_explanation: str
    research_attempts : int


class DeepResearchAgent:
    """
    An AI agent that performs deep research on a question.
    """
    def __init__(self, llm_model: str, model_key: str, tavily_api_key: str, vectorstore: Any, number_of_tasks: int = 3, MAX_RESEARCH_ATTEMPTS: int = 3):
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=model_key)
        self.python_repl = PythonREPL()
        self.tavily_client = TavilyClient(tavily_api_key)
        self.vectorstore = vectorstore
        self.tools = [
            Tool.from_function(self.calculator, name="calculator", description=self.calculator.__doc__),
            Tool.from_function(self.tax_return_doc_search, name="tax_return_doc_search", description=self.tax_return_doc_search.__doc__),
            Tool.from_function(self.web_search, name="web_search", description=self.web_search.__doc__)
        ]
        self.number_of_tasks = number_of_tasks
        self.MAX_RESEARCH_ATTEMPTS = MAX_RESEARCH_ATTEMPTS
        self.graph = self._build_graph()

    
    def _calculate_max_node_visits(self) -> int:
        # Calculate the maximum number of node visits based on the number of tasks and research attempts
        planner_visits = 1  # Planner node is always visited once
        context_enhancer_visits = 1  # Context Enhancer node is always visited once
        # TODO CHECK THIS MATH FOR RESEARCHER
        sub_task_node_visits = (1 + 1 + 1 + 1 + 1) * (self.MAX_RESEARCH_ATTEMPTS + 1)  # Researcher, Analyzer, Validator, Confidence Scoring, and Subtask Router nodes in a single subtask
        synthesizer_visits = 1  # Synthesizer node is always visited once
        BUFFER = 2 # Buffer to account for potienial start or end nodes
        return planner_visits + context_enhancer_visits + (sub_task_node_visits * self.number_of_tasks) + synthesizer_visits + BUFFER
    

    def _create_initial_state(self, users_question: str) -> AgentState:
        return {
            "users_question": users_question,
            "conversation_history": [],
            "sub_tasks": [],
            "current_subtask": "",
            "retrieval_history": [],
            "sub_task_solutions": [],
            "next_node": "",
            "chain_of_thought": "",
            "final_answer": "",
            "enhanced_context": "",
            "validator_feedback": "",
            "validation_flag": "",
            "confidence_score": "",
            "confidence_explanation": "",
            "research_attempts": 0
        }
    

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner)
        graph.add_node("context_enhancer", self._context_enhancer)
        graph.add_node("researcher", self._researcher)
        graph.add_node("analyzer", self._analyzer)
        graph.add_node("validator", self._validator)
        graph.add_node("confidence_scoring", self._confidence_scoring)
        graph.add_node("subtask_router", self._subtask_router)
        graph.add_node("synthesizer", self._synthesizer)

        # Flow: START -> planner -> context_enhancer -> researcher -> analyzer -> validator ->
        # confidence_scoring -> subtask_router (conditional edge) -> synthesizer -> END
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "context_enhancer")
        graph.add_edge("context_enhancer", "researcher")
        graph.add_edge("researcher", "analyzer")
        graph.add_edge("analyzer", "validator")
        graph.add_edge("validator", "confidence_scoring")
        graph.add_edge("confidence_scoring", "subtask_router")
        graph.add_conditional_edges("subtask_router", lambda state: state["next_node"], {
            "researcher": "researcher",
            "synthesizer": "synthesizer",
        })
        graph.add_edge("synthesizer", END)

        return graph.compile()
    
    
    def view_graph(self):
        return self.graph.get_graph(xray=True).draw_mermaid_png()


    def run(self, user_question: str) -> AgentState:
        initial_state : AgentState = self._create_initial_state(user_question)
        return self.graph.invoke(initial_state, config={"recursion_limit": self._calculate_max_node_visits()})
    

    """
    Tools used by the agent in research phase:
    """

    def calculator(self, code: str) -> str:
        """Execute a Python expression or snippet and return the printed result."""
        return self.python_repl.run(code)


    def tax_return_doc_search(self, query: str) -> str:
        """Search the embedded tax document for relevant information."""
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])


    def web_search(self, query: str) -> str:
        """Perform a web search and return the results."""
        try:
            query = query.replace('"', '')
            response = self.tavily_client.search(query=query, max_results=3)
            hits = response.get("results", [])
            if not hits:
                return "No relevant web results found for that query."
            return "\n\n".join(f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}" for r in hits)
        except Exception as e:
            return f"Web search failed due to: {e}"
        
        
    def fallback_tool(self, state: AgentState, failed_tool: str, tool_input: str) -> str: # TODO : comments
        state["chain_of_thought"] += (
            f"\nFallback: Detected failure from '{failed_tool}' with input '{tool_input}'. "
            "Attempting fallback with 'web_search'.\n"
        )
        fallback_result = self._invoke_tool("web_search", tool_input)
        state["chain_of_thought"] += f"\nFallback tool result: {fallback_result}\n"
        return fallback_result
    

    """
    Nodes for the agent's state graph:
    """

    # TODO COMMENTS FOR ALL THE NODES ON WHAT THEY ARE SUPPOSED TO DO

    def _planner(self, state: AgentState) -> AgentState:
        state["chain_of_thought"] += f"Planner Node: deriving sub_tasks based on '{state['users_question']}'...\n"
        planner_prompt = f"""You are a planning assistant.
        The user's main question is: "{state['users_question']}".
        Conversation so far (if relevant):
        {state["conversation_history"]}
        Please break the user's question into EXACTLY {self.number_of_tasks} actionable sub-tasks or key points we should investigate.
        Each sub-task should be concise, focusing on a specific aspect of the question or problem.
        Only list the sub-tasks, with no extra explanation.
        """
        plan_response = self._call_llm(planner_prompt)
        generated_subtasks = self._parse_subtasks(plan_response)
        state["sub_tasks"] = generated_subtasks
        if generated_subtasks:
            state["current_subtask"] = generated_subtasks[0]
        state["chain_of_thought"] += f"Planner Node: LLM sub-task output:\n{plan_response}\n"
        return state
    

    def _context_enhancer(self, state: AgentState) -> AgentState: # TODO : comments FIX THIS 
        conversation_text = "\n".join(
            f"{msg.get('role', 'UNKNOWN')}: {msg.get('content', '')}"
            for msg in state.get("conversation_history", [])
        )
        recent_solutions = state.get("sub_task_solutions", [])
        latest_solutions = "\n".join(recent_solutions[-3:]) if recent_solutions else ""
        enhanced_context = f"Conversation History:\n{conversation_text}\n\nRecent Solutions:\n{latest_solutions}"
        state["enhanced_context"] = enhanced_context
        state["chain_of_thought"] += "\nContext Enhancer Node: Enhanced context built for subsequent nodes.\n"
        return state


    def _researcher(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state
        state["chain_of_thought"] += f"\nResearcher Node: deciding tool for '{current_subtask}'\n"
        solutions_text = "\n".join(state["sub_task_solutions"])
        researcher_prompt = f"""You are a researcher that decides which tool to use.
        Current subtask: "{current_subtask}"
        Enhanced Context: {state.get("enhanced_context", "")}
        Context from sub_task_solutions:
        {solutions_text}
        Tools available:
        1) calculator
        2) tax_return_doc_search
        3) web_search
        Which tool should we use next to gather info for this subtask?
        Also provide the input we should pass to that tool.
        Output format:
        Tool: <one of calculator/source_doc_search/web_search>
        ToolInput: <string input for that tool>
        """
        researcher_response = self._call_llm(researcher_prompt)
        state["research_attempts"] += 1
        state["chain_of_thought"] += f"Researcher Node: Response for which tool to use:\n{researcher_response}\n"
        chosen_tool, tool_input = self._parse_tool_decision(researcher_response)
        try:
            content = self._invoke_tool(chosen_tool, tool_input)
        except Exception as e:
            content = self.fallback_tool(state, chosen_tool, tool_input)
        entry = {
            "subtask": current_subtask,
            "tool_used": chosen_tool,
            "tool_input": tool_input,
            "content": content
        }
        state["retrieval_history"].append(entry)
        state["chain_of_thought"] += f"Researcher Node: Content found for use:\n{content}\n"
        return state


    def _analyzer(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state
        relevant_entries = [rh for rh in state["retrieval_history"] if rh["subtask"] == current_subtask]
        subtask_content = "\n".join(e["content"] for e in relevant_entries)
        analyzer_prompt = f"""You are an analyzer.
        The current subtask is: "{current_subtask}".
        Here is the information we retrieved:
        {subtask_content}
        Please create or update a partial solution for this subtask that integrates the new findings.
        Your answer should be thorough and include any relevant details previously found
        """
        analyzer_response = self._call_llm(analyzer_prompt)
        state["sub_task_solutions"].append(analyzer_response)
        state["chain_of_thought"] += f"\nAnalyzer Node: Solution for '{current_subtask}':\n{analyzer_response}\n"
        return state
    
    
    def _validator(self, state: AgentState) -> AgentState:
        current_subtask = state.get("current_subtask", "")
        if not current_subtask:
            return state
        relevant_entries = [
            entry for entry in state.get("retrieval_history", [])
            if entry.get("subtask") == current_subtask
        ]
        aggregated_content = "\n".join([entry.get("content", "") for entry in relevant_entries])
        validator_prompt = (
            f"You are a validator node. The current subtask is: '{current_subtask}'.\n"
            f"Below is the aggregated partial solution information:\n{aggregated_content}\n\n"
            "Please evaluate whether the solution has any inconsistencies or gaps. "
            "If everything looks good, respond with 'OK'. Otherwise, describe what should be improved."
        )
        validator_response = self._call_llm(validator_prompt)
        state["validator_feedback"] = validator_response
        if "OK" in validator_response.upper():
            state["validation_flag"] = "ACCEPTED"
        else:
            state["validation_flag"] = "REVIEW_NEEDED"
        state["chain_of_thought"] += f"\nValidator Node: Feedback for '{current_subtask}': {validator_response}\n"
        return state
    

    def _confidence_scoring(self, state: AgentState) -> AgentState:
        current_subtask = state.get("current_subtask", "")
        if not current_subtask:
            return state
        latest_solution = state["sub_task_solutions"][-1] if state.get("sub_task_solutions") else "No solution found."
        score_prompt = (
            f"You are a confidence scoring assistant. For the subtask '{current_subtask}', please evaluate the "
            f"following solution on a scale from 1 (low confidence) to 10 (high confidence), and provide a brief "
            f"justification for the score. Here is the solution:\n{latest_solution}\n\n"
            "Provide your answer in the format: Score: <number>, Explanation: <brief explanation>."
        )
        score_response = self._call_llm(score_prompt)
        parts = score_response.split("Explanation:")
        if len(parts) == 2:
            score_str = parts[0].split("Score:")[-1].strip()
            explanation = parts[1].strip()
            state["confidence_score"] = score_str
            state["confidence_explanation"] = explanation
        else:
            state["confidence_score"] = score_response
            state["confidence_explanation"] = ""
        state["chain_of_thought"] += f"\nConfidence Scoring Node: Response for '{current_subtask}': {score_response}\n"
        return state


    def _subtask_router(self, state: AgentState) -> AgentState:
        if "research_attempts" not in state:
            state["research_attempts"] = 0

        sub_tasks = state["sub_tasks"]
        current_subtask = state["current_subtask"]

        if not sub_tasks or current_subtask not in sub_tasks:
            state["chain_of_thought"] += "\nSubtask Router: no valid subtask. Going to synthesizer.\n"
            state["next_node"] = "synthesizer"
            return state

        # Check if the current subtask exceeds the maximum attempts.
        if state["research_attempts"] >= self.MAX_RESEARCH_ATTEMPTS:
            state["chain_of_thought"] += (
                f"\nSubtask Router: Reached max research attempts ({self.MAX_RESEARCH_ATTEMPTS}) "
                f"for subtask '{current_subtask}'. Finalizing this subtask.\n"
            )
            state["sub_task_solutions"].append(
                f"[Note: Maximum research attempts reached for subtask '{current_subtask}'. Proceeding with best available findings.]"
            )
            state["research_attempts"] = 0  # Reset for the next subtask

            current_index = sub_tasks.index(current_subtask)
            if current_index + 1 < len(sub_tasks):
                state["current_subtask"] = sub_tasks[current_index + 1]
                state["chain_of_thought"] += f"Subtask Router: Moving to next subtask '{state['current_subtask']}'.\n"
                state["next_node"] = "researcher"
            else:
                state["chain_of_thought"] += "Subtask Router: No more subtasks left. Proceeding to synthesizer.\n"
                state["next_node"] = "synthesizer"
            return state

        # Generate the router prompt
        router_prompt = f"""You are a subtask router deciding if we have gathered sufficient information for the subtask '{current_subtask}'.
        The user's main question is: "{state['users_question']}".
        Validator Feedback: "{state.get("validator_feedback", "None")}".
        Validation Flag: "{state.get("validation_flag", "None")}".
        Confidence Score: "{state.get("confidence_score", "None")}".
        
        If the validator feedback shows that the solution is complete (Validation Flag of ACCEPTED) and the confidence score is high (7 or above on a scale of 1-10), please answer 'DONE'.
        Otherwise, if further research is needed, answer 'MORE'.

        Also, if additional research is unlikely to significantly improve the subtask after {self.MAX_RESEARCH_ATTEMPTS - state['research_attempts']} more attempts, answer 'DONE'.

        Provide only one word: either 'MORE' or 'DONE' as your answer.
        """

        router_response = self._call_llm(router_prompt).strip().lower()

        state["chain_of_thought"] += f"\nSubtask Router Node: Decision for '{current_subtask}': {router_response}\n"
        if "more" in router_response:
            state["next_node"] = "researcher"
            return state
        else:
            state["research_attempts"] = 0
            current_index = sub_tasks.index(current_subtask)
            if current_index + 1 < len(sub_tasks):
                state["current_subtask"] = sub_tasks[current_index + 1]
                state["chain_of_thought"] += f"Subtask Router Node: Moving to next subtask '{state['current_subtask']}'.\n"
                state["next_node"] = "researcher"
                return state
            else:
                state["chain_of_thought"] += "Subtask Router Node: All subtasks complete. Proceeding to synthesizer.\n"
                state["next_node"] = "synthesizer"
                return state


    def _synthesizer(self, state: AgentState) -> AgentState:
        question = state["users_question"]
        solutions_text = "\n\n".join(state["sub_task_solutions"])
        synth_prompt = f"""You are the final synthesizer.
        User's question: "{question}"
        Below are the solutions for each subtask:
        {solutions_text}
        Validator Feedback: {state.get("validator_feedback", "")}
        Confidence Score: {state.get("confidence_score", "")}
        Please synthesize these findings into a final, coherent response that thoroughly addresses the user's question.
        Structure your answer with an introduction, a clear body, 
        and a conclusion. The goal is to provide a comprehensive and
        well-organized answer that doesn't give suggestions on who to go to but to give specific advice.
        """
        final_response = self._call_llm(synth_prompt)
        state["final_answer"] = final_response
        state["chain_of_thought"] += "\nSynthesizer Node: Final answer generated.\n"
        return state


    """
    Helper Methods
    """


    def _parse_subtasks(self, text: str) -> List[str]:
        lines = [line.strip("-• ").strip() for line in text.split("\n") if line.strip()]
        return [line for line in lines if line]


    def _parse_tool_decision(self, researcher_response: str) -> (str, str):
        chosen_tool = None
        tool_input = None
        for line in researcher_response.split("\n"):
            line_lower = line.lower()
            if line_lower.startswith("tool:"):
                chosen_tool = line.split(":", 1)[1].strip().strip("`'\"")
            elif line_lower.startswith("toolinput:"):
                tool_input = line.split(":", 1)[1].strip().strip("`'\"")
        if chosen_tool is None:
            chosen_tool = "web_search"
        if tool_input is None:
            tool_input = researcher_response
        return chosen_tool, tool_input


    def _invoke_tool(self, tool_name: str, tool_input: str) -> str:
        for t in self.tools:
            if t.name == tool_name:
                result = t.invoke(tool_input)
                return self._format_result(result)
        raise ValueError(f"Tool '{tool_name}' not found.")


    def _format_result(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            return "\n".join(str(item) for item in data)
        return str(data)
    

    def _call_llm(self, prompt: str) -> str: # TODO : Comments, error handling should be added to chain of thought?
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error invoking LLM: {e}"