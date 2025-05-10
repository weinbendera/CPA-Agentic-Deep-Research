import re
from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langchain_experimental.utilities import PythonREPL
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field


class SubtaskRecord(TypedDict):
    """
    A class representing a subtask record for each individual subtask
    """
    retrieval_entries: List[Dict[str, Any]]
    solution: str

class PlannerOutput(BaseModel):
    """Structure for planner’s JSON output"""
    subtasks: List[str] = Field(..., description="A list of distinct, concise subtasks")


class AgentState(TypedDict):
    """
    A class representing the state of the agent and the data stored.
    """
    # The user's current question to the chatbot
    user_question: str
    # The images from the processed PDFs as b64 strings
    vision_payloads: List[dict]
    # The conversation history, including the user's question and the agent's responses
    conversation_history: List[Dict[str, str]]

    # The subtasks that are being researched,
    # includes the name and its SubtaskRecord
    subtask_content: Dict[str, SubtaskRecord]
    sub_tasks: List[str]
    current_subtask: str
    next_node: str
    chain_of_thought: str
    final_answer: str

class DeepResearchAgent:
    """
    An AI agent that performs deep research on a question.
    """
    def __init__(self, llm_model: str, model_key: str, tavily_api_key: str, vectorstore: Any=None, number_of_tasks: int = 3):
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=model_key)
        self.structured_planner = self.llm.with_structured_output(PlannerOutput, method="function_calling")

        self.python_repl = PythonREPL()
        self.tavily_client = TavilyClient(tavily_api_key)
        self.vectorstore = vectorstore
        self.tools = [
            Tool.from_function(self.calculator, name="calculator", description=self.calculator.__doc__),
            Tool.from_function(self.tax_return_doc_search, name="tax_return_doc_search", description=self.tax_return_doc_search.__doc__),
            Tool.from_function(self.web_search, name="web_search", description=self.web_search.__doc__)
        ]
        self.number_of_tasks = number_of_tasks
        self.graph = self._build_graph()


    def run(self, user_question: str, vision_payloads: list[dict] = None) -> AgentState:
        """
        params: user_question: str, what the user has entered
        params: vision_payloads: list[dict], the payloads from the vision model
        """
        initial_state: AgentState = {
            "user_question": user_question,
            "conversation_history": [],
            "vision_payloads": vision_payloads or [],
            "subtask_content": {},
            "sub_tasks": [],
            "current_subtask": "",
            "next_node": "",
            "chain_of_thought": "",
            "final_answer": ""
        }
        return self.graph.invoke(initial_state)

    """
    Tools used by the agent in research phase:
    """

    def calculator(self, code: str) -> str:
        """Execute a Python expression or snippet and return the printed result."""
        return self.python_repl.run(code)


    def tax_return_doc_search(self, query: str) -> str:
        """Search the embedded tax document for relevant information."""
        if self.vectorstore is None:
            return "No document vectorstore available for search."

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

    """
    Nodes for the agent's state graph:
    """

    def _planner(self, state: AgentState) -> AgentState:
        state["chain_of_thought"] += f"Planner Node: deriving sub_tasks based on '{state['user_question']}'.\n"
        try :
            scenarios = self.load_scenarios("./../data/scenarios.md")
        except Exception as e:
            print(f"Failed to load scenarios: {e}")
            scenarios = []

        planner_prompt = f"""You are an expert CPA planning assistant.
        The user's main question is: "{state['user_question']}".
        Conversation so far (if relevant):
        {state["conversation_history"]}
        The user has provided a PDF document with tax information and images that has been sent to you.
        
        Based on these inputs, break the user's question into around {self.number_of_tasks} actionable sub-tasks or key points to investigate.
        Each sub-task should be concise, focusing on a specific aspect of the question or problem.
        You have the user's documents indexed in a vectorstore.
        Plan subtasks that would require retrieving relevant chunks from that vectorstore or searching the online web.
        Only list the sub-tasks, with no extra explanation.
        Example subtasks: {[scenarios]}
        Return them as a JSON object matching the PlannerOutput schema.
        """

        MAX_B64 = 5 * 1_048_576
        batches = self.chunk_by_b64_chars(state["vision_payloads"], MAX_B64)

        # Add the prompt first
        content = [{"type": "text", "text": planner_prompt}]
        # Add the vision payloads as batches to the content and create subtasks from the single batch
        all_subtasks = []
        for batch in batches:
            content = [{"type": "text", "text": planner_prompt}] + batch
            result: PlannerOutput = self.structured_planner.invoke(input=[{"role": "user", "content": content}])
            all_subtasks.extend(result.subtasks)

        generated_subtasks = all_subtasks

        # Cut out similar subtasks
        subtasks = self.deduplicate_subtasks(generated_subtasks, threshold=0.7)
        state["sub_tasks"] = subtasks

        # Initialize the subtask content with empty retrieval entries and solutions
        state["subtask_content"] = {
            subtask: SubtaskRecord(retrieval_entries=[], solution="")
            for subtask in subtasks
        }
        state["current_subtask"] = subtasks[0]
        state["chain_of_thought"] += f"Planner Node: LLM sub-task output:\n{generated_subtasks}\n"
        return state


    def _researcher(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state
        state["chain_of_thought"] += f"\nResearcher Node: deciding tool for '{current_subtask}'\n"

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": ""})
        retrieval_entries = subtask_record["retrieval_entries"]
        current_solution = subtask_record["solution"]

        researcher_prompt = f"""You are a researcher that decides which tool to use.
        Current subtask: 
        "{current_subtask}"
        The current solution for this subtask is:
        {current_solution}
        The retrieval entries found so far for this subtask are:
        {retrieval_entries}
        
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

        researcher_response = self.llm.invoke(researcher_prompt).content

        state["chain_of_thought"] += f"Researcher Node: Response for which tool to use:\n{researcher_response}\n"
        chosen_tool, tool_input = self._parse_tool_decision(researcher_response)
        content = self._invoke_tool(chosen_tool, tool_input)

        entry = {"tool_used": chosen_tool, "tool_input": tool_input, "content": content}

        subtask_record["retrieval_entries"].append(entry)
        state["subtask_content"][current_subtask] = subtask_record

        state["chain_of_thought"] += f"Researcher Node: Content found for use:\n{content}\n"
        return state


    def _analyzer(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": ""})
        retrieval_entries = subtask_record["retrieval_entries"]
        current_solution = subtask_record["solution"]

        analyzer_prompt = f"""You are an expert tax CPA analyzer that summarizes findings from research.
        The current subtask is: "{current_subtask}".
        The current solution for this subtask is:
        {current_solution}
        The retrieval entries found so far for this subtask are:
        {retrieval_entries}
        
        Please create or update a partial solution for this subtask that integrates the new findings.
        Your answer should be thorough and include any relevant details previously found.
        """
        analyzer_response = self.llm.invoke(analyzer_prompt).content

        subtask_record["solution"] = analyzer_response
        state["subtask_content"][current_subtask] = subtask_record

        state["chain_of_thought"] += f"\nAnalyzer Node: Solution for '{current_subtask}':\n{analyzer_response}\n"
        return state


    def _subtask_router(self, state: AgentState) -> AgentState:
        sub_tasks = state["sub_tasks"]
        current_subtask = state["current_subtask"]
        if not sub_tasks or current_subtask not in sub_tasks:
            state["chain_of_thought"] += "\nSubtask Router: no valid subtask. Going to synthesizer.\n"
            state["next_node"] = "synthesizer"
            return state

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": ""})
        current_solution = subtask_record["solution"]

        router_prompt = f"""You are an expert tax CPA that is deciding whether a subtask is fully researched or not.
        The user's main question is:
         "{state['user_question']}".
        The current solution for this subtask is:
        {current_solution}

        Do we have enough information to finalize this subtask, or do we need more research?
        Answer 'MORE' if we should research more, or 'DONE' if we can move on.
        Research more if the information found is not relevant to the subtask or not relevant to the user's main question.
        """
        router_response = self.llm.invoke(router_prompt).content.lower()
        state["chain_of_thought"] += f"\nSubtask Router Node: Router's decision: {router_response}\n"
        if "more" in router_response:
            state["chain_of_thought"] += "Subtask Router Node: Continuing research for this subtask.\n"
            state["next_node"] = "researcher"
            return state
        else:
            index = sub_tasks.index(current_subtask) + 1
            if index < len(sub_tasks):
                new_subtask = sub_tasks[index]
                state["current_subtask"] = new_subtask
                state["chain_of_thought"] += f"Subtask Router Node: Next subtask '{new_subtask}'.\n"
                state["next_node"] = "researcher"
                return state
            else:
                state["chain_of_thought"] += f"Subtask Router Node: All subtasks done. Proceed to synthesizer.\n"
                state["next_node"] = "synthesizer"
                return state


    def _synthesizer(self, state: AgentState) -> AgentState:
        # Combine all subtasks, their final solutions, and their retrieval entries into a final answer.
        subtask_blocks = []
        for subtask in state.get("sub_tasks", []):
            subtask_record = state["subtask_content"].get(subtask, {"retrieval_entries": [], "solution": ""})
            solution = (subtask_record.get("solution") or "").strip() or "_[no solution yet]_"
            retrieval_entries = "\n".join(
                f"- Tool: {entry['tool_used']}\n  Tool Input: {entry['tool_input']}\n  Content: {entry['content']}"
                for entry in subtask_record["retrieval_entries"]
            )
            subtask_blocks.append(f"### {subtask}\n{solution}\n\nResearch Found:{retrieval_entries}")

        findings = "\n\n".join(subtask_blocks)

        synthesizer_prompt = f"""You are an expert tax CPA synthesizer that combines findings from research 
        to give a comprehensive fully researched response based on the user's question and the research findings.
        The user's main question is:
         "{state['user_question']}".
         
        Subtasks, their solutions, and their research findings:
        {findings}
        
        Please synthesize these findings into a final response that thoroughly addresses the user's main question and findings that can help the user.
        Provide a short introduction of the findings and how you have found them.
        The body should be well-organized with sections that address the subtasks you have researched, 
        a comprehensive look at their solutions that have been found through your research,
        the sections should also includes actionable steps the user can take based on the findings,
        Provide a conclusion that gives a brief overview of what was talked about in the body and how it relates to the user's main question.
        """
        final_response = self.llm.invoke(synthesizer_prompt)
        state["final_answer"] = final_response.content
        state["chain_of_thought"] += "\nSynthesizer Node: Produced final answer.\n"
        return state


    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner)
        graph.add_node("researcher", self._researcher)
        graph.add_node("analyzer", self._analyzer)
        graph.add_node("subtask_router", self._subtask_router)
        graph.add_node("synthesizer", self._synthesizer)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "analyzer")
        graph.add_edge("analyzer", "subtask_router")
        graph.add_conditional_edges("subtask_router", lambda state: state["next_node"], {
            "researcher": "researcher",
            "synthesizer": "synthesizer",
        })
        graph.add_edge("synthesizer", END)
        return graph.compile()

    """
    Helper Methods
    """

    def deduplicate_subtasks(self, subtasks: List[str], threshold: float = 0.8) -> List[str]:
        # embed the names of each subtask
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        embeddings = embedder.embed_documents(subtasks)
        kept_texts = []
        kept_embeds: List[List[float]] = []
        for text, emb in zip(reversed(subtasks), reversed(embeddings)):
            if not kept_embeds:
                kept_texts.append(text)
                kept_embeds.append(emb)
                continue
            sims = cosine_similarity([emb], kept_embeds)[0]
            if max(sims) < threshold:
                kept_texts.append(text)
                kept_embeds.append(emb)

        return kept_texts

    def chunk_by_b64_chars(self, payloads: list[dict], max_chars: int):
        batches, current, total = [], [], 0
        for p in payloads:
            # extract just the Base64 portion (after the comma)
            b64 = p["image_url"]["url"].split(",", 1)[1]
            length = len(b64)
            # if adding this page would overflow, start a new batch
            if current and total + length > max_chars:
                batches.append(current)
                current, total = [], 0
            current.append(p)
            total += length
        if current:
            batches.append(current)
        return batches


    def load_scenarios(self, scenarios_file: str) -> List[str]:
        with open(scenarios_file, 'r', encoding='utf-8') as file:
            content = file.read()
        pattern = r'^\d+\.\s*\*\*(.*?)\*\*\s*:\s*(.*?)(?=\n\d+\.|\Z)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        scenarios = []
        for title, desc in matches:
            # collapse newlines, trim whitespace
            desc = ' '.join(desc.split())
            scenarios.append(f"{title.strip()}: {desc}")
        return scenarios


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
            lines = []
            for item in data:
                lines.append(str(item))
            return "\n".join(lines)
        return str(data)