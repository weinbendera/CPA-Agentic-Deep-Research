import re
from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
import math
import asyncio
import copy
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Spacer, Paragraph, SimpleDocTemplate, PageBreak, Frame, BaseDocTemplate, PageTemplate
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch
import re



class SubtaskRecord(TypedDict):
    """
    A class representing a subtask record for each individual subtask
    Contains the -retrieval_entries- found during research,
    the current -solution- to the subtask,
    the -feedback- from the validator node on how to improve the solution through research,
    and the current number of -research_attempts- used for the subtask.
    """
    retrieval_entries: List[Dict[str, Any]]
    solution: str
    validator_feedback: str
    research_attempts: int


class PlannerOutput(BaseModel):
    """Structure for planner’s JSON output"""
    subtasks: List[str] = Field(..., description="A list of distinct, user-specific subtasks")


class ResearcherOutput(BaseModel):
    """Structure for researcher's JSON output"""
    tool: str = Field(..., description="The tool to use for research")
    tool_input: str = Field(..., description="The input for the chosen tool")
    reason: str = Field(..., description="The reason for the chosen tool and input")


class ValidatorOutput(BaseModel):
    """Structure for validator's JSON output"""
    feedback: str = Field(..., description="Feedback on the solution")
    validation_flag: str = Field(..., description="Validation status (e.g., 'ACCEPTED', 'REVIEW_NEEDED')")


class ConfidenceScoringOutput(BaseModel):
    """Structure for confidence scoring's JSON output"""
    score: str = Field(..., description="Confidence score (e.g., '1-10')")
    explanation: str = Field(..., description="Explanation of the score")


class RouterOutput(BaseModel):
    """Structure for router's JSON output"""
    decision: str = Field(..., description="Decision on whether to continue research or finalize the subtask (e.g., 'MORE', 'DONE')")


class AgentState(TypedDict):
    """
    A class representing the state of the agent and the data stored.
    """
    user_question: str
    vision_payloads: List[dict]
    conversation_history: List[Dict[str, str]]
    subtask_content: Dict[str, SubtaskRecord]
    sub_tasks: List[str]
    current_subtask: str
    next_node: str
    chain_of_thought: str
    report: str
    final_answer: str
    validation_flag: str
    confidence_score: str
    confidence_explanation: str

class NumberedCanvas(Canvas):
        def __init__(self, *args, **kwargs):
            Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_number(num_pages)
                Canvas.showPage(self)
            Canvas.save(self)

        def draw_page_number(self, page_count):
            self.setFont("Helvetica", 8)
            self.drawRightString(7.5 * inch, 0.5 * inch, f"Page {self._pageNumber} of {page_count}")

class DeepResearchAgent:
    """
    An AI agent that performs deep research on a question.
    """

    def __init__(self, llm_model: str, model_key: str, tavily_api_key: str, vectorstore: Any = None, number_of_tasks: int = 3, MAX_RESEARCH_ATTEMPTS: int = 3):
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=model_key)
        self.structured_planner = self.llm.with_structured_output(PlannerOutput, method="function_calling")
        self.structured_researcher = self.llm.with_structured_output(ResearcherOutput, method="function_calling")
        self.structured_validator = self.llm.with_structured_output(ValidatorOutput, method="function_calling")
        self.structured_scorer = self.llm.with_structured_output(ConfidenceScoringOutput, method="function_calling")
        self.structured_router = self.llm.with_structured_output(RouterOutput, method="function_calling")

        self.python_repl = PythonREPL()
        self.tavily_client = TavilyClient(tavily_api_key)
        self.vectorstore = vectorstore
        self.tools = [
            Tool.from_function(self.calculator, name="calculator", description=self.calculator.__doc__),
            Tool.from_function(self.tax_return_doc_search, name="tax_return_doc_search",
                               description=self.tax_return_doc_search.__doc__),
            Tool.from_function(self.web_search, name="web_search", description=self.web_search.__doc__)
        ]
        self.number_of_tasks = number_of_tasks
        self.MAX_RESEARCH_ATTEMPTS = MAX_RESEARCH_ATTEMPTS


    def _calculate_max_node_visits(self, payload_size) -> int:
        # Calculate the maximum number of node visits based on the number of tasks and research attempts
        planner_visits = 1  # Planner node is always visited once
        context_enhancer_visits = 1  # Context Enhancer node is always visited once
        # TODO CHECK THIS MATH FOR RESEARCHER
        payload_size = payload_size / 1_048_576  # Convert to MB
        total_plans = math.ceil(payload_size / 5)  # Assuming 5MB per plan
        total_tasks = total_plans * self.number_of_tasks
        print(f"Total Number of tasks possible: {total_tasks}")
        sub_task_node_visits = (1 + 1 + 1 + 1 + 1) * (
                    self.MAX_RESEARCH_ATTEMPTS + 1)  # Researcher, Analyzer, Validator, Confidence Scoring, and Subtask Router nodes in a single subtask
        synthesizer_visits = 1  # Synthesizer node is always visited once
        BUFFER = 2  # Buffer to account for potienial start or end nodes
        recursion_limit = planner_visits + context_enhancer_visits + (
                    sub_task_node_visits * total_tasks) + synthesizer_visits + BUFFER
        print(f"Recursion Limit: {recursion_limit}")
        return recursion_limit


    def _create_initial_state(self, user_question: str, vision_payloads: list[dict] = None) -> AgentState:
        return {
            "user_question": user_question,
            "conversation_history": [],
            "vision_payloads": vision_payloads or [],
            "subtask_content": {},
            "sub_tasks": [],
            "current_subtask": "",
            "next_node": "",
            "chain_of_thought": "",
            "validation_flag": "",
            "confidence_score": "",
            "confidence_explanation": "",
            "report": "",
            "final_answer": ""
        }
    

    def _build_subtask_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("researcher", self._researcher)
        graph.add_node("analyzer", self._analyzer)
        graph.add_node("validator", self._validator)
        graph.add_node("confidence_scoring", self._confidence_scoring)
        graph.add_node("subtask_router", self._subtask_router)

        graph.add_edge(START, "researcher")
        graph.add_edge("researcher", "analyzer")
        graph.add_edge("analyzer", "validator")
        graph.add_edge("validator", "confidence_scoring")
        graph.add_edge("confidence_scoring", "subtask_router")
        graph.add_conditional_edges("subtask_router",
            lambda state: state["next_node"],
            {"researcher": "researcher", "END": END}
        )
        return graph.compile()
    

    async def _run_one_subtask(self, master_state: AgentState, subtask: str, payload_size) -> AgentState:
        # make a deep copy so each subgraph run is isolated
        sub_state = copy.deepcopy(master_state)
        sub_state["current_subtask"] = subtask
        # compile & invoke your subtask graph (synchronous), but wrap it in a thread
        return await asyncio.to_thread(self.subtask_graph.invoke,
                                       sub_state,
                                       {"recursion_limit": self._calculate_max_node_visits(payload_size=payload_size)})
    

    async def run(self, user_question: str, vision_payloads: list[dict] = None) -> AgentState:
        # 1) create initial state, run planner
        state = self._create_initial_state(user_question, vision_payloads)
        state = await self._planner(state)

        # 2) build subtask‐only graph once
        self.subtask_graph = self._build_subtask_graph()

        # 3) fan‐out: launch one task per subtask

        payload_size = sum(len(p["image_url"]["url"].split(",", 1)[1]) for p in vision_payloads) if vision_payloads else 0
        print("Size of payloads: ", payload_size)

        tasks = [
            self._run_one_subtask(state, sub, payload_size)
            for sub in state["sub_tasks"]
        ]
        completed_states = await asyncio.gather(*tasks)

        # 4) merge all the results back into the master state
        for sub_state in completed_states:
            sub = sub_state["current_subtask"]
            record = sub_state["subtask_content"][sub]
            state["subtask_content"][sub] = record
            state["report"] += "\n" + record["solution"]
            state["chain_of_thought"] += (
            f"\n=== Chain of Thought for '{sub}' ===\n"
            + sub_state["chain_of_thought"]
            + f"\n=== End of Chain of Thought for '{sub}' ===\n")

        # 5) finally, synthesize everything once
        state = self._synthesizer(state)

        self._save_final_answer_as_pdf(state, filename="final_answer.pdf")
        return state

    """
    Tools used by the agent in research phase:
    """

    def calculator(self, code: str) -> str:
        """Execute a Python expression to make your own calculations."""
        return self.python_repl.run(code)

    def tax_return_doc_search(self, query: str) -> str:
        """Search the users embedded tax document for relevant information."""
        if self.vectorstore is None:
            return "No document vectorstore available for search."

        results = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    def web_search(self, query: str) -> str:
        """Perform a web search to retrieve new tax laws or other information you are looking for."""
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

    async def _planner(self, state: AgentState) -> AgentState:
        state["chain_of_thought"] += f"Planner Node: deriving sub_tasks based on '{state['user_question']}'.\n"
        try:
            scenarios = self._load_scenarios("../data/scenarios.md")
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
        batches = self._chunk_by_b64_chars(state["vision_payloads"], MAX_B64)

        async def call_batch(batch):
            content = [{"type":"text","text":planner_prompt}] + batch
            # wrap the blocking .invoke(...) in a Thread
            resp: PlannerOutput = await asyncio.to_thread(
                self.structured_planner.invoke,
                input=[{"role":"user","content":content}],
            )
            return resp.subtasks

        tasks = [call_batch(batch) for batch in batches]
        # 3) fire them all at once
        results = await asyncio.gather(*tasks)

        all_subtasks = [sub for batch_subs in results for sub in batch_subs]
        state["chain_of_thought"] += f"Got {len(all_subtasks)} raw subtasks, now deduplicating…\n"

        subtasks = self._deduplicate_subtasks(all_subtasks, threshold=0.7)
        state["sub_tasks"] = subtasks
        state["subtask_content"] = {
            sub: SubtaskRecord(retrieval_entries=[], solution="", validator_feedback="", research_attempts=0)
            for sub in subtasks
        }
        state["current_subtask"] = subtasks[0] if subtasks else ""
        state["chain_of_thought"] += f"Planner Node: Final subtasks: {subtasks}\n"

        return state


    def _researcher(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state
        state["chain_of_thought"] += f"\nResearcher Node: deciding tool for '{current_subtask}'\n"

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": "",
                                                                        "validator_feedback": "", "research_attempts": 0})
        retrieval_entries = subtask_record["retrieval_entries"]
        current_solution = subtask_record["solution"]
        validator_feedback = subtask_record["validator_feedback"]

        researcher_prompt = f"""You are a researcher that decides which tool and what to enter as an input to conduct research.
        Current subtask: 
        "{current_subtask}"
        The current solution for this subtask is:
        {current_solution}
        An expert tax CPA has validated the current solution to this subtask and has provided feedback
        about gaps in the solution to research and provided insight into which tool to use:
        {validator_feedback}
        The retrieval entries found so far for this subtask are:
        {retrieval_entries}
        
        Tools available:
        1) calculator
        2) tax_return_doc_search
        3) web_search
        Which tool should we use next to gather info for this subtask?
        Provide the input we should pass to that tool.
        Also provide a short explanation of why you chose this tool and input.
        Return them as a JSON object matching the ResearcherOutput schema.
        """

        researcher_response: ResearcherOutput = self.structured_researcher.invoke(input=[{"role": "user", "content": researcher_prompt}])
        chosen_tool = researcher_response.tool
        tool_input = researcher_response.tool_input

        state["chain_of_thought"] += f"\nResearcher Node: Tool chosen: {chosen_tool}\n Tool input: {tool_input}\n Reason: {researcher_response.reason}\n"
        content = self._invoke_tool(chosen_tool, tool_input)
        entry = {"tool_used": chosen_tool, "tool_input": tool_input, "content": content}

        subtask_record["retrieval_entries"].append(entry)
        state["subtask_content"][current_subtask] = subtask_record

        research_attempts = subtask_record.get("research_attempts", 0) + 1
        subtask_record["research_attempts"] = research_attempts
        state["chain_of_thought"] += f"Researcher Node: Content found for use:\n{content}\n Attempt Number: {research_attempts}\n"
        print(f"\nResearcher: Current Subtask: {current_subtask}\nThis is research attempt: {research_attempts} , and this is the content found:\n{content}\n")
        return state


    def _analyzer(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": "",
                                                                        "validator_feedback": "", "research_attempts": 0})
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
        state["chain_of_thought"] += f"\nAnalyzer Node: Solution for '{current_subtask}' is:\n{analyzer_response}\n"
        print(f"\nAnalyzer: Current Solution: \n{analyzer_response}\n")
        return state


    def _validator(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": "",
                                                                        "validator_feedback": "", "research_attempts": 0})
        retrieval_entries = subtask_record["retrieval_entries"]
        current_solution = subtask_record["solution"]

        validator_prompt = f"""You are a validator node. The current subtask is: '{current_subtask}'.\n"
        The current solution for this subtask is:
        {current_solution}
        The retrieval entries found so far for this subtask are:
        {retrieval_entries}
        Evaluate whether the solution has any inconsistencies or gaps.
        If everything is accurate and complete, respond with 'ACCEPTED'. Otherwise, respond with 'REVIEW_NEEDED' as your validation_flag.
        For your feedback field, explain why you think the solution is acceptable why it needs to be reviewed.
        If you are explaining why it needs to be reviewed, please provide a list of the gaps or inconsistencies you found
        and tools and their inputs that you believe could help to fill in those gaps.
        Return these two as a JSON object matching the ValidatorOutput schema.
        """

        validator_response: ValidatorOutput = self.structured_validator.invoke(input=[{"role": "user", "content": validator_prompt}])

        subtask_record["validator_feedback"] = validator_response.feedback

        state["validation_flag"] = validator_response.validation_flag
        state["chain_of_thought"] += f"\nValidator Node: Feedback for '{current_subtask}': {validator_response.feedback}\n"
        print(f"\nValidator: Validator's Flag: {validator_response.validation_flag}\nValidator's Feedback: {validator_response.feedback}\n")
        return state


    def _confidence_scoring(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        if not current_subtask:
            return state

        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": ""})
        retrieval_entries = subtask_record["retrieval_entries"]
        current_solution = subtask_record["solution"]

        score_prompt = f"""You are a confidence scoring assistant.
        For the subtask '{current_subtask}', please evaluate the following solution on a scale from 1 (low confidence) to 10 (high confidence),
         and provide a brief justification for the score. Here is the solution:\n{current_solution}\n\n"
        Return these two as a JSON object matching the ConfidenceScoringOutput schema.
        """

        score_response: ConfidenceScoringOutput = self.structured_scorer.invoke(input=[{"role": "user", "content": score_prompt}])

        state["confidence_score"] = score_response.score
        state["confidence_explanation"] = score_response.explanation

        state["chain_of_thought"] += f"\nConfidence Scoring Node: Response for '{current_subtask}': {score_response}\n"
        print(f"\nConfidence Scorer: Scoring -> {score_response}\n")
        return state


    def _subtask_router(self, state: AgentState) -> AgentState:
        current_subtask = state["current_subtask"]
        subtask_record = state["subtask_content"].get(current_subtask, {"retrieval_entries": [], "solution": "",
                                                                        "validator_feedback": "", "research_attempts": 0})
        research_attempts = subtask_record["research_attempts"]

        # Check if the current subtask exceeds the maximum attempts.
        if research_attempts >= self.MAX_RESEARCH_ATTEMPTS:
            print("\nRouter: Reached max research attempts!\n")
            state["chain_of_thought"] += (
                f"\nSubtask Router: Reached max research attempts ({self.MAX_RESEARCH_ATTEMPTS}) "
                f"for subtask '{current_subtask}'. Finalizing this subtask.\n"
            )

            state["next_node"] = "END"
        else:
            router_prompt = f"""You are a subtask router deciding if we have gathered sufficient information for the subtask '{current_subtask}'.
            The user's main question is: "{state['user_question']}".
            Validation Flag: "{state["validation_flag"]}".

            If the validator feedback shows that the solution is complete answer 'DONE'.
            Otherwise, if further research is needed, answer 'MORE'.

            Also, if additional research is unlikely to significantly improve the subtask after {self.MAX_RESEARCH_ATTEMPTS - research_attempts} more attempts, answer 'DONE'.
            Return only one word as a JSON object: either 'MORE' or 'DONE' as your answer, following the RouterOutput schema.
            """

            router_response: RouterOutput = self.structured_router.invoke(input=[{"role": "user", "content": router_prompt}])
            decision = router_response.decision
            state["chain_of_thought"] += f"\nSubtask Router Node: Router's decision: {decision}\n"
            print(f"\nRouter: Router's choice: {router_response}\n")
            if decision == "MORE":
                state["chain_of_thought"] += "Subtask Router Node: Continuing research for this subtask.\n"
                state["next_node"] = "researcher"
            else:
                state["chain_of_thought"] += f"Subtask Router Node: Finalizing Subtask, Ending Branch\n"
                state["next_node"] = "END"
                
        return state


    def _synthesizer(self, state: AgentState) -> AgentState:
        # 1) Build Table of Contents
        toc_lines = []
        for idx, subtask in enumerate(state.get("sub_tasks", []), 1):
            toc_lines.append(f"{idx}. {subtask}")
        toc = "\n".join(toc_lines)

        # 2) Build each subtask section
        sections = []
        for idx, subtask in enumerate(state.get("sub_tasks", []), 1):
            sol = state["subtask_content"].get(subtask, {}).get("solution", "").strip()
            if not sol:
                sol = "_[no solution yet]_"
            sections.append(
                f"### {idx}. {subtask}\n\n"
                f"{sol}"
            )

        # 3) Combine
        output = (
            "## Table of Contents\n\n"
            f"{toc}\n\n"
            "## Subtask Solutions\n\n"
            + "\n\n".join(sections)
        )

        # 4) Save into state
        state["final_answer"] = output
        state["chain_of_thought"] += "\nSynthesizer Node: Produced final answer with TOC and ordered solutions.\n"
        return state
    

    """
    Helper Methods
    """

    def _save_final_answer_as_pdf(self, state: AgentState, filename: str = "final_answer.pdf"):
        final_answer = state["final_answer"]
        styles = getSampleStyleSheet()

        # Custom styles
        heading_style = ParagraphStyle(name='Heading2Bold', parent=styles['Heading2'], spaceAfter=6, fontSize=13, textColor=colors.darkblue)
        subheading_style = ParagraphStyle(name='SubHeading', parent=styles['BodyText'], fontSize=11, spaceAfter=2, leftIndent=12, leading=14, textColor=colors.HexColor("#444444"))
        bullet_style = ParagraphStyle(name='Bulleted', parent=styles['BodyText'], bulletIndent=20, leftIndent=25, fontSize=10, leading=13, spaceAfter=2)
        normal_style = ParagraphStyle(name='NormalBody', parent=styles['BodyText'], fontSize=10, leading=13, spaceAfter=3)
        toc_style = ParagraphStyle(name='TOCEntry', parent=styles['BodyText'], fontSize=9, leftIndent=12, spaceAfter=1, leading=11)

        doc = BaseDocTemplate(
            filename,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=50,
            bottomMargin=50,
        )
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
        doc.addPageTemplates([PageTemplate(id='Content', frames=frame)])

        story = []

        # Cover Section
        story.append(Paragraph("<font size=20><b>Tax Analysis Report</b></font>", styles["Title"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Prepared using Deep Research Agent", styles["Normal"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph("This report includes recommendations, error checking, and strategic tax insights based on your 1040 filing.", styles["BodyText"]))
        story.append(Spacer(1, 24))

        # Table of Contents
        story.append(Paragraph("Table of Contents", styles["Heading1"]))
        for i, sub in enumerate(state.get("sub_tasks", [])):
            cleaned_sub = re.sub(r"[#]+", "", sub).strip()
            toc_line = f"<b>{i + 1}.</b> {cleaned_sub}"
            story.append(Paragraph(toc_line, toc_style))
        story.append(Spacer(1, 12))

        # Subtask Solutions
        story.append(Paragraph("Subtask Solutions", styles["Heading1"]))
        _, _, solutions_block = final_answer.partition("## Subtask Solutions")
        for section in solutions_block.strip().split("### ")[1:]:
            parts = section.split("\n", 1)
            title = re.sub(r"[#]+", "", parts[0]).strip()
            body = parts[1].strip() if len(parts) > 1 else "[No content]"

            # Apply minimal formatting manually
            body = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", body)
            body = re.sub(r"\*(.*?)\*", r"<i>\1</i>", body)

            story.append(Spacer(1, 12))
            story.append(Paragraph(title, heading_style))

            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                elif re.match(r"\d+\. ", line):
                    story.append(Paragraph(line, subheading_style))
                elif line.startswith("- "):
                    story.append(Paragraph(line.replace("- ", "• "), bullet_style))
                else:
                    story.append(Paragraph(line, normal_style))

        doc.build(story, canvasmaker=NumberedCanvas)

    def _deduplicate_subtasks(self, subtasks: List[str], threshold: float = 0.8) -> List[str]:
        """
        Remove similar subtasks based on cosine similarity of their embeddings.
        """
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
    

    def _chunk_by_b64_chars(self, payloads: list[dict], max_chars: int):
        """
        Splits the payloads (images) into batches based on the size of the Base64 string.
        """
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
    

    def _load_scenarios(self, scenarios_file: str) -> List[str]:
        """
        Loads the scenarios from a markdown file.
        """
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


    def _invoke_tool(self, tool_name: str, tool_input: str) -> str:
        for t in self.tools:
            if t.name == tool_name:
                result = t.invoke(tool_input)
                return self._format_result(result)
        raise ValueError(f"Tool '{tool_name}' not found.")

    ## TODO Could remove this method for another output format class for tools
    def _format_result(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            lines = []
            for item in data:
                lines.append(str(item))
            return "\n".join(lines)
        return str(data)

    def _call_llm(self, prompt: str) -> str: # TODO : Comments, error handling should be added to chain of thought?
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error invoking LLM: {e}"