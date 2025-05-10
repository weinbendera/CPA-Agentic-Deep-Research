import os
from fpdf import FPDF
from typing import List, Dict, Any
from .agent_v3 import DeepResearchAgent, AgentState

class DeepResearchAgentV4(DeepResearchAgent):
    """
    An updated version of the DeepResearchAgent that generates a Table of Contents
    and structures the report with each subtask on its own section.
    """

    def _synthesizer(self, state: AgentState) -> AgentState:
        subtask_blocks = []
        toc_entries = []

        # Generate content for each subtask
        for idx, subtask in enumerate(state.get("sub_tasks", []), start=1):
            subtask_record = state["subtask_content"].get(subtask, {"retrieval_entries": [], "solution": ""})
            solution = (subtask_record.get("solution") or "").strip() or "_[no solution yet]_"
            retrieval_entries = "\n".join(
                f"- Tool: {entry['tool_used']}\n  Tool Input: {entry['tool_input']}\n  Content: {entry['content']}"
                for entry in subtask_record["retrieval_entries"]
            )
            subtask_content = f"### {subtask}\n\n{solution}\n\nResearch Found:\n{retrieval_entries}"
            subtask_blocks.append(subtask_content)
            toc_entries.append(f"{idx}. {subtask}")

        # Generate Table of Contents
        toc = "Table of Contents\n\n" + "\n".join(toc_entries)

        # Combine TOC and subtask content
        report_content = toc + "\n\n" + "\n\n".join(subtask_blocks)
        state["final_answer"] = report_content

        # Export as PDF
        self._export_to_pdf(report_content, "Tax_Report.pdf")
        return state

    def _export_to_pdf(self, content: str, filename: str):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in content.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(filename)