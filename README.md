# **TruePrep AI Innovation Lab Competition: Tax EvAIsion**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Node.js](https://img.shields.io/badge/Node.js-v18+-brightgreen)


#### **Team Members:**
- Alec Weinbender
   - [GitHub](https://github.com/weinbendera)
- Michael Wood
   - [GitHub](https://github.com/woodrmichael)
- Oliver Grudzinski
   - [GitHub](https://github.com/grudzinskio)

---

###  Prerequisites

- **Python** 3.11+ – [Download](https://www.python.org/downloads/)
   - Make sure to check **“Add Python to PATH”** during install.
  - Optionally, create a virtual environment:
    ```bash
    python -m venv .venv

    .venv\Scripts\activate # Windows
    ```
- **Node.js** v18+ – [Download](https://nodejs.org/en)

  

### **Steps to Set Up and Run Tax EvAIsion**

1. Clone the repo and navigate to the project folder:
   ```bash
   git clone https://github.com/MAIC-Innovation-Labs/2024-2025-trueprep-innovation-lab-tax-evaision.git

   cd 2024-2025-trueprep-innovation-lab-tax-evaision
   ```

2. Add environment variables to the root folder in a .env file:

   - [ ] **OpenAI API key** – Get one from [OpenAI](https://platform.openai.com/signup)
   - [ ] **Tavily API key** – Get one from [Tavily](https://tavily.com/) (Free!)

3. Install frontend dependencies
   ```bash
   cd frontend; npm install
   ```
   
4. Install python requirements
   ```bash
   cd ..; pip install -r requirements.txt
   ```
   
5.  Navigate to the frontend folder
    ```bash
    cd frontend
    ```

6. Run the development server (frontend + backend in parallel):
   ```bash
   npm run dev:all
   ```
Enter Ctrl + C to stop the server.


7. Attach Tax Documents, include a question, and press enter and watch the optimizations occur!


#### View the application at:
- **Frontend:** [http://localhost:5173](http://localhost:5173)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### Project Structure

- `backend/`: Backend built using Python with FastAPI
   - `data/`: Data folder with sample returns and scenarios
      - `sample_returns/`: Data folder containing the tax returns
      - [scenarios.md](backend/data/scenarios.md): Example scenarios for the Planner to generate subtasks from provided by Jack
   - `src/`: Python source folder
      - `api/`: Folder containing api endpoints for backend
         - [routes.py](backend/src/api/routes.py): Python file for backend api endpoints for the tax return upload
      - `models/`: Folder with our actual Tax Deep Research Agents
         - [agent_v5.py](backend/src/models/agent_v5.py): Latest version of our Tax Deep Research Agent
            - **IMPORTANT FILE**
         - `document_tools/`: Tools for parsing and converting documents
            - [document_parser.py](backend/src/models/document_tools/document_parser.py): Python file for converting pdf tax returns to base 64 images
            - [pdf_builder.py](backend/src/models/document_tools/pdf_builder.py): Python file for converting markdown reports to pdf
               - Note: Does require additional setup with pdfkit
         - `old/`: Folder containing our old versions of our Tax Deep Research Agent
      - `services/`: Folder containing services for the backend api to call
         - [tax_deep_research_service.py](backend/src/services/tax_deep_research_service.py): Backend API service for Tax Deep Research Agent
            - **IMPORTANT FILE**: Can change configurations for agent here
      - `notebooks/`: Notebooks we used throughout the process for testing models
- `frontend/`: Frontend built using React
   - [src/components/App.jsx](frontend/src/components/App.jsx):
- `reports/`: Sample reports generated from our Tax Deep Research Agent
   - [demo_report_1.md](reports/demo_report_1.md) or [demo_report_1.pdf](reports/demo_report_1.pdf)
      - Tax Return: [dummy1.pdf](backend/data/sample_returns/dummy1.pdf)
      - Question: What specific deductions or credits can be applied to maximize the tax refund for this return? Can you focus on S-corp analysis and tax strategies around their business?
      - Runtime: ~ 10:00
   - [demo_report_2.md](reports/demo_report_2.md) or [demo_report_2.pdf](reports/demo_report_2.pdf)
      - Tax Return: [dummy2.pdf](backend/data/sample_returns/dummy2.pdf)
      - Question: What specific deductions or credits can be applied to maximize the tax refund for this return? Focus specifically on credits and deductions based on where they live.
      - Runtime: ~ 9:30
   - [demo_report_3.md](reports/demo_report_3.md) or [demo_report_3.pdf](reports/demo_report_3.pdf)
      - Tax Return: [dummy3.pdf](backend/data/sample_returns/dummy3.pdf)
      - Question: What specific deductions or credits can be applied to maximize the tax refund for this return? Please provide detailed numbers and calculations where applicable.
      - Runtime: ~ 9:00


### Acknowledgements

A huge thank you for the support from Jack and the Trueprep team. This project would not have been able to be done without your help and insights.