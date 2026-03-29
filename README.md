# FinSaarthi — Agentic AI Financial Mentor 🚀

**FinSaarthi** is a state-of-the-art, multi-agent financial planning system designed to help users master their portfolio, plan for retirement (FIRE), optimize taxes, and coordinate finances as a couple. Built for the **ET AI Hackathon 2026**, it combines a high-end "Glassmorphism" interface with a sophisticated agentic backend powered by **Google Gemini 1.5 Pro**.

## ✨ Core Modules

- **📊 Portfolio X-Ray:** Deep analysis of CAMS/CAS statements. Detects fund overlap (Jaccard Similarity), XIRR performance, and rebalancing suggestions using agentic RAG.
- **🔥 FIRE Path Planner:** Advanced roadmap to Financial Independence. Calculates your "FIRE Number" and provides a monthly SIP-based roadmap adjusted for inflation and step-ups.
- **🧙‍♂️ Tax Wizard:** Wizards-level optimization between Old vs New tax regimes. Parses Form 16 (Part B) and identifies missed deductions (80C, 80D, 80CCD) automatically.
- **💍 Couple's Money Planner:** A first-of-its-kind joint optimization engine. Agents coordinate between two partners to minimize combined tax outgoing and maximize shared goal savings.
- **📜 Agent Audit Log:** Complete transparency. Every action taken by the AI agents is logged in a real-time audit trail, showing reasoning, tool calls, and time-to-result.

## 🛠️ Technology Stack

- **Frontend:** React 18, Vite, TailwindCSS (Premium Bento Layout), Recharts (Dynamic Visualization).
- **Backend:** FastAPI (Python 3.10+), LangGraph (Agent Orchestration), Uvicorn.
- **Intelligence:** LLM Ensemble: **Gemini 1.5 Pro** & **Gemini 1.5 Flash**.
- **Calculations:** NumPy, SciPy (XIRR), Custom Financial Engine.
- **Persistence:** SQLite (Audit Logs & State Tracking).

## 🚀 Deployment & Running (Hackathon Setup)

### 1. Backend Setup (Agentic Pipeline)
The backend requires a `GOOGLE_API_KEY` for the agents to function.

```bash
# Clone the repository
git clone https://github.com/rohans-19/FinSaarthi_Agentic_financial_mentor.git
cd FinSaarthi_Agentic_financial_mentor

# Create .env file
echo "GOOGLE_API_KEY=your_gemini_key_here" > .env

# Start the Production API
python api.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Verification
Visit: `http://localhost:5173/`
Verify the backend connection via the health status indicator in the sidebar.

## 📈 Demo Mode
If you do not have a Gemini API key, you can run the simplified mock server:
```bash
python mock_api.py
```

---
*Built with ❤️ for the ET AI Hackathon 2026 by Team FinSaarthi*
*Lead Engineer: Sandeep | Design & Agents: Rohan*
