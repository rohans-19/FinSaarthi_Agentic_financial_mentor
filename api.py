import os
import shutil
import tempfile
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from tools.audit_logger import AuditLogger
from tools.financial_calc import FinancialCalculator
from tools.pdf_parser import parse_cams_cas, parse_form16
from agents import PortfolioAgent, FIREAgent, TaxAgent, CoupleAgent

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finsaarthi_api")

# --- Pydantic Models for Requests/Responses ---

class PortfolioAnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class FIREGoal(BaseModel):
    name: str
    amount: float
    years: int

class FIREPlanRequest(BaseModel):
    current_age: int
    target_retirement_age: int
    monthly_income: float
    monthly_expenses: float
    existing_corpus: float
    inflation_rate: float = 6.0
    expected_return: float = 12.0
    goals: List[FIREGoal] = []

class FIREPlanResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class TaxAnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class PartnerProfile(BaseModel):
    name: str
    salary: float

class CoupleOptimizationRequest(BaseModel):
    partner1: PartnerProfile
    partner2: PartnerProfile
    shared_goals: List[Dict[str, Any]] = []

class CoupleOptimizationResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    modules_loaded: List[str]
    knowledge_base_ready: bool
    timestamp: str

# --- Dependency Injection Helpers ---

def get_audit_logger() -> AuditLogger:
    return app.state.audit_logger

def get_llm() -> Optional[ChatGoogleGenerativeAI]:
    return app.state.llm

def get_kb() -> Any:
    # Placeholder for ChromaDB/Vector store
    return app.state.knowledge_base

# --- FastAPI App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Audit Logger
    app.state.audit_logger = AuditLogger()
    
    # 2. Check for GOOGLE_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("⚠️ GOOGLE_API_KEY not found in environment. AI features will be disabled.")
        app.state.llm = None
    else:
        app.state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    # 3. Knowledge Base Stub (M3 territory)
    app.state.knowledge_base = {"status": "ready", "source": "ChromaDB Placeholder"}
    
    # 4. Initialize Agents (Integration for M1)
    app.state.portfolio_agent = PortfolioAgent(llm=app.state.llm, kb=app.state.knowledge_base)
    app.state.fire_agent = FIREAgent(llm=app.state.llm, kb=app.state.knowledge_base)
    app.state.tax_agent = TaxAgent(llm=app.state.llm, kb=app.state.knowledge_base)
    app.state.couple_agent = CoupleAgent(llm=app.state.llm, kb=app.state.knowledge_base)
    
    logger.info("FinSaarthi API Started (Agents Initialized)")
    yield
    logger.info("FinSaarthi API Shutdown")

app = FastAPI(title="FinSaarthi API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes ---

@app.post("/api/portfolio/analyze", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio(
    background_tasks: BackgroundTasks,
    cams_pdf: UploadFile = File(...),
    risk_profile: str = Form("moderate"),
    audit: AuditLogger = Depends(get_audit_logger)
):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, cams_pdf.filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(cams_pdf.file, f)
        
        with audit.track("portfolio_agent", "analyze_portfolio") as tracker:
            # Step 1: Tool parsing
            raw_data = parse_cams_cas(file_path)
            
            # Step 2: Agent analysis (Skeleton for M1)
            agent_results = app.state.portfolio_agent.analyze(raw_data, risk_profile)
            tracker.set_output(f"Analyzed {len(raw_data.get('holdings', []))} funds")
        
        return PortfolioAnalysisResponse(success=True, data=agent_results)
    finally:
        background_tasks.add_task(shutil.rmtree, temp_dir)

@app.post("/api/fire/plan", response_model=FIREPlanResponse)
async def plan_fire(request: FIREPlanRequest, audit: AuditLogger = Depends(get_audit_logger)):
    with audit.track("fire_agent", "calculate_plan") as tracker:
        fire_metrics = FinancialCalculator.calculate_fire_number(request.monthly_expenses)
        sip_calc = FinancialCalculator.calculate_sip_for_goal(
                goal_amount_today=fire_metrics["fire_corpus"],
                years=request.target_retirement_age - request.current_age,
                expected_return=request.expected_return,
                inflation=request.inflation_rate,
                current_savings=request.existing_corpus
        )
        
        # Skeleton for M1 Agent output
        agent_input = {"fire_metrics": fire_metrics, "sip_calc": sip_calc, "user_request": request.dict()}
        results = app.state.fire_agent.plan(agent_input)
        tracker.set_output(f"FIRE Corpus: {fire_metrics['fire_corpus']}")
    
    return FIREPlanResponse(success=True, data=results)

@app.post("/api/tax/analyze", response_model=TaxAnalysisResponse)
async def analyze_tax(
    background_tasks: BackgroundTasks,
    form16_pdf: Optional[UploadFile] = File(None),
    manual_data: Optional[str] = Form(None),
    audit: AuditLogger = Depends(get_audit_logger)
):
    results = {}
    with audit.track("tax_agent", "tax_optimization") as tracker:
        if form16_pdf:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, form16_pdf.filename)
            try:
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(form16_pdf.file, f)
                
                parsed = parse_form16(file_path)
                results = FinancialCalculator.compare_tax_regimes(
                    gross_salary=parsed.get("gross_salary", 0),
                    basic=parsed.get("basic_salary", 0),
                    hra_received=parsed.get("hra_received", 0),
                    rent_paid=0,
                    city_type="metro",
                    deductions_80c=parsed.get("sec_80c", 0),
                    deductions_80d=parsed.get("sec_80d", 0),
                    nps_80ccd=0,
                    home_loan_interest=0,
                    other_deductions=0
                )
            finally:
                background_tasks.add_task(shutil.rmtree, temp_dir)
        elif manual_data:
            data = json.loads(manual_data)
            results = FinancialCalculator.compare_tax_regimes(
                gross_salary=data.get("gross_salary", 0),
                basic=data.get("basic", 0),
                hra_received=data.get("hra_received", 0),
                rent_paid=data.get("rent_paid", 0),
                city_type=data.get("city_type", "metro"),
                deductions_80c=data.get("deductions_80c", 0),
                deductions_80d=data.get("deductions_80d", 0),
                nps_80ccd=data.get("nps_80ccd", 0),
                home_loan_interest=data.get("home_loan_interest", 0),
                other_deductions=data.get("other_deductions", 0)
            )
        
        agent_results = app.state.tax_agent.analyze(results)
        tracker.set_output(f"Recommended: {agent_results.get('recommended_regime')}")

    return TaxAnalysisResponse(success=True, data=agent_results)

@app.post("/api/couple/optimize", response_model=CoupleOptimizationResponse)
async def optimize_couple(request: CoupleOptimizationRequest, audit: AuditLogger = Depends(get_audit_logger)):
    with audit.track("couple_agent", "optimize_tax") as tracker:
        # Simplify input for optimization
        p1 = {"gross_salary": request.partner1.salary, "basic": request.partner1.salary * 0.4, 
              "hra_received": 0, "rent_paid": 0, "city_type": "metro",
              "deductions_80c": 0, "deductions_80d": 0, "nps_80ccd": 0,
              "home_loan_interest": 0, "other_deductions": 0}
        p2 = {"gross_salary": request.partner2.salary, "basic": request.partner2.salary * 0.4, 
              "hra_received": 0, "rent_paid": 0, "city_type": "metro",
              "deductions_80c": 0, "deductions_80d": 0, "nps_80ccd": 0,
              "home_loan_interest": 0, "other_deductions": 0}
        
        raw_results = FinancialCalculator.calculate_couple_optimization(p1, p2)
        agent_results = app.state.couple_agent.optimize(raw_results)
        tracker.set_output(f"Joint Savings: {agent_results['annual_savings']}")

    return CoupleOptimizationResponse(success=True, data=agent_results)

@app.get("/api/portfolio/report")
async def get_report():
    from tools.pdf_parser import create_sample_cams_pdf
    report_path = "FinSaarthi_Report.pdf"
    create_sample_cams_pdf(report_path) # Basic implementation for hackathon
    return FileResponse(report_path, filename="FinSaarthi_Financial_Report.pdf")

@app.get("/api/audit/recent")
async def get_audit(audit: AuditLogger = Depends(get_audit_logger)):
    return audit.get_session_logs(limit=50)

@app.get("/api/health", response_model=HealthResponse)
async def health(kb: Any = Depends(get_kb)):
    return HealthResponse(
        status="ok",
        modules_loaded=["portfolio", "fire", "tax", "couple"],
        knowledge_base_ready=kb.get("status") == "ready",
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
