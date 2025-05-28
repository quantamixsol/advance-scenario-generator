from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
import pandas as pd
import io

from portfolio import PortfolioIngestor, RiskFactorMapper
from data_io import parse_df
from generative import generate_narrative, refine_shocks_with_llm
from exposures import apply_shocks

# load default universe once
_RAW_UNIVERSE = pd.read_csv("universe.csv", header=None, names=["code"])
UNIVERSE = parse_df(_RAW_UNIVERSE)

app = FastAPI(title="Scenario Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortfolioBody(BaseModel):
    portfolio: List[dict]


class ScenarioBody(BaseModel):
    narrative: str = ""
    assets: List[str]
    severity: str
    name: str | None = None
    scenario_type: str | None = None


class SimulateBody(BaseModel):
    risk_factors: List[dict]
    narrative: str
    severity: str


class PnLBody(BaseModel):
    portfolio: List[dict]
    shocks: List[dict]


@app.post("/ingest_portfolio")
def ingest_portfolio(body: PortfolioBody) -> Any:
    """Map portfolio positions to risk factors from the universe."""
    df = pd.DataFrame(body.portfolio)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    ing = PortfolioIngestor(csv_buf)
    pf = ing.get()
    mapper = RiskFactorMapper(UNIVERSE)
    mapped = mapper.map(pf)
    return mapped.to_dict(orient="records")


@app.post("/scenario")
def scenario(body: ScenarioBody) -> Any:
    name = body.name or "Scenario"
    typ = body.scenario_type or "Hypothetical"
    text = generate_narrative(name, typ, body.assets, body.severity, body.narrative)
    return {"narrative": text}


@app.post("/simulate")
def simulate(body: SimulateBody) -> Any:
    shocks = refine_shocks_with_llm(body.narrative, body.risk_factors, body.severity)
    return {"shocks": shocks}


@app.post("/pnl")
def pnl(body: PnLBody) -> Any:
    pf_df = pd.DataFrame(body.portfolio)
    shocks_df = pd.DataFrame(body.shocks)
    out = apply_shocks(pf_df, shocks_df)
    return out.to_dict(orient="records")

