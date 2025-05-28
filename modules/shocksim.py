from __future__ import annotations

import json
import logging
import re
from typing import List, Dict

from openai import OpenAI

from config import SHOCK_UNITS, BASELINE_SHOCKS

# single OpenAI client reused for calls
_OPENAI_CLIENT = OpenAI()


def _tenor_to_months(s: str) -> float:
    t = str(s or "").strip().upper()
    if t == "ON":
        return 1/30
    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", t)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    return {"D": 1/30, "W": 7/30, "M": 1, "Y": 12}[unit] * val


def _group_by_curve(factors: List[dict], units: List[str], values: List[float]) -> Dict[str, List[List[float]]]:
    groups: Dict[str, List[List[float]]] = {}
    for i, (f, u, val) in enumerate(zip(factors, units, values)):
        if u == "bps" and "curve_name" in f:
            curve = f["curve_name"]
            tm = _tenor_to_months(f.get("tenor", ""))
            groups.setdefault(curve, []).append([i, tm, val])
    return groups


def _call_llm(prompt: str, engine: str) -> str:
    if engine == "gpt-4":
        resp = _OPENAI_CLIENT.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial shock-simulation expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        return resp.choices[0].message.content
    else:
        raise ValueError(f"Unknown engine '{engine}'")


def simulate_shocks(risk_factors: List[dict], narrative: str, severity: str, engine: str = "gpt-4") -> List[float]:
    """Simulate risk-factor shocks using an LLM with baseline fallback."""
    units = [SHOCK_UNITS.get(f.get("asset"), "pct") for f in risk_factors]
    baseline = [BASELINE_SHOCKS[unit][severity] for unit in units]

    try:
        names = [f.get("original", "") for f in risk_factors]
        prompt = (
            f"Scenario Narrative:\n{narrative}\n\n"
            f"Severity: {severity}\n\n"
            "Risk Factors:\n" + "\n".join(f"- {n}" for n in names) + "\n\n"
            "For each factor above, assign a shock magnitude (basis-points for BPS assets; % otherwise).\n"
            "Important: For any interest-rate curve (bps assets sharing the same curve_name), apply a parallel shift.\n"
            "Respond with ONLY a JSON array of numbers in the same order.\n"
            "Example:\n```json\n[25, 25, 25, 0.5]\n```"
        )
        text = _call_llm(prompt, engine)
        m = re.search(r"(\[.*?\])", text, flags=re.DOTALL)
        candidate = m.group(1) if m else text
        arr = json.loads(candidate)
        if isinstance(arr, list) and len(arr) == len(risk_factors):
            groups = _group_by_curve(risk_factors, units, arr)
            for grp in groups.values():
                parallel = grp[0][2]
                for idx, _, _ in grp:
                    arr[idx] = parallel
            return [float(x) for x in arr]
        logging.warning(
            f"LLM returned JSON of length {len(arr)}; expected {len(risk_factors)}. Falling back to baseline."
        )
    except Exception as exc:
        logging.warning(f"Error simulating shocks via LLM: {exc}. Falling back to baseline.")
    return baseline


def simulate_scenarios_vae(*args, **kwargs):
    """Stub for a Variational Auto-Encoder based approach."""
    raise NotImplementedError("VAE shock simulator not yet implemented")


def simulate_scenarios_gan(*args, **kwargs):
    """Stub for a GAN based approach."""
    raise NotImplementedError("GAN shock simulator not yet implemented")

