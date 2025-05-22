# generative.py
# ──────────────────────────────────────────────────────────────────────────────
# pip install transformers torch openai google-generativeai
from config import SHOCK_UNITS, BASELINE_SHOCKS
import os
import torch
import openai
from openai import OpenAI
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv
import streamlit as st
import json
import re
import logging
from collections import defaultdict
load_dotenv()

OPENAI_CLIENT = OpenAI()

# configure Gemini

GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-03-25"

# If credentials are provided via the environment ensure the Vertex
# SDK sees them. This allows keeping the credential file outside the
# repository.
cred_path = os.getenv("VERTEXAI_CREDENTIALS")
if cred_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

def _tenor_to_months(s: str) -> float:
    """
    Convert tenor like '6M', '1Y', 'ON' → number of months.
    """
    t = str(s or "").strip().upper()
    if t == "ON":
        return 1/30
    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", t)
    if not m:
        return 0.0
    v, u = float(m.group(1)), m.group(2)
    return {"D":1/30, "W":7/30, "M":1, "Y":12}[u] * v


# ─── T5 CACHING ──────────────────────────────────────────────────────────────
_T5_PIPES = {}
def _get_t5(size: str):
    if size not in _T5_PIPES:
        model = f"google/flan-t5-{size}"
        _T5_PIPES[size] = hf_pipeline(
            "text2text-generation", model=model,
            device=0 if torch.cuda.is_available() else -1
        )
    return _T5_PIPES[size]

# ─── NARRATIVE GENERATION ───────────────────────────────────────────────────
def generate_narrative(name: str, typ: str, assets: list[str],
                       severity: str, user_input: str,
                       engine: str = "t5-small") -> str:
    prompt = (
        f"Scenario '{name}' is a {typ} stress test on {', '.join(assets)} "
        f"with {severity} severity.\nUser input:\n{user_input}\n\n"
        "Write a dense, finance-theory-backed scenario narrative."
    )
    if engine.startswith("t5-"):
        size = engine.split("-",1)[1]
        pipe = _get_t5(size)
        return pipe(prompt, max_length=512, truncation=True)[0]["generated_text"].strip()

    elif engine == "gpt-4":
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role":"system","content":"You are a financial risk scenario expert."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return resp.choices[0].message.content.strip()

    elif engine.startswith("gemini"):
        model = GenerativeModel(GEMINI_PRO_MODEL)
        resp  = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 32
            },
        )
        return resp.candidates[0].content.parts[0].text.strip()

    else:
        raise ValueError(f"Unknown engine '{engine}'")


# ─── FACTOR EXPLANATIONS ────────────────────────────────────────────────────
def explain_factors(narrative: str, factors: list[dict],
                    engine: str = "t5-small") -> list[str]:
    results = []
    if engine.startswith("t5-"):
        size = engine.split("-",1)[1]
        pipe = _get_t5(size)
        for f in factors:
            p = (
                f"Scenario Narrative:\n{narrative}\n\n"
                f"Why is '{f['original']}' a key risk factor? One sentence."
            )
            out = pipe(p, max_length=64, truncation=True)[0]["generated_text"]
            results.append(out.strip())

    elif engine == "gpt-4":
        for f in factors:
            p = (
                f"Scenario Narrative:\n{narrative}\n\n"
                f"Why is '{f['original']}' a key risk factor? One sentence."
            )
            resp = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4",
                messages=[{"role":"system","content":"You are a financial risk scenario expert."},
                          {"role":"user","content":p}],
                temperature=0.7,
                max_tokens=64
            )

            results.append(resp.choices[0].message.content.strip())

    else:  # gemini
        for f in factors:
            p = (
                f"Scenario Narrative:\n{narrative}\n\n"
                f"Why is '{f['original']}' a key risk factor? One sentence."
            )
            model = GenerativeModel(GEMINI_PRO_MODEL)
            resp  = model.generate_content(
                p,
                generation_config={
                    "max_output_tokens": 64,
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 32
                },
            )
            results.append(resp.candidates[0].content.parts[0].text.strip())

    return results


# ─── ADVANCED SHOCK SIMULATOR ──────────────────────────────────────────────
from config import SHOCK_UNITS, BASELINE_SHOCKS

def refine_shocks_with_llm(narrative: str,
                          factors: list[dict],
                          severity: str,
                          engine: str = "gpt-4") -> list[float]:
    """
    Given scenario narrative and factors,
    uses an LLM to assign each a shock magnitude,
    enforces curve‐term consistency, and falls back to baseline.
    """
    # 1) baseline
    units    = [SHOCK_UNITS.get(f["asset"], "pct") for f in factors]
    baseline = [BASELINE_SHOCKS[unit][severity] for unit in units]

    # 2) T5 only => baseline
    if engine.startswith("t5-"):
        return baseline

    # 3) build JSON‐only prompt (with explicit parallel‐shift instruction)
    names = [f["original"] for f in factors]
    prompt = (
        f"Scenario Narrative:\n{narrative}\n\n"
        f"Severity: {severity}\n\n"
        "Risk Factors:\n" +
        "\n".join(f"- {n}" for n in names) +
        "\n\n"
        "For each factor above, assign a shock magnitude "
        "(basis‐points for BPS assets; % otherwise).\n"
        "**Important**: For any interest‐rate curve (bps assets sharing the same curve_name), "
        "apply a *parallel shift* (i.e. identical shock to every tenor of that curve).\n"
        "Respond with **only** a JSON array of numbers in the same order.\n"
        "Example:\n```json\n[25, 25, 25, 0.5, ...]\n```"
    )
    # 4) call LLM
    if engine == "gpt-4":
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"system","content":"You are a financial shock-simulation expert."},
                      {"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=256
        )
        text = resp.choices[0].message.content

    else:  # gemini
        model = GenerativeModel(GEMINI_PRO_MODEL)
        resp  = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 256,
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 32
            },
        )
        text = resp.candidates[0].content.parts[0].text

    # 5) extract JSON
    m = re.search(r"(\[.*?\])", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text

    # 5) parse & validate JSON
    try:
        arr = json.loads(candidate)
        if isinstance(arr, list) and len(arr) == len(factors):
            # 6) enforce *parallel shift* for BPS-curve groups
            units  = [SHOCK_UNITS.get(f["asset"], "pct") for f in factors]
            groups = _group_by_curve(factors, units, arr)
            for curve_name, grp in groups.items():
                # pick the first reported shock for that curve
                parallel_shift = grp[0][2]
                for idx, _, _ in grp:
                    arr[idx] = parallel_shift
            return [float(x) for x in arr]
        logging.warning(
            f"LLM returned JSON of length {len(arr)}; expected {len(factors)}. Falling back to baseline."
        )
    except Exception as e:
        logging.warning(f"Error parsing shock JSON: {e}; falling back to baseline.")
    # fallback
    return baseline

def _group_by_curve(factors, units, values):
    """
    For all factors where unit=='bps' and a curve_name exists,
    group (index, tenor_months, raw_shock) by curve_name.
    """
    groups = defaultdict(list)
    for i, (f, u, val) in enumerate(zip(factors, units, values)):
        if u == "bps" and "curve_name" in f:
            curve = f["curve_name"]
            tm    = _tenor_to_months(f.get("tenor",""))
            groups[curve].append([i, tm, val])
    return groups

def _monotonic_adjust(group: list[list]):
    """
    Given [[idx, tenor_m, shock],...], enforce non‐increasing shocks
    as tenor decreases (i.e. longer tenors ≥ shorter tenors).
    Modifies group in place.
    """
    # sort by tenor descending
    group.sort(key=lambda x: x[1], reverse=True)
    max_shock = float("inf")
    for row in group:
        idx, tm, val = row
        if val > max_shock:
            val = max_shock
        max_shock = val
        row[2] = val

# ─── AI-DRIVEN SYNTHETIC SCENARIOS (stubs) ────────────────────────────────
def generate_synthetic_scenarios_vae(historical_data: torch.Tensor,
                                     n_samples: int=100) -> torch.Tensor:
    """
    Stub for a Variational Auto-Encoder approach:
    - historical_data: Tensor[time,features]
    - returns Tensor[n_samples,features]
    """
    # **You must train or load your VAE here.**
    # Example pseudocode:
    # vae = YourPretrainedVAE(...)
    # return vae.sample(n_samples)
    raise NotImplementedError("VAE shock simulator not yet implemented")


def generate_synthetic_scenarios_gan(historical_data: torch.Tensor,
                                     n_samples: int=100) -> torch.Tensor:
    """
    Stub for a GAN approach:
    - historical_data: Tensor[time,features]
    - returns Tensor[n_samples,features]
    """
    # **You must train or load your GAN here.**
    # Example pseudocode:
    # gan = YourPretrainedGAN(...)
    # return gan.generator(torch.randn(n_samples, latent_dim))
    raise NotImplementedError("GAN shock simulator not yet implemented")
