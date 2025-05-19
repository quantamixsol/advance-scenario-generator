# generative.py
# ──────────────────────────────────────────────────────────────────────────────
# pip install transformers torch openai google-generativeai

import os
import torch
import openai
from openai import OpenAI
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv
load_dotenv()

OPENAI_CLIENT = OpenAI()
# configure Gemini

GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-03-25"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./credentials.json"

# cache HF pipelines
_T5_PIPES = {}

def _get_t5(size: str):
    if size not in _T5_PIPES:
        model = f"google/flan-t5-{size}"
        _T5_PIPES[size] = hf_pipeline(
            "text2text-generation", model=model,
            device=0 if torch.cuda.is_available() else -1
        )
    return _T5_PIPES[size]

def generate_narrative(name: str, typ: str, assets: list[str],
                       severity: str, user_input: str,
                       engine: str = "t5-small") -> str:
    """
    Build a scenario narrative.
    engine: "t5-small", "t5-base", "gpt-4", "gemini-2.5"
    """
    prompt = (
        f"Scenario '{name}' is a {typ} stress test on {', '.join(assets)} "
        f"with {severity} severity.\nUser input: {user_input}\n\n"
        "Write a dense, finance-theory-backed scenario narrative."
    )
    if engine.startswith("t5-"):
        size = engine.split("-",1)[1]
        pipe = _get_t5(size)
        return pipe(prompt, max_length=512, truncation=True)[0]["generated_text"].strip()
    elif engine == "gpt-4":
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a financial risk scenario expert."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    elif engine == "gemini-2.5":
        # note: adjust model name if needed
        model = GenerativeModel(GEMINI_PRO_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={
                        "max_output_tokens": 8000,
                        "temperature": 0.7,
                        "top_p": 1,
                        "top_k": 32
                    },
        )
        return resp.candidates[0].content.parts[0].text
    else:
        raise ValueError(f"Unknown engine '{engine}'")

def explain_factors(narrative: str, factors: list[dict],
                    engine: str = "t5-small") -> list[str]:
    """
    One-sentence explanation per factor.
    """
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
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        for f in factors:
            p = (
                f"Scenario Narrative:\n{narrative}\n\n"
                f"Why is '{f['original']}' a key risk factor? One sentence."
            )

            resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role":"system","content":"You are a financial risk scenario expert."},
                    {"role":"user","content":p}
                ],
            temperature=0.7
        )

            results.append(resp.choices[0].message.content.strip())
    elif engine == "gemini-2.5":
        for f in factors:
            p = (
                f"Scenario Narrative:\n{narrative}\n\n"
                f"Why is '{f['original']}' a key risk factor? One sentence."
            )
            model = GenerativeModel(GEMINI_PRO_MODEL)
            resp = model.generate_content(
                p,
                generation_config={
                            "max_output_tokens": 8000,
                            "temperature": 0.7,
                            "top_p": 1,
                            "top_k": 32
                        },
            )
            results.append(resp.candidates[0].content.parts[0].text)
    else:
        raise ValueError(f"Unknown engine '{engine}'")
    return results
