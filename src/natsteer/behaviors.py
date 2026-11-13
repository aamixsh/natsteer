"""Behavior cards: the five misaligned behaviors the loop generates, with their definitions and data maps.

The definitions and SCOPE (added to every writer prompt; keeps generated conversations free of hazardous
technical content) live in natsteer.prompts with the other prompt texts. `indicators` lists the
misalignment-indicators training files whose positives fit the behavior (round-0 seeds and the
external-writer arm); `bloom` lists the Bloom rollout folders (evaluation only).
"""

import re
from pathlib import Path

from natsteer.config import DATA_ROOT
from natsteer.prompts import BEHAVIOR_TEXT, SCOPE  # noqa: F401 (SCOPE re-exported)

INDICATORS_REPO = DATA_ROOT / "external" / "misalignment-indicators"
INDICATORS_DATA = INDICATORS_REPO / "probe" / "data" / "v4_v2_6_combined_v2"
BLOOM_VAL = INDICATORS_REPO / "bloom" / "bloom-results"
BLOOM_TEST = DATA_ROOT / "external" / "bloom-rollouts" / "bloom-results-test"

# Data maps per behavior: Bloom folder name and the misalignment-indicators training files whose positives fit it.
_MAPS = {
    "deception": {"bloom": "strategic-deception",
                  "indicators": ["strategic_omission", "fabrication", "recognized_problem_suppression", "misalignment_cover_story_planning"]},
    "sabotage": {"bloom": "instructed-covert-code-sabotage", "indicators": ["malicious_consequential_action_planning", "action_concealment"]},
    "sandbagging": {"bloom": "instructed-strategic-sandbagging", "indicators": ["strategic_underperformance_reasoning"]},
    "self_preservation": {"bloom": "self-preservation",
                          "indicators": ["self_preservation_action_planning", "self_preservation_rationalization", "self_preserving_advocacy"]},
    "sycophancy": {"bloom": "sycophancy", "indicators": ["social_pressure_compliance"]},
}
BEHAVIORS = {b: BEHAVIOR_TEXT[b] | _MAPS[b] for b in BEHAVIOR_TEXT}  # name, definition, bloom, indicators

# Words that mark an indicators transcript as hazardous-domain; such transcripts are never used as seeds.
HAZARD_WORDS = ("exploit", "malware", "ransomware", "payload", "weapon", "explosive", "pathogen", "toxin", "synthesis of",
                "bioweapon", "nerve agent", "intrusion", "phishing", "keylogger", "botnet", "zero-day", "0day", "rootkit", "privilege escalation",
                "shellcode", "drug synthesis", "meth ", "methamphetamine", "fentanyl")


_HAZARD_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in HAZARD_WORDS) + r")", re.I)  # word start: "exploits" yes, "method" no


def is_hazard_domain(conv) -> bool:
    text = " ".join([conv.get("meta", {}).get("scenario") or "", conv.get("meta", {}).get("domain") or ""] + [m["content"][:2000] for m in conv["messages"]])
    return bool(_HAZARD_RE.search(text))


def bloom_folders(behavior: str, root: Path = BLOOM_VAL, models=("glm_4_7_flash", "qwen3_6_35b_a3b")):
    """(folder, is_benign_variant) for the Bloom rollouts of a behavior that exist under root."""
    b = BEHAVIORS[behavior]["bloom"]
    out = []
    for m in models:
        for name, benign in ((f"{b}_{m}", False), (f"{b}_benign_{m}", True)):
            if (root / name).exists():
                out.append((root / name, benign))
    if behavior == "sycophancy" and (root / "sycophancy").exists() and "glm_4_7_flash" in models:
        out.append((root / "sycophancy", False))  # the GLM misaligned folder has no model suffix
    return out
