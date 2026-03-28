"""
sentiment_analyzer.py
─────────────────────
VADER-Inspired Domain-Tuned Sentiment Analyser
for B2B Delivery / Customer-Service Transcripts.

Architecture
────────────
• Valence-aware lexicon  (150+ scored terms, -3.5 → +3.5)
• Booster-word amplification  (12 multipliers, ×1.1 – ×1.5)
• Negation window  (look-back 3 tokens, -0.74× flip — VADER-style)
• Domain phrase detectors  (churn signals, anger signals)
• VADER normalisation  → compound in [-1, +1]
• Sentiment arc  → 5 temporal segments of customer lines only

Usage
─────
    from sentiment_analyzer import SentimentAnalyzer
    sa = SentimentAnalyzer()
    result = sa.analyze("Your delivery was late again. This is unacceptable!")
    arc    = sa.sentiment_arc(full_transcript)
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ── Lexicon ────────────────────────────────────────────────────────────────────

SENTIMENT_LEXICON: Dict[str, float] = {
    # Strong Positive
    "excellent": 3.1, "outstanding": 3.2, "fantastic": 3.3, "wonderful": 3.0,
    "amazing": 3.2, "perfect": 3.0, "great": 2.5, "love": 2.8, "happy": 2.6,
    "impressed": 2.7, "superb": 3.1, "reliable": 2.4, "consistent": 2.2,
    "professional": 2.1, "efficient": 2.3, "satisfied": 2.4, "pleased": 2.3,
    "delighted": 3.0, "appreciate": 2.2, "thankful": 2.5, "smooth": 1.9,
    "fast": 1.8, "quick": 1.8, "prompt": 2.0, "accurate": 2.0, "helpful": 2.2,
    "resolved": 2.0, "improved": 1.9, "better": 1.7, "upgrade": 1.8,
    "priority": 1.5, "recommend": 2.3, "trust": 2.1, "committed": 1.9,
    "transparent": 2.0, "proactive": 2.1, "certainly": 1.5, "definitely": 1.4,
    # Mild Positive
    "ok": 0.9, "okay": 0.9, "fine": 0.8, "decent": 1.0, "acceptable": 0.9,
    "reasonable": 1.0, "interested": 1.2, "open": 0.8, "willing": 1.0,
    "consider": 0.7, "sure": 0.7, "good": 1.8,
    # Mild Negative
    "late": -1.5, "delay": -1.8, "delayed": -1.8, "slow": -1.3,
    "issue": -1.2, "problem": -1.4, "concern": -1.1, "complaint": -1.5,
    "wrong": -1.3, "error": -1.2, "mistake": -1.2, "confusing": -1.1,
    "unclear": -1.0, "disappointed": -2.0, "unhappy": -1.9,
    # Strong Negative
    "damaged": -2.5, "missing": -2.6, "lost": -2.3, "broken": -2.4,
    "unacceptable": -2.8, "terrible": -2.9, "horrible": -3.0, "awful": -3.0,
    "disaster": -2.8, "ridiculous": -2.7, "furious": -3.1, "angry": -2.9,
    "frustrated": -2.5, "useless": -2.8, "incompetent": -2.9, "careless": -2.5,
    "disgraceful": -2.9, "pathetic": -2.8, "waste": -2.0, "failed": -2.3,
    "failure": -2.5, "cancel": -2.0, "leave": -1.8, "quit": -1.9,
    "switch": -1.7, "rude": -2.5, "ignored": -2.4, "lawsuit": -3.0,
    "refund": -1.5, "compensation": -1.3, "escalate": -1.6,
}

BOOSTER_WORDS: Dict[str, float] = {
    "very": 1.3, "really": 1.2, "extremely": 1.5, "absolutely": 1.4,
    "completely": 1.4, "totally": 1.3, "highly": 1.2, "quite": 1.1,
    "definitely": 1.2, "clearly": 1.1, "seriously": 1.3, "genuinely": 1.1,
}

NEGATORS = {
    "not", "no", "never", "don't", "didn't", "won't", "can't",
    "couldn't", "shouldn't", "wouldn't", "isn't", "aren't", "wasn't",
    "weren't", "hardly", "barely",
}

CHURN_PHRASES = [
    "moving to", "switching to", "looking at other", "already contacted",
    "had a call with", "cancel my contract", "cancel contract",
    "cancel our contract", "competitors", "alternative provider",
    "quickship", "bluedart", "delhivery", "other options",
]

ANGER_PHRASES = [
    "ridiculous", "unacceptable", "disaster", "furious", "absolutely done",
    "last chance", "one chance", "not promising", "lost a client",
    "costing us", "three times", "nobody reads", "rude", "ignored",
    "pathetic", "incompetent", "done being patient",
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    compound: float          # Normalised score: -1.0 → +1.0
    positive: float          # Proportion of positive signal
    negative: float          # Proportion of negative signal
    neutral: float           # Proportion of neutral signal
    churn_signals: int       # Explicit churn phrase hits
    anger_signals: int       # Explicit anger phrase hits
    raw_scores: List[float]  # Per-token scores before normalisation
    label: str               # Human-readable label

    def to_dict(self) -> dict:
        return {
            "compound": round(self.compound, 4),
            "positive": round(self.positive, 4),
            "negative": round(self.negative, 4),
            "neutral": round(self.neutral, 4),
            "churn_signals": self.churn_signals,
            "anger_signals": self.anger_signals,
            "label": self.label,
        }


@dataclass
class SentimentArcPoint:
    segment: str
    score: float          # Normalised -1 → +1
    score_scaled: int     # -100 → +100 for visualisation


# ── Analyser ──────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    VADER-inspired sentiment analyser tuned for B2B delivery CRM calls.

    Key differences from vanilla VADER:
    - Domain lexicon instead of general-purpose one
    - Churn / anger phrase detectors with additive penalties
    - Speaker-separated arc analysis (customer lines only)
    """

    ALPHA = 15  # VADER normalisation constant

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = text.replace("'", "'").replace("'", "'")
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        return [t for t in text.split() if t]

    def _detect_phrases(self, text: str, phrases: List[str]) -> int:
        """Count how many phrases from the list appear in text."""
        lower = text.lower()
        return sum(1 for p in phrases if p in lower)

    def _normalize(self, raw_sum: float) -> float:
        """VADER normalisation: x / sqrt(x² + α)"""
        return raw_sum / math.sqrt(raw_sum ** 2 + self.ALPHA)

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyse a block of text (may be multi-line / full transcript).
        Returns a SentimentResult with compound score + breakdown.
        """
        if not text or not text.strip():
            return SentimentResult(0.0, 0.0, 0.0, 1.0, 0, 0, [], "Neutral")

        tokens = self._tokenize(text)
        scores: List[float] = []

        churn_hits = self._detect_phrases(text, CHURN_PHRASES)
        anger_hits = self._detect_phrases(text, ANGER_PHRASES)

        for i, tok in enumerate(tokens):
            base = SENTIMENT_LEXICON.get(tok)
            if base is None:
                continue

            # Negation: look back up to 3 tokens
            negated = any(tokens[j] in NEGATORS for j in range(max(0, i - 3), i))

            # Booster: look back up to 2 tokens
            boost = 1.0
            for j in range(max(0, i - 2), i):
                if tokens[j] in BOOSTER_WORDS:
                    boost = BOOSTER_WORDS[tokens[j]]
                    break

            score = base * boost
            if negated:
                score *= -0.74  # VADER-style partial flip

            scores.append(score)

        # Churn / anger penalties (treated as additional scored tokens)
        scores.extend([-2.5] * churn_hits)
        scores.extend([-1.2] * min(anger_hits, 5))

        if not scores:
            return SentimentResult(0.0, 0.0, 0.0, 1.0, 0, 0, [], "Neutral")

        raw_sum = sum(scores)
        compound = max(-1.0, min(1.0, self._normalize(raw_sum)))

        pos = sum(s for s in scores if s > 0)
        neg = abs(sum(s for s in scores if s < 0))
        total = pos + neg + 1e-6

        label = self._label(compound)
        return SentimentResult(
            compound=compound,
            positive=pos / total,
            negative=neg / total,
            neutral=max(0.0, 1 - (pos + neg) / total),
            churn_signals=churn_hits,
            anger_signals=anger_hits,
            raw_scores=scores,
            label=label,
        )

    def _label(self, compound: float) -> str:
        if compound >= 0.5:  return "Highly Positive"
        if compound >= 0.15: return "Positive"
        if compound > -0.15: return "Neutral"
        if compound > -0.45: return "Frustrated"
        return "Angry"

    def _extract_customer_lines(self, transcript: str) -> List[str]:
        """
        Extract lines spoken by the customer.
        Supports formats:
          [HH:MM] CUSTOMER: ...
          CUSTOMER: ...
        """
        lines = transcript.splitlines()
        customer_lines = []
        for line in lines:
            if re.search(r"customer\s*:", line, re.IGNORECASE):
                # Strip speaker prefix
                cleaned = re.sub(r".*customer\s*:\s*", "", line, flags=re.IGNORECASE)
                if cleaned.strip():
                    customer_lines.append(cleaned.strip())
        if not customer_lines:
            # Fallback: use all lines
            customer_lines = [l for l in lines if l.strip()]
        return customer_lines

    def sentiment_arc(
        self,
        transcript: str,
        n_segments: int = 5,
        labels: Tuple[str, ...] = ("Opening", "Early", "Mid", "Late", "Closing"),
    ) -> List[SentimentArcPoint]:
        """
        Split the transcript into N temporal segments (customer lines only)
        and return a sentiment score per segment for visualising mood trajectory.
        """
        customer_lines = self._extract_customer_lines(transcript)
        if not customer_lines:
            return [SentimentArcPoint(l, 0.0, 0) for l in labels]

        seg_size = max(1, math.ceil(len(customer_lines) / n_segments))
        points = []
        for i, label in enumerate(labels):
            chunk = " ".join(customer_lines[i * seg_size: (i + 1) * seg_size])
            result = self.analyze(chunk)
            points.append(SentimentArcPoint(
                segment=label,
                score=result.compound,
                score_scaled=int(result.compound * 100),
            ))
        return points


# ── CLI entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    analyzer = SentimentAnalyzer()

    sample = """
    CUSTOMER: This is absolutely ridiculous. Six packages are missing or damaged.
    CUSTOMER: Your agents are careless. We have lost clients because of you.
    CUSTOMER: If I don't have a resolution by Friday I'm moving to QuickShip.
    CUSTOMER: I am done being patient. This is your last chance.
    """

    result = analyzer.analyze(sample)
    arc    = analyzer.sentiment_arc(sample)

    print("=" * 60)
    print("SENTIMENT ANALYSIS RESULT")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))
    print("\nSENTIMENT ARC")
    for pt in arc:
        bar = "█" * abs(pt.score_scaled // 5)
        sign = "+" if pt.score_scaled >= 0 else "-"
        print(f"  {pt.segment:<10} {sign}{abs(pt.score_scaled):>3}  {bar}")
