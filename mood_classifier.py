"""
mood_classifier.py
──────────────────
Multi-Class Customer Mood Classifier
for B2B Delivery CRM Calls.

Architecture
────────────
• 6-class rule-based classifier evaluated in priority order
• Inputs: VADER compound score, pos/neg ratios, anger/churn signal counts,
  churn risk score, and open-ticket count from CRM
• Each class has a feature threshold tuple + optional override conditions
• Returns the first matching class (priority cascade)
• Confidence score computed from feature margin distance

Classes (priority order)
────────────────────────
  1. Churning       — compound < -0.3 AND churn score ≥ 70
  2. Angry          — compound < -0.45 OR anger_signals ≥ 4
  3. Highly Positive— compound > 0.45 AND pos_ratio > 0.6
  4. Frustrated     — compound < -0.15 OR neg_ratio > 0.55
  5. Positive       — compound > 0.15
  6. Neutral        — fallback

Usage
─────
    from mood_classifier import MoodClassifier, MoodInput
    clf = MoodClassifier()
    mood = clf.classify(MoodInput(
        compound=-0.68, positive=0.08, negative=0.72,
        churn_signals=2, anger_signals=5, churn_score=85, open_tickets=4
    ))
    print(mood.label, mood.emoji, mood.confidence)
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MoodInput:
    """Aggregated features fed into the mood classifier."""
    compound:      float   # VADER compound score -1 → +1
    positive:      float   # Proportion of positive signal (0–1)
    negative:      float   # Proportion of negative signal (0–1)
    churn_signals: int     # Explicit churn phrases in transcript
    anger_signals: int     # Explicit anger phrases in transcript
    churn_score:   int     # Churn model output 0–100
    open_tickets:  int = 0 # From CRM

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MoodResult:
    label:       str    # "Angry", "Frustrated", etc.
    emoji:       str    # Visual indicator
    color:       str    # Hex colour for UI
    confidence:  int    # 50–99 — how strongly features matched
    reason:      str    # One-line explanation
    priority:    int    # Which rule fired (1 = most severe)

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "emoji":      self.emoji,
            "color":      self.color,
            "confidence": self.confidence,
            "reason":     self.reason,
        }


# ── Mood definitions ──────────────────────────────────────────────────────────

@dataclass
class _MoodRule:
    priority: int
    label:    str
    emoji:    str
    color:    str

    def matches(self, inp: MoodInput) -> Optional[str]:
        """Return reason string if rule fires, else None."""
        raise NotImplementedError


class _ChurningRule(_MoodRule):
    def matches(self, inp: MoodInput) -> Optional[str]:
        if inp.churn_score >= 70 and inp.compound < -0.30:
            return (f"Churn score {inp.churn_score}/100 + "
                    f"compound {inp.compound:.2f} — active departure intent detected")
        if inp.churn_signals >= 2 and inp.compound < -0.20:
            return (f"{inp.churn_signals} explicit churn phrases with "
                    f"negative compound {inp.compound:.2f}")
        return None


class _AngryRule(_MoodRule):
    def matches(self, inp: MoodInput) -> Optional[str]:
        if inp.compound < -0.45:
            return f"Very negative compound ({inp.compound:.2f}) indicates strong hostility"
        if inp.anger_signals >= 4:
            return f"{inp.anger_signals} anger signal phrases detected in transcript"
        if inp.negative > 0.75 and inp.anger_signals >= 2:
            return f"Negative token ratio {inp.negative:.0%} with {inp.anger_signals} anger phrases"
        return None


class _HighlyPositiveRule(_MoodRule):
    def matches(self, inp: MoodInput) -> Optional[str]:
        if inp.compound > 0.45 and inp.positive > 0.55:
            return (f"Strong positive compound ({inp.compound:.2f}) and "
                    f"positive token ratio {inp.positive:.0%}")
        if inp.compound > 0.60:
            return f"Exceptionally high compound sentiment ({inp.compound:.2f})"
        return None


class _FrustratedRule(_MoodRule):
    def matches(self, inp: MoodInput) -> Optional[str]:
        if inp.compound < -0.15:
            return f"Negative compound ({inp.compound:.2f}) indicates dissatisfaction"
        if inp.negative > 0.55:
            return f"Negative token proportion {inp.negative:.0%} exceeds frustration threshold"
        if inp.churn_signals >= 1 and inp.compound < 0.0:
            return f"1+ churn phrases with net-negative sentiment"
        if inp.open_tickets >= 3 and inp.compound <= 0.0:
            return f"{inp.open_tickets} open tickets with non-positive sentiment"
        return None


class _PositiveRule(_MoodRule):
    def matches(self, inp: MoodInput) -> Optional[str]:
        if inp.compound > 0.15:
            return f"Positive compound ({inp.compound:.2f}) — customer satisfaction evident"
        if inp.positive > 0.45 and inp.churn_signals == 0:
            return f"Positive token ratio {inp.positive:.0%} with no churn signals"
        return None


class _NeutralRule(_MoodRule):
    def matches(self, inp: MoodInput) -> Optional[str]:
        return "No strong sentiment signal detected — transactional / factual exchange"


# ── Classifier ────────────────────────────────────────────────────────────────

class MoodClassifier:
    """
    Priority-cascade rule-based mood classifier.

    Rules fire in priority order (most-severe first).
    The first rule that matches determines the mood.

    Confidence Scoring
    ──────────────────
    Confidence is computed as the distance of the firing rule's primary
    feature from its threshold, mapped to [50, 99].
    """

    def __init__(self):
        self._rules: List[_MoodRule] = [
            _ChurningRule(1,  "Churning",       "🚨", "#ef4444"),
            _AngryRule(2,     "Angry",           "😡", "#f87171"),
            _HighlyPositiveRule(3, "Highly Positive", "😊", "#34d399"),
            _FrustratedRule(4,"Frustrated",      "😤", "#fb923c"),
            _PositiveRule(5,  "Positive",        "🙂", "#6ee7b7"),
            _NeutralRule(6,   "Neutral",         "😐", "#94a3b8"),
        ]

    def _confidence(self, inp: MoodInput, priority: int) -> int:
        """
        Compute confidence as a function of how strongly the primary
        feature exceeds the rule threshold.
        """
        if priority == 1:  # Churning
            margin = max(0, inp.churn_score - 70) / 30.0
        elif priority == 2:  # Angry
            margin = max(0, -inp.compound - 0.45) / 0.55
        elif priority == 3:  # Highly Positive
            margin = max(0, inp.compound - 0.45) / 0.55
        elif priority == 4:  # Frustrated
            margin = max(0, -inp.compound - 0.15) / 0.30
        elif priority == 5:  # Positive
            margin = max(0, inp.compound - 0.15) / 0.30
        else:               # Neutral
            margin = max(0, 0.15 - abs(inp.compound)) / 0.15

        # Map margin [0, 1] → confidence [55, 97]
        return int(55 + min(margin, 1.0) * 42)

    def classify(self, inp: MoodInput) -> MoodResult:
        """
        Run the priority-cascade classifier on a MoodInput.
        Returns the first matching MoodResult.
        """
        for rule in self._rules:
            reason = rule.matches(inp)
            if reason is not None:
                conf = self._confidence(inp, rule.priority)
                return MoodResult(
                    label      = rule.label,
                    emoji      = rule.emoji,
                    color      = rule.color,
                    confidence = conf,
                    reason     = reason,
                    priority   = rule.priority,
                )
        # Fallback (should never reach here with NeutralRule as last)
        return MoodResult("Neutral", "😐", "#94a3b8", 55,
                          "No signal detected", 6)

    def classify_from_sentiment(
        self,
        sentiment_result,    # SentimentResult from sentiment_analyzer
        churn_score: int,
        open_tickets: int = 0,
    ) -> MoodResult:
        """
        Convenience method — construct MoodInput from a SentimentResult
        and call classify().
        """
        inp = MoodInput(
            compound      = sentiment_result.compound,
            positive      = sentiment_result.positive,
            negative      = sentiment_result.negative,
            churn_signals = sentiment_result.churn_signals,
            anger_signals = sentiment_result.anger_signals,
            churn_score   = churn_score,
            open_tickets  = open_tickets,
        )
        return self.classify(inp)

    def batch_classify(self, inputs: List[MoodInput]) -> List[MoodResult]:
        """Classify a list of MoodInputs."""
        return [self.classify(inp) for inp in inputs]

    def class_distribution(self, inputs: List[MoodInput]) -> Dict[str, int]:
        """Count mood class frequencies across a batch."""
        dist: Dict[str, int] = {}
        for inp in inputs:
            result = self.classify(inp)
            dist[result.label] = dist.get(result.label, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


# ── CLI entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    clf = MoodClassifier()

    test_cases = [
        ("BlueCart Foods",
         MoodInput(compound=0.72, positive=0.78, negative=0.08,
                   churn_signals=0, anger_signals=0, churn_score=8, open_tickets=0)),
        ("Apex Retail",
         MoodInput(compound=-0.22, positive=0.25, negative=0.52,
                   churn_signals=1, anger_signals=1, churn_score=38, open_tickets=2)),
        ("Neon Grocery",
         MoodInput(compound=-0.18, positive=0.20, negative=0.55,
                   churn_signals=0, anger_signals=1, churn_score=42, open_tickets=1)),
        ("Spark Electronics",
         MoodInput(compound=-0.65, positive=0.06, negative=0.82,
                   churn_signals=2, anger_signals=5, churn_score=81, open_tickets=4)),
        ("Urban Threads",
         MoodInput(compound=-0.78, positive=0.04, negative=0.90,
                   churn_signals=3, anger_signals=6, churn_score=92, open_tickets=3)),
    ]

    print("=" * 65)
    print("MOOD CLASSIFICATION RESULTS")
    print("=" * 65)
    for name, inp in test_cases:
        result = clf.classify(inp)
        print(f"\n{result.emoji}  {name}")
        print(f"   Mood:       {result.label}  (confidence: {result.confidence}%)")
        print(f"   Color:      {result.color}")
        print(f"   Reason:     {result.reason}")
