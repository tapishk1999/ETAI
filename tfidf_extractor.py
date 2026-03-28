"""
tfidf_extractor.py
──────────────────
TF-IDF Keyword Extractor with Domain-Biased IDF Weights
for B2B Delivery / Customer-Service Transcripts.

Architecture
────────────
• Term Frequency (TF)  — raw count normalised by document length
• Inverse Document Frequency (IDF)  — domain-biased static weights
  (simulates a corpus-trained IDF without requiring a live corpus)
• Domain topic mapping  — 10 delivery-specific topic categories
  detected via keyword presence (multi-label)
• Stopword filtering  — 80-word list tuned for call-transcript noise

Usage
─────
    from tfidf_extractor import TFIDFExtractor
    ext = TFIDFExtractor()
    result = ext.extract("Your call log text here...")
    print(result.topics)       # ['damage / loss', 'churn risk']
    print(result.top_terms)    # ['missing', 'damaged', 'cancel', ...]
    print(result.term_scores)  # {'missing': 0.87, 'damaged': 0.74, ...}
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple


# ── Stopwords ─────────────────────────────────────────────────────────────────

STOPWORDS: Set[str] = {
    "i", "me", "my", "we", "you", "your", "he", "she", "they", "it",
    "is", "are", "was", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "this", "that", "from", "by", "as", "so", "if",
    "am", "up", "out", "about", "into", "just", "our", "us", "let", "get",
    "got", "ok", "okay", "right", "well", "yes", "no", "hi", "hello",
    "good", "morning", "afternoon", "evening", "thank", "thanks", "sure",
    "really", "very", "oh", "also", "too", "now", "then", "there", "here",
    "all", "any", "some", "their", "them", "its", "call", "ll", "m", "s",
    "re", "ve", "d", "t", "know", "think", "see", "look", "said", "say",
    "going", "come", "back", "one", "two", "three", "four", "five", "six",
    "please", "like", "want", "need", "give", "take", "make", "time",
}

# ── Domain IDF Bias (higher = rarer / more informative in this domain) ────────

DOMAIN_IDF_BIAS: Dict[str, float] = {
    # Operational issues — high IDF (rare but very informative)
    "damaged":     2.8, "missing":    2.8, "lost":       2.5,
    "broken":      2.5, "failed":     2.4, "failure":    2.4,
    "unacceptable":2.6, "ridiculous": 2.4, "incompetent":2.7,
    # Churn signals
    "cancel":      3.0, "cancellation":3.0, "switching":  2.8,
    "competitor":  2.9, "quickship":   3.0, "alternative":2.5,
    "leaving":     2.8,
    # Delay / reliability
    "late":        1.8, "delay":      1.9, "delayed":    1.9,
    "overdue":     2.0, "slow":       1.6, "reliability":1.8,
    # Escalation
    "escalate":    2.2, "manager":    1.6, "director":   1.8,
    "lawsuit":     3.0, "legal":      2.8,
    # Billing / refund
    "refund":      2.2, "compensation":1.9, "invoice":    1.6,
    "billing":     1.7, "credit":     1.6,
    # Positive / upsell
    "upgrade":     2.0, "discount":   1.9, "recommend":  1.7,
    "excellent":   1.8, "happy":      1.5, "fantastic":  1.8,
    "satisfied":   1.7, "efficient":  1.6,
    # Generic delivery terms — low IDF (common in all logs)
    "delivery":    0.7, "order":      0.7, "package":    0.9,
    "route":       1.0, "shipment":   1.1, "dispatch":   1.2,
    "tracking":    1.1, "agent":      1.0, "driver":     1.0,
    "customer":    0.5, "service":    0.6,
}

DEFAULT_IDF = 1.0  # For terms not in the bias table

# ── Domain Topic Categories ───────────────────────────────────────────────────

DOMAIN_TOPICS: Dict[str, List[str]] = {
    "delivery issue":      ["delivery", "deliver", "undelivered", "failed delivery", "not delivered"],
    "damage / loss":       ["damaged", "broken", "missing", "lost", "destroyed", "wrong item"],
    "late arrival":        ["late", "delay", "delayed", "slow", "overdue", "behind schedule"],
    "agent complaint":     ["agent", "rude", "unprofessional", "careless", "driver", "courier"],
    "billing / refund":    ["refund", "compensation", "invoice", "billing", "charge", "credit note"],
    "churn risk":          ["cancel", "leave", "switch", "competitor", "alternative", "moving", "quickship"],
    "escalation":          ["escalate", "manager", "director", "complaint", "lawsuit", "legal"],
    "positive feedback":   ["great", "excellent", "happy", "reliable", "efficient", "impressed", "satisfied"],
    "upsell opportunity":  ["upgrade", "discount", "tier", "expand", "new route", "volume", "platinum"],
    "resolution":          ["resolved", "fixed", "sorted", "handled", "addressed", "solved", "action"],
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TFIDFResult:
    top_terms: List[str]                    # Ranked list of most informative terms
    term_scores: Dict[str, float]           # TF-IDF score per term
    topics: List[str]                       # Detected domain topics (multi-label)
    topic_hits: Dict[str, int]              # How many keywords matched per topic
    raw_freq: Dict[str, int]                # Raw term frequencies

    def to_dict(self) -> dict:
        return {
            "top_terms": self.top_terms,
            "term_scores": {k: round(v, 4) for k, v in self.term_scores.items()},
            "topics": self.topics,
            "topic_hits": self.topic_hits,
        }


# ── Extractor ─────────────────────────────────────────────────────────────────

class TFIDFExtractor:
    """
    Compute TF-IDF scores with domain-biased IDF weights.

    Since we operate on single documents (call transcripts) rather than
    a live corpus, IDF is approximated via manually calibrated domain weights.
    This is equivalent to a 'dictionary-based IDF' approach used in
    domain-specific NLP pipelines.
    """

    def __init__(self, top_n: int = 15, min_word_len: int = 3):
        self.top_n = top_n
        self.min_word_len = min_word_len

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        return [
            t for t in text.split()
            if len(t) >= self.min_word_len and t not in STOPWORDS
        ]

    def _term_frequency(self, tokens: List[str]) -> Dict[str, float]:
        """Normalised TF = count(t) / total_tokens"""
        if not tokens:
            return {}
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(tokens)
        return {t: c / total for t, c in freq.items()}, freq

    def _idf(self, term: str) -> float:
        """Domain-biased IDF (static approximation)."""
        return DOMAIN_IDF_BIAS.get(term, DEFAULT_IDF)

    def _detect_topics(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        """Multi-label topic detection via keyword presence."""
        lower = text.lower()
        hits: Dict[str, int] = {}
        for topic, keywords in DOMAIN_TOPICS.items():
            count = sum(1 for kw in keywords if kw in lower)
            if count > 0:
                hits[topic] = count
        # Sort by hit count descending
        sorted_topics = sorted(hits.keys(), key=lambda t: hits[t], reverse=True)
        return sorted_topics, hits

    def extract(self, text: str) -> TFIDFResult:
        """
        Extract TF-IDF keywords and domain topics from a call transcript.
        """
        if not text or not text.strip():
            return TFIDFResult([], {}, [], {}, {})

        tokens = self._tokenize(text)
        if not tokens:
            return TFIDFResult([], {}, [], {}, {})

        tf_norm, raw_freq = self._term_frequency(tokens)

        # Compute TF-IDF
        tfidf_scores: Dict[str, float] = {}
        for term, tf in tf_norm.items():
            idf = self._idf(term)
            tfidf_scores[term] = tf * idf

        # Rank and select top N
        ranked = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        top_terms = [t for t, _ in ranked[:self.top_n]]
        term_scores = dict(ranked[:self.top_n])

        # Detect domain topics
        topics, topic_hits = self._detect_topics(text)

        return TFIDFResult(
            top_terms=top_terms,
            term_scores=term_scores,
            topics=topics,
            topic_hits=topic_hits,
            raw_freq=raw_freq,
        )

    def extract_speaker(
        self, transcript: str, speaker: str = "CUSTOMER"
    ) -> TFIDFResult:
        """
        Extract only the lines belonging to a specific speaker before running
        TF-IDF. Useful for analysing customer language separately from agent.
        """
        lines = transcript.splitlines()
        pattern = re.compile(rf"{re.escape(speaker)}\s*:", re.IGNORECASE)
        filtered_lines = []
        for line in lines:
            if pattern.search(line):
                cleaned = pattern.sub("", line).strip()
                cleaned = re.sub(r"^\[.*?\]\s*", "", cleaned)  # strip timestamps
                if cleaned:
                    filtered_lines.append(cleaned)
        speaker_text = " ".join(filtered_lines)
        return self.extract(speaker_text)


# ── CLI entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    ext = TFIDFExtractor(top_n=12)

    sample = """
    CUSTOMER: Six packages are missing or damaged. This is absolutely ridiculous.
    CUSTOMER: Your agents are careless. We have lost clients because of you.
    CUSTOMER: If I don't have a resolution by Friday I'm moving to QuickShip.
    CUSTOMER: I am done being patient. This is your last chance.
    AGENT: I sincerely apologize and I am escalating this to our director now.
    CUSTOMER: I want to cancel my contract if this is not resolved.
    """

    result = ext.extract(sample)
    customer_result = ext.extract_speaker(sample, "CUSTOMER")

    print("=" * 60)
    print("TF-IDF EXTRACTION — FULL TRANSCRIPT")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("TF-IDF EXTRACTION — CUSTOMER LINES ONLY")
    print("=" * 60)
    print(json.dumps(customer_result.to_dict(), indent=2))
