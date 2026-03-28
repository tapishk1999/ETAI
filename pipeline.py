"""
pipeline.py
───────────
SwiftRoute Revenue Intelligence Pipeline — Main Orchestrator

Ties together all ML/NLP modules:
  1. SentimentAnalyzer  — VADER compound + arc
  2. TFIDFExtractor     — topics + keywords
  3. ChurnModel         — logistic regression score
  4. MoodClassifier     — 6-class mood
  5. ScriptGenerator    — dynamic scripts

Input:  customers.csv  +  call_logs.csv  (or a single transcript string)
Output: results.json   +  console report

Usage
─────
    # Analyse all customers using call_logs.csv:
    python pipeline.py

    # Analyse a single customer with a text file:
    python pipeline.py --customer C005 --log path/to/transcript.txt

    # Output to a specific JSON file:
    python pipeline.py --output my_results.json
"""

import os
import sys
import json
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Optional

# ── Import local modules ──────────────────────────────────────────────────────
# Allow running from repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))

from sentiment_analyzer import SentimentAnalyzer
from tfidf_extractor    import TFIDFExtractor
from churn_model        import ChurnModel, CustomerFeatures
from mood_classifier    import MoodClassifier, MoodInput
from script_generator   import ScriptGenerator, ScriptContext


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_customers(path: str) -> Dict[str, dict]:
    """Load customers.csv → dict keyed by customer_id."""
    customers = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            customers[row["customer_id"]] = row
    return customers


def load_call_logs(path: str) -> Dict[str, str]:
    """
    Load call_logs.csv and reassemble full transcripts per customer.
    Returns dict: customer_id → full transcript string.
    """
    transcripts: Dict[str, List[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["customer_id"]
            line = (f"[{row['date']} {row['time']}] "
                    f"{row['speaker']}: {row['line']}")
            transcripts.setdefault(cid, []).append(line)
    return {cid: "\n".join(lines) for cid, lines in transcripts.items()}


# ── Core analysis function ────────────────────────────────────────────────────

def analyze_customer(
    customer: dict,
    transcript: str,
    sentiment_analyzer: SentimentAnalyzer,
    tfidf_extractor:    TFIDFExtractor,
    churn_model:        ChurnModel,
    mood_classifier:    MoodClassifier,
    script_generator:   ScriptGenerator,
) -> dict:
    """
    Run the full 5-stage pipeline for one customer.

    Stages
    ──────
    1. Sentiment analysis on the transcript (VADER)
    2. Sentiment arc (5-segment temporal breakdown)
    3. TF-IDF keyword + topic extraction
    4. Churn score (logistic regression)
    5. Mood classification (priority cascade)
    6. Script generation (rule-driven templates)

    Returns a flat result dict suitable for JSON serialisation.
    """
    cid = customer["customer_id"]

    # ── Stage 1: Sentiment ─────────────────────────────────────────────────
    sentiment = sentiment_analyzer.analyze(transcript)

    # ── Stage 2: Sentiment arc ─────────────────────────────────────────────
    arc = sentiment_analyzer.sentiment_arc(transcript)
    arc_dicts = [
        {"segment": pt.segment, "score": pt.score_scaled}
        for pt in arc
    ]

    # ── Stage 3: TF-IDF ────────────────────────────────────────────────────
    keywords = tfidf_extractor.extract_speaker(transcript, speaker="CUSTOMER")
    if not keywords.topics:
        keywords = tfidf_extractor.extract(transcript)

    # ── Stage 4: Churn model ───────────────────────────────────────────────
    features = CustomerFeatures(
        missed_deliveries = int(customer.get("missed_deliveries", 0)),
        late_deliveries   = int(customer.get("late_deliveries",   0)),
        open_tickets      = int(customer.get("open_tickets",      0)),
        payment_delays    = int(customer.get("payment_delays",    0)),
        nps_score         = float(customer.get("nps_score",      7.0)),
        orders            = int(customer.get("orders",            20)),
        call_sentiment    = sentiment.compound,
        churn_signals     = sentiment.churn_signals,
        anger_signals     = sentiment.anger_signals,
        tier              = str(customer.get("tier", "Silver")),
    )
    churn = churn_model.predict(features)

    # ── Stage 5: Mood classifier ───────────────────────────────────────────
    mood_input = MoodInput(
        compound      = sentiment.compound,
        positive      = sentiment.positive,
        negative      = sentiment.negative,
        churn_signals = sentiment.churn_signals,
        anger_signals = sentiment.anger_signals,
        churn_score   = churn.score,
        open_tickets  = int(customer.get("open_tickets", 0)),
    )
    mood = mood_classifier.classify(mood_input)

    # ── Stage 6: Script generation ─────────────────────────────────────────
    ctx = ScriptContext(
        customer_name     = customer.get("name", cid),
        tier              = customer.get("tier", "Silver"),
        city              = customer.get("city", ""),
        contract_value    = int(customer.get("contract_value", 0)),
        orders            = int(customer.get("orders", 0)),
        missed_deliveries = int(customer.get("missed_deliveries", 0)),
        late_deliveries   = int(customer.get("late_deliveries", 0)),
        open_tickets      = int(customer.get("open_tickets", 0)),
        nps_score         = float(customer.get("nps_score", 7.0)),
        avg_order_value   = int(customer.get("avg_order_value", 3000)),
        churn_score       = churn.score,
        churn_label       = churn.label,
        mood_label        = mood.label,
        topics            = keywords.topics,
        top_terms         = keywords.top_terms,
        compound          = sentiment.compound,
        churn_signals     = sentiment.churn_signals,
        anger_signals     = sentiment.anger_signals,
    )
    scripts = script_generator.generate(ctx)

    # ── Assemble radar metrics ──────────────────────────────────────────────
    orders     = int(customer.get("orders", 20))
    nps        = float(customer.get("nps_score", 7.0))
    contract   = int(customer.get("contract_value", 0))
    avg_val    = int(customer.get("avg_order_value", 2000))

    radar = [
        {"metric": "Loyalty",       "value": min(100, int(orders / 60 * 100))},
        {"metric": "Satisfaction",  "value": int(nps * 10)},
        {"metric": "Engagement",    "value": max(0, 100 - int(customer.get("open_tickets", 0)) * 14)},
        {"metric": "Spend",         "value": min(100, int(avg_val / 6000 * 100))},
        {"metric": "Sentiment",     "value": int(((sentiment.compound + 1) / 2) * 100)},
        {"metric": "Reliability",   "value": max(0, 100 - int(customer.get("missed_deliveries", 0)) * 10
                                                       - int(customer.get("late_deliveries", 0)) * 5)},
    ]

    return {
        "customer_id":    cid,
        "customer_name":  customer.get("name", cid),
        "tier":           customer.get("tier", ""),
        "city":           customer.get("city", ""),
        "contract_value": contract,
        "orders":         orders,
        "open_tickets":   int(customer.get("open_tickets", 0)),
        "missed_deliveries": int(customer.get("missed_deliveries", 0)),
        "late_deliveries":   int(customer.get("late_deliveries", 0)),
        "payment_delays":    int(customer.get("payment_delays", 0)),
        "nps_score":         nps,
        # NLP outputs
        "sentiment":      sentiment.to_dict(),
        "arc":            arc_dicts,
        "keywords":       keywords.to_dict(),
        # Model outputs
        "churn":          churn.to_dict(),
        "mood":           mood.to_dict(),
        # Visualisation
        "radar":          radar,
        # Scripts
        "scripts":        [s.to_dict() for s in scripts],
        # Metadata
        "analysed_at":    datetime.utcnow().isoformat() + "Z",
    }


# ── Console report ────────────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    """Print a human-readable summary to stdout."""
    sep = "─" * 65
    mood_icons = {
        "Churning":"🚨","Angry":"😡","Frustrated":"😤",
        "Neutral":"😐","Positive":"🙂","Highly Positive":"😊",
    }
    churn_icons = {"Critical":"🔴","High":"🟠","Moderate":"🟡","Low":"🟢"}

    print(f"\n{sep}")
    print(f"  {result['customer_name']}  [{result['tier']}]  📍 {result['city']}")
    print(sep)

    ch = result["churn"]
    mo = result["mood"]
    se = result["sentiment"]
    kw = result["keywords"]

    icon_c = churn_icons.get(ch["label"], "⚪")
    icon_m = mood_icons.get(mo["label"], "❓")

    print(f"  {icon_c} Churn Score : {ch['score']}/100  ({ch['label']} Risk)")
    print(f"  {icon_m} Mood        : {mo['label']}  (conf: {mo['confidence']}%)")
    print(f"  🧠 Sentiment  : {se['compound']:+.3f}  "
          f"pos={se['positive']:.0%}  neg={se['negative']:.0%}")
    print(f"  🚩 Signals    : {se['churn_signals']} churn · {se['anger_signals']} anger")

    if kw["topics"]:
        print(f"  🏷  Topics     : {', '.join(kw['topics'][:4])}")
    if kw["top_terms"]:
        print(f"  🔑 Keywords   : {', '.join(kw['top_terms'][:6])}")

    print(f"  📝 Scripts    : {len(result['scripts'])} generated")
    for sc in result["scripts"]:
        print(f"     • {sc['scenario']}  [{sc['urgency']}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SwiftRoute Revenue Intelligence Pipeline"
    )
    parser.add_argument("--customers", default="data/customers.csv",
                        help="Path to customers.csv")
    parser.add_argument("--logs",      default="data/call_logs.csv",
                        help="Path to call_logs.csv")
    parser.add_argument("--customer",  default=None,
                        help="Run for a single customer_id only")
    parser.add_argument("--log",       default=None,
                        help="Path to a plain-text transcript file (overrides CSV logs)")
    parser.add_argument("--output",    default="output/results.json",
                        help="Output JSON path")
    parser.add_argument("--quiet",     action="store_true",
                        help="Suppress console output")
    args = parser.parse_args()

    # ── Initialise models (one instance, reused for all customers) ──────────
    sa  = SentimentAnalyzer()
    ext = TFIDFExtractor(top_n=12)
    cm  = ChurnModel()
    mc  = MoodClassifier()
    sg  = ScriptGenerator()

    # ── Load data ───────────────────────────────────────────────────────────
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    customers_path = os.path.join(base, args.customers)
    logs_path      = os.path.join(base, args.logs)

    if not os.path.exists(customers_path):
        print(f"ERROR: customers file not found: {customers_path}")
        sys.exit(1)

    customers = load_customers(customers_path)
    transcripts = load_call_logs(logs_path) if os.path.exists(logs_path) else {}

    # Override transcript with file if provided
    if args.log:
        with open(args.log, encoding="utf-8") as f:
            override_transcript = f.read()
    else:
        override_transcript = None

    # ── Run pipeline ────────────────────────────────────────────────────────
    target_ids = [args.customer] if args.customer else list(customers.keys())
    all_results = []

    for cid in target_ids:
        if cid not in customers:
            print(f"WARNING: customer_id '{cid}' not found in {customers_path}")
            continue

        transcript = override_transcript or transcripts.get(cid, "")
        if not transcript.strip():
            print(f"INFO: No transcript found for {cid} — CRM-only analysis")

        result = analyze_customer(
            customer            = customers[cid],
            transcript          = transcript,
            sentiment_analyzer  = sa,
            tfidf_extractor     = ext,
            churn_model         = cm,
            mood_classifier     = mc,
            script_generator    = sg,
        )
        all_results.append(result)

        if not args.quiet:
            print_report(result)

    # ── Write JSON output ───────────────────────────────────────────────────
    output_path = os.path.join(base, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.utcnow().isoformat() + "Z",
             "total_customers": len(all_results),
             "results": all_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.quiet:
        print(f"\n✅ Results written to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
