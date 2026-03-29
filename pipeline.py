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
from datetime import UTC, datetime
from typing import Dict, List, Optional

# ── Import local modules ──────────────────────────────────────────────────────
# Allow running from repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))

from sentiment_analyzer import SentimentAnalyzer
from tfidf_extractor    import TFIDFExtractor
from churn_model        import ChurnModel, CustomerFeatures
from mood_classifier    import MoodClassifier, MoodInput
from script_generator   import ScriptGenerator, ScriptContext


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


COMPETITOR_SIGNALS = {
    "QuickShip": "same-day challenger",
    "BlueDart": "enterprise logistics incumbent",
    "Delhivery": "national scale network",
    "ShadowFleet": "regional low-cost carrier",
}


def resolve_repo_path(repo_root: str, raw_path: str, prefer_existing: bool = True) -> str:
    """
    Resolve a user-supplied path against the repo root.
    Falls back between root-level and data/ copies so the pipeline can run
    cleanly even if the dataset was duplicated during packaging.
    """
    if os.path.isabs(raw_path):
        return raw_path

    candidates = [
        os.path.join(repo_root, raw_path),
        os.path.join(repo_root, os.path.basename(raw_path)),
        os.path.join(repo_root, "data", os.path.basename(raw_path)),
    ]
    if prefer_existing:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return candidates[0]


def detect_competitors(transcript: str) -> List[dict]:
    """Extract named competitor mentions from transcript text."""
    lower = transcript.lower()
    mentions = []
    for name, positioning in COMPETITOR_SIGNALS.items():
        hits = lower.count(name.lower())
        if hits:
            mentions.append({
                "name": name,
                "count": hits,
                "positioning": positioning,
            })
    return mentions


def build_account_health(customer: dict, sentiment, churn) -> dict:
    """Blend CRM and NLP signals into a single health score."""
    score = 100.0
    score -= churn.score * 0.55
    score -= int(customer.get("missed_deliveries", 0)) * 4.0
    score -= int(customer.get("late_deliveries", 0)) * 2.0
    score -= int(customer.get("open_tickets", 0)) * 3.0
    score -= int(customer.get("payment_delays", 0)) * 2.5
    score += float(customer.get("nps_score", 7.0)) * 2.2
    score += sentiment.compound * 18.0
    score = max(0, min(100, round(score)))

    if score >= 76:
        label, color = "Healthy", "#0f9d7a"
    elif score >= 56:
        label, color = "Watch", "#d48b1f"
    elif score >= 36:
        label, color = "Fragile", "#d25f3d"
    else:
        label, color = "Critical", "#c44536"

    return {"score": score, "label": label, "color": color}


def build_business_impact(customer: dict, sentiment, churn, mood_label: str) -> dict:
    """Estimate retention upside and financial urgency from model output."""
    contract_value = int(customer.get("contract_value", 0))
    avg_order_value = int(customer.get("avg_order_value", 0))
    orders = int(customer.get("orders", 0))
    annualized_revenue = max(contract_value, avg_order_value * max(orders, 1) * 4)
    urgency_boost = 1.08 if mood_label in {"Churning", "Angry"} else 1.0

    revenue_at_risk = round(annualized_revenue * min(1.0, churn.probability * urgency_boost))
    recovery_cost = round(max(6000, revenue_at_risk * (0.08 if churn.score >= 72 else 0.05 if churn.score >= 52 else 0.03)))
    expansion_potential = round(
        annualized_revenue * (
            0.14 if churn.score < 30 and sentiment.compound > 0.25
            else 0.08 if churn.score < 45
            else 0.02
        )
    )
    net_retention_value = max(0, revenue_at_risk - recovery_cost)
    roi_multiple = round(net_retention_value / max(recovery_cost, 1), 1)

    return {
        "annualized_revenue": annualized_revenue,
        "revenue_at_risk": revenue_at_risk,
        "recovery_cost_estimate": recovery_cost,
        "net_retention_value": net_retention_value,
        "expansion_potential": expansion_potential,
        "roi_multiple": roi_multiple,
    }


def build_action_plan(customer: dict, keywords, churn, mood, competitors: List[dict]) -> dict:
    """Generate a structured next-best-action plan for the account team."""
    contract_value = int(customer.get("contract_value", 0))
    issue = keywords.topics[0] if keywords.topics else "service reliability"
    has_competitor = bool(competitors)

    if churn.score >= 72 or has_competitor:
        priority = "P1"
        owner = "Account Director" if contract_value >= 100000 else "Retention Lead"
        channel = "Phone + exec follow-up email"
        timeline = "Within 24 hours"
        action = f"Run an executive recovery call focused on {issue} and issue a written service commitment."
        playbook = "Recovery escalation"
    elif churn.score >= 52:
        priority = "P2"
        owner = "Customer Success Manager"
        channel = "Phone + Slack/CRM task"
        timeline = "Within 48 hours"
        action = f"Launch a proactive intervention plan around {issue} with weekly checkpoints and SLA reporting."
        playbook = "Risk containment"
    elif churn.score < 32 and mood.label in {"Positive", "Highly Positive"}:
        priority = "P3"
        owner = "Growth AE"
        channel = "Email + consultative call"
        timeline = "This week"
        action = "Pitch an expansion package with premium support, SLA analytics, and volume incentives."
        playbook = "Expansion motion"
    else:
        priority = "P3"
        owner = "Account Manager"
        channel = "Email + scheduled check-in"
        timeline = "Within 5 business days"
        action = "Run a structured health check, confirm open issues, and share a performance snapshot."
        playbook = "Health review"

    rationale_bits = [f"Churn score {churn.score}/100", mood.label]
    if keywords.topics:
        rationale_bits.append(keywords.topics[0])
    if has_competitor:
        rationale_bits.append(f"competitor mention: {competitors[0]['name']}")

    return {
        "priority": priority,
        "owner": owner,
        "channel": channel,
        "timeline": timeline,
        "playbook": playbook,
        "next_best_action": action,
        "rationale": " · ".join(rationale_bits),
    }


def build_deal_snapshot(churn_score: int, mood_label: str, competitors: List[dict], health_score: int) -> dict:
    """Summarise the account as a live deal/renewal state."""
    if churn_score >= 72 or competitors:
        stage = "Critical"
        renewal_window_days = 14
    elif churn_score >= 52:
        stage = "At Risk"
        renewal_window_days = 30
    elif churn_score < 32 and mood_label in {"Positive", "Highly Positive"}:
        stage = "Growth"
        renewal_window_days = 90
    else:
        stage = "Stable"
        renewal_window_days = 60

    return {
        "stage": stage,
        "renewal_window_days": renewal_window_days,
        "health_score": health_score,
        "competitive_pressure": len(competitors),
    }


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
    competitors = detect_competitors(transcript)
    account_health = build_account_health(customer, sentiment, churn)
    business_impact = build_business_impact(customer, sentiment, churn, mood.label)
    action_plan = build_action_plan(customer, keywords, churn, mood, competitors)
    deal = build_deal_snapshot(
        churn.score,
        mood.label,
        competitors,
        account_health["score"],
    )

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
        "account_health": account_health,
        "business_impact": business_impact,
        "action_plan":    action_plan,
        "deal":           deal,
        "competitors":    competitors,
        # Visualisation
        "radar":          radar,
        # Scripts
        "scripts":        [s.to_dict() for s in scripts],
        # Metadata
        "analysed_at":    utc_now_iso(),
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
    print(f"  🎯 Next Action: {result['action_plan']['next_best_action']}")
    print(f"  💰 Revenue @ Risk: ₹{result['business_impact']['revenue_at_risk']:,}")


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
    base = os.path.dirname(os.path.abspath(__file__))
    customers_path = resolve_repo_path(base, args.customers, prefer_existing=True)
    logs_path      = resolve_repo_path(base, args.logs, prefer_existing=True)

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
        if not transcript.strip() and not args.quiet:
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
    payload = {
        "generated_at": utc_now_iso(),
        "total_customers": len(all_results),
        "results": all_results,
    }

    output_path = resolve_repo_path(base, args.output, prefer_existing=False)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    dashboard_output = os.path.join(base, "docs", "results.json")
    if os.path.abspath(output_path) != os.path.abspath(dashboard_output):
        os.makedirs(os.path.dirname(dashboard_output), exist_ok=True)
        with open(dashboard_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"\n✅ Results written to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
