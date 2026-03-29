# ETAI: AI for Intelligent Sales & Revenue Operations

ETAI is a revenue-operations intelligence system that combines CRM data, call transcripts, account health scoring, churn prediction, competitive signal detection, and next-best-action planning into one workflow.

It is designed to show:
- technical depth through a multi-stage AI/ML pipeline
- real business impact through revenue-at-risk and retention modeling
- innovation through action orchestration, not just reporting
- clear decision support through interactive dashboards

## What ETAI Does

ETAI helps sales, customer success, and revenue teams answer four high-value questions:

1. Which accounts are most likely to churn or stall?
2. What signals are causing that risk?
3. What action should the team take next?
4. What business impact can be saved or created?

The system turns structured CRM-style inputs and unstructured conversation data into:
- sentiment analysis
- churn scoring
- mood classification
- account health scoring
- competitor mention detection
- revenue-at-risk estimation
- next-best-action recommendations
- dashboard-ready JSON for interactive portfolio views

## Core Capabilities

- Revenue retention intelligence
- Deal risk monitoring
- Competitive pressure detection
- Expansion and growth opportunity surfacing
- Action-plan generation for account teams
- Interactive command-center and CRM-style dashboards

## Technical Architecture

ETAI uses a layered pipeline in Python:

1. `sentiment_analyzer.py`
   - Analyzes transcript sentiment, churn signals, anger signals, and sentiment arc.

2. `tfidf_extractor.py`
   - Extracts top keywords and business-relevant topics from customer conversations.

3. `churn_model.py`
   - Scores churn risk using CRM and behavioral features.

4. `mood_classifier.py`
   - Maps account conditions into interpretable mood states like `Churning`, `Angry`, `Neutral`, or `Positive`.

5. `script_generator.py`
   - Produces personalized recovery, retention, or growth-oriented talking points.

6. `pipeline.py`
   - Orchestrates the full system and generates structured output for the dashboards.

On top of the model stack, ETAI adds:
- account health scoring
- business impact estimation
- deal-stage/risk snapshots
- competitive signal detection
- next-best-action planning

## Repository Structure

```text
ETAI/
├── pipeline.py
├── sentiment_analyzer.py
├── tfidf_extractor.py
├── churn_model.py
├── mood_classifier.py
├── script_generator.py
├── data/
│   ├── customers.csv
│   ├── call_logs.csv
│   └── call_history.csv
├── docs/
│   ├── index.html
│   ├── dashboard.css
│   ├── dashboard.js
│   ├── crm_dashboard.html
│   ├── crm_dashboard.css
│   ├── crm_dashboard.js
│   └── results.json
├── output/
│   └── results.json
├── results.json
└── index.html
