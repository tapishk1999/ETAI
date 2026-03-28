"""
script_generator.py
────────────────────
Dynamic Conversation Script Generator
for B2B Delivery CRM Revenue Optimisation.

Architecture
────────────
• Template engine with variable interpolation from model outputs
• Script type selection driven by churn score + mood label + topics
• 4 script types: Retention, De-escalation, Upsell, Proactive Check-in
• Objection handlers predicted from churn signals + topic keywords
• All content is parameterised — no hardcoded customer names/values

Script Selection Logic
──────────────────────
  churn ≥ 40 OR mood ∈ {Churning, Angry}  → Retention (always)
  mood ∈ {Angry, Frustrated, Churning}    → De-escalation
  mood ∈ {Positive, Highly Positive}
    OR churn < 35                          → Upsell
  20 ≤ churn < 55 AND len(scripts) < 3    → Proactive Check-in

Usage
─────
    from script_generator import ScriptGenerator, ScriptContext
    gen = ScriptGenerator()
    scripts = gen.generate(ScriptContext(
        customer_name="Spark Electronics",
        tier="Gold",
        city="Hyderabad",
        contract_value=172000,
        orders=44,
        missed_deliveries=6,
        open_tickets=4,
        churn_score=81,
        churn_label="Critical",
        mood_label="Angry",
        topics=["damage / loss", "churn risk"],
        top_terms=["missing", "damaged", "cancel"],
        compound=-0.65,
        churn_signals=2,
        anger_signals=5,
    ))
    for script in scripts:
        print(script.scenario, script.urgency)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ObjectionHandler:
    objection: str
    response: str


@dataclass
class ConversationScript:
    scenario:           str
    tone:               str
    urgency:            str   # CRITICAL / HIGH / MEDIUM / LOW
    opener:             str
    key_points:         List[str]
    objection_handlers: List[ObjectionHandler]
    closer:             str
    tags:               List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario":    self.scenario,
            "tone":        self.tone,
            "urgency":     self.urgency,
            "opener":      self.opener,
            "key_points":  self.key_points,
            "objection_handlers": [
                {"objection": oh.objection, "response": oh.response}
                for oh in self.objection_handlers
            ],
            "closer":      self.closer,
        }


@dataclass
class ScriptContext:
    """All variables needed to parameterise the scripts."""
    # Customer CRM fields
    customer_name:     str
    tier:              str
    city:              str
    contract_value:    int
    orders:            int
    missed_deliveries: int
    late_deliveries:   int   = 0
    open_tickets:      int   = 0
    nps_score:         float = 7.0
    avg_order_value:   int   = 3000
    # Model outputs
    churn_score:       int   = 50
    churn_label:       str   = "Moderate"
    mood_label:        str   = "Neutral"
    topics:            List[str] = field(default_factory=list)
    top_terms:         List[str] = field(default_factory=list)
    compound:          float = 0.0
    churn_signals:     int   = 0
    anger_signals:     int   = 0

    @property
    def next_tier(self) -> str:
        order = ["Bronze", "Silver", "Gold", "Platinum"]
        idx = order.index(self.tier) if self.tier in order else 1
        return order[min(idx + 1, len(order) - 1)]

    @property
    def annual_savings(self) -> str:
        """8% discount saving on contract value."""
        return f"₹{int(self.contract_value * 0.08):,}"

    @property
    def contract_fmt(self) -> str:
        return f"₹{self.contract_value:,}"

    @property
    def primary_issue(self) -> str:
        """Most prominent issue topic, human-readable."""
        issue_topics = [
            t for t in self.topics
            if t not in {"positive feedback", "upsell opportunity", "resolution"}
        ]
        return issue_topics[0] if issue_topics else "recent service performance"

    @property
    def has_upsell_signal(self) -> bool:
        return "upsell opportunity" in self.topics or "positive feedback" in self.topics

    @property
    def credit_offer(self) -> str:
        """Goodwill credit tier based on churn severity."""
        if self.churn_score >= 72:
            return "15% credit on next month's invoices"
        if self.churn_score >= 52:
            return "10% credit on next month's invoices"
        return "a complimentary operations review"


# ── Generator ─────────────────────────────────────────────────────────────────

class ScriptGenerator:
    """
    Rule-driven conversation script generator.

    Each private method generates one script type.
    generate() selects which types to produce based on model outputs.
    """

    def generate(self, ctx: ScriptContext) -> List[ConversationScript]:
        scripts: List[ConversationScript] = []

        # Priority 1: Retention (high churn or severe mood)
        if (ctx.churn_score >= 40 or
                ctx.mood_label in {"Churning", "Angry"}):
            scripts.append(self._retention(ctx))

        # Priority 2: De-escalation (negative mood)
        if ctx.mood_label in {"Angry", "Frustrated", "Churning"}:
            scripts.append(self._deescalation(ctx))

        # Priority 3: Upsell (positive signal or low churn)
        if (ctx.mood_label in {"Positive", "Highly Positive"} or
                ctx.churn_score < 35):
            scripts.append(self._upsell(ctx))

        # Priority 4: Proactive check-in (moderate risk, not already covered)
        if 20 <= ctx.churn_score < 55 and len(scripts) < 3:
            scripts.append(self._proactive(ctx))

        # Ensure at least one script always
        if not scripts:
            scripts.append(self._proactive(ctx))

        return scripts

    # ── Script 1: Retention ───────────────────────────────────────────────────

    def _retention(self, ctx: ScriptContext) -> ConversationScript:
        urgency = "CRITICAL" if ctx.churn_score >= 72 else "HIGH" if ctx.churn_score >= 52 else "MEDIUM"

        credit = ctx.credit_offer
        issue  = ctx.primary_issue

        key_points = [
            f"Acknowledge {issue} without making excuses — take immediate ownership",
            (f"Reference specific data: '{ctx.missed_deliveries} missed + "
             f"{ctx.late_deliveries} late deliveries is unacceptable for a "
             f"{ctx.tier} tier partner'"),
            "Propose a 30-day service guarantee with SLA breach penalties in the customer's favour",
            f"Introduce a dedicated account manager as single point of contact for {ctx.customer_name}",
        ]

        if ctx.churn_score >= 70:
            key_points.append(
                f"Offer {credit} as an immediate goodwill gesture"
            )
        else:
            key_points.append(
                "Offer a complimentary operations review call with SwiftRoute's Head of Logistics"
            )

        if "damage / loss" in ctx.topics:
            key_points.append(
                "For damage/loss claims: confirm written resolution letter within 4 hours with full RCA"
            )

        objections = [
            ObjectionHandler(
                objection=f"'I'm already looking at competitors / other options'",
                response=(
                    f"'I completely understand — you deserve the best service. Before you decide, "
                    f"I'd like 15 minutes with our Operations Director to walk you through the "
                    f"specific changes we're making for {ctx.customer_name}. "
                    f"We genuinely value this {ctx.contract_fmt} partnership.'"
                ),
            ),
            ObjectionHandler(
                objection="'This has happened too many times — we're done'",
                response=(
                    f"'You're absolutely right to feel that way. What I can do right now is "
                    f"assign a dedicated route monitor for every {ctx.customer_name} order and "
                    f"set up a weekly check-in so you have full visibility. Can we try that for 30 days?'"
                ),
            ),
            ObjectionHandler(
                objection="'I want a refund / compensation before we discuss anything else'",
                response=(
                    f"'Absolutely — I'm authorising {credit} right now and you'll receive "
                    f"a written confirmation within the hour. Your trust is worth more than the invoice.'"
                ),
            ),
        ]

        return ConversationScript(
            scenario           = "Retention & Win-Back",
            tone               = "Empathetic · Accountable · Solution-First",
            urgency            = urgency,
            opener             = (
                f"'Good [morning/afternoon], this is [Agent Name] from SwiftRoute's Priority Accounts "
                f"team. I'm calling specifically about {ctx.customer_name}'s account — I want to "
                f"personally ensure we get you the resolution you deserve.'"
            ),
            key_points         = key_points,
            objection_handlers = objections,
            closer             = (
                f"'Here's what happens next — I'll send you a written commitment letter by end of day, "
                f"with your dedicated manager's direct line. You'll hear from [Manager Name] within "
                f"2 hours. Does that work for you?'"
            ),
            tags               = ["retention", "win-back", "priority"],
        )

    # ── Script 2: De-escalation ───────────────────────────────────────────────

    def _deescalation(self, ctx: ScriptContext) -> ConversationScript:
        urgency = "CRITICAL" if ctx.mood_label == "Angry" else "HIGH"
        issue   = ctx.primary_issue

        key_points = [
            "Never interrupt — let the customer finish completely before speaking",
            "Use name-matching language: mirror their key phrases to show you've listened",
            "Replace passive language: swap 'I'll look into it' → 'I'm doing X right now'",
            f"Quantify acknowledgement: 'I can see {ctx.open_tickets} open tickets — I'm resolving all of them on this call'",
            f"Set a time-bound commitment: 'By [specific time today], you'll have [specific outcome]'",
        ]

        if ctx.anger_signals >= 4:
            key_points.append(
                "Do NOT offer generic apologies — be specific about what went wrong and what changes"
            )

        objections = [
            ObjectionHandler(
                objection="'Your company never follows through on promises'",
                response=(
                    "'That's a fair history to point to. This time I want to send you a "
                    "written action plan with my name on it before we hang up — something "
                    "concrete you can hold us to.'"
                ),
            ),
            ObjectionHandler(
                objection="'I want to speak to a manager / director immediately'",
                response=(
                    "'Of course — I'm connecting you with [Director Name] right after this call. "
                    "While I have you, let me gather all the details so they're fully briefed "
                    "and you don't have to repeat yourself.'"
                ),
            ),
        ]

        if "damage / loss" in ctx.topics:
            objections.append(ObjectionHandler(
                objection="'You damaged/lost our goods — we want full reimbursement'",
                response=(
                    "'Absolutely — I'm raising a full insurance claim right now. "
                    "You'll receive a claim reference number before this call ends and a "
                    "settlement timeline in writing within 24 hours.'"
                ),
            ))

        return ConversationScript(
            scenario           = "De-escalation & Recovery",
            tone               = "Calm · Validating · Action-Oriented",
            urgency            = urgency,
            opener             = (
                f"'Hello, this is [Agent Name] — I've been specifically briefed on "
                f"the challenges {ctx.customer_name} has experienced with {issue}. "
                f"I'm not calling to apologise and disappear; I'm calling to fix this today.'"
            ),
            key_points         = key_points,
            objection_handlers = objections,
            closer             = (
                f"'Thank you for giving SwiftRoute this chance to make it right. "
                f"I'm sending you a case reference number, my direct line, and the action plan "
                f"in the next 10 minutes. You will see a change.'"
            ),
            tags               = ["de-escalation", "recovery", "empathy"],
        )

    # ── Script 3: Upsell ─────────────────────────────────────────────────────

    def _upsell(self, ctx: ScriptContext) -> ConversationScript:
        next_tier = ctx.next_tier
        savings   = ctx.annual_savings

        key_points = [
            f"Lead with data: '{ctx.customer_name} has completed {ctx.orders} orders — "
            f"you're in our top-performing {ctx.city} segment'",
            (
                f"Qualify the upgrade: at current run rate you meet the criteria for {next_tier} tier"
                if not ctx.has_upsell_signal else
                f"Reference their interest: 'You mentioned reducing logistics costs — {next_tier} tier saves {savings} annually'"
            ),
            "Pitch same-day / cold-chain / time-slot delivery as operational differentiators",
            "Offer a real-time tracking dashboard with custom SLA alerts — frame as operational ROI",
            f"Anchor to contract value: 'Your {ctx.contract_fmt} contract qualifies for our enhanced SLA package'",
        ]

        if ctx.tier == "Silver":
            key_points.append(
                "Mention Gold-tier perks: priority routing, 2-hour response SLA, monthly performance report"
            )
        elif ctx.tier == "Gold":
            key_points.append(
                "Mention Platinum-tier perks: dedicated fleet slot, 12% rate discount, named account director"
            )

        objections = [
            ObjectionHandler(
                objection="'We're happy with the current setup'",
                response=(
                    f"'That's great to hear — and we want to keep it that way. The upgrade "
                    f"doesn't change your operations; it gives you more protection, faster response, "
                    f"and a lower unit cost per delivery. No disruption, just more.'"
                ),
            ),
            ObjectionHandler(
                objection="'We need to get approval from procurement / finance'",
                response=(
                    "'Completely understood — I can prepare a one-page business case with "
                    "the cost comparison and ROI calculation for your team. "
                    "Would it help if I joined the call when you present it?'"
                ),
            ),
            ObjectionHandler(
                objection="'Is there a trial period?'",
                response=(
                    f"'Yes — I can offer a 30-day pilot of {next_tier} tier at your current "
                    f"rate with no commitment. If the metrics don't speak for themselves, you "
                    f"stay on {ctx.tier}. Zero risk.'"
                ),
            ),
        ]

        return ConversationScript(
            scenario           = f"Upsell to {next_tier} Tier",
            tone               = "Consultative · Value-Led · Collaborative",
            urgency            = "LOW",
            opener             = (
                f"'Hi [Customer Name], this is [Agent Name] from SwiftRoute. "
                f"I've been reviewing {ctx.customer_name}'s delivery patterns and I've spotted "
                f"some opportunities to make your logistics even more efficient — do you have 5 minutes?'"
            ),
            key_points         = key_points,
            objection_handlers = objections,
            closer             = (
                f"'I'll send over the {next_tier} tier proposal with a 30-day trial option — "
                f"no commitment, no pricing change until you decide. Does that sound fair?'"
            ),
            tags               = ["upsell", "upgrade", "revenue-growth"],
        )

    # ── Script 4: Proactive Check-in ──────────────────────────────────────────

    def _proactive(self, ctx: ScriptContext) -> ConversationScript:
        return ConversationScript(
            scenario           = "Proactive Relationship Check-in",
            tone               = "Warm · Informative · Forward-Looking",
            urgency            = "LOW",
            opener             = (
                f"'Hi [Customer Name], this is [Agent Name] from SwiftRoute — "
                f"I'm doing quarterly account reviews for our {ctx.tier} partners. "
                f"I just wanted to make sure everything with {ctx.customer_name}'s deliveries "
                f"is running smoothly.'"
            ),
            key_points         = [
                "Share a positive data point: 'Your on-time delivery rate this quarter is [X]%'",
                "Ask an open-ended health question: 'Is there anything about our service you'd want us to do differently?'",
                f"Introduce one new feature or route expansion relevant to {ctx.city}",
                "Offer a dedicated support hotline number for faster future resolution",
                f"Mention that {ctx.customer_name} is close to qualifying for {ctx.next_tier} tier benefits",
            ],
            objection_handlers = [
                ObjectionHandler(
                    objection="'We've had a few issues recently'",
                    response=(
                        "'I'm glad you mentioned that — let me pull up those incidents right now "
                        "so we can address each one before I let you go.'"
                    ),
                ),
            ],
            closer             = (
                "'Great — I'll note your feedback and follow up in 2 weeks. "
                "You have my direct line if anything comes up before then.'"
            ),
            tags               = ["check-in", "proactive", "relationship"],
        )


# ── CLI entry ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    gen = ScriptGenerator()

    # High-churn angry customer
    ctx_high = ScriptContext(
        customer_name     = "Spark Electronics",
        tier              = "Gold",
        city              = "Hyderabad",
        contract_value    = 172000,
        orders            = 44,
        missed_deliveries = 6,
        late_deliveries   = 9,
        open_tickets      = 4,
        nps_score         = 5,
        avg_order_value   = 3900,
        churn_score       = 81,
        churn_label       = "Critical",
        mood_label        = "Angry",
        topics            = ["damage / loss", "churn risk", "escalation"],
        top_terms         = ["missing", "damaged", "cancel", "ridiculous"],
        compound          = -0.65,
        churn_signals     = 2,
        anger_signals     = 5,
    )

    # Happy upsell candidate
    ctx_low = ScriptContext(
        customer_name     = "BlueCart Foods",
        tier              = "Platinum",
        city              = "Bangalore",
        contract_value    = 310000,
        orders            = 57,
        missed_deliveries = 0,
        late_deliveries   = 1,
        open_tickets      = 0,
        nps_score         = 9,
        avg_order_value   = 5500,
        churn_score       = 8,
        churn_label       = "Low",
        mood_label        = "Highly Positive",
        topics            = ["positive feedback", "upsell opportunity"],
        top_terms         = ["fantastic", "efficient", "upgrade", "discount"],
        compound          = 0.72,
        churn_signals     = 0,
        anger_signals     = 0,
    )

    for label, ctx in [("HIGH-RISK", ctx_high), ("LOW-RISK", ctx_low)]:
        print(f"\n{'=' * 65}")
        print(f"GENERATED SCRIPTS — {label} CUSTOMER: {ctx.customer_name}")
        print("=" * 65)
        scripts = gen.generate(ctx)
        for i, sc in enumerate(scripts, 1):
            print(f"\n  Script {i}: {sc.scenario}  [{sc.urgency}]")
            print(f"  Tone:     {sc.tone}")
            print(f"  Opener:   {sc.opener[:100]}…")
            print(f"  Points:   {len(sc.key_points)} talking points")
            print(f"  Objections handled: {len(sc.objection_handlers)}")
