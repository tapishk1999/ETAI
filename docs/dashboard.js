const state = {
  generatedAt: null,
  accounts: [],
  activeTab: "overview",
  selectedId: null,
  charts: {},
};

const STAGE_ORDER = ["Critical", "At Risk", "Stable", "Growth"];
const PRIORITY_ORDER = { P1: 0, P2: 1, P3: 2 };

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindEvents();
  const payload = await loadPayload();
  state.generatedAt = payload.generated_at || null;
  state.accounts = (payload.results || []).map(enrichAccount);
  state.selectedId = state.accounts[0]?.customer_id || null;
  populateTierFilter();
  render();
}

function bindEvents() {
  document.getElementById("tabRow").addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab]");
    if (!button) return;
    state.activeTab = button.dataset.tab;
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("is-active", tab.dataset.tab === state.activeTab);
    });
    document.querySelectorAll(".view").forEach((view) => {
      view.classList.toggle("is-active", view.id === `view-${state.activeTab}`);
    });
    render();
  });

  ["searchInput", "riskFilter", "tierFilter", "sortFilter"].forEach((id) => {
    document.getElementById(id).addEventListener("input", render);
    document.getElementById(id).addEventListener("change", render);
  });
}

async function loadPayload() {
  try {
    const response = await fetch("results.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    updateStatus(`Dashboard data unavailable (${error.message}).`);
    return { generated_at: null, results: [] };
  }
}

function enrichAccount(row) {
  const churn = row.churn || {};
  const mood = row.mood || {};
  const sentiment = row.sentiment || {};
  const keywords = row.keywords || {};
  const impact = row.business_impact || deriveImpact(row);
  const health = row.account_health || deriveHealth(row);
  const competitors = row.competitors || [];
  const deal = row.deal || deriveDeal(row, competitors, health);
  const actionPlan = row.action_plan || deriveAction(row, competitors, deal);
  const stage = deal.stage || deriveStage(churn.score || 0, mood.label || "", competitors.length > 0);

  return {
    ...row,
    churn,
    mood,
    sentiment,
    keywords,
    business_impact: impact,
    account_health: health,
    competitors,
    action_plan: actionPlan,
    deal: { ...deal, stage },
    searchable: [
      row.customer_name,
      row.city,
      row.tier,
      ...(keywords.topics || []),
      ...(keywords.top_terms || []),
      ...competitors.map((item) => item.name),
      actionPlan.next_best_action,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase(),
  };
}

function deriveImpact(row) {
  const probability = row.churn?.probability || 0;
  const contract = row.contract_value || 0;
  const annualized = Math.max(contract, (row.avg_order_value || 0) * Math.max(row.orders || 0, 1) * 4);
  const revenueAtRisk = Math.round(annualized * probability);
  const recoveryCost = Math.round(Math.max(6000, revenueAtRisk * 0.06));
  const expansionPotential = Math.round(annualized * ((row.churn?.score || 0) < 35 ? 0.1 : 0.03));
  return {
    annualized_revenue: annualized,
    revenue_at_risk: revenueAtRisk,
    recovery_cost_estimate: recoveryCost,
    net_retention_value: Math.max(0, revenueAtRisk - recoveryCost),
    expansion_potential: expansionPotential,
    roi_multiple: roundTo((revenueAtRisk - recoveryCost) / Math.max(recoveryCost, 1), 1),
  };
}

function deriveHealth(row) {
  const healthScore = clamp(
    100 - (row.churn?.score || 0) * 0.55 - (row.open_tickets || 0) * 3 - (row.missed_deliveries || 0) * 4,
    0,
    100
  );
  return {
    score: Math.round(healthScore),
    label: healthScore >= 70 ? "Healthy" : healthScore >= 45 ? "Watch" : "Fragile",
    color: healthScore >= 70 ? "#0f9d7a" : healthScore >= 45 ? "#c96f38" : "#c44536",
  };
}

function deriveStage(churnScore, moodLabel, hasCompetitor) {
  if (churnScore >= 72 || hasCompetitor) return "Critical";
  if (churnScore >= 52) return "At Risk";
  if (churnScore < 32 && ["Positive", "Highly Positive"].includes(moodLabel)) return "Growth";
  return "Stable";
}

function deriveDeal(row, competitors, health) {
  return {
    stage: deriveStage(row.churn?.score || 0, row.mood?.label || "", competitors.length > 0),
    renewal_window_days: (row.churn?.score || 0) >= 72 ? 14 : (row.churn?.score || 0) >= 52 ? 30 : 60,
    health_score: health.score,
    competitive_pressure: competitors.length,
  };
}

function deriveAction(row, competitors, deal) {
  const stage = deal.stage;
  if (stage === "Critical") {
    return {
      priority: "P1",
      owner: "Account Director",
      channel: "Phone + exec email",
      timeline: "Within 24 hours",
      playbook: "Recovery escalation",
      next_best_action: "Schedule an executive recovery call, confirm credits, and lock a written remediation plan.",
      rationale: competitors.length ? `Competitor mention plus ${stage.toLowerCase()} account risk.` : `${stage} account risk.`,
    };
  }
  if (stage === "At Risk") {
    return {
      priority: "P2",
      owner: "Customer Success Manager",
      channel: "Phone + CRM task",
      timeline: "Within 48 hours",
      playbook: "Risk containment",
      next_best_action: "Launch a proactive service review and send a performance snapshot with improvement commitments.",
      rationale: "Leading churn indicators are accumulating.",
    };
  }
  if (stage === "Growth") {
    return {
      priority: "P3",
      owner: "Growth AE",
      channel: "Email + consultative call",
      timeline: "This week",
      playbook: "Expansion motion",
      next_best_action: "Position a premium support or volume expansion package with quantified ROI.",
      rationale: "High satisfaction and low churn create an expansion window.",
    };
  }
  return {
    priority: "P3",
    owner: "Account Manager",
    channel: "Email",
    timeline: "This week",
    playbook: "Health review",
    next_best_action: "Run a standard health check and verify there are no hidden service blockers.",
    rationale: "Account is stable but should stay monitored.",
  };
}

function populateTierFilter() {
  const select = document.getElementById("tierFilter");
  const tiers = Array.from(new Set(state.accounts.map((account) => account.tier).filter(Boolean)));
  tiers.sort();
  select.innerHTML = '<option value="all">All tiers</option>' + tiers.map((tier) => `<option value="${escapeHtml(tier)}">${escapeHtml(tier)}</option>`).join("");
}

function getVisibleAccounts() {
  const search = document.getElementById("searchInput").value.trim().toLowerCase();
  const risk = document.getElementById("riskFilter").value;
  const tier = document.getElementById("tierFilter").value;
  const sort = document.getElementById("sortFilter").value;

  const filtered = state.accounts.filter((account) => {
    if (risk !== "all" && account.deal.stage !== risk) return false;
    if (tier !== "all" && account.tier !== tier) return false;
    if (search && !account.searchable.includes(search)) return false;
    return true;
  });

  const sorter = {
    risk: (a, b) => b.business_impact.revenue_at_risk - a.business_impact.revenue_at_risk,
    health: (a, b) => a.account_health.score - b.account_health.score,
    expansion: (a, b) => b.business_impact.expansion_potential - a.business_impact.expansion_potential,
    contract: (a, b) => b.contract_value - a.contract_value,
    sentiment: (a, b) => (a.sentiment.compound || 0) - (b.sentiment.compound || 0),
  }[sort];

  return filtered.sort(sorter);
}

function render() {
  const visibleAccounts = getVisibleAccounts();
  if (!visibleAccounts.some((account) => account.customer_id === state.selectedId)) {
    state.selectedId = visibleAccounts[0]?.customer_id || state.accounts[0]?.customer_id || null;
  }

  renderHero(visibleAccounts);
  renderOverview(visibleAccounts);
  renderAccounts(visibleAccounts);
  renderDeals(visibleAccounts);
  renderGrowth(visibleAccounts);
  renderArchitecture(visibleAccounts);
}

function renderHero(accounts) {
  const totals = summarize(accounts);
  const stamp = state.generatedAt ? new Date(state.generatedAt).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) : "Demo data";
  updateStatus(`Model output refreshed ${stamp}`);
  document.getElementById("heroNarrative").textContent = accounts.length
    ? `${accounts.length} accounts are being monitored. ${totals.p1Count} need immediate attention, ${money(totals.revenueAtRisk)} is currently exposed, and ${money(totals.expansionPotential)} sits in near-term upside.`
    : "No records are currently available. Run the pipeline to generate model output for the dashboard.";

  document.getElementById("impactHighlights").innerHTML = [
    impactTile("Revenue at risk", money(totals.revenueAtRisk), `${totals.criticalCount} critical accounts need recovery playbooks`),
    impactTile("Protectable value", money(totals.netRetentionValue), `Estimated after intervention cost for the visible portfolio`),
    impactTile("Expansion headroom", money(totals.expansionPotential), `${totals.growthCount} accounts show favorable growth signals`),
  ].join("");
}

function renderOverview(accounts) {
  renderKpis(accounts);
  renderActionQueue(accounts);
  renderCompetitiveBoard(accounts);
  drawStageChart(accounts);
  drawTierChart(accounts);
}

function renderKpis(accounts) {
  const totals = summarize(accounts);
  const metrics = [
    ["ARR monitored", money(totals.annualizedRevenue), `${accounts.length} active accounts in current view`],
    ["Revenue at risk", money(totals.revenueAtRisk), `${totals.criticalCount} critical, ${totals.atRiskCount} at risk`],
    ["Avg health", `${totals.avgHealth}/100`, `Portfolio resilience across CRM and call signals`],
    ["P1 interventions", String(totals.p1Count), `High-urgency actions with named owners`],
    ["Competitive pressure", String(totals.competitorAccounts), `${totals.competitorMentions} named market threat signals`],
    ["Projected ROI", `${totals.avgRoi}x`, `Retention value vs. estimated intervention spend`],
  ];

  document.getElementById("kpiGrid").innerHTML = metrics
    .map(
      ([label, value, sub]) => `
        <article class="kpi">
          <div class="kpi-label">${escapeHtml(label)}</div>
          <div class="kpi-value">${escapeHtml(value)}</div>
          <div class="kpi-sub">${escapeHtml(sub)}</div>
        </article>
      `
    )
    .join("");
}

function renderActionQueue(accounts) {
  const queue = [...accounts]
    .sort((a, b) => {
      const priorityDelta = PRIORITY_ORDER[a.action_plan.priority] - PRIORITY_ORDER[b.action_plan.priority];
      if (priorityDelta !== 0) return priorityDelta;
      return b.business_impact.revenue_at_risk - a.business_impact.revenue_at_risk;
    })
    .slice(0, 6);

  document.getElementById("actionQueue").innerHTML = queue.length
    ? queue
        .map(
          (account) => `
            <article class="queue-card">
              <div class="row-top">
                <div>
                  <div class="list-title">${escapeHtml(account.customer_name)}</div>
                  <div class="list-copy">${escapeHtml(account.action_plan.next_best_action)}</div>
                </div>
                <span class="score-chip ${stageClass(account.deal.stage)}">${escapeHtml(account.action_plan.priority)}</span>
              </div>
              <div class="row-bottom tiny">
                <div><strong>${escapeHtml(account.action_plan.owner)}</strong><br>${escapeHtml(account.action_plan.timeline)}</div>
                <div><strong>${money(account.business_impact.revenue_at_risk)}</strong><br>value at risk</div>
                <div><strong>${escapeHtml(account.action_plan.channel)}</strong><br>recommended channel</div>
              </div>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">No accounts match the current filters.</div>';
}

function renderCompetitiveBoard(accounts) {
  const pressured = accounts.filter((account) => account.competitors.length > 0);
  document.getElementById("competitiveBoard").innerHTML = pressured.length
    ? pressured
        .map(
          (account) => `
            <article class="competitor-card">
              <div class="row-top">
                <div>
                  <div class="list-title">${escapeHtml(account.customer_name)}</div>
                  <div class="list-copy">${escapeHtml(account.competitors.map((item) => `${item.name} (${item.positioning})`).join(", "))}</div>
                </div>
                <span class="score-chip risk-critical">${escapeHtml(account.deal.stage)}</span>
              </div>
              <div class="tiny">Counter-play: ${escapeHtml(account.action_plan.next_best_action)}</div>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">No explicit competitor mentions were detected in the visible portfolio.</div>';
}

function renderAccounts(accounts) {
  document.getElementById("accountList").innerHTML = accounts.length
    ? accounts
        .map((account) => {
          const isSelected = account.customer_id === state.selectedId;
          return `
            <article class="account-row ${isSelected ? "is-selected" : ""}" data-account="${escapeHtml(account.customer_id)}">
              <div class="row-top">
                <div>
                  <div class="list-title">${escapeHtml(account.customer_name)}</div>
                  <div class="tiny">${escapeHtml(account.city || "Unknown city")} · ${escapeHtml(account.tier || "Tier not set")}</div>
                </div>
                <span class="score-chip ${stageClass(account.deal.stage)}">${escapeHtml(account.deal.stage)}</span>
              </div>
              <div class="row-bottom tiny">
                <div><strong>${account.churn.score || 0}/100</strong><br>churn risk</div>
                <div><strong>${account.account_health.score}</strong><br>health</div>
                <div><strong>${money(account.business_impact.revenue_at_risk)}</strong><br>value at risk</div>
              </div>
              <div class="bar-track"><div class="bar-fill" style="width:${account.account_health.score}%"></div></div>
            </article>
          `;
        })
        .join("")
    : '<div class="empty-state">No accounts match the current filters.</div>';

  document.getElementById("accountList").querySelectorAll("[data-account]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedId = row.dataset.account;
      render();
    });
  });

  const account = accounts.find((item) => item.customer_id === state.selectedId);
  renderAccountDetail(account);
}

function renderAccountDetail(account) {
  const container = document.getElementById("accountDetail");
  if (!account) {
    container.innerHTML = '<div class="empty-state">Select an account to inspect its health, risk drivers, scripts, and recovery plan.</div>';
    return;
  }

  const importances = (account.churn.importances || []).slice(0, 4);
  const scripts = (account.scripts || []).slice(0, 3);

  container.innerHTML = `
    <div class="detail-top">
      <div>
        <div class="eyebrow">Account intelligence</div>
        <h3>${escapeHtml(account.customer_name)}</h3>
        <p class="detail-copy">${escapeHtml(account.action_plan.rationale)}</p>
      </div>
      <span class="score-chip ${stageClass(account.deal.stage)}">${escapeHtml(account.deal.stage)}</span>
    </div>

    <div class="detail-metrics">
      ${metric("Health score", `${account.account_health.score}/100`, account.account_health.label)}
      ${metric("Churn score", `${account.churn.score || 0}/100`, account.churn.label || "Unlabeled")}
      ${metric("Revenue at risk", money(account.business_impact.revenue_at_risk), `${account.action_plan.priority} priority`)}
      ${metric("Expansion potential", money(account.business_impact.expansion_potential), `${account.business_impact.roi_multiple}x ROI profile`)}
    </div>

    <div class="split-grid">
      <div class="detail-box">
        <div class="panel-head">
          <div>
            <div class="eyebrow">Next-best action</div>
            <h3>${escapeHtml(account.action_plan.playbook)}</h3>
          </div>
        </div>
        <div class="stack">
          ${actionLine("Owner", account.action_plan.owner)}
          ${actionLine("Timeline", account.action_plan.timeline)}
          ${actionLine("Channel", account.action_plan.channel)}
          ${actionLine("Action", account.action_plan.next_best_action)}
        </div>
      </div>

      <div class="detail-box">
        <div class="panel-head">
          <div>
            <div class="eyebrow">Signal summary</div>
            <h3>${escapeHtml(account.mood.label || "Neutral")} sentiment state</h3>
          </div>
        </div>
        <div class="chip-row">
          ${(account.keywords.topics || []).map((topic) => `<span class="chip">${escapeHtml(topic)}</span>`).join("") || '<span class="chip">No topics detected</span>'}
          ${account.competitors.map((item) => `<span class="chip">${escapeHtml(item.name)}</span>`).join("")}
        </div>
        <div class="tiny">
          Sentiment ${roundTo(account.sentiment.compound || 0, 2)}, mood confidence ${account.mood.confidence || 0}%,
          ${account.sentiment.churn_signals || 0} churn phrases, ${account.sentiment.anger_signals || 0} anger phrases.
        </div>
      </div>
    </div>

    <div class="split-grid">
      <div class="detail-box">
        <div class="panel-head">
          <div>
            <div class="eyebrow">Explainability</div>
            <h3>Top risk drivers</h3>
          </div>
        </div>
        <div class="stack">
          ${
            importances.length
              ? importances
                  .map(
                    (item) => `
                      <div class="architecture-card">
                        <div class="row-top">
                          <div class="list-title">${escapeHtml(item.feature)}</div>
                          <span class="badge ${item.direction === "risk" ? "risk-critical" : "health-good"}">${escapeHtml(item.direction)}</span>
                        </div>
                        <div class="tiny">${item.importance_pct}% contribution to the current churn score.</div>
                      </div>
                    `
                  )
                  .join("")
              : '<div class="tiny">No feature importance data available.</div>'
          }
        </div>
      </div>

      <div class="detail-box">
        <div class="panel-head">
          <div>
            <div class="eyebrow">Sentiment journey</div>
            <h3>Customer call trajectory</h3>
          </div>
        </div>
        <canvas id="arcChart" height="210"></canvas>
      </div>
    </div>

    <div class="detail-box">
      <div class="panel-head">
        <div>
          <div class="eyebrow">Generated playbooks</div>
          <h3>Conversation scripts for the account team</h3>
        </div>
      </div>
      <div class="stack">
        ${
          scripts.length
            ? scripts
                .map(
                  (script) => `
                    <article class="script-card">
                      <div class="row-top">
                        <div>
                          <div class="list-title">${escapeHtml(script.scenario)}</div>
                          <div class="tiny">${escapeHtml(script.tone)} · ${escapeHtml(script.urgency)}</div>
                        </div>
                      </div>
                      <div class="list-copy">${escapeHtml(script.opener)}</div>
                      <div class="stack">
                        ${(script.key_points || []).slice(0, 3).map((point) => `<div class="tiny">- ${escapeHtml(point)}</div>`).join("")}
                      </div>
                    </article>
                  `
                )
                .join("")
            : '<div class="tiny">No scripts available.</div>'
        }
      </div>
    </div>
  `;

  drawArcChart(account);
}

function renderDeals(accounts) {
  const columns = STAGE_ORDER.map((stage) => {
    const stageAccounts = accounts.filter((account) => account.deal.stage === stage);
    return `
      <section class="kanban-column">
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="eyebrow">Deal stage</div>
              <h3>${escapeHtml(stage)}</h3>
            </div>
            <span class="score-chip ${stageClass(stage)}">${stageAccounts.length}</span>
          </div>
          <div class="stack">
            ${
              stageAccounts.length
                ? stageAccounts
                    .map(
                      (account) => `
                        <article class="deal-card">
                          <div class="list-title">${escapeHtml(account.customer_name)}</div>
                          <div class="tiny">${money(account.contract_value || 0)} contract · renew in ${account.deal.renewal_window_days}d</div>
                          <div class="tiny">Health ${account.account_health.score}/100 · ${account.competitors.length} competitor signal(s)</div>
                          <div class="list-copy">${escapeHtml(account.action_plan.next_best_action)}</div>
                        </article>
                      `
                    )
                    .join("")
                : '<div class="tiny">No accounts in this stage.</div>'
            }
          </div>
        </div>
      </section>
    `;
  });
  document.getElementById("dealBoard").innerHTML = columns.join("");
}

function renderGrowth(accounts) {
  const opportunities = [...accounts]
    .filter((account) => account.business_impact.expansion_potential > 0)
    .sort((a, b) => b.business_impact.expansion_potential - a.business_impact.expansion_potential)
    .slice(0, 6);

  document.getElementById("growthList").innerHTML = opportunities.length
    ? opportunities
        .map(
          (account) => `
            <article class="motion-card">
              <div class="row-top">
                <div>
                  <div class="list-title">${escapeHtml(account.customer_name)}</div>
                  <div class="tiny">${escapeHtml(account.city || "Unknown city")} · ${escapeHtml(account.tier || "Tier not set")}</div>
                </div>
                <span class="score-chip risk-growth">${money(account.business_impact.expansion_potential)}</span>
              </div>
              <div class="list-copy">${escapeHtml(account.action_plan.playbook === "Expansion motion" ? account.action_plan.next_best_action : "Use healthy account signals to position premium analytics, support SLAs, or volume-based pricing.")}</div>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">No expansion candidates under the current filters.</div>';

  const prospectHooks = buildProspectingHooks(opportunities.length ? opportunities : accounts);
  document.getElementById("prospectList").innerHTML = prospectHooks.length
    ? prospectHooks
        .map(
          (hook) => `
            <article class="motion-card">
              <div class="list-title">${escapeHtml(hook.title)}</div>
              <div class="list-copy">${escapeHtml(hook.copy)}</div>
              <div class="tiny">${escapeHtml(hook.reason)}</div>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">No prospecting hooks available for the current slice.</div>';
}

function buildProspectingHooks(accounts) {
  return accounts.slice(0, 5).map((account, index) => ({
    title: `${account.city || "Regional"} lookalike motion ${index + 1}`,
    copy: `Target accounts that resemble ${account.customer_name} and lead with an outcome around ${firstTopic(account)} plus a quantified ROI story.`,
    reason: `${money(account.business_impact.expansion_potential)} upside signal derived from account behavior, sentiment, and tier profile.`,
  }));
}

function renderArchitecture(accounts) {
  document.getElementById("architectureLayers").innerHTML = [
    architectureCard("1. CRM + conversation ingestion", "Customer fields, operational metrics, and call transcripts are merged into a shared account view."),
    architectureCard("2. NLP signal layer", "The stack applies sentiment analysis, topic extraction, anger/churn phrase detection, and competitor lookup."),
    architectureCard("3. Scoring layer", "Churn, health, business-impact, and deal-stage scores convert raw signals into prioritization."),
    architectureCard("4. Decision layer", "A next-best-action engine assigns owner, timeline, channel, and playbook instead of stopping at analytics."),
    architectureCard("5. Operator surface", "Portfolio filters, explainability, action queue, and account playbooks make the system usable for RevOps and account teams."),
  ].join("");

  const totals = summarize(accounts);
  document.getElementById("innovationList").innerHTML = [
    architectureCard("Unified intelligence", "Retention, deal intelligence, competitive pressure, and growth motions live in one portfolio model instead of isolated demos."),
    architectureCard("Business-native outputs", "The UI leads with revenue at risk, protectable value, and expansion upside, not just model scores."),
    architectureCard("Action orchestration", "Every account has a named owner, recommended channel, and execution window for operational follow-through."),
  ].join("");

  document.getElementById("businessImpactList").innerHTML = [
    architectureCard("Cycle-time reduction", "The priority queue compresses triage by surfacing which accounts need intervention first and why."),
    architectureCard("Revenue protection", `${money(totals.revenueAtRisk)} is visible as current portfolio exposure for the filtered set.`),
    architectureCard("Expansion focus", `${money(totals.expansionPotential)} in modeled upside helps account teams shift from firefighting to growth.`),
  ].join("");
}

function drawStageChart(accounts) {
  const counts = STAGE_ORDER.map((stage) => accounts.filter((account) => account.deal.stage === stage).length);
  drawChart("stageChart", {
    type: "doughnut",
    data: {
      labels: STAGE_ORDER,
      datasets: [{
        data: counts,
        backgroundColor: ["#c44536", "#c96f38", "#ba8b26", "#0f9d7a"],
        borderWidth: 0,
      }],
    },
    options: baseChartOptions({
      plugins: { legend: { position: "bottom" } },
      cutout: "62%",
    }),
  });
}

function drawTierChart(accounts) {
  const tiers = Array.from(new Set(accounts.map((account) => account.tier).filter(Boolean)));
  const values = tiers.map((tier) =>
    accounts
      .filter((account) => account.tier === tier)
      .reduce((sum, account) => sum + account.business_impact.revenue_at_risk, 0)
  );

  drawChart("tierChart", {
    type: "bar",
    data: {
      labels: tiers,
      datasets: [{
        label: "Revenue at risk",
        data: values,
        backgroundColor: ["#0f9d7a", "#ba8b26", "#c96f38", "#c44536"],
        borderRadius: 10,
      }],
    },
    options: baseChartOptions({
      scales: {
        x: { grid: { display: false }, ticks: { color: "#65717d" } },
        y: {
          grid: { color: "rgba(220, 207, 188, 0.8)" },
          ticks: { color: "#65717d", callback: (value) => money(value) },
        },
      },
    }),
  });
}

function drawArcChart(account) {
  const points = account.arc || [];
  drawChart("arcChart", {
    type: "line",
    data: {
      labels: points.map((point) => point.segment),
      datasets: [{
        label: "Sentiment arc",
        data: points.map((point) => point.score),
        borderColor: "#0f9d7a",
        backgroundColor: "rgba(15, 157, 122, 0.15)",
        fill: true,
        tension: 0.35,
      }],
    },
    options: baseChartOptions({
      scales: {
        x: { grid: { display: false }, ticks: { color: "#65717d" } },
        y: {
          min: -100,
          max: 100,
          grid: { color: "rgba(220, 207, 188, 0.8)" },
          ticks: { color: "#65717d" },
        },
      },
    }),
  });
}

function drawChart(canvasId, config) {
  const existing = state.charts[canvasId];
  if (existing) existing.destroy();
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined") return;
  state.charts[canvasId] = new Chart(canvas, config);
}

function baseChartOptions(extra) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#65717d", font: { family: "Manrope" } } },
      tooltip: {
        backgroundColor: "#1f2a36",
        titleColor: "#fff",
        bodyColor: "#fff",
      },
    },
    ...extra,
  };
}

function summarize(accounts) {
  const initial = {
    annualizedRevenue: 0,
    revenueAtRisk: 0,
    netRetentionValue: 0,
    expansionPotential: 0,
    competitorMentions: 0,
    competitorAccounts: 0,
    avgHealth: 0,
    criticalCount: 0,
    atRiskCount: 0,
    growthCount: 0,
    p1Count: 0,
    avgRoi: 0,
  };

  const totals = accounts.reduce((acc, account) => {
    acc.annualizedRevenue += account.business_impact.annualized_revenue || 0;
    acc.revenueAtRisk += account.business_impact.revenue_at_risk || 0;
    acc.netRetentionValue += account.business_impact.net_retention_value || 0;
    acc.expansionPotential += account.business_impact.expansion_potential || 0;
    acc.competitorMentions += account.competitors.reduce((sum, item) => sum + item.count, 0);
    acc.competitorAccounts += account.competitors.length ? 1 : 0;
    acc.avgHealth += account.account_health.score || 0;
    acc.criticalCount += account.deal.stage === "Critical" ? 1 : 0;
    acc.atRiskCount += account.deal.stage === "At Risk" ? 1 : 0;
    acc.growthCount += account.deal.stage === "Growth" ? 1 : 0;
    acc.p1Count += account.action_plan.priority === "P1" ? 1 : 0;
    acc.avgRoi += Number(account.business_impact.roi_multiple || 0);
    return acc;
  }, initial);

  const count = Math.max(accounts.length, 1);
  totals.avgHealth = Math.round(totals.avgHealth / count);
  totals.avgRoi = roundTo(totals.avgRoi / count, 1);
  return totals;
}

function impactTile(label, value, copy) {
  return `
    <article class="architecture-card">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-sub">${escapeHtml(copy)}</div>
    </article>
  `;
}

function architectureCard(title, copy) {
  return `
    <article class="architecture-card">
      <div class="list-title">${escapeHtml(title)}</div>
      <div class="list-copy">${escapeHtml(copy)}</div>
    </article>
  `;
}

function metric(label, value, sub) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-sub">${escapeHtml(sub)}</div>
    </div>
  `;
}

function actionLine(label, value) {
  return `
    <div class="architecture-card">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="list-copy"><strong>${escapeHtml(value)}</strong></div>
    </div>
  `;
}

function stageClass(stage) {
  return {
    "Critical": "risk-critical",
    "At Risk": "risk-atrisk",
    "Stable": "risk-stable",
    "Growth": "risk-growth",
  }[stage] || "risk-stable";
}

function firstTopic(account) {
  return account.keywords.topics?.[0] || "service reliability";
}

function money(value) {
  return `Rs ${new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(Math.round(value || 0))}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundTo(value, places) {
  const scale = Math.pow(10, places);
  return Math.round((value || 0) * scale) / scale;
}

function updateStatus(text) {
  document.getElementById("dataStamp").textContent = text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
