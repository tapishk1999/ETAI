const opsState = {
  activeTab: "overview",
  charts: {},
  accounts: [],
  agents: [
    { id: "A001", name: "Marcus Reid", zone: "West", vehicle: "Van" },
    { id: "A002", name: "Priya Sharma", zone: "South", vehicle: "Bike" },
    { id: "A003", name: "Chen Wei", zone: "North", vehicle: "Scooter" },
    { id: "A004", name: "Fatima Noor", zone: "Central", vehicle: "Van" },
    { id: "A005", name: "James Okafor", zone: "East", vehicle: "Truck" },
  ],
  orders: [],
};

document.addEventListener("DOMContentLoaded", initOps);

async function initOps() {
  bindOpsEvents();
  const payload = await loadOpsPayload();
  opsState.accounts = (payload.results || []).map((row) => ({
    ...row,
    deal: row.deal || { stage: "Stable", renewal_window_days: 60 },
    action_plan: row.action_plan || { owner: "Account Manager", timeline: "This week", next_best_action: "Review account health." },
    account_health: row.account_health || { score: 70 },
    business_impact: row.business_impact || { revenue_at_risk: 0, expansion_potential: 0 },
    searchable: [row.customer_name, row.city, row.tier].filter(Boolean).join(" ").toLowerCase(),
  }));
  opsState.orders = buildOrders(opsState.accounts, opsState.agents);
  populateZones();
  renderOps();
}

function bindOpsEvents() {
  document.getElementById("opsTabs").addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab]");
    if (!button) return;
    opsState.activeTab = button.dataset.tab;
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === opsState.activeTab));
    document.querySelectorAll(".view").forEach((view) => view.classList.toggle("is-active", view.id === `ops-${opsState.activeTab}`));
    renderOps();
  });

  ["opsSearch", "statusFilter", "zoneFilter"].forEach((id) => {
    document.getElementById(id).addEventListener("input", renderOps);
    document.getElementById(id).addEventListener("change", renderOps);
  });
}

async function loadOpsPayload() {
  if (window.__ETAI_RESULTS__?.results?.length) {
    const stamp = window.__ETAI_RESULTS__.generated_at
      ? new Date(window.__ETAI_RESULTS__.generated_at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })
      : "Bundled snapshot";
    document.getElementById("opsStamp").textContent = `Ops surface refreshed ${stamp}`;
    return window.__ETAI_RESULTS__;
  }

  const candidates = ["results.json", "./results.json", "../docs/results.json"];
  try {
    let payload = null;
    for (const path of candidates) {
      const response = await fetch(path, { cache: "no-store" });
      if (response.ok) {
        payload = await response.json();
        break;
      }
    }
    if (!payload) throw new Error("No readable results.json found");
    const stamp = payload.generated_at ? new Date(payload.generated_at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) : "Demo data";
    document.getElementById("opsStamp").textContent = `Ops surface refreshed ${stamp}`;
    return payload;
  } catch (error) {
    const localHint = window.location.protocol === "file:"
      ? " Open the GitHub Pages link or serve the docs folder over HTTP."
      : "";
    document.getElementById("opsStamp").textContent = `Ops data unavailable (${error.message}).${localHint}`;
    return { results: [] };
  }
}

function populateZones() {
  const zones = Array.from(new Set(opsState.agents.map((agent) => agent.zone)));
  document.getElementById("zoneFilter").innerHTML =
    '<option value="all">All zones</option>' +
    zones.map((zone) => `<option value="${escapeHtml(zone)}">${escapeHtml(zone)}</option>`).join("");
}

function buildOrders(accounts, agents) {
  const cityZone = {
    Mumbai: "West",
    Pune: "West",
    Delhi: "North",
    Bangalore: "South",
    Hyderabad: "South",
    Chennai: "East",
  };

  return accounts.flatMap((account, index) => {
    const zone = cityZone[account.city] || agents[index % agents.length].zone;
    const agent = agents.filter((item) => item.zone === zone)[0] || agents[index % agents.length];
    const baseStatuses = account.deal.stage === "Critical"
      ? ["Exception", "Pending"]
      : account.deal.stage === "At Risk"
        ? ["In Transit", "Pending"]
        : account.deal.stage === "Growth"
          ? ["Delivered", "In Transit"]
          : ["Delivered", "Delivered"];

    return baseStatuses.map((status, offset) => ({
      id: `ORD-${String(index * 2 + offset + 101).padStart(4, "0")}`,
      customer: account.customer_name,
      city: account.city,
      zone,
      agent: agent.name,
      agent_zone: agent.zone,
      status,
      priority: account.action_plan.priority,
      value: Math.round((account.contract_value || 0) / (offset === 0 ? 18 : 24)) || 1800,
      eta: offset === 0 ? "10:30" : "14:45",
      riskStage: account.deal.stage,
    }));
  });
}

function getVisibleOrders() {
  const search = document.getElementById("opsSearch").value.trim().toLowerCase();
  const status = document.getElementById("statusFilter").value;
  const zone = document.getElementById("zoneFilter").value;

  return opsState.orders.filter((order) => {
    if (status !== "all" && order.status !== status) return false;
    if (zone !== "all" && order.zone !== zone) return false;
    if (search && !`${order.id} ${order.customer} ${order.city} ${order.agent}`.toLowerCase().includes(search)) return false;
    return true;
  });
}

function renderOps() {
  const visibleOrders = getVisibleOrders();
  const visibleAccounts = getVisibleAccounts(visibleOrders);
  renderOpsOverview(visibleOrders, visibleAccounts);
  renderOrders(visibleOrders);
  renderAgents(visibleOrders);
  renderOpsAccounts(visibleAccounts);
  renderAnalytics(visibleOrders);
}

function getVisibleAccounts(orders) {
  const names = new Set(orders.map((order) => order.customer));
  if (!names.size) return [];
  return opsState.accounts.filter((account) => names.has(account.customer_name));
}

function renderOpsOverview(orders, accounts) {
  const delivered = orders.filter((order) => order.status === "Delivered").length;
  const activeAgents = new Set(orders.map((order) => order.agent)).size;
  const exceptionOrders = orders.filter((order) => order.status === "Exception").length;
  const pending = orders.filter((order) => order.status === "Pending").length;
  const revenueAtRisk = accounts.reduce((sum, account) => sum + (account.business_impact.revenue_at_risk || 0), 0);

  const kpis = [
    ["Orders in view", String(orders.length), `${pending} pending and ${exceptionOrders} in exception state`],
    ["On-time proxy", `${orders.length ? Math.round((delivered / orders.length) * 100) : 0}%`, "Delivered orders across the active filter set"],
    ["Active agents", String(activeAgents), "Agents assigned to visible deliveries"],
    ["Ops escalations", String(exceptionOrders + pending), "Orders that can trigger account-risk conversations"],
    ["Revenue linked", money(revenueAtRisk), "Portfolio value connected to the active operations surface"],
  ];

  document.getElementById("opsKpis").innerHTML = kpis.map(([label, value, sub]) => `
    <article class="kpi">
      <div class="kpi-label">${escapeHtml(label)}</div>
      <div class="kpi-value">${escapeHtml(value)}</div>
      <div class="kpi-sub">${escapeHtml(sub)}</div>
    </article>
  `).join("");

  const escalations = [...accounts]
    .sort((a, b) => (b.business_impact.revenue_at_risk || 0) - (a.business_impact.revenue_at_risk || 0))
    .slice(0, 5);

  document.getElementById("escalationQueue").innerHTML = escalations.map((account) => `
    <article class="queue-card">
      <div class="row-top">
        <div>
          <div class="copy"><strong>${escapeHtml(account.customer_name)}</strong></div>
          <div class="tiny">${escapeHtml(account.action_plan.next_best_action)}</div>
        </div>
        <span class="score-chip ${stageClass(account.deal.stage)}">${escapeHtml(account.deal.stage)}</span>
      </div>
      <div class="row-bottom tiny">
        <div><strong>${escapeHtml(account.action_plan.owner)}</strong><br>owner</div>
        <div><strong>${money(account.business_impact.revenue_at_risk || 0)}</strong><br>revenue at risk</div>
        <div><strong>${escapeHtml(account.action_plan.timeline)}</strong><br>timeline</div>
      </div>
    </article>
  `).join("") || '<div class="chart-empty">No linked accounts match the current filters.</div>';

  drawStatusChart(orders);
}

function renderOrders(orders) {
  const wrap = document.getElementById("ordersTableWrap");
  wrap.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Order</th>
            <th>Customer</th>
            <th>Zone</th>
            <th>Agent</th>
            <th>Status</th>
            <th>Priority</th>
            <th>Value</th>
            <th>ETA</th>
          </tr>
        </thead>
        <tbody>
          ${orders.map((order) => `
            <tr>
              <td>${escapeHtml(order.id)}</td>
              <td>${escapeHtml(order.customer)}</td>
              <td>${escapeHtml(order.zone)}</td>
              <td>${escapeHtml(order.agent)}</td>
              <td><span class="score-chip ${statusClass(order.status)}">${escapeHtml(order.status)}</span></td>
              <td>${escapeHtml(order.priority)}</td>
              <td>${money(order.value)}</td>
              <td>${escapeHtml(order.eta)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderAgents(orders) {
  const counts = countAssignments(orders);
  document.getElementById("agentGrid").innerHTML = opsState.agents.map((agent) => {
    const assignments = counts[agent.name] || [];
    const delivered = assignments.filter((item) => item.status === "Delivered").length;
    const onTime = assignments.length ? Math.round((delivered / assignments.length) * 100) : 0;
    return `
      <article class="agent-card">
        <div class="row-top">
          <div>
            <h3>${escapeHtml(agent.name)}</h3>
            <div class="tiny">${escapeHtml(agent.zone)} zone · ${escapeHtml(agent.vehicle)}</div>
          </div>
          <span class="score-chip risk-ok">${assignments.length} active</span>
        </div>
        <div class="agent-meta tiny">
          <div><strong>${assignments.length}</strong><br>assignments</div>
          <div><strong>${onTime}%</strong><br>delivery rate</div>
          <div><strong>${escapeHtml(agent.zone)}</strong><br>coverage</div>
        </div>
      </article>
    `;
  }).join("");
}

function renderOpsAccounts(accounts) {
  const ranked = [...accounts]
    .sort((a, b) => (b.business_impact.revenue_at_risk || 0) - (a.business_impact.revenue_at_risk || 0));

  document.getElementById("opsAccounts").innerHTML = ranked.length ? ranked.map((account) => `
    <article class="account-card">
      <div class="row-top">
        <div>
          <h3>${escapeHtml(account.customer_name)}</h3>
          <div class="tiny">${escapeHtml(account.city || "Unknown city")} · ${escapeHtml(account.tier || "Tier not set")}</div>
        </div>
        <span class="score-chip ${stageClass(account.deal.stage)}">${escapeHtml(account.deal.stage)}</span>
      </div>
      <div class="row-bottom tiny">
        <div><strong>${account.open_tickets || 0}</strong><br>open tickets</div>
        <div><strong>${account.missed_deliveries || 0}</strong><br>missed deliveries</div>
        <div><strong>${account.account_health.score || 0}/100</strong><br>health</div>
      </div>
      <div class="copy">${escapeHtml(account.action_plan.next_best_action)}</div>
    </article>
  `).join("") : '<div class="chart-empty">No accounts match the current filters.</div>';
}

function renderAnalytics(orders) {
  drawAgentChart(orders);
  drawCityChart(orders);
}

function drawStatusChart(orders) {
  const labels = ["Delivered", "In Transit", "Pending", "Exception"];
  const data = labels.map((label) => orders.filter((order) => order.status === label).length);
  drawChart("statusChart", {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: ["#0f9d7a", "#b68828", "#c96f38", "#c44536"],
        borderWidth: 0,
      }],
    },
    options: baseOptions({ cutout: "62%" }),
  });
}

function drawAgentChart(orders) {
  const counts = countAssignments(orders);
  const labels = opsState.agents.map((agent) => agent.name);
  const assignments = labels.map((label) => (counts[label] || []).length);
  const delivered = labels.map((label) => {
    const rows = counts[label] || [];
    return rows.length ? Math.round((rows.filter((row) => row.status === "Delivered").length / rows.length) * 100) : 0;
  });

  drawChart("agentChart", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Assignments", data: assignments, backgroundColor: "#c96f38", borderRadius: 8, yAxisID: "y" },
        { label: "On-time proxy", data: delivered, backgroundColor: "#0f9d7a", borderRadius: 8, yAxisID: "y2" },
      ],
    },
    options: baseOptions({
      scales: {
        x: { grid: { display: false }, ticks: { color: "#66717b" } },
        y: { grid: { color: "#ede4d7" }, ticks: { color: "#66717b" }, position: "left" },
        y2: { grid: { display: false }, ticks: { color: "#66717b" }, position: "right", min: 0, max: 100 },
      },
    }),
  });
}

function drawCityChart(orders) {
  const cityMap = {};
  orders.forEach((order) => { cityMap[order.city] = (cityMap[order.city] || 0) + order.value; });
  drawChart("cityChart", {
    type: "bar",
    data: {
      labels: Object.keys(cityMap),
      datasets: [{
        label: "Order value",
        data: Object.values(cityMap),
        backgroundColor: ["#0f9d7a", "#b68828", "#c96f38", "#c44536", "#8d6b2c", "#4a9072"],
        borderRadius: 8,
      }],
    },
    options: baseOptions({
      scales: {
        x: { grid: { display: false }, ticks: { color: "#66717b" } },
        y: { grid: { color: "#ede4d7" }, ticks: { color: "#66717b", callback: (value) => money(value) } },
      },
    }),
  });
}

function countAssignments(orders) {
  return orders.reduce((acc, order) => {
    acc[order.agent] = acc[order.agent] || [];
    acc[order.agent].push(order);
    return acc;
  }, {});
}

function drawChart(id, config) {
  if (opsState.charts[id]) opsState.charts[id].destroy();
  const canvas = document.getElementById(id);
  if (!canvas) return;
  clearChartMessage(canvas);

  if (!chartHasData(config)) {
    showChartMessage(canvas, "No data available for this view.");
    return;
  }

  if (typeof Chart === "undefined") {
    showChartMessage(canvas, "Chart library unavailable. Summary cards still reflect the live data.");
    return;
  }

  canvas.classList.remove("is-hidden");
  opsState.charts[id] = new Chart(canvas, config);
}

function chartHasData(config) {
  const labels = config?.data?.labels || [];
  const datasets = config?.data?.datasets || [];
  if (!labels.length || !datasets.length) return false;

  const values = datasets.flatMap((dataset) => Array.isArray(dataset.data) ? dataset.data : []);
  if (!values.length) return false;

  if (["doughnut", "pie", "polarArea"].includes(config.type)) {
    return values.some((value) => Number(value || 0) > 0);
  }

  return values.some((value) => value !== null && value !== undefined);
}

function showChartMessage(canvas, message) {
  canvas.classList.add("is-hidden");
  const fallback = document.createElement("div");
  fallback.className = "chart-empty";
  fallback.textContent = message;
  canvas.insertAdjacentElement("afterend", fallback);
}

function clearChartMessage(canvas) {
  canvas.classList.remove("is-hidden");
  const next = canvas.nextElementSibling;
  if (next?.classList.contains("chart-empty")) next.remove();
}

function baseOptions(extra) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#66717b" } },
      tooltip: {
        backgroundColor: "#1f2a36",
        titleColor: "#fff",
        bodyColor: "#fff",
      },
    },
    ...extra,
  };
}

function stageClass(stage) {
  return {
    "Critical": "risk-critical",
    "At Risk": "risk-atrisk",
    "Stable": "risk-stable",
    "Growth": "risk-ok",
  }[stage] || "risk-stable";
}

function statusClass(status) {
  return {
    "Delivered": "risk-ok",
    "In Transit": "risk-stable",
    "Pending": "risk-atrisk",
    "Exception": "risk-critical",
  }[status] || "risk-stable";
}

function money(value) {
  return `Rs ${new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(Math.round(value || 0))}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
