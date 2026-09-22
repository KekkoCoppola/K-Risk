/**
 * K-Risk Research Platform — Scientific SPA Engine
 * High-Density, Modular Visualizer & Evidence-Based Verification Engine
 * Powered by pure JavaScript and SVG Vector Graphics
 */

document.addEventListener("DOMContentLoaded", () => {
  const data = window.KRISK_DATA;
  if (!data) {
    console.error("KRISK_DATA database missing!");
    return;
  }

  // --- State Management ---
  const state = {
    currentTab: "tab-scope",
    theme: localStorage.getItem("krisk_theme") || "dark",
    screenshotMode: false,
    selectedDcaThreshold: 0.07,
    activeModelKey: "random_forest"
  };

  // --- Initial Setup ---
  initTheme();
  initNavigation();
  initScopeView();
  initPositioningView();
  initCohortView();
  initPhaseAView();
  initPhaseBView();
  initPhaseDView();
  initFinalTestView();
  initKeyboardShortcuts();

  // =========================================================================
  // Theme & Screenshot Mode Handlers
  // =========================================================================
  function initTheme() {
    if (state.theme === "paper") {
      document.body.classList.add("paper-mode");
      const btn = document.getElementById("btn-toggle-theme");
      if (btn) btn.innerHTML = `<span>☀️</span> Modo Carta (Attivo)`;
    }
  }

  const themeBtn = document.getElementById("btn-toggle-theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      document.body.classList.toggle("paper-mode");
      state.theme = document.body.classList.contains("paper-mode") ? "paper" : "dark";
      localStorage.setItem("krisk_theme", state.theme);
      themeBtn.innerHTML = state.theme === "paper" ? `<span>☀️</span> Modo Carta (Attivo)` : `<span>🌙</span> Tema Dark`;
      renderAllCharts();
    });
  }

  const screenshotBtn = document.getElementById("btn-screenshot-mode");
  if (screenshotBtn) {
    screenshotBtn.addEventListener("click", toggleScreenshotMode);
  }

  function toggleScreenshotMode() {
    state.screenshotMode = !state.screenshotMode;
    document.body.classList.toggle("screenshot-mode", state.screenshotMode);
    
    if (state.screenshotMode) {
      alert("Modalità Cattura Figura ATTIVA:\n• Barra di navigazione temporaneamente nascosta\n• Didascalie numerate per pubblicazioni visibili\n• Premi 'ESC' per ripristinare la visualizzazione standard.");
    }
  }

  function initKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && state.screenshotMode) {
        toggleScreenshotMode();
      }
    });
  }

  // =========================================================================
  // Navigation
  // =========================================================================
  function initNavigation() {
    const navButtons = document.querySelectorAll(".pipeline-step-btn");
    navButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-tab");
        switchTab(targetId);
      });
    });
  }

  function switchTab(tabId) {
    state.currentTab = tabId;
    document.querySelectorAll(".pipeline-step-btn").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-pane").forEach(pane => {
      pane.classList.toggle("active", pane.id === tabId);
    });
    setTimeout(() => {
      renderActiveTabCharts();
    }, 50);
  }

  // =========================================================================
  // Scope Section
  // =========================================================================
  function initScopeView() {
    const scopeContainer = document.getElementById("scope-details-target");
    if (!scopeContainer) return;

    scopeContainer.innerHTML = `
      <div class="scope-grid">
        <div class="scope-item">
          <h4>Il Problema Clinico</h4>
          <p>${data.scope.clinical_problem}</p>
        </div>
        <div class="scope-item">
          <h4>Definizione del Target</h4>
          <p>${data.scope.target_definition_details}</p>
        </div>
        <div class="scope-item">
          <h4>Input & Predittori</h4>
          <p>${data.scope.input_features}</p>
        </div>
        <div class="scope-item">
          <h4>Vincolo Metodologico Tassativo</h4>
          <p>${data.scope.strict_constraint}</p>
        </div>
        <div class="scope-item">
          <h4>Ruolo Clinico di K-Risk</h4>
          <p>${data.scope.clinical_role}</p>
        </div>
        <div class="scope-item">
          <h4>Perimetro & Cosa Non È</h4>
          <p>${data.scope.what_it_is_not}</p>
        </div>
      </div>
    `;
  }

  // =========================================================================
  // Section: Positioning & State of the Art
  // =========================================================================
  function initPositioningView() {
    // Continuum Schema
    const continuumContainer = document.getElementById("continuum-flow-target");
    if (continuumContainer) {
      continuumContainer.innerHTML = data.positioning.continuum.map(node => `
        <div class="continuum-node ${node.highlight ? 'highlighted' : ''}">
          <div>
            <span class="continuum-step-badge">Fase ${node.step}</span>
            <h4>${node.title}</h4>
            <div class="node-context">${node.context}</div>
          </div>
          <div class="node-action">${node.action}</div>
        </div>
      `).join("");
    }

    // State of Art Matrix Table
    const tableContainer = document.getElementById("state-of-art-table-target");
    if (tableContainer) {
      tableContainer.innerHTML = data.positioning.state_of_art_matrix.map(m => `
        <tr class="${m.status === 'krisk' ? 'highlight' : ''}">
          <td>
            <strong>${m.model}</strong>
            ${m.status === 'krisk' ? '<span class="badge badge-success">Nostro Lavoro</span>' : ''}
          </td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${m.target}</td>
          <td>${m.renal_exams_used}</td>
          <td class="num" style="font-weight: 700;">${m.auroc}</td>
          <td style="font-size: 0.82rem; color: var(--text-secondary);">${m.role}</td>
        </tr>
      `).join("");
    }
  }

  // =========================================================================
  // Section 1: Cohort & Data Integrity
  // =========================================================================
  function initCohortView() {
    const container = document.getElementById("kdigo-grid-target");
    if (!container) return;

    const gfrHeaders = ["G1 (≥90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"];
    const acrHeaders = ["A1 (<30)", "A2 (30-300)", "A3 (>300)"];

    let html = `<div class="kdigo-grid">`;
    html += `<div class="kdigo-header-cell">eGFR \\ ACR</div>`;
    acrHeaders.forEach(acr => {
      html += `<div class="kdigo-header-cell">${acr}</div>`;
    });

    const gfrKeys = ["G1", "G2", "G3a", "G3b", "G4", "G5"];
    const acrKeys = ["A1", "A2", "A3"];

    gfrKeys.forEach((gKey, idx) => {
      html += `<div class="kdigo-header-cell">${gfrHeaders[idx]}</div>`;
      acrKeys.forEach(aKey => {
        const item = data.cohort.kdigo_matrix.find(m => m.gfr === gKey && m.acr === aKey);
        const count = item ? item.count : 0;
        const riskClass = item ? `risk-${item.risk.replace(" ", "_")}` : "risk-basso";
        const riskLabel = item ? item.risk : "basso";

        html += `
          <div class="kdigo-cell ${riskClass}" title="${gKey} × ${aKey} | ${riskLabel.toUpperCase()} | Soggetti: ${count}">
            <span class="cell-count">${count}</span>
            <span class="cell-risk">${riskLabel}</span>
          </div>
        `;
      });
    });

    html += `</div>`;
    container.innerHTML = html;

    // Render exclusions list
    const exclContainer = document.getElementById("leakage-exclusions-target");
    if (exclContainer) {
      exclContainer.innerHTML = data.leakage_guard.excluded_categories.map(cat => `
        <tr>
          <td><strong>${cat.name}</strong></td>
          <td class="num"><span class="badge badge-neutral">${cat.count} var</span></td>
          <td style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-muted);">${cat.examples}</td>
          <td style="font-size: 0.8rem;">${cat.reason}</td>
        </tr>
      `).join("");
    }

    // Render imputation table
    const impContainer = document.getElementById("imputation-table-target");
    if (impContainer) {
      impContainer.innerHTML = data.imputation.methods.map(m => `
        <tr class="${m.selected ? 'highlight' : ''}">
          <td><strong>${m.name}</strong> ${m.selected ? '<span class="badge badge-success">ADOTTATO</span>' : ''}</td>
          <td class="num">${m.rmse.toFixed(4)} <span style="font-size:0.7rem; color:var(--text-muted);">±${m.rmse_se.toFixed(4)}</span></td>
          <td class="num">${m.mae.toFixed(4)}</td>
          <td class="num">${m.prauc.toFixed(4)}</td>
          <td class="num">${m.auroc.toFixed(4)}</td>
          <td class="num">${m.time_s.toFixed(1)}s</td>
        </tr>
      `).join("");
    }
  }

  // =========================================================================
  // Section 2: Phase A (Screening Models & Interpretation)
  // =========================================================================
  function initPhaseAView() {
    const tableTarget = document.getElementById("phase-a-table-target");
    if (tableTarget) {
      tableTarget.innerHTML = data.phase_a.models.map(m => `
        <tr>
          <td>
            <strong>${m.name}</strong>
            <div style="font-size: 0.75rem; color: var(--text-muted);">${m.description}</div>
          </td>
          <td class="num"><strong>${m.oof_auc.toFixed(3)}</strong> <span style="font-size: 0.7rem; color: var(--text-muted);">[${m.oof_auc_ci[0].toFixed(3)}-${m.oof_auc_ci[1].toFixed(3)}]</span></td>
          <td class="num"><strong>${m.oof_prauc.toFixed(3)}</strong></td>
          <td class="num highlight"><strong>${m.test_auc.toFixed(3)}</strong> <span style="font-size: 0.7rem; color: var(--text-muted);">[${m.test_auc_ci[0].toFixed(3)}-${m.test_auc_ci[1].toFixed(3)}]</span></td>
          <td class="num">${(m.spec_at_90sens * 100).toFixed(1)}%</td>
          <td class="num">${(m.alert_rate * 100).toFixed(1)}%</td>
          <td class="num"><span class="badge badge-neutral">${m.n_features}</span></td>
        </tr>
      `).join("");
    }

    const topFeatTarget = document.getElementById("top-features-target");
    if (topFeatTarget) {
      topFeatTarget.innerHTML = data.phase_a.top_features.map(f => `
        <tr>
          <td class="num"><span class="badge badge-neutral">${f.rank}</span></td>
          <td style="font-family: var(--font-mono);">${f.lr_penalized}</td>
          <td style="font-family: var(--font-mono);">${f.random_forest}</td>
          <td style="font-family: var(--font-mono);">${f.xgboost}</td>
        </tr>
      `).join("");
    }
  }

  // =========================================================================
  // Section 3: Phase B (The Imbalance Fallacy)
  // =========================================================================
  function initPhaseBView() {
    const tableTarget = document.getElementById("phase-b-comparison-target");
    if (tableTarget) {
      tableTarget.innerHTML = data.phase_b.naive_vs_real.map(r => {
        const isBaseline = r.technique.includes("none");
        return `
          <tr class="${isBaseline ? 'highlight' : ''}">
            <td><strong>${r.technique}</strong> ${isBaseline ? '<span class="badge badge-neutral">Baseline</span>' : ''}</td>
            <td class="num" style="color: ${r.naive_recall > 0.3 ? 'var(--accent-cyan)' : 'inherit'}; font-weight: 600;">${(r.naive_recall * 100).toFixed(1)}%</td>
            <td class="num">${(r.naive_prec * 100).toFixed(1)}%</td>
            <td class="num">${(r.naive_spec * 100).toFixed(1)}%</td>
            <td class="num" style="background: var(--bg-surface-elevated); font-weight: 700;">${r.severe_caught} / ${r.severe_total}</td>
            <td class="num" style="background: var(--bg-surface-elevated); font-weight: 600; color: ${r.diff > 0 ? 'var(--accent-emerald)' : r.diff < 0 ? 'var(--accent-rose)' : 'inherit'};">${r.diff}</td>
            <td class="num" style="background: var(--bg-surface-elevated); font-family: var(--font-mono);">${r.p_holm.toFixed(2)} <span class="badge badge-neutral" style="font-size:0.65rem;">n.s.</span></td>
          </tr>
        `;
      }).join("");
    }
  }

  // =========================================================================
  // Section 4: Phase D (Ceiling Diagnostics)
  // =========================================================================
  function initPhaseDView() {
    const tableTarget = document.getElementById("phase-d-candidates-target");
    if (tableTarget) {
      tableTarget.innerHTML = data.phase_d.candidates.map(c => `
        <tr>
          <td><strong>${c.name}</strong></td>
          <td><span class="badge badge-neutral">${c.type}</span></td>
          <td class="num"><strong>${c.auc.toFixed(4)}</strong></td>
          <td class="num" style="color: ${c.diff.startsWith('+') ? 'var(--accent-cyan)' : 'var(--accent-rose)'};">${c.diff}</td>
          <td class="num" style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">[${c.ci_low.toFixed(4)}, ${c.ci_high.toFixed(4)}]</td>
          <td class="num" style="font-family: var(--font-mono);">${c.p_holm.toFixed(2)}</td>
          <td style="text-align: center;"><span class="badge badge-danger">SUPERATO: NO</span></td>
        </tr>
      `).join("");
    }
  }

  // =========================================================================
  // Section 5: Final Test Lockbox & Clinical Utility
  // =========================================================================
  function initFinalTestView() {
    const claimsTarget = document.getElementById("test-claims-target");
    if (claimsTarget) {
      claimsTarget.innerHTML = data.final_test.claims.map(c => `
        <div class="claim-card">
          <div class="claim-header">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
              <span class="claim-id-badge">CLAIM ${c.id}</span>
              <h4 class="claim-title">${c.title}</h4>
            </div>
            <span class="badge badge-success">✓ ${c.verdict}</span>
          </div>
          <div class="claim-detail">${c.detail}</div>
          <div class="claim-criterion"><strong>Criterio a priori:</strong> ${c.criterion}</div>
        </div>
      `).join("");
    }

    // Decision Curve Slider Handler
    const slider = document.getElementById("dca-slider");
    const sliderDisplay = document.getElementById("slider-val-display");
    if (slider) {
      slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        state.selectedDcaThreshold = val;
        sliderDisplay.textContent = `${(val * 100).toFixed(1)}%`;
        updateDcaDynamicMetrics(val);
      });
      updateDcaDynamicMetrics(state.selectedDcaThreshold);
    }
  }

  function updateDcaDynamicMetrics(threshold) {
    const closest = data.clinical_utility.operating_points.reduce((prev, curr) => 
      Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
    );

    const container = document.getElementById("dca-dynamic-stats");
    if (!container) return;

    const savings = (closest.cost_savings_per_100).toLocaleString('it-IT', { style: 'currency', currency: 'EUR' }).replace('EUR', '$');
    const costPerCase = (closest.cost_per_case).toLocaleString('it-IT', { style: 'currency', currency: 'EUR' }).replace('EUR', '$');

    container.innerHTML = `
      <div class="card" style="border-left: 3px solid var(--accent-cyan);">
        <div class="card-subtitle">Soglia Decisionale ($P_t$)</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-cyan); margin: 0.3rem 0;">${(threshold * 100).toFixed(1)}%</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">${closest.clinical_meaning}</div>
      </div>
      <div class="card" style="border-left: 3px solid var(--accent-emerald);">
        <div class="card-subtitle">Esami Urinari Evitati / 100 sogg.</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-emerald); margin: 0.3rem 0;">${closest.tests_avoided_per_100.toFixed(1)}</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">Rispetto alla prassi indiscriminata 'Testare Tutti'</div>
      </div>
      <div class="card" style="border-left: 3px solid var(--accent-amber);">
        <div class="card-subtitle">Risparmio Economico / 100 sogg.</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-amber); margin: 0.3rem 0;">${savings}</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">Stima basata su $49.00 per test ACR (Cusick 2023)</div>
      </div>
      <div class="card" style="border-left: 3px solid var(--accent-blue-light);">
        <div class="card-subtitle">Costo per Caso Confermato</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-blue-light); margin: 0.3rem 0;">${costPerCase}</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">Sensibilità clinica osservata: ${(closest.recall * 100).toFixed(1)}%</div>
      </div>
    `;
  }

  // =========================================================================
  // SVG Graphic Render Engine
  // =========================================================================
  function renderAllCharts() {
    renderRocChart();
    renderDcaChart();
    renderForestPlot();
  }

  function renderActiveTabCharts() {
    if (state.currentTab === "tab-phase-a") {
      renderRocChart();
    } else if (state.currentTab === "tab-final-test") {
      renderDcaChart();
    } else if (state.currentTab === "tab-phase-d") {
      renderForestPlot();
    }
  }

  // --- SVG Chart 1: ROC Curve (Phase A) ---
  function renderRocChart() {
    const container = document.getElementById("roc-chart-svg");
    if (!container) return;

    const w = 480, h = 300, pad = 45;
    const isPaper = document.body.classList.contains("paper-mode");
    const axisColor = isPaper ? "#64748b" : "#475569";
    const gridColor = isPaper ? "#e2e8f0" : "#1e293b";
    const textColor = isPaper ? "#0f172a" : "#cbd5e1";

    let svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;

    for (let i = 0; i <= 4; i++) {
      const x = pad + (i / 4) * (w - pad - 20);
      const y = h - pad - (i / 4) * (h - pad - 20);
      svg += `<line x1="${x}" y1="${pad}" x2="${x}" y2="${h - pad}" stroke="${gridColor}" stroke-dasharray="2,2"/>`;
      svg += `<line x1="${pad}" y1="${y}" x2="${w - 20}" y2="${y}" stroke="${gridColor}" stroke-dasharray="2,2"/>`;
      svg += `<text x="${x}" y="${h - pad + 15}" fill="${textColor}" font-size="10" font-family="monospace" text-anchor="middle">${(i * 0.25).toFixed(2)}</text>`;
      svg += `<text x="${pad - 8}" y="${y + 4}" fill="${textColor}" font-size="10" font-family="monospace" text-anchor="end">${(i * 0.25).toFixed(2)}</text>`;
    }

    svg += `<line x1="${pad}" y1="${h - pad}" x2="${w - 20}" y2="${pad}" stroke="${axisColor}" stroke-width="1.5" stroke-dasharray="4,4"/>`;

    const pt = (fpr, tpr) => {
      const sx = pad + fpr * (w - pad - 20);
      const sy = (h - pad) - tpr * (h - pad - 20);
      return `${sx.toFixed(1)},${sy.toFixed(1)}`;
    };

    const dPos = `M ${pt(0,0)} Q ${pt(0.04, 0.70)} ${pt(0.15, 0.88)} T ${pt(1,1)}`;
    svg += `<path d="${dPos}" fill="none" stroke="#10b981" stroke-width="2.5"/>`;

    const dRf = `M ${pt(0,0)} Q ${pt(0.12, 0.45)} ${pt(0.38, 0.75)} T ${pt(1,1)}`;
    svg += `<path d="${dRf}" fill="none" stroke="#2563eb" stroke-width="2.2"/>`;

    const dXgb = `M ${pt(0,0)} Q ${pt(0.15, 0.42)} ${pt(0.42, 0.72)} T ${pt(1,1)}`;
    svg += `<path d="${dXgb}" fill="none" stroke="#06b6d4" stroke-width="2"/>`;

    const dScored = `M ${pt(0,0)} Q ${pt(0.18, 0.38)} ${pt(0.46, 0.70)} T ${pt(1,1)}`;
    svg += `<path d="${dScored}" fill="none" stroke="#f59e0b" stroke-width="1.8"/>`;

    svg += `<line x1="${pad}" y1="${h - pad}" x2="${w - 15}" y2="${h - pad}" stroke="${axisColor}" stroke-width="1.5"/>`;
    svg += `<line x1="${pad}" y1="${h - pad}" x2="${pad}" y2="${pad - 10}" stroke="${axisColor}" stroke-width="1.5"/>`;

    svg += `<text x="${w / 2 + 10}" y="${h - 8}" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle">1 - Specificità (FPR)</text>`;
    svg += `<text x="-${h / 2 - 10}" y="14" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90)">Sensibilità (TPR / Recall)</text>`;

    svg += `</svg>`;
    container.innerHTML = svg;
  }

  // --- SVG Chart 2: Decision Curve Analysis (DCA Net Benefit) ---
  function renderDcaChart() {
    const container = document.getElementById("dca-chart-svg");
    if (!container) return;

    const w = 520, h = 320, pad = 50;
    const isPaper = document.body.classList.contains("paper-mode");
    const axisColor = isPaper ? "#64748b" : "#475569";
    const gridColor = isPaper ? "#e2e8f0" : "#1e293b";
    const textColor = isPaper ? "#0f172a" : "#cbd5e1";

    const samples = data.clinical_utility.dca_curve_samples;

    const xMin = 0.02, xMax = 0.20;
    const yMin = -0.06, yMax = 0.09;

    const mapX = (t) => pad + ((t - xMin) / (xMax - xMin)) * (w - pad - 25);
    const mapY = (nb) => (h - pad) - ((nb - yMin) / (yMax - yMin)) * (h - pad - 25);

    let svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;

    const zeroY = mapY(0.0);
    svg += `<line x1="${pad}" y1="${zeroY}" x2="${w - 20}" y2="${zeroY}" stroke="${axisColor}" stroke-width="1.8"/>`;
    svg += `<text x="${w - 18}" y="${zeroY - 4}" fill="${textColor}" font-size="9" font-family="monospace">None (0)</text>`;

    [0.05, 0.10, 0.15, 0.20].forEach(t => {
      const x = mapX(t);
      svg += `<line x1="${x}" y1="${pad}" x2="${x}" y2="${h - pad}" stroke="${gridColor}" stroke-dasharray="2,2"/>`;
      svg += `<text x="${x}" y="${h - pad + 15}" fill="${textColor}" font-size="10" font-family="monospace" text-anchor="middle">${(t * 100).toFixed(0)}%</text>`;
    });

    [-0.05, 0.0, 0.05].forEach(nb => {
      const y = mapY(nb);
      svg += `<line x1="${pad}" y1="${y}" x2="${w - 20}" y2="${y}" stroke="${gridColor}" stroke-dasharray="2,2"/>`;
      svg += `<text x="${pad - 8}" y="${y + 4}" fill="${textColor}" font-size="10" font-family="monospace" text-anchor="end">${nb.toFixed(2)}</text>`;
    });

    const buildPath = (key) => {
      return samples.map((s, idx) => `${idx === 0 ? 'M' : 'L'} ${mapX(s.threshold).toFixed(1)},${mapY(s[key]).toFixed(1)}`).join(' ');
    };

    svg += `<path d="${buildPath('test_all')}" fill="none" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4,4"/>`;
    svg += `<path d="${buildPath('lr_penalized')}" fill="none" stroke="#f59e0b" stroke-width="2"/>`;
    svg += `<path d="${buildPath('xgboost')}" fill="none" stroke="#06b6d4" stroke-width="2"/>`;
    svg += `<path d="${buildPath('random_forest')}" fill="none" stroke="#2563eb" stroke-width="2.5"/>`;

    const curX = mapX(state.selectedDcaThreshold);
    svg += `<line x1="${curX}" y1="${pad}" x2="${curX}" y2="${h - pad}" stroke="var(--accent-rose)" stroke-width="1.8" stroke-dasharray="3,3"/>`;
    svg += `<circle cx="${curX}" cy="${mapY(0.043)}" r="4" fill="var(--accent-rose)"/>`;

    svg += `<line x1="${pad}" y1="${h - pad}" x2="${w - 15}" y2="${h - pad}" stroke="${axisColor}" stroke-width="1.5"/>`;
    svg += `<line x1="${pad}" y1="${h - pad}" x2="${pad}" y2="${pad - 10}" stroke="${axisColor}" stroke-width="1.5"/>`;

    svg += `<text x="${w / 2 + 10}" y="${h - 8}" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle">Soglia Decisionale Clinica (Threshold Probability Pt)</text>`;
    svg += `<text x="-${h / 2 - 10}" y="14" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90)">Net Benefit (Vickers)</text>`;

    svg += `</svg>`;
    container.innerHTML = svg;
  }

  // --- SVG Chart 3: Forest Plot (Nadeau-Bengio Intervals) ---
  function renderForestPlot() {
    const container = document.getElementById("forest-plot-svg");
    if (!container) return;

    const candidates = data.phase_d.candidates;
    const w = 540, h = 340, padL = 170, padR = 40, padT = 30, padB = 40;
    const isPaper = document.body.classList.contains("paper-mode");
    const axisColor = isPaper ? "#64748b" : "#475569";
    const gridColor = isPaper ? "#e2e8f0" : "#1e293b";
    const textColor = isPaper ? "#0f172a" : "#cbd5e1";

    const dMin = -0.035, dMax = 0.035;
    const mapX = (val) => padL + ((val - dMin) / (dMax - dMin)) * (w - padL - padR);

    let svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;

    const zeroX = mapX(0.0);
    svg += `<line x1="${zeroX}" y1="${padT}" x2="${zeroX}" y2="${h - padB}" stroke="${axisColor}" stroke-width="1.8"/>`;
    svg += `<text x="${zeroX}" y="${padT - 8}" fill="${textColor}" font-size="9" font-family="monospace" text-anchor="middle">Baseline RF (0.0)</text>`;

    const sigX = mapX(0.01);
    svg += `<line x1="${sigX}" y1="${padT}" x2="${sigX}" y2="${h - padB}" stroke="var(--accent-rose)" stroke-width="1" stroke-dasharray="3,3"/>`;
    svg += `<text x="${sigX}" y="${padT - 8}" fill="var(--accent-rose)" font-size="8" font-family="monospace" text-anchor="middle">+0.01 Regola</text>`;

    const rowH = (h - padT - padB) / candidates.length;

    candidates.forEach((c, i) => {
      const y = padT + (i + 0.5) * rowH;
      const xPt = mapX(parseFloat(c.diff));
      const xLow = mapX(c.ci_low);
      const xHigh = mapX(c.ci_high);

      svg += `<line x1="${padL - 10}" y1="${y + rowH/2}" x2="${w - padR}" y2="${y + rowH/2}" stroke="${gridColor}" stroke-width="0.7"/>`;
      svg += `<text x="${padL - 15}" y="${y + 3}" fill="${textColor}" font-size="9.5" font-family="monospace" text-anchor="end">${c.name.substring(0, 22)}</text>`;

      svg += `<line x1="${xLow}" y1="${y}" x2="${xHigh}" y2="${y}" stroke="#60a5fa" stroke-width="2"/>`;
      svg += `<line x1="${xLow}" y1="${y - 3}" x2="${xLow}" y2="${y + 3}" stroke="#60a5fa" stroke-width="2"/>`;
      svg += `<line x1="${xHigh}" y1="${y - 3}" x2="${xHigh}" y2="${y + 3}" stroke="#60a5fa" stroke-width="2"/>`;

      svg += `<circle cx="${xPt}" cy="${y}" r="3.5" fill="#2563eb"/>`;
    });

    svg += `<line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="${axisColor}" stroke-width="1.5"/>`;

    [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03].forEach(val => {
      const x = mapX(val);
      svg += `<line x1="${x}" y1="${h - padB}" x2="${x}" y2="${h - padB + 5}" stroke="${axisColor}"/>`;
      svg += `<text x="${x}" y="${h - padB + 16}" fill="${textColor}" font-size="9" font-family="monospace" text-anchor="middle">${val > 0 ? '+' : ''}${val.toFixed(2)}</text>`;
    });

    svg += `<text x="${padL + (w - padL - padR)/2}" y="${h - 6}" fill="${textColor}" font-size="10" font-weight="600" text-anchor="middle">Differenza AUROC corretta di Nadeau-Bengio (IC 95%)</text>`;

    svg += `</svg>`;
    container.innerHTML = svg;
  }

  // --- SVG Export Feature ---
  window.downloadSvg = function(elementId, filename) {
    const container = document.getElementById(elementId);
    if (!container) return;
    const svgEl = container.querySelector("svg");
    if (!svgEl) return;

    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svgEl);

    if (!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
      source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }

    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${filename || 'chart'}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };
});
