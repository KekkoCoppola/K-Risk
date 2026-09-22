/**
 * K-Risk Research Platform — Scientific SPA Engine
 * High-Density, Modular Visualizer & Evidence-Based Verification Engine
 * Powered by pure JavaScript and SVG Vector Graphics
 * Supports Full Real-Time Internationalization (IT / EN)
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
    theme: localStorage.getItem("krisk_theme") || "light",
    lang: localStorage.getItem("krisk_lang") || "it",
    metricFormat: localStorage.getItem("krisk_metric_format") || "decimal",
    screenshotMode: false,
    selectedDcaThreshold: 0.07,
    activeModelKey: "random_forest"
  };

  // --- Initial Setup ---
  initTheme();
  initLanguageToggle();
  initNavigation();
  initTableFormatToggle();
  applyLanguage(state.lang);

  // =========================================================================
  // Theme Toggle (Dark / Light Mode)
  // =========================================================================
  function initTheme() {
    const isLight = state.theme === "light";
    document.body.classList.toggle("light-mode", isLight);
  }

  const themeToggle = document.getElementById("btn-theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const isLight = document.body.classList.toggle("light-mode");
      state.theme = isLight ? "light" : "dark";
      localStorage.setItem("krisk_theme", state.theme);
      renderAllCharts();
    });
  }

  // =========================================================================
  // Language Toggle (Italian / English)
  // =========================================================================
  function initLanguageToggle() {
    const langBtn = document.getElementById("btn-lang-toggle");
    if (langBtn) {
      langBtn.addEventListener("click", () => {
        const nextLang = state.lang === "it" ? "en" : "it";
        applyLanguage(nextLang);
      });
    }
  }

  function applyLanguage(lang) {
    state.lang = lang;
    localStorage.setItem("krisk_lang", lang);
    document.documentElement.lang = lang;

    // 1. Update Lang button appearance
    const flagTarget = document.getElementById("lang-flag-target");
    const labelTarget = document.getElementById("lang-label-target");
    const langBtn = document.getElementById("btn-lang-toggle");

    const ukFlagSvg = `<svg class="flag-icon" width="20" height="14" viewBox="0 0 60 40">
      <clipPath id="uk-flag-clip"><rect width="60" height="40" rx="3"/></clipPath>
      <g clip-path="url(#uk-flag-clip)">
        <path d="M0 0v40h60V0z" fill="#012169"/>
        <path d="M0 0l60 40m0-40L0 40" stroke="#fff" stroke-width="8"/>
        <path d="M0 0l60 40m0-40L0 40" stroke="#c8102e" stroke-width="4"/>
        <path d="M30 0v40M0 20h60" stroke="#fff" stroke-width="12"/>
        <path d="M30 0v40M0 20h60" stroke="#c8102e" stroke-width="8"/>
      </g>
    </svg>`;

    const itFlagSvg = `<svg class="flag-icon" width="20" height="14" viewBox="0 0 60 40">
      <clipPath id="it-flag-clip"><rect width="60" height="40" rx="3"/></clipPath>
      <g clip-path="url(#it-flag-clip)">
        <rect width="20" height="40" x="0" fill="#009246"/>
        <rect width="20" height="40" x="20" fill="#ffffff"/>
        <rect width="20" height="40" x="40" fill="#ce2b37"/>
      </g>
    </svg>`;

    // When viewing in Italian, the toggle shows the UK flag + EN (invitation to translate to English)
    // When viewing in English, the toggle shows the Italian flag + IT (invitation to return to Italian)
    if (lang === "it") {
      if (flagTarget) flagTarget.innerHTML = ukFlagSvg;
      if (labelTarget) labelTarget.textContent = "EN";
      if (langBtn) {
        langBtn.setAttribute("title", "Translate entire platform to English");
        langBtn.setAttribute("aria-label", "Translate to English");
      }
    } else {
      if (flagTarget) flagTarget.innerHTML = itFlagSvg;
      if (labelTarget) labelTarget.textContent = "IT";
      if (langBtn) {
        langBtn.setAttribute("title", "Traduci l'intera piattaforma in Italiano");
        langBtn.setAttribute("aria-label", "Traduci in Italiano");
      }
    }

    // 2. Translate static DOM elements
    const dict = window.KRISK_I18N?.translations?.[lang];
    if (dict) {
      if (dict.page_title) document.title = dict.page_title;
      const metaDesc = document.querySelector('meta[data-i18n-meta="page_description"]');
      if (metaDesc && dict.page_description) {
        metaDesc.setAttribute('content', dict.page_description);
      }

      document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if (dict[key] !== undefined) {
          el.innerHTML = dict[key];
        }
      });

      document.querySelectorAll("[data-i18n-title]").forEach(el => {
        const key = el.getAttribute("data-i18n-title");
        if (dict[key] !== undefined) {
          el.setAttribute("title", dict[key]);
        }
      });
    }

    // 3. Re-render dynamic views
    initScopeView();
    initPositioningView();
    initCohortView();
    initPhaseAView();
    initPhaseBView();
    initPhaseDView();
    initFinalTestView();
    updateTableFormatButtons();
    renderAllCharts();
  }

  // =========================================================================
  // Navigation
  // =========================================================================
  function initNavigation() {
    const navButtons = document.querySelectorAll(".pipeline-step-btn, .pill-nav-item, .nav-step-col");
    navButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const targetId = btn.getAttribute("data-tab");
        if (targetId) switchTab(targetId);
      });
    });
  }

  function switchTab(tabId) {
    state.currentTab = tabId;
    document.querySelectorAll(".pipeline-step-btn, .pill-nav-item").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".nav-step-col").forEach(col => {
      const isActive = col.getAttribute("data-tab") === tabId;
      col.classList.toggle("active", isActive);
      if (isActive && window.innerWidth <= 992) {
        col.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
      }
    });
    document.querySelectorAll(".tab-pane").forEach(pane => {
      pane.classList.toggle("active", pane.id === tabId);
    });
    setTimeout(() => {
      renderActiveTabCharts();
    }, 50);
  }

  // =========================================================================
  // Table Metric Format Toggle (Decimale / Percentuale)
  // =========================================================================
  function initTableFormatToggle() {
    updateTableFormatButtons();
    const formatButtons = document.querySelectorAll(".btn-toggle-format");
    formatButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        state.metricFormat = state.metricFormat === "percent" ? "decimal" : "percent";
        localStorage.setItem("krisk_metric_format", state.metricFormat);
        updateTableFormatButtons();
        // Re-render all data tables
        initPhaseAView();
        initPhaseBView();
        initPhaseDView();
        initPositioningView();
        initCohortView();
      });
    });
  }

  function updateTableFormatButtons() {
    const isPct = state.metricFormat === "percent";
    const isEn = state.lang === "en";
    const decTitle = isEn ? "View values in decimal (0.00)" : "Visualizza valori in decimale (0.00)";
    const pctTitle = isEn ? "View values in percent (%)" : "Visualizza valori in percentuale (%)";

    document.querySelectorAll(".btn-toggle-format").forEach(btn => {
      btn.classList.toggle("mode-percent", isPct);
      btn.classList.toggle("mode-decimal", !isPct);
      btn.setAttribute("title", isPct ? decTitle : pctTitle);
      btn.setAttribute("aria-label", isPct ? decTitle : pctTitle);
    });
  }

  // =========================================================================
  // Section 00: Scope
  // =========================================================================
  function initScopeView() {
    const scopeContainer = document.getElementById("scope-details-target");
    if (!scopeContainer) return;

    const langData = window.KRISK_I18N?.data?.[state.lang]?.scope_cards;
    if (langData) {
      scopeContainer.innerHTML = `
        <div class="scope-grid">
          ${langData.map(c => `
            <div class="scope-item">
              <h4>${c.title}</h4>
              <p>${c.detail}</p>
            </div>
          `).join("")}
        </div>
      `;
      return;
    }

    // Fallback to data.js
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
  // Section 06: Positioning & State of the Art
  // =========================================================================
  function initPositioningView() {
    const isEn = state.lang === "en";
    const posData = window.KRISK_I18N?.data?.[state.lang]?.positioning;

    // Continuum Flow Schema
    const continuumContainer = document.getElementById("continuum-flow-target");
    if (continuumContainer) {
      const flow = posData?.continuum_flow || data.positioning.continuum;
      continuumContainer.innerHTML = flow.map(node => `
        <div class="continuum-node ${node.highlight ? 'highlighted' : ''}">
          <div>
            <span class="continuum-step-badge">${isEn ? 'Phase' : 'Fase'} ${node.step}</span>
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
      const matrix = posData?.state_of_art_matrix || data.positioning.state_of_art_matrix;
      const isPct = state.metricFormat === "percent";
      tableContainer.innerHTML = matrix.map(m => {
        const aurocNum = parseFloat(m.auroc);
        const auroc = isPct && !isNaN(aurocNum) ? (aurocNum * 100).toFixed(1) + "%" : m.auroc;
        return `
        <tr class="${m.status === 'krisk' ? 'highlight' : ''}">
          <td>
            <strong>${m.model}</strong>
            ${m.status === 'krisk' ? '<span class="badge badge-success">K-Risk</span>' : ''}
          </td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${m.target}</td>
          <td>${m.renal_exams_used}</td>
          <td class="num" style="font-weight: 700;">${auroc}</td>
          <td style="font-size: 0.82rem; color: var(--text-secondary);">${m.role}</td>
        </tr>
      `;
      }).join("");
    }
  }

  // =========================================================================
  // Section 01: Cohort & Data Integrity
  // =========================================================================
  function initCohortView() {
    const container = document.getElementById("kdigo-grid-target");
    if (!container) return;

    const isEn = state.lang === "en";
    const gfrHeaders = ["G1 (≥90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"];
    const acrHeaders = ["A1 (<30)", "A2 (30-300)", "A3 (>300)"];

    let html = `<div class="kdigo-grid">`;
    html += `<div class="kdigo-header-cell">eGFR \ ACR</div>`;
    acrHeaders.forEach(acr => {
      html += `<div class="kdigo-header-cell">${acr}</div>`;
    });

    const gfrKeys = ["G1", "G2", "G3a", "G3b", "G4", "G5"];
    const acrKeys = ["A1", "A2", "A3"];

    const riskLabels = {
      it: { basso: "Basso", moderato: "Moderato", alto: "Alto", molto_alto: "Molto Alto" },
      en: { basso: "Low", moderato: "Moderate", alto: "High", molto_alto: "Very High" }
    };

    gfrKeys.forEach((gKey, idx) => {
      html += `<div class="kdigo-header-cell">${gfrHeaders[idx]}</div>`;
      acrKeys.forEach(aKey => {
        const item = data.cohort.kdigo_matrix.find(m => m.gfr === gKey && m.acr === aKey);
        const count = item ? item.count : 0;
        const rawRisk = item ? item.risk : "basso";
        const riskKey = rawRisk.replace(" ", "_");
        const riskClass = `risk-${riskKey}`;
        const displayedRisk = riskLabels[state.lang]?.[riskKey] || rawRisk;

        const titleText = isEn
          ? `${gKey} × ${aKey} | ${displayedRisk.toUpperCase()} | Subjects: ${count}`
          : `${gKey} × ${aKey} | ${displayedRisk.toUpperCase()} | Soggetti: ${count}`;

        html += `
          <div class="kdigo-cell ${riskClass}" title="${titleText}">
            <span class="cell-count">${count}</span>
            <span class="cell-risk">${displayedRisk}</span>
          </div>
        `;
      });
    });

    html += `</div>`;
    container.innerHTML = html;

    // Render exclusions list
    const exclContainer = document.getElementById("leakage-exclusions-target");
    const leakCats = window.KRISK_I18N?.data?.[state.lang]?.leakage_guard?.excluded_categories || data.leakage_guard.excluded_categories;
    if (exclContainer && leakCats) {
      exclContainer.innerHTML = leakCats.map(cat => `
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
      const isPct = state.metricFormat === "percent";
      const adoptedLabel = isEn ? "ADOPTED" : "ADOTTATO";
      impContainer.innerHTML = data.imputation.methods.map(m => {
        const prauc = isPct ? (m.prauc * 100).toFixed(2) + "%" : m.prauc.toFixed(4);
        const auroc = isPct ? (m.auroc * 100).toFixed(2) + "%" : m.auroc.toFixed(4);
        return `
        <tr class="${m.selected ? 'highlight' : ''}">
          <td><strong>${m.name}</strong> ${m.selected ? `<span class="badge badge-success">${adoptedLabel}</span>` : ''}</td>
          <td class="num">${m.rmse.toFixed(4)} <span style="font-size:0.7rem; color:var(--text-muted);">±${m.rmse_se.toFixed(4)}</span></td>
          <td class="num">${m.mae.toFixed(4)}</td>
          <td class="num">${prauc}</td>
          <td class="num">${auroc}</td>
          <td class="num">${m.time_s.toFixed(1)}s</td>
        </tr>
      `;
      }).join("");
    }
  }

  // =========================================================================
  // Section 02: Phase A (Screening Models & Interpretation)
  // =========================================================================
  function initPhaseAView() {
    const isPct = state.metricFormat === "percent";
    const phaseADescs = window.KRISK_I18N?.data?.[state.lang]?.phase_a_descriptions || {};
    const tableTarget = document.getElementById("phase-a-table-target");
    if (tableTarget) {
      tableTarget.innerHTML = data.phase_a.models.map(m => {
        const oofAuc = isPct ? (m.oof_auc * 100).toFixed(1) + "%" : m.oof_auc.toFixed(3);
        const oofCi = isPct 
          ? `[${(m.oof_auc_ci[0] * 100).toFixed(1)}%-${(m.oof_auc_ci[1] * 100).toFixed(1)}%]`
          : `[${m.oof_auc_ci[0].toFixed(3)}-${m.oof_auc_ci[1].toFixed(3)}]`;
        const oofPrauc = isPct ? (m.oof_prauc * 100).toFixed(1) + "%" : m.oof_prauc.toFixed(3);
        const testAuc = isPct ? (m.test_auc * 100).toFixed(1) + "%" : m.test_auc.toFixed(3);
        const testCi = isPct
          ? `[${(m.test_auc_ci[0] * 100).toFixed(1)}%-${(m.test_auc_ci[1] * 100).toFixed(1)}%]`
          : `[${m.test_auc_ci[0].toFixed(3)}-${m.test_auc_ci[1].toFixed(3)}]`;
        const spec = isPct ? (m.spec_at_90sens * 100).toFixed(1) + "%" : m.spec_at_90sens.toFixed(3);
        const alertRate = isPct ? (m.alert_rate * 100).toFixed(1) + "%" : m.alert_rate.toFixed(3);
        const desc = phaseADescs[m.name] || m.description;

        return `
        <tr>
          <td>
            <strong>${m.name}</strong>
            <div style="font-size: 0.75rem; color: var(--text-muted);">${desc}</div>
          </td>
          <td class="num"><strong>${oofAuc}</strong> <span style="font-size: 0.7rem; color: var(--text-muted);">${oofCi}</span></td>
          <td class="num"><strong>${oofPrauc}</strong></td>
          <td class="num highlight"><strong>${testAuc}</strong> <span style="font-size: 0.7rem; color: var(--text-muted);">${testCi}</span></td>
          <td class="num">${spec}</td>
          <td class="num">${alertRate}</td>
          <td class="num"><span class="badge badge-neutral">${m.n_features}</span></td>
        </tr>
      `;
      }).join("");
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
  // Section 03: Phase B (The Imbalance Fallacy)
  // =========================================================================
  function initPhaseBView() {
    const isPct = state.metricFormat === "percent";
    const tableTarget = document.getElementById("phase-b-comparison-target");
    if (tableTarget) {
      tableTarget.innerHTML = data.phase_b.naive_vs_real.map(r => {
        const isBaseline = r.technique.includes("none");
        const rec = isPct ? (r.naive_recall * 100).toFixed(1) + "%" : r.naive_recall.toFixed(3);
        const prec = isPct ? (r.naive_prec * 100).toFixed(1) + "%" : r.naive_prec.toFixed(3);
        const spec = isPct ? (r.naive_spec * 100).toFixed(1) + "%" : r.naive_spec.toFixed(3);

        return `
          <tr class="${isBaseline ? 'highlight' : ''}">
            <td><strong>${r.technique}</strong> ${isBaseline ? '<span class="badge badge-neutral">Baseline</span>' : ''}</td>
            <td class="num" style="color: ${r.naive_recall > 0.3 ? 'var(--accent-cyan)' : 'inherit'}; font-weight: 600;">${rec}</td>
            <td class="num">${prec}</td>
            <td class="num">${spec}</td>
            <td class="num" style="background: var(--bg-surface-elevated); font-weight: 700;">${r.severe_caught} / ${r.severe_total}</td>
            <td class="num" style="background: var(--bg-surface-elevated); font-weight: 600; color: ${r.diff > 0 ? 'var(--accent-emerald)' : r.diff < 0 ? 'var(--accent-rose)' : 'inherit'};">${r.diff}</td>
            <td class="num" style="background: var(--bg-surface-elevated); font-family: var(--font-mono);">${r.p_holm.toFixed(2)} <span class="badge badge-neutral" style="font-size:0.65rem;">n.s.</span></td>
          </tr>
        `;
      }).join("");
    }
  }

  // =========================================================================
  // Section 04: Phase D (Ceiling Diagnostics)
  // =========================================================================
  function initPhaseDView() {
    const isPct = state.metricFormat === "percent";
    const isEn = state.lang === "en";
    const outcomeText = isEn ? "SURPASSED: NO" : "SUPERATO: NO";
    const tableTarget = document.getElementById("phase-d-candidates-target");
    if (tableTarget) {
      tableTarget.innerHTML = data.phase_d.candidates.map(c => {
        const auc = isPct ? (c.auc * 100).toFixed(2) + "%" : c.auc.toFixed(4);
        const diffNum = parseFloat(c.diff);
        const diff = isPct 
          ? (diffNum >= 0 ? "+" : "") + (diffNum * 100).toFixed(2) + "%" 
          : c.diff;
        const ci = isPct
          ? `[${(c.ci_low * 100).toFixed(2)}%, ${(c.ci_high * 100).toFixed(2)}%]`
          : `[${c.ci_low.toFixed(4)}, ${c.ci_high.toFixed(4)}]`;

        return `
        <tr>
          <td><strong>${c.name}</strong></td>
          <td><span class="badge badge-neutral">${c.type}</span></td>
          <td class="num"><strong>${auc}</strong></td>
          <td class="num" style="color: ${c.diff.startsWith('+') ? 'var(--accent-cyan)' : 'var(--accent-rose)'};">${diff}</td>
          <td class="num" style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">${ci}</td>
          <td class="num" style="font-family: var(--font-mono);">${c.p_holm.toFixed(2)}</td>
          <td style="text-align: center;"><span class="badge badge-danger">${outcomeText}</span></td>
        </tr>
      `;
      }).join("");
    }
  }

  // =========================================================================
  // Section 05: Final Test Lockbox & Clinical Utility
  // =========================================================================
  function initFinalTestView() {
    const claimsTarget = document.getElementById("test-claims-target");
    const isEn = state.lang === "en";
    const claims = window.KRISK_I18N?.data?.[state.lang]?.final_test?.claims;
    const aPrioriText = isEn ? "A priori criterion:" : "Criterio a priori:";

    if (claimsTarget && claims) {
      claimsTarget.innerHTML = claims.map(c => `
        <div class="claim-card">
          <div class="claim-header">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
              <span class="claim-id-badge">CLAIM ${c.id}</span>
              <h4 class="claim-title">${c.title}</h4>
            </div>
            <span class="badge badge-success">✓ ${c.verdict}</span>
          </div>
          <div class="claim-detail">${c.detail}</div>
          <div class="claim-criterion"><strong>${aPrioriText}</strong> ${c.criterion}</div>
        </div>
      `).join("");
    }

    // Decision Curve Slider Handler
    const slider = document.getElementById("dca-slider");
    const sliderDisplay = document.getElementById("slider-val-display");
    if (slider) {
      slider.oninput = (e) => {
        const val = parseFloat(e.target.value);
        state.selectedDcaThreshold = val;
        if (sliderDisplay) sliderDisplay.textContent = `${(val * 100).toFixed(1)}%`;
        updateDcaDynamicMetrics(val);
      };
      updateDcaDynamicMetrics(state.selectedDcaThreshold);
    }
  }

  function updateDcaDynamicMetrics(threshold) {
    const closest = data.clinical_utility.operating_points.reduce((prev, curr) => 
      Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev
    );

    const container = document.getElementById("dca-dynamic-stats");
    if (!container) return;

    const isEn = state.lang === "en";
    const savingsVal = Math.round(closest.cost_savings_per_100).toLocaleString(isEn ? 'en-US' : 'it-IT');
    const costPerCaseVal = Math.round(closest.cost_per_case).toLocaleString(isEn ? 'en-US' : 'it-IT');

    const clinicalMeaning = isEn
      ? (threshold <= 0.05 ? "Pre-screening triage focus (High sensitivity)" :
         threshold <= 0.10 ? "Recommended standard clinical practice" :
         "Restricted resources / Low tolerance for false alerts")
      : closest.clinical_meaning;

    const card1Title = isEn ? "Decision Threshold (P<sub>t</sub>)" : "Soglia Decisionale (P<sub>t</sub>)";
    const card2Title = isEn ? "Urinary Exams Spared / 100 pts" : "Esami Urinari Evitati / 100 sogg.";
    const card2Sub = isEn ? "Compared to indiscriminate 'Test All' practice" : "Rispetto alla prassi indiscriminata 'Testare Tutti'";
    const card3Title = isEn ? "Economic Savings / 100 pts" : "Risparmio Economico / 100 sogg.";
    const card3Sub = isEn ? "Estimate based on 49 € per ACR test (Cusick 2023)" : "Stima basata su 49 € per test ACR (Cusick 2023)";
    const card4Title = isEn ? "Cost per Confirmed Case" : "Costo per Caso Confermato";
    const card4Sub = isEn ? `Observed clinical sensitivity: ${(closest.recall * 100).toFixed(1)}%` : `Sensibilità clinica osservata: ${(closest.recall * 100).toFixed(1)}%`;

    container.innerHTML = `
      <div class="card" style="border-left: 3px solid var(--accent-cyan);">
        <div class="card-subtitle">${card1Title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-cyan); margin: 0.3rem 0;">${(threshold * 100).toFixed(1)}%</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">${clinicalMeaning}</div>
      </div>
      <div class="card" style="border-left: 3px solid var(--accent-emerald);">
        <div class="card-subtitle">${card2Title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-emerald); margin: 0.3rem 0;">${closest.tests_avoided_per_100.toFixed(1)}</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">${card2Sub}</div>
      </div>
      <div class="card" style="border-left: 3px solid var(--accent-amber);">
        <div class="card-subtitle">${card3Title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-amber); margin: 0.3rem 0;">${savingsVal} €</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">${card3Sub}</div>
      </div>
      <div class="card" style="border-left: 3px solid var(--accent-blue-light);">
        <div class="card-subtitle">${card4Title}</div>
        <div style="font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); color: var(--accent-blue-light); margin: 0.3rem 0;">${costPerCaseVal} €</div>
        <div style="font-size: 0.78rem; color: var(--text-muted);">${card4Sub}</div>
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

    const isEn = state.lang === "en";
    const w = 480, h = 300, pad = 45;
    const isLight = document.body.classList.contains("light-mode");
    const axisColor = isLight ? "#64748b" : "#475569";
    const gridColor = isLight ? "#e2e8f0" : "#1e293b";
    const textColor = isLight ? "#0f172a" : "#cbd5e1";

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

    const xLabel = isEn ? "1 - Specificity (FPR)" : "1 - Specificità (FPR)";
    const yLabel = isEn ? "Sensitivity (TPR / Recall)" : "Sensibilità (TPR / Recall)";

    svg += `<text x="${w / 2 + 10}" y="${h - 8}" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle">${xLabel}</text>`;
    svg += `<text x="-${h / 2 - 10}" y="14" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90)">${yLabel}</text>`;

    svg += `</svg>`;
    container.innerHTML = svg;
  }

  // --- SVG Chart 2: Decision Curve Analysis (DCA Net Benefit) ---
  function renderDcaChart() {
    const container = document.getElementById("dca-chart-svg");
    if (!container) return;

    const isEn = state.lang === "en";
    const w = 520, h = 320, pad = 50;
    const isLight = document.body.classList.contains("light-mode");
    const axisColor = isLight ? "#64748b" : "#475569";
    const gridColor = isLight ? "#e2e8f0" : "#1e293b";
    const textColor = isLight ? "#0f172a" : "#cbd5e1";

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

    const xLabel = isEn ? "Clinical Decision Threshold (Threshold Probability Pt)" : "Soglia Decisionale Clinica (Threshold Probability Pt)";
    const yLabel = isEn ? "Standardized Net Benefit (Vickers)" : "Net Benefit (Vickers)";

    svg += `<text x="${w / 2 + 10}" y="${h - 8}" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle">${xLabel}</text>`;
    svg += `<text x="-${h / 2 - 10}" y="14" fill="${textColor}" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90)">${yLabel}</text>`;

    svg += `</svg>`;
    container.innerHTML = svg;
  }

  // --- SVG Chart 3: Forest Plot (Nadeau-Bengio Intervals) ---
  function renderForestPlot() {
    const container = document.getElementById("forest-plot-svg");
    if (!container) return;

    const isEn = state.lang === "en";
    const candidates = data.phase_d.candidates;
    const w = 540, h = 340, padL = 170, padR = 40, padT = 30, padB = 40;
    const isLight = document.body.classList.contains("light-mode");
    const axisColor = isLight ? "#64748b" : "#475569";
    const gridColor = isLight ? "#e2e8f0" : "#1e293b";
    const textColor = isLight ? "#0f172a" : "#cbd5e1";

    const dMin = -0.035, dMax = 0.035;
    const mapX = (val) => padL + ((val - dMin) / (dMax - dMin)) * (w - padL - padR);

    let svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;

    const zeroX = mapX(0.0);
    svg += `<line x1="${zeroX}" y1="${padT}" x2="${zeroX}" y2="${h - padB}" stroke="${axisColor}" stroke-width="1.8"/>`;
    svg += `<text x="${zeroX}" y="${padT - 8}" fill="${textColor}" font-size="9" font-family="monospace" text-anchor="middle">Baseline RF (0.0)</text>`;

    const sigX = mapX(0.01);
    const ruleLabel = isEn ? "+0.01 Rule" : "+0.01 Regola";
    svg += `<line x1="${sigX}" y1="${padT}" x2="${sigX}" y2="${h - padB}" stroke="var(--accent-rose)" stroke-width="1" stroke-dasharray="3,3"/>`;
    svg += `<text x="${sigX}" y="${padT - 8}" fill="var(--accent-rose)" font-size="8" font-family="monospace" text-anchor="middle">${ruleLabel}</text>`;

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

    const axisTitle = isEn 
      ? "Nadeau-Bengio Adjusted AUROC Difference (95% CI)" 
      : "Differenza AUROC corretta di Nadeau-Bengio (IC 95%)";
    svg += `<text x="${padL + (w - padL - padR)/2}" y="${h - 6}" fill="${textColor}" font-size="10" font-weight="600" text-anchor="middle">${axisTitle}</text>`;

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
