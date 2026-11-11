// K-Risk Research Platform — Scientific Dataset & Results Database
// Rigorosamente allineato e certificato sui dati di esecuzione in src/, analytics/ e analisi_impatto/

window.KRISK_DATA = {
  metadata: {
    project_name: "K-Risk",
    full_title: "K-Risk: Pre-screening Opportunistico Non-Invasivo della Malattia Renale Cronica",
    tagline: "Machine Learning Clinico, Triage di 1° Livello, Apprendimento Non Supervisionato e Validazione Trans-Popolazione",
    version: "1.0.0-thesis-final",
    commit: "9b4e92c",
    timestamp: "2026-11-11T23:30:00",
    total_cohort: 5801,
    train_n: 4350,
    test_n: 1451,
    test_ratio: 0.25,
    seed: 42,
    formula_egfr: "CKD-EPI 2021 (senza coefficiente etnico)",
    target_definition: "eGFR < 60 ml/min/1.73 m²  OPPURE  ACR ≥ 30 mg/g",
    features_total: 81,
    features_numeric: 60,
    features_categorical: 14,
    features_excluded: 85,
    external_cohort_n: 34494,
    external_cohort_name: "NHANES 2005–2018 (Stati Uniti, CDC NCHS)",
    university: "Università degli Studi di Salerno",
    department: "Dipartimento di Informatica",
    candidate: "Francesco Coppola",
    supervisors: "Dott.ssa Fabiola De Marco, Dott. Domenico Rossi",
    academic_year: "2026/2027"
  },

  scope: {
    in_brief: "Stimare la presenza di marcatori di malattia renale cronica (KDIGO) SENZA ricorrere ad esami renali specifici, definendo un triage di primo livello per stabilire la priorità di prescrizione degli esami di laboratorio.",
    clinical_problem: "La malattia renale cronica (CKD) colpisce oltre il 10% della popolazione adulta mondiale, ma fino al 90% delle persone affette non ne è a conoscenza (CDC 2026, sorveglianza Wang et al. 2023). Negli adulti senza diabete noto, il test urinario dell'albumina (ACR) viene prescritto raramente, determinando ritardi diagnostici significativi.",
    target_definition_details: "Classificazione binaria conforme ai criteri KDIGO: y = 1 se ACR ≥ 30 mg/g oppure eGFR < 60 ml/min/1.73 m². Prevalenza osservata del 9.77% (567 positivi su 5.801 soggetti).",
    input_features: "Dati anagrafici (età, sesso), antropometrici (BMI, vita, fianchi), pressori (sistolica, diastolica, frequenza), ematologici ed ematochimici di routine (glicemia, HbA1c, lipidi, enzimi epatici, emocromo completo, peptide C).",
    strict_constraint: "Zero esami renali in input: esclusi categoricamente creatinina sierica (SCRE), albumina urinaria (UmALB, UMAUCR), azotemia (BUN), eGFR e tutte le diagnosi renali pregresse.",
    clinical_role: "A monte del laboratorio: K-Risk è uno strumento di triage pre-test per la medicina generale e le visite di controllo, volto a ottimizzare le risorse sanitarie prescrivendo gli esami di secondo livello (ACR ed eGFR) solo ai soggetti a reale rischio.",
    what_it_is_not: "Non sostituisce la diagnosi clinica formale (che richiede test di laboratorio confermati oltre 3 mesi secondo le linee guida KDIGO 2024); non è un modello di prognosi dialitica."
  },

  cohort: {
    train: {
      n: 4350,
      positives: 425,
      prevalence_pct: 9.77,
      diabetic_n: 263,
      diabetic_positives: 68,
      diabetic_prevalence_pct: 25.86,
      non_diabetic_n: 4087,
      non_diabetic_positives: 357,
      non_diabetic_prevalence_pct: 8.74,
      kdigo_counts: {
        basso: 3925,
        moderato: 354,
        alto: 50,
        molto_alto: 21
      }
    },
    test: {
      n: 1451,
      positives: 142,
      prevalence_pct: 9.79,
      diabetic_n: 88,
      diabetic_positives: 23,
      diabetic_prevalence_pct: 26.14,
      non_diabetic_n: 1363,
      non_diabetic_positives: 119,
      non_diabetic_prevalence_pct: 8.73,
      kdigo_counts: {
        basso: 1309,
        moderato: 119,
        alto: 17,
        molto_alto: 6
      }
    },
    total: {
      n: 5801,
      positives: 567,
      prevalence_pct: 9.77,
      kdigo_counts: {
        basso: 5234,
        moderato: 473,
        alto: 67,
        molto_alto: 27
      }
    },
    external_nhanes: {
      n: 34494,
      population: "Adulti statunitensi (NHANES cicli 2005–2018)",
      mortality_followup: "CDC National Death Index (NDI) fino al 31 dicembre 2019",
      kdigo_positives: 4710,
      healthy_kdigo_low: 28433,
      features_mapped: 74
    },
    kdigo_matrix: [
      { gfr: "G1", acr: "A1", risk: "basso", count: 2842, egfr_range: "≥ 90", acr_range: "< 30" },
      { gfr: "G1", acr: "A2", risk: "moderato", count: 261, egfr_range: "≥ 90", acr_range: "30–300" },
      { gfr: "G1", acr: "A3", risk: "alto", count: 32, egfr_range: "≥ 90", acr_range: "> 300" },
      { gfr: "G2", acr: "A1", risk: "basso", count: 2392, egfr_range: "60–89", acr_range: "< 30" },
      { gfr: "G2", acr: "A2", risk: "moderato", count: 169, egfr_range: "60–89", acr_range: "30–300" },
      { gfr: "G2", acr: "A3", risk: "alto", count: 39, egfr_range: "60–89", acr_range: "> 300" },
      { gfr: "G3a", acr: "A1", risk: "moderato", count: 23, egfr_range: "45–59", acr_range: "< 30" },
      { gfr: "G3a", acr: "A2", risk: "alto", count: 14, egfr_range: "45–59", acr_range: "30–300" },
      { gfr: "G3a", acr: "A3", risk: "molto alto", count: 12, egfr_range: "45–59", acr_range: "> 300" },
      { gfr: "G3b", acr: "A1", risk: "alto", count: 2, egfr_range: "30–44", acr_range: "< 30" },
      { gfr: "G3b", acr: "A2", risk: "molto alto", count: 4, egfr_range: "30–44", acr_range: "30–300" },
      { gfr: "G3b", acr: "A3", risk: "molto alto", count: 6, egfr_range: "30–44", acr_range: "> 300" },
      { gfr: "G4", acr: "A1", risk: "molto alto", count: 0, egfr_range: "15–29", acr_range: "< 30" },
      { gfr: "G4", acr: "A2", risk: "molto alto", count: 2, egfr_range: "15–29", acr_range: "30–300" },
      { gfr: "G4", acr: "A3", risk: "molto alto", count: 1, egfr_range: "15–29", acr_range: "> 300" },
      { gfr: "G5", acr: "A1", risk: "molto alto", count: 0, egfr_range: "< 15", acr_range: "< 30" },
      { gfr: "G5", acr: "A2", risk: "molto alto", count: 1, egfr_range: "< 15", acr_range: "30–300" },
      { gfr: "G5", acr: "A3", risk: "molto alto", count: 1, egfr_range: "< 15", acr_range: "> 300" }
    ]
  },

  phase_a: {
    models: [
      {
        key: "dummy",
        name: "Dummy (Maggioranza)",
        family: "Baseline",
        auroc: 0.500,
        auroc_ci: [0.500, 0.500],
        pr_auc: 0.098,
        brier: 0.088,
        severe_recall: "0/71 (0.0%)",
        color: "#94a3b8"
      },
      {
        key: "lr_scored",
        name: "Logistica SCORED",
        family: "Modello Clinico Ristimato",
        auroc: 0.675,
        auroc_ci: [0.648, 0.702],
        pr_auc: 0.228,
        brier: 0.085,
        severe_recall: "62/71 (87.3%)",
        color: "#f59e0b"
      },
      {
        key: "lr_penalized",
        name: "Logistica Elastic Net",
        family: "Modello Lineare Regolarizzato",
        auroc: 0.696,
        auroc_ci: [0.669, 0.722],
        pr_auc: 0.248,
        brier: 0.083,
        severe_recall: "64/71 (90.1%)",
        color: "#3b82f6"
      },
      {
        key: "random_forest",
        name: "Random Forest",
        family: "Ensemble Bagging",
        auroc: 0.703,
        auroc_ci: [0.676, 0.731],
        pr_auc: 0.258,
        brier: 0.082,
        severe_recall: "65/71 (91.5%)",
        color: "#10b981"
      },
      {
        key: "xgboost",
        name: "XGBoost (3–12)",
        family: "Gradient Boosting",
        auroc: 0.698,
        auroc_ci: [0.671, 0.725],
        pr_auc: 0.252,
        brier: 0.083,
        severe_recall: "65/71 (91.5%)",
        color: "#8b5cf6"
      }
    ],
    operating_points: [
      { name: "Max Sensibilità (90%)", threshold: 0.052, sens: 0.901, spec: 0.384, cases_caught: 383, alert_rate: 0.654 },
      { name: "Soglia Clinica Youden (7.7%)", threshold: 0.077, sens: 0.812, spec: 0.485, cases_caught: 345, alert_rate: 0.551 },
      { name: "Bilanciata (10%)", threshold: 0.100, sens: 0.664, spec: 0.648, cases_caught: 282, alert_rate: 0.382 },
      { name: "Alta Specificità (15%)", threshold: 0.150, sens: 0.386, spec: 0.835, cases_caught: 164, alert_rate: 0.187 }
    ]
  },

  phase_b: {
    techniques: [
      { key: "none", name: "Dati Naturali (None)", auroc: 0.703, recall_naive_05: 0.044, recall_calibrated: 0.901, severe_caught: 65, status: "Riferimento Ideale" },
      { key: "class_weight", name: "Pesi Inversi di Classe", auroc: 0.699, recall_naive_05: 0.556, recall_calibrated: 0.898, severe_caught: 64, status: "Soglia Traslata" },
      { key: "level_weight", name: "Pesi per Livello KDIGO", auroc: 0.700, recall_naive_05: 0.570, recall_calibrated: 0.899, severe_caught: 65, status: "Soglia Traslata" },
      { key: "undersampling", name: "Undersampling Random", auroc: 0.692, recall_naive_05: 0.646, recall_calibrated: 0.892, severe_caught: 63, status: "Perdita Varianza" },
      { key: "oversampling", name: "Oversampling Random", auroc: 0.697, recall_naive_05: 0.502, recall_calibrated: 0.895, severe_caught: 64, status: "Rischio Overfit" },
      { key: "smote_nc", name: "SMOTE-NC", auroc: 0.698, recall_naive_05: 0.403, recall_calibrated: 0.896, severe_caught: 64, status: "Nessun Guadagno" },
      { key: "smote_nc_level", name: "SMOTE-NC Condizionato", auroc: 0.699, recall_naive_05: 0.433, recall_calibrated: 0.897, severe_caught: 64, status: "Nessun Guadagno" },
      { key: "ctgan", name: "CTGAN (Generativo WGAN)", auroc: 0.672, recall_naive_05: 0.214, recall_calibrated: 0.865, severe_caught: 58, status: "Degradazione" },
      { key: "ctgan_level", name: "CTGAN Condizionato", auroc: 0.675, recall_naive_05: 0.235, recall_calibrated: 0.868, severe_caught: 59, status: "Degradazione" }
    ],
    finding: "Il presunto incremento di recall fornito dal bilanciamento delle classi (dal 4.4% al 64.6% alla soglia convenzionale 0.50) è un'illusione ottica dovuta alla traslazione della distribuzione delle probabilità. Ricalibrate le probabilità (Platt scaling) o valutando a parità di tasso di allarme, nessuna tecnica di bilanciamento supera i dati naturali."
  },

  phase_c: {
    diabetic_subgroup: {
      n: 351,
      positives: 91,
      prevalence_pct: 25.86,
      auroc_rf: 0.718,
      auroc_xgb: 0.714,
      alert_rate_at_global_10: "97.7% – 100.0%",
      clinical_implication: "Nella coorte con diabete, il modello segnala la quasi totalità dei pazienti. Ciò conferma che il triage ML non è necessario nei diabetici (che già beneficiano dello screening universale annuale da linee guida ADA/KDIGO), mentre esprime il suo massimo valore di costo-efficacia nei non diabetici."
    }
  },

  phase_d: {
    candidates: [
      { name: "TabPFN (Transformer tabulare)", diff: "+0.007", ci_low: -0.006, ci_high: +0.020, auroc: 0.710 },
      { name: "Ensemble Mean (Meta-modello)", diff: "+0.005", ci_low: -0.005, ci_high: +0.015, auroc: 0.708 },
      { name: "CatBoost (Ordered boosting)", diff: "+0.004", ci_low: -0.007, ci_high: +0.015, auroc: 0.707 },
      { name: "Tri-Ensemble (Top-21 feature)", diff: "+0.003", ci_low: -0.007, ci_high: +0.013, auroc: 0.706 },
      { name: "Explainable Boosting Machine (EBM)", diff: "+0.001", ci_low: -0.011, ci_high: +0.013, auroc: 0.704 },
      { name: "Random Forest (Baseline Fase A)", diff: "0.000", ci_low: 0.000, ci_high: 0.000, auroc: 0.703 },
      { name: "LightGBM", diff: "-0.002", ci_low: -0.014, ci_high: +0.010, auroc: 0.701 },
      { name: "Target Decomposition (Modelli separati)", diff: "-0.005", ci_low: -0.018, ci_high: +0.008, auroc: 0.698 },
      { name: "Continuous Target (Regressione log-ACR)", diff: "-0.008", ci_low: -0.021, ci_high: +0.005, auroc: 0.695 },
      { name: "Logistica Penalizzata (All scaled)", diff: "-0.011", ci_low: -0.023, ci_high: +0.001, auroc: 0.692 }
    ],
    controls: {
      positive_control_auroc: 0.933,
      positive_control_desc: "Inserendo l'albumina urinaria tra le feature (violando il vincolo non-invasivo), l'AUROC balza istantaneamente a 0.933.",
      semi_synthetic_75: 0.751,
      semi_synthetic_80: 0.803,
      ceiling_conclusion: "Tutti i 12 candidati avanzati rimangono statisticamente confinati tra 0.692 e 0.710 (nessuno supera la soglia di significatività pre-registrata di +0.01 rispetto alla baseline). Il tetto è di natura puramente biologica: il sangue non contiene informazione sufficiente per rilevare la microalbuminuria isolata precoce senza esame delle urine."
    }
  },

  unsupervised: {
    title: "Apprendimento Non Supervisionato & 'Soglia AI'",
    subtitle: "Esperimento richiesto dai relatori: rimozione di formule KDIGO, soglie cliniche ed esami renali per comprendere il ragionamento autonomo dell'AI",
    splits: [
      {
        name: "Split T (Tutte le 190 colonne grezze)",
        features_n: 190,
        content: "Include tutte le colonne del dataset, comprese quelle che contengono la definizione o le componenti renali.",
        auroc_with_kdigo: 0.842,
        finding: "L'AI ricostruisce spontaneamente la separazione clinica con un'accuratezza dell'84%."
      },
      {
        name: "Split R-Regola (Solo colonne con la regola diagnostica)",
        features_n: 4,
        content: "Creatinina, eGFR, ACR e albumina urinaria.",
        auroc_with_kdigo: 0.991,
        finding: "Riproduzione quasi perfetta (AUROC 0.991) della regola formale KDIGO."
      },
      {
        name: "Split P-Sangue (67 feature rigorose senza esami renali)",
        features_n: 67,
        content: "Solo parametri ematochimici e clinici ordinari, zero esami renali.",
        auroc_with_kdigo: 0.654,
        finding: "In assenza di etichette e senza esami renali, l'AI non si allinea allo scalino artificiale di KDIGO, ma ricalca un asse metabolico continuo dominato dall'età e dall'acido urico."
      }
    ],
    continuous_gradient: {
      finding: "Il punteggio non supervisionato cresce in modo lineare e continuo con l'albuminuria (+0.092 deviazioni standard per ogni raddoppio di ACR, IC 95%: 0.080–0.104) GIÀ AL DI SOTTO DEI 30 mg/g.",
      break_point_analysis: "Nessun punto di rottura statistico (elbow / knot) identificato a 30 mg/g. L'AI scopre che la biologia del danno renale è un continuum senza gradini artificiali."
    }
  },

  external_validation: {
    cohort_name: "NHANES 2005–2018 (National Health and Nutrition Examination Survey)",
    sample_size: 34494,
    positives_count: 4710,
    transfer_method: "Zero-Shot Puro (nessun riaddestramento, pesi congelati sulla coorte Jinuo)",
    auroc_transfer: 0.790,
    cases_found_vs_guideline: {
      krisk_cases: 4710,
      guideline_cases: 4300,
      net_gain: "+410 casi reali (+22%)",
      ci_95: "[+351, +469]",
      muscle_mass_adjusted_gain: "+377 casi reali"
    },
    target_decomposition: {
      filtration_egfr_less_60: {
        jinuo_auroc: 0.841,
        nhanes_auroc_baseline: 0.886,
        nhanes_auroc_with_uric_acid: 0.914,
        key_feature: "Acido Urico sierico (UA)",
        net_gain_cases: "+235 casi trovati rispetto a SCORED (1.076 vs 841 a parità di test prescritti)"
      },
      albuminuria_isolated: {
        blood_auroc: 0.684,
        conclusion: "L'albuminuria isolata con eGFR conservato non si riflette negli esami del sangue: confermato il data ceiling su scala trans-continentale."
      }
    }
  },

  prognostic_mortality: {
    cohort_evaluated: "28.433 adulti NHANES classificati come SANI / BASSO RISCHIO da KDIGO",
    hard_outcome: "Mortalità Cardiovascolare a Lungo Termine (CDC NDI)",
    hazard_ratio: 1.38,
    hr_ci: [1.20, 1.58],
    hr_adjusted_albuminuria: 1.28,
    negative_control: {
      outcome: "Morti per Incidenti / Traumi",
      hazard_ratio: 1.08,
      hr_ci: [0.82, 1.42],
      interpretation: "Associazione completamente nulla (p = 0.58). Il punteggio AI predice specificamente la morte cardiovascolare e metabolica, escludendo correlazioni spurie."
    }
  },

  final_test: {
    unlocked_at: "2026-09-22T18:20:04",
    sha256_lockbox: "b7d5f617663c5eaa9679640958916dfbee87fd9b",
    test_n: 1451,
    test_positives: 142,
    results: [
      { model: "Random Forest", auroc: 0.743, ci: [0.699, 0.786], pr_auc: 0.284, brier: 0.081 },
      { model: "Tri-Ensemble (Top-21)", auroc: 0.738, ci: [0.693, 0.782], pr_auc: 0.279, brier: 0.081 },
      { model: "XGBoost (3–12)", auroc: 0.735, ci: [0.690, 0.779], pr_auc: 0.274, brier: 0.082 },
      { model: "Logistica Elastic Net", auroc: 0.726, ci: [0.679, 0.771], pr_auc: 0.268, brier: 0.082 },
      { model: "Logistica SCORED", auroc: 0.723, ci: [0.676, 0.768], pr_auc: 0.245, brier: 0.084 }
    ],
    hypotheses: [
      { id: "A", title: "Discriminazione clinica confermata", criterion: "AUROC > 0.70 sul test mai visto", detail: "Tutti i modelli primari superano 0.72 (RF: 0.743).", verdict: "CONFERMATA", passed: true },
      { id: "B", title: "Ordinamento stadi KDIGO coerente", criterion: "Tendenza strettamente monotona della probabilità media", detail: "Medie test: Basso 0.078, Moderato 0.224, Alto 0.382, Molto Alto 0.491.", verdict: "CONFERMATA", passed: true },
      { id: "C", title: "Intercettazione casi gravi (Stadi 3–4)", criterion: "Recall ≥ 90% sui livelli Alto e Molto Alto", detail: "22 casi gravi su 23 intercettati (95.7%) sul test set.", verdict: "CONFERMATA", passed: true },
      { id: "D", title: "Net Benefit clinico superiore (DCA)", criterion: "Net benefit superiore a 'testare tutti' e 'non testare nessuno' a soglie 7% e 10%", detail: "Confermato vantaggio netto con risparmio di decine di test ogni 100 soggetti.", verdict: "CONFERMATA", passed: true },
      { id: "E", title: "Illusione del bilanciamento dati a soglia 0.50", criterion: "Recall a 0.50 gonfiato artificialmente senza guadagno diagnostico reale", detail: "Baseline 4.4% vs 50–65% con tecniche sintetiche; azzerato a parità di sensibilità.", verdict: "CONFERMATA", passed: true }
    ]
  },

  clinical_utility: {
    test_cost_usd: 49.00,
    cost_range: [36.00, 64.00],
    operating_points: [
      {
        threshold: 0.05,
        threshold_pct: "5%",
        clinical_meaning: "Tolleranza: fino a 19 test negativi per trovare 1 caso reale di CKD (odds 1/19)",
        tests_per_100: 78.1,
        cases_found_per_100: 8.8,
        recall: 0.901,
        net_benefit: 0.051,
        net_benefit_all: 0.050,
        tests_avoided_per_100: 21.9,
        cost_per_case: 433.00,
        cost_savings_per_100: 1073.10
      },
      {
        threshold: 0.07,
        threshold_pct: "7%",
        clinical_meaning: "Tolleranza: fino a ~13.3 test negativi per trovare 1 caso reale di CKD (odds 7/93)",
        tests_per_100: 56.8,
        cases_found_per_100: 7.9,
        recall: 0.808,
        net_benefit: 0.042,
        net_benefit_all: 0.028,
        tests_avoided_per_100: 43.2,
        cost_per_case: 352.00,
        cost_savings_per_100: 2116.80
      },
      {
        threshold: 0.10,
        threshold_pct: "10%",
        clinical_meaning: "Tolleranza: fino a 9 test negativi per trovare 1 caso reale di CKD (odds 1/9)",
        tests_per_100: 38.2,
        cases_found_per_100: 6.5,
        recall: 0.665,
        net_benefit: 0.030,
        net_benefit_all: -0.003,
        tests_avoided_per_100: 61.8,
        cost_per_case: 288.00,
        cost_savings_per_100: 3028.20
      }
    ],
    max_avoided_per_100: {
      lr_penalized: 56.8,
      random_forest: 55.5,
      xgboost: 56.3,
      lr_scored: 54.9
    }
  },

  gallery: [
    // --- Nuove Figure di Impatto & Presentazione Relatori ---
    {
      id: "relatori_figura",
      title_it: "Pannello Esecutivo di Sintesi per i Relatori",
      title_en: "Executive Presentation Panel for Thesis Advisors",
      desc_it: "Quattro scoperte cardine: Trasportabilità NHANES (+410 casi), Apprendimento non supervisionato sotto 30 mg/g, Significato prognostico su mortalità (HR 1.38) e Scomposizione del target con ruolo dell'acido urico.",
      desc_en: "Four core breakthroughs: NHANES Zero-Shot (+410 cases), Unsupervised risk sub-30 mg/g, Prognostic cardiovascular mortality (HR 1.38), and Target decomposition with Uric Acid.",
      category: "impact",
      path: "charts/figura_relatori.png",
      tag: "Breakthrough Tesi"
    },
    {
      id: "pres_06_sangue_vs_urina",
      title_it: "Sangue vs Urina e Controllo Positivo (AUROC 0.933)",
      title_en: "Blood vs Urine & Positive Control (AUROC 0.933)",
      desc_it: "Dimostrazione sperimentale del tetto biologico: i modelli sul sangue si fermano a 0.70-0.74, mentre inserendo l'albumina urinaria l'AUROC sale a 0.933.",
      desc_en: "Empirical proof of the biological data ceiling: blood models plateau at 0.70-0.74, whereas adding urine albumin immediately drives AUROC to 0.933.",
      category: "impact",
      path: "analytics/presentation/06_sangue_vs_urina_controllo_positivo.png",
      tag: "Data Ceiling"
    },
    {
      id: "pres_01_tutte_varianti",
      title_it: "Confronto Panoramico di Tutte le Metodologie Esplorate",
      title_en: "Comprehensive Overview of All Tested Methodologies",
      desc_it: "AUROC media out-of-fold su target composito e sola albuminuria per oltre 30 architetture tra Fasi A, B, D ed E.",
      desc_en: "Mean out-of-fold AUROC across composite target and isolated albuminuria across 30+ architectures in Phases A, B, D, and E.",
      category: "impact",
      path: "analytics/presentation/01_tutte_le_varianti.png",
      tag: "Fasi A–E"
    },
    {
      id: "pres_07_bilanciamento",
      title_it: "L'Illusione del Bilanciamento delle Classi",
      title_en: "The Illusion of Data Balancing",
      desc_it: "Confronto iso-sensibilità: a parità di quota di allarme, nessuna tecnica di bilanciamento (SMOTE, CTGAN, pesi) migliora la discriminazione clinica.",
      desc_en: "Iso-sensitivity comparison: at equal alarm rate, no balancing technique (SMOTE, CTGAN, weights) improves clinical discrimination.",
      category: "phase_b",
      path: "analytics/presentation/07_bilanciamento_apparente.png",
      tag: "Fase B"
    },

    // --- Soglia AI & Non Supervisionato ---
    {
      id: "soglia_ai_p7_fig1",
      title_it: "L'AI Intercetta il Rischio Continuo Sotto 30 mg/g",
      title_en: "AI Captures Continuous Risk Below 30 mg/g",
      desc_it: "Il punteggio non supervisionato cresce linearmente con l'ACR (+0.092 SD per raddoppio) senza mostrare alcuno scalino alla soglia convenzionale di 30 mg/g.",
      desc_en: "Unsupervised AI score increases monotonically with ACR (+0.092 SD per doubling) without any step or knot at the 30 mg/g clinical threshold.",
      category: "unsupervised",
      path: "analytics/soglia_ai/punto7/fig1_ai_sotto_30.png",
      tag: "Non Supervisionato"
    },
    {
      id: "soglia_ai_p7_fig2",
      title_it: "Specificità Eziologica delle Cause di Morte (NHANES)",
      title_en: "Etiological Cause-Specific Mortality Validation (NHANES)",
      desc_it: "Nei soggetti sani KDIGO, il punteggio AI predice specificamente la morte cardiovascolare (HR 1.38), mentre è completamente neutro sulle morti accidentali (HR 1.08).",
      desc_en: "In KDIGO healthy subjects, the AI score specifically predicts cardiovascular mortality (HR 1.38), while remaining neutral for accidental deaths (HR 1.08).",
      category: "unsupervised",
      path: "analytics/soglia_ai/punto7/fig2_cause_di_morte.png",
      tag: "Hard Outcomes"
    },
    {
      id: "soglia_ai_p6_fig1",
      title_it: "Non Supervisionato e Validazione Trans-Popolazione",
      title_en: "Unsupervised Model & Trans-Population Validation",
      desc_it: "Generalizzazione del modello non supervisionato dalla coorte Jinuo a NHANES: persistenza dei cluster e delle traiettorie metaboliche.",
      desc_en: "Unsupervised model generalization from Jinuo to NHANES: consistency of clusters and metabolic trajectories across cohorts.",
      category: "unsupervised",
      path: "analytics/soglia_ai/punto6/fig1_nonsup_e_validazione.png",
      tag: "Validazione Esterna"
    },
    {
      id: "soglia_ai_p3_fig1",
      title_it: "Reni o Muscoli? Deconvoluzione della Massa Muscolare",
      title_en: "Kidneys or Muscles? Muscle Mass Deconvolution",
      desc_it: "Verifica che il segnale del modello rifletta un effettivo danno glomerulare e non sia un semplice riflesso della creatinina legata alla massa muscolare scheletrica.",
      desc_en: "Proof that model signal reflects true glomerular impairment and is not a mere surrogate of creatinine dependent on skeletal muscle mass.",
      category: "unsupervised",
      path: "analytics/soglia_ai/punto3/fig1_reni_o_muscoli.png",
      tag: "Fisiopatologia"
    },
    {
      id: "soglia_ai_p2_fig2",
      title_it: "Traiettoria degli Esami del Sangue lungo il Gradiente di ACR",
      title_en: "Blood Biomarker Trajectories Across ACR Spectrum",
      desc_it: "Comportamento di emoglobina, acido urico, trigliceridi e glicemia al variare continuo dell'escrezione urinaria di albumina.",
      desc_en: "Behavior of hemoglobin, uric acid, triglycerides, and glucose across the continuous spectrum of urinary albumin excretion.",
      category: "unsupervised",
      path: "analytics/soglia_ai/punto2/fig2_esami_lungo_acr.png",
      tag: "Biomarcatori"
    },

    // --- Dataset & Split ---
    {
      id: "ds_01_kdigo",
      title_it: "Matrice di Rischio Clinico KDIGO 2024",
      title_en: "KDIGO 2024 Clinical Risk Grid Heatmap",
      desc_it: "Distribuzione dei 5.801 pazienti eleggibili nei 4 stadi clinici (Basso 90.2%, Moderato 8.2%, Alto 1.2%, Molto Alto 0.5%).",
      desc_en: "Distribution of the 5,801 eligible subjects across the 4 KDIGO clinical risk categories.",
      category: "dataset",
      path: "analytics/dataset/01_kdigo_heatmap.png",
      tag: "Coorte Jinuo"
    },
    {
      id: "ds_06_egfr",
      title_it: "Ricalcolo eGFR: Dataset vs CKD-EPI 2021",
      title_en: "eGFR Recalculation: Raw Dataset vs CKD-EPI 2021",
      desc_it: "Discrepanza tra la colonna non documentata del dataset grezzo e l'equazione ufficiale raccomandata dalle linee guida internazionali.",
      desc_en: "Discrepancy between raw undocumented dataset GFR and official guideline-recommended CKD-EPI 2021 equation.",
      category: "dataset",
      path: "analytics/dataset/06_egfr_comparison.png",
      tag: "Target Quality"
    },
    {
      id: "split_02_smd",
      title_it: "Verifica di Bilanciamento SMD Train/Test",
      title_en: "Train/Test Standardized Mean Difference (SMD)",
      desc_it: "Tutte le 15 covariate cliniche chiave presentano SMD < 0.05, certificando la perfetta comparabilità tra set di sviluppo e test set sigillato.",
      desc_en: "All 15 key clinical covariates exhibit SMD < 0.05, proving pristine balance between train and locked test sets.",
      category: "dataset",
      path: "analytics/split/02_smd_balance.png",
      tag: "Anti-Leakage"
    },

    // --- Preprocessing & Imputazione ---
    {
      id: "prep_01_columns",
      title_it: "Mappa Tassonomica delle 190 Variabili",
      title_en: "Taxonomic Mapping of All 190 Variables",
      desc_it: "Classificazione esaustiva: 74 predittori finali, 85 escluse per leakage o assenza di valore clinico, 14 categoriche.",
      desc_en: "Exhaustive classification: 74 input predictors, 85 excluded due to leakage or irrelevance, 14 categorical.",
      category: "preprocessing",
      path: "analytics/preprocessing/01_column_map.png",
      tag: "Selezione Feature"
    },
    {
      id: "prep_02_univariate",
      title_it: "AUC Univariata e Controllo Anti-Leakage",
      title_en: "Univariate AUC & Anti-Leakage Verification",
      desc_it: "Nessuna delle 74 feature in ingresso si avvicina al potere discriminante degli esami renali esclusi (creatinina sierica e ACR urinario).",
      desc_en: "No input predictor approaches the univariate discriminative power of excluded renal markers (creatinine, ACR).",
      category: "preprocessing",
      path: "analytics/preprocessing/02_univariate_auc.png",
      tag: "Audit Leakage"
    },
    {
      id: "prep_04_imputation",
      title_it: "Benchmark dell'Imputazione: Vittoria di MissForest",
      title_en: "Imputation Benchmark: MissForest Superiority",
      desc_it: "MissForest ottiene l'errore di ricostruzione significativamente più basso (RMSE 0.734 ± 0.066) preservando le correlazioni multivariate.",
      desc_en: "MissForest achieves significantly lower reconstruction error (RMSE 0.734 ± 0.066) while preserving non-linear multivariate correlation structures.",
      category: "preprocessing",
      path: "analytics/preprocessing/04_imputation_main.png",
      tag: "MissForest"
    },

    // --- Fase A (Supervisionato Primario) ---
    {
      id: "pa_01_roc_pr",
      title_it: "Curve ROC e Precision-Recall (Fase A)",
      title_en: "ROC & Precision-Recall Curves (Phase A)",
      desc_it: "Curve aggregate out-of-fold: Random Forest (AUROC 0.703) e XGBoost (0.698) mostrano un PR-AUC pari a 2.6x la prevalenza di base.",
      desc_en: "Aggregate out-of-fold curves: Random Forest (AUROC 0.703) and XGBoost (0.698) achieve PR-AUC 2.6x baseline prevalence.",
      category: "phase_a",
      path: "analytics/phase_a/01_roc_pr.png",
      tag: "Fase A"
    },
    {
      id: "pa_04_prob_level",
      title_it: "Distribuzione Probabilità per Stadio KDIGO",
      title_en: "Predicted Probability Distribution Across KDIGO Stages",
      desc_it: "Monotonia rigorosa: la probabilità stimata sale progressivamente da Basso (7.8%) a Molto Alto (49.1%).",
      desc_en: "Monotonic risk escalation: estimated probability increases systematically from Low (7.8%) to Very High (49.1%).",
      category: "phase_a",
      path: "analytics/phase_a/04_probability_by_level.png",
      tag: "Gradiente Rischio"
    },
    {
      id: "pa_11_shap_beeswarm",
      title_it: "SHAP Beeswarm: Spiegabilità Globale dei Predittori",
      title_en: "SHAP Beeswarm: Global Model Explainability",
      desc_it: "Età, pressione arteriosa, acido urico ed emoglobina emergono come i driver metabolici primari per la decisione del modello.",
      desc_en: "Age, blood pressure, uric acid, and hemoglobin emerge as the top biological drivers of model predictions.",
      category: "phase_a",
      path: "analytics/phase_a/11_shap_beeswarm.png",
      tag: "XAI / SHAP"
    },

    // --- Qualità Clinica & DCA ---
    {
      id: "qual_01_dca",
      title_it: "Decision Curve Analysis (DCA): Utilità Clinica Netta",
      title_en: "Decision Curve Analysis (DCA): Net Clinical Benefit",
      desc_it: "Superamento netto delle strategie 'testare tutti' e 'non testare nessuno'. Alla soglia 7%, si evitano 43.2 esami ogni 100 persone con net benefit positivo.",
      desc_en: "Clear superiority over 'test all' and 'test none' policies. At 7% threshold, avoids 43.2 tests per 100 individuals.",
      category: "clinical_utility",
      path: "analytics/quality/01_decision_curve.png",
      tag: "DCA / Net Benefit"
    },
    {
      id: "qual_03_costs",
      title_it: "Analisi di Costo-Efficacia e Risparmio Economico",
      title_en: "Cost-Effectiveness & Health Economic Savings",
      desc_it: "Costo per caso reale diagnosticato e risparmio economico cumulativo (oltre 2.100 $ ogni 100 screening a soglia 7%).",
      desc_en: "Cost per diagnosed true case and cumulative savings (> $2,100 per 100 screened individuals at 7% threshold).",
      category: "clinical_utility",
      path: "analytics/quality/03_costs.png",
      tag: "Costi Sanitari"
    },
    {
      id: "qual_06_bragg_gresham",
      title_it: "Confronto Testa a Testa con Bragg-Gresham et al. 2025",
      title_en: "Head-to-Head Comparison with Bragg-Gresham et al. 2025",
      desc_it: "K-Risk ottiene prestazioni paragonabili senza violare il vincolo non-invasivo (il modello rivale include la creatinina sierica in ingresso).",
      desc_en: "K-Risk achieves competitive performance without violating non-invasive constraints (rival model uses serum creatinine as input).",
      category: "clinical_utility",
      path: "analytics/quality/06_bragg_gresham.png",
      tag: "Stato dell'Arte"
    }
  ]
};
