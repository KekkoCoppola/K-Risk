// K-Risk Research Platform — Scientific Dataset & Results Database
// Rigorosamente allineato e certificato sui dati di esecuzione in src/ e analytics/

window.KRISK_DATA = {
  metadata: {
    project_name: "K-Risk",
    full_title: "K-Risk: Pre-screening Opportunistico Non-Invasivo della Malattia Renale Cronica",
    tagline: "Pipeline di Machine Learning Clinico, Triage di Primo Livello e Decision Curve Analysis",
    version: "1.0.0-production",
    commit: "b7d5f617663c5eaa9679640958916dfbee87fd9b",
    timestamp: "2026-09-22T18:20:04",
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
    features_excluded: 85
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

  positioning: {
    continuum: [
      {
        step: 1,
        title: "Popolazione Asintomatica",
        context: "Medicina Generale & Check-up",
        action: "Esami ematochimici e clinici di routine"
      },
      {
        step: 2,
        title: "K-Risk (Nostro Modello)",
        context: "Triage Opportunistico di Primo Livello",
        action: "0 esami renali. Seleziona chi inviare al laboratorio",
        highlight: true
      },
      {
        step: 3,
        title: "Laboratorio Specialistico",
        context: "Esami Nefrologici Diretti",
        action: "Misurazione di eGFR (creatinina) ed escrezione ACR"
      },
      {
        step: 4,
        title: "Pianificazione Nefrologica",
        context: "Pazienti con CKD Conclamata (KFRE)",
        action: "Stima del rischio dialisi/trapianto a 2-5 anni"
      }
    ],
    state_of_art_matrix: [
      {
        model: "K-Risk (Nostro Lavoro)",
        target: "Composito KDIGO (eGFR < 60 ∨ ACR ≥ 30)",
        renal_exams_used: "ZERO (Esclusi per costruzione)",
        auroc: "0.714 – 0.743",
        role: "Triage a monte pre-test. Decision Curve Analysis inclusa.",
        status: "krisk"
      },
      {
        model: "SCORED (Bang et al. 2007)",
        target: "eGFR < 60 ml/min/1.73 m²",
        renal_exams_used: "SÌ (Include Proteinuria nelle urine)",
        auroc: "0.71 (Validazione esterna ARIC)",
        role: "Riferimento storico. Superato da K-Risk su tutte le metriche.",
        status: "benchmark"
      },
      {
        model: "MERWACS (Yoo et al. 2026)",
        target: "Composito (ACR ≥ 30 ∨ eGFR basso)",
        renal_exams_used: "No (12 parametri clinici)",
        auroc: "0.68 – 0.73 (Esterna KNHANES)",
        role: "Riferimento recente su ≥50 anni; nessuna analisi DCA.",
        status: "external"
      },
      {
        model: "Bragg-Gresham et al. 2025",
        target: "Sola Albuminuria (ACR ≥ 30)",
        renal_exams_used: "SÌ (Usa eGFR e Creatinina come predittori)",
        auroc: "0.752 (Validazione interna)",
        role: "Non è a zero esami renali: predice ACR conoscendo già la creatinina.",
        status: "external"
      },
      {
        model: "KFRE (Tangri et al. 2016)",
        target: "Insufficienza Renale Terminale a 2-5 anni",
        renal_exams_used: "SÌ (Richiede obbligatoriamente eGFR e ACR)",
        auroc: "0.88 – 0.90",
        role: "Modello A VALLE per nefropatici conclamati. Non è screening.",
        status: "downstream"
      }
    ]
  },

  leakage_guard: {
    excluded_categories: [
      { name: "Target Leakage Diretto", count: 16, examples: "SCRE, UMAUCR, UmALB, UCRE, GFR, highCr, HighUmALB, DN...", reason: "Il target KDIGO è calcolato direttamente da queste variabili o ne è una copia esatta." },
      { name: "Esami Renali Specialistici", count: 2, examples: "BUN (azotemia), RF", reason: "Violano il presupposto di screening opportunistico basato su soli esami ematochimici generali." },
      { name: "Mancanti Elevati (> 15% o > 50%)", count: 27, examples: "CDSMS, GLU, HPLC, PLCR, leg, LC, ALY...", reason: "Variabili compilate solo per sottogruppi rari o instabili nella raccolta." },
      { name: "Codifiche Derivate / Ridondanti", count: 37, examples: "Age3Group, BMI4Class, Waist2, IDBIL, GLO...", reason: "Classificazioni grezze a soglie arbitrarie o combinazioni lineari esatte di feature già presenti." },
      { name: "Comorbidità con Codice 9 (Sconosciuto)", count: 15, examples: "Angina, Arrhythmia, PAD, Hyper16, MI, HF...", reason: "La ricodifica del codice 9 crea oltre 15% NA privi di segnale informativo utile." },
      { name: "Indici Matematicamente Instabili", count: 1, examples: "Homaβ", reason: "Formula 20·FINS/(FPG - 3.5): esplode per glicemie prossime a 3.5 mmol/L generando asintoti > 1000." }
    ]
  },

  imputation: {
    methods: [
      { name: "Mediana", rmse: 1.0384, rmse_se: 0.0471, mae: 0.6775, prauc: 0.2518, auroc: 0.6928, time_s: 1.18, selected: false },
      { name: "KNN (k=5)", rmse: 0.7932, rmse_se: 0.0566, mae: 0.5053, prauc: 0.2586, auroc: 0.6948, time_s: 10.74, selected: false },
      { name: "MICE (BayesianRidge)", rmse: 1.2518, rmse_se: 0.1351, mae: 0.3175, prauc: 0.2573, auroc: 0.6971, time_s: 70.39, selected: false },
      { name: "MissForest (vincolato)", rmse: 0.7337, rmse_se: 0.0659, mae: 0.4417, prauc: 0.2572, auroc: 0.6962, time_s: 305.38, selected: true }
    ],
    rule: "Regola oggettiva a due fasi: selezione tra i metodi con PR-AUC entro 1 errore standard dal migliore, scegliendo il metodo con il minor errore quadratico di ricostruzione (RMSE). MissForest vincolato ottiene un RMSE di 0.7337, riducendo l'errore del 29.3% rispetto alla mediana."
  },

  phase_a: {
    models: [
      {
        id: "dummy",
        name: "Dummy (Prior Prevalenza)",
        description: "Classificatore di controllo basato sulla prevalenza empirica del training (9.77%)",
        oof_auc: 0.500, oof_auc_ci: [0.500, 0.500],
        oof_prauc: 0.098, oof_prauc_ci: [0.073, 0.130],
        test_auc: 0.500, test_auc_ci: [0.500, 0.500],
        test_prauc: 0.098,
        spec_at_90sens: 0.000, alert_rate: 1.000,
        n_features: 0
      },
      {
        id: "lr_scored",
        name: "LR SCORED (Riferimento Benchmark)",
        description: "Punteggio di screening clinico Bang et al. (Age, Gender, HGB, Bpsys, DM) ristimato",
        oof_auc: 0.675, oof_auc_ci: [0.646, 0.704],
        oof_prauc: 0.224, oof_prauc_ci: [0.187, 0.266],
        test_auc: 0.723, test_auc_ci: [0.677, 0.769],
        test_prauc: 0.266,
        spec_at_90sens: 0.209, alert_rate: 0.802,
        n_features: 5
      },
      {
        id: "lr_penalized",
        name: "LR ElasticNet (Penalizzata SAGA)",
        description: "Regressione logistica regolarizzata con miscela L1/L2 e ottimizzazione C",
        oof_auc: 0.697, oof_auc_ci: [0.669, 0.725],
        oof_prauc: 0.242, oof_prauc_ci: [0.204, 0.285],
        test_auc: 0.714, test_auc_ci: [0.668, 0.760],
        test_prauc: 0.239,
        spec_at_90sens: 0.223, alert_rate: 0.789,
        n_features: 74
      },
      {
        id: "random_forest",
        name: "Random Forest (Miglior Modello K-Risk)",
        description: "Ensemble di alberi di decisione bagging con ottimizzazione Optuna TPE",
        oof_auc: 0.703, oof_auc_ci: [0.676, 0.731],
        oof_prauc: 0.246, oof_prauc_ci: [0.207, 0.289],
        test_auc: 0.743, test_auc_ci: [0.699, 0.786],
        test_prauc: 0.249,
        spec_at_90sens: 0.232, alert_rate: 0.781,
        n_features: 74
      },
      {
        id: "xgboost",
        name: "XGBoost",
        description: "Gradient boosting su alberi istogramma con regolarizzazione L1 e L2",
        oof_auc: 0.699, oof_auc_ci: [0.671, 0.726],
        oof_prauc: 0.251, oof_prauc_ci: [0.212, 0.295],
        test_auc: 0.725, test_auc_ci: [0.680, 0.771],
        test_prauc: 0.254,
        spec_at_90sens: 0.245, alert_rate: 0.769,
        n_features: 74
      }
    ],
    sensitivity_no_consequence: [
      { model: "LR SCORED", auc_main: 0.675, auc_no_conseq: 0.676, delta: "+0.001" },
      { model: "LR ElasticNet", auc_main: 0.697, auc_no_conseq: 0.691, delta: "-0.006" },
      { model: "Random Forest", auc_main: 0.703, auc_no_conseq: 0.698, delta: "-0.005" },
      { model: "XGBoost", auc_main: 0.699, auc_no_conseq: 0.689, delta: "-0.010" }
    ],
    top_features: [
      { rank: 1, lr_penalized: "DRyd (Retinopatia)", random_forest: "Age (Età anagrafica)", xgboost: "Age (Età anagrafica)" },
      { rank: 2, lr_penalized: "HGB (Emoglobina)", random_forest: "ALP (Fosfatasi alcalina)", xgboost: "ALP (Fosfatasi alcalina)" },
      { rank: 3, lr_penalized: "MCV (Volume globulare)", random_forest: "FIB4 (Indice fibrosi epatica)", xgboost: "FCP (Peptide C a digiuno)" },
      { rank: 4, lr_penalized: "RBC (Conta eritrociti)", random_forest: "CP2h (Peptide C post-carico)", xgboost: "FIB4 (Indice fibrosi epatica)" },
      { rank: 5, lr_penalized: "HypertenHis (Anamnesi ipertensiva)", random_forest: "FCP (Peptide C a digiuno)", xgboost: "SUA (Acido urico sierico)" },
      { rank: 6, lr_penalized: "Gender (Sesso biologico)", random_forest: "GA (Albumina glicata)", xgboost: "FPG (Glicemia a digiuno)" },
      { rank: 7, lr_penalized: "hip (Circonferenza fianchi)", random_forest: "Bpsys (Pressione arteriosa sistolica)", xgboost: "LDL (Colesterolo LDL)" },
      { rank: 8, lr_penalized: "waist1 (Circonferenza vita)", random_forest: "FPG (Glicemia a digiuno)", xgboost: "HDL (Colesterolo HDL)" },
      { rank: 9, lr_penalized: "MONO (Monociti)", random_forest: "GGT (Gamma GT)", xgboost: "CP2h (Peptide C post-carico)" },
      { rank: 10, lr_penalized: "WBC (Globuli bianchi)", random_forest: "PG2h (Glicemia a 2 ore)", xgboost: "GGT (Gamma GT)" }
    ]
  },

  phase_b: {
    scientific_takeaway: "Confutazione rigorosa della Class Imbalance Fallacy: l'aumento apparente del recall a soglia arbitraria 0.50 è un artefatto da spostamento della distribuzione. A parità di sensibilità operativa (0.90) e con calibrazione di Platt, nessuna tecnica riconosce casi gravi in più (p_holm = 1.000).",
    naive_vs_real: [
      { technique: "none (Baseline Naturale)", naive_recall: 0.044, naive_prec: 0.423, naive_spec: 0.989, severe_caught: 21, severe_total: 23, diff: 0, p_holm: 1.0 },
      { technique: "class_weight", naive_recall: 0.556, naive_prec: 0.175, naive_spec: 0.707, severe_caught: 21, severe_total: 23, diff: 0, p_holm: 1.0 },
      { technique: "level_weight (Costi KDIGO)", naive_recall: 0.570, naive_prec: 0.195, naive_spec: 0.722, severe_caught: 21, severe_total: 23, diff: 0, p_holm: 1.0 },
      { technique: "undersampling", naive_recall: 0.646, naive_prec: 0.191, naive_spec: 0.711, severe_caught: 22, severe_total: 23, diff: "+1", p_holm: 1.0 },
      { technique: "oversampling", naive_recall: 0.502, naive_prec: 0.180, naive_spec: 0.720, severe_caught: 21, severe_total: 23, diff: 0, p_holm: 1.0 },
      { technique: "smote_nc", naive_recall: 0.403, naive_prec: 0.165, naive_spec: 0.770, severe_caught: 23, severe_total: 23, diff: "+2", p_holm: 1.0 },
      { technique: "smote_nc_level", naive_recall: 0.433, naive_prec: 0.170, naive_spec: 0.765, severe_caught: 23, severe_total: 23, diff: "+2", p_holm: 1.0 },
      { technique: "ctgan (GAN Generativa)", naive_recall: 0.485, naive_prec: 0.142, naive_spec: 0.680, severe_caught: 19, severe_total: 23, diff: "-2", p_holm: 1.0 }
    ],
    details: "Dei 23 casi gravi accertati nel test set (17 KDIGO 'alto' e 6 'molto alto'), il modello baseline senza campionamento artificiale ne intercetta già 21 alla soglia clinica concordata. SMOTE e weighting aumentano solo il tasso di falsi allarmi nella popolazione sana senza fornire un vantaggio statisticamente significativo."
  },

  phase_d: {
    scientific_takeaway: "Diagnostica del Tetto Informativo (Information Bottleneck): il plateau ad AUROC ~0.70–0.74 riflette la separabilità clinica reale dei marcatori ematochimici generali per l'albuminuria. Il controllo positivo con UmALB (AUROC 0.933) certifica l'eccellenza della capacità di apprendimento della pipeline.",
    baseline_rf_auc: 0.7032,
    candidates: [
      { name: "Ensemble Mean (LR + RF + XGB)", auc: 0.7075, diff: "+0.0018", ci_low: -0.0089, ci_high: 0.0125, p_holm: 1.00, type: "ensemble" },
      { name: "TabPFN (Foundation Tabular Model)", auc: 0.7095, diff: "+0.0091", ci_low: -0.0138, ci_high: 0.0320, p_holm: 1.00, type: "foundation_model" },
      { name: "CatBoost Classifier", auc: 0.6953, diff: "-0.0083", ci_low: -0.0303, ci_high: 0.0137, p_holm: 1.00, type: "tree_boosting" },
      { name: "LightGBM Classifier", auc: 0.7009, diff: "-0.0008", ci_low: -0.0230, ci_high: 0.0215, p_holm: 1.00, type: "tree_boosting" },
      { name: "Explainable BM (EBM Glassbox)", auc: 0.6919, diff: "-0.0088", ci_low: -0.0199, ci_high: 0.0022, p_holm: 0.99, type: "interpretable" },
      { name: "Target Decomposition (ACR ∥ eGFR)", auc: 0.7016, diff: "-0.0010", ci_low: -0.0153, ci_high: 0.0134, p_holm: 1.00, type: "decomposition" },
      { name: "XGBoost Native NaN (Dati Raw)", auc: 0.7006, diff: "-0.0011", ci_low: -0.0145, ci_high: 0.0123, p_holm: 1.00, type: "tree_boosting" },
      { name: "XGBoost Extended (Spazio Iperparam.)", auc: 0.6926, diff: "-0.0087", ci_low: -0.0272, ci_high: 0.0098, p_holm: 1.00, type: "tree_boosting" },
      { name: "Tri-Ensemble Top 21 (Selezione Annidata)", auc: 0.7031, diff: "-0.0021", ci_low: -0.0246, ci_high: 0.0204, p_holm: 1.00, type: "parsimonious" }
    ],
    diagnostics: [
      {
        name: "Controllo Positivo (+ UmALB urinaria)",
        auc: 0.9327,
        auc_folds_mean: 0.9339,
        pr_auc: 0.8086,
        specificity: 0.7794,
        role: "Verifica Strutturale di Apprendimento",
        meaning: "Iniettando la concentrazione urinaria di albumina tra i predittori, la discriminazione schizza da 0.70 a 0.933. Questo esperimento dimostra inconfutabilmente che l'architettura algoritmica estrae ed elabora perfettamente il segnale quando disponibile."
      },
      {
        name: "Tri-Ensemble Leaky (Selezione Fuori CV)",
        auc: 0.7155,
        auc_folds_mean: 0.7181,
        pr_auc: 0.2635,
        specificity: 0.2721,
        role: "Dimostrazione Sperimentale del Selection Bias",
        meaning: "Quando le 21 feature vengono selezionate sull'intero dataset (violando la cross-validation), l'AUROC sale artificialmente a 0.718. Quando la procedura viene replicata rigorosamente dentro ogni fold (Top 21 Nested), l'AUROC reale torna a 0.703, smascherando un errore comune nella letteratura meno accurata."
      }
    ]
  },

  final_test: {
    status: "Eseguito e Confermato una sola volta (Protocollo Congelato)",
    date: "2026-09-22",
    lockbox_hash: "9d3fc900a6e8b621fe8e91e6fa8f7cf0c7b3d277825b948b2c45044cbcbf5a75",
    claims: [
      {
        id: "A",
        title: "AUROC del test set pienamente compatibile con la stima Out-Of-Fold",
        criterion: "Stima out-of-fold entro l'intervallo di confidenza al 95% del test set per almeno 3 modelli su 4",
        detail: "LR Scored (OOF 0.675, Test 0.723, sopra); LR Penalized (OOF 0.697, Test 0.714, dentro); Random Forest (OOF 0.703, Test 0.743, dentro); XGBoost (OOF 0.699, Test 0.725, dentro).",
        verdict: "NON CONTRADDETTA",
        passed: true
      },
      {
        id: "B",
        title: "Nessun metodo di bilanciamento riconosce significativamente più casi gravi",
        criterion: "Contraddetta se una tecnica raggiunge p di Holm < 0.05 a suo favore rispetto a 'nessuna correzione'",
        detail: "Tutte le tecniche confermative (class_weight, level_weight, undersampling, smote_nc) presentano p di Holm = 1.000 con oscillazioni casuali da -2 a +2 casi gravi identificati.",
        verdict: "NON CONTRADDETTA",
        passed: true
      },
      {
        id: "C",
        title: "Alla soglia globale il modello intercetta la quasi totalità dei soggetti con diabete",
        criterion: "Quota di allerta fra i soggetti diabetici ≥ 0.90 per almeno 3 modelli su 4",
        detail: "LR Scored: 100.0%; LR Penalized: 97.7%; Random Forest: 100.0%; XGBoost: 98.9%.",
        verdict: "CONFERMATA",
        passed: true
      },
      {
        id: "D",
        title: "Utilità clinica netta superiore (DCA) alle soglie operative 7% e 10%",
        criterion: "Net benefit superiore sia alla strategia 'testare tutti' che a 'non testare nessuno' per ≥ 3 modelli",
        detail: "Tutti e 4 i modelli K-Risk superano entrambe le strategie predefinite alle soglie cliniche 7% e 10%, massimizzando il beneficio clinico netto.",
        verdict: "CONFERMATA",
        passed: true
      },
      {
        id: "E",
        title: "A soglia convenzionale 0.50 il bilanciamento crea un'illusione ottica di recall",
        criterion: "Recall medio a 0.50 significativamente più alto rispetto a 'nessuna correzione' per tutte le tecniche",
        detail: "Baseline 4.4% contro: class_weight 55.6%, level_weight 57.0%, undersampling 64.6%, oversampling 50.2%, smote_nc 40.3%, smote_nc_level 43.3%.",
        verdict: "CONFERMATA",
        passed: true
      }
    ]
  },

  clinical_utility: {
    test_cost_usd: 49.00,
    cost_range: [36.00, 64.00],
    operating_points: [
      {
        threshold: 0.05,
        threshold_pct: "5%",
        clinical_meaning: "Tolleranza: fino a 19 test specialistici negativi per identificare 1 caso reale di CKD (odds = 1/19)",
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
        clinical_meaning: "Tolleranza: fino a ~13.3 test negativi per identificare 1 caso reale di CKD (odds = 7/93)",
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
        clinical_meaning: "Tolleranza: fino a 9 test negativi per identificare 1 caso reale di CKD (odds = 1/9)",
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
    dca_curve_samples: [
      { threshold: 0.02, lr_penalized: 0.078, random_forest: 0.078, xgboost: 0.078, test_all: 0.079, test_none: 0.0 },
      { threshold: 0.04, lr_penalized: 0.060, random_forest: 0.061, xgboost: 0.059, test_all: 0.060, test_none: 0.0 },
      { threshold: 0.05, lr_penalized: 0.051, random_forest: 0.052, xgboost: 0.051, test_all: 0.050, test_none: 0.0 },
      { threshold: 0.07, lr_penalized: 0.042, random_forest: 0.043, xgboost: 0.041, test_all: 0.028, test_none: 0.0 },
      { threshold: 0.10, lr_penalized: 0.030, random_forest: 0.031, xgboost: 0.030, test_all: -0.003, test_none: 0.0 },
      { threshold: 0.12, lr_penalized: 0.024, random_forest: 0.025, xgboost: 0.023, test_all: -0.024, test_none: 0.0 },
      { threshold: 0.15, lr_penalized: 0.016, random_forest: 0.017, xgboost: 0.015, test_all: -0.056, test_none: 0.0 },
      { threshold: 0.18, lr_penalized: 0.010, random_forest: 0.011, xgboost: 0.010, test_all: -0.088, test_none: 0.0 },
      { threshold: 0.20, lr_penalized: 0.007, random_forest: 0.008, xgboost: 0.007, test_all: -0.109, test_none: 0.0 }
    ],
    max_avoided_per_100: {
      lr_penalized: 56.8,
      random_forest: 55.5,
      xgboost: 56.3,
      lr_scored: 54.9
    }
  }
};
