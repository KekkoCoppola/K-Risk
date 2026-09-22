/**
 * K-Risk Research Platform — Internationalization (i18n) Module
 * English & Italian Full Scientific Translation Dictionary & Bilingual Datasets
 */

window.KRISK_I18N = {
  // Current active language (default 'it')
  currentLang: localStorage.getItem("krisk_lang") || "it",

  // Static DOM Elements Translations (keyed by data-i18n attribute)
  translations: {
    it: {
      // Document Metadata
      page_title: "K-Risk — Piattaforma Scientifica di Screening Non-Invasivo della Malattia Renale Cronica",
      page_description: "Piattaforma scientifica K-Risk: triage clinico di primo livello per la stima dei marcatori di malattia renale cronica senza esami specialistici. Decision Curve Analysis e diagnostica del tetto informativo.",
      
      // Topbar & Navigation
      git_chip_title: "Commit hash registrato al congelamento del protocollo",
      theme_btn_title: "Cambia tema (Chiaro / Scuro)",
      nav_scope: "Scope",
      nav_cohort: "Coorte",
      nav_phase_a: "Fase A",
      nav_phase_b: "Fase B",
      nav_phase_d: "Fase D",
      nav_final_test: "Lockbox & DCA",
      nav_results: "Risultati",

      // Hero Bento Header
      hero_title: "Welcome in, K-Risk",
      hero_subtitle: "Screening Non-Invasivo della Malattia Renale Cronica · Triage di 1° Livello & Decision Curve Analysis",
      prev_label_1: "KDIGO Basso (90.2%)",
      prev_label_2: "Moderato (8.2%)",
      prev_label_3: "Alto (1.2%)",
      prev_label_4: "Molto Alto (0.5%)",
      stat_cohort_label: "Coorte Totale",
      stat_positives_label: "Positivi KDIGO",
      stat_renal_exams_label: "Esami Renali",
      stat_auroc_label: "AUROC Test",

      // Tab 00: Scope
      scope_h2: "00. Scope Formale del Progetto K-Risk",
      scope_p: "Definizione degli obiettivi clinici, perimetro operativo, popolazione di riferimento, vincoli architetturali e presupposti scientifici.",
      scope_badge: "Protocollo Registrato",
      scope_callout: "<strong>Sintesi clinica ed operativa:</strong> K-Risk allena e valida modelli di classificazione che, <strong>senza alcun esame renale</strong>, stimano chi presenta marcatori di malattia renale cronica (CKD), confrontano le stime con la stratificazione clinica <strong>KDIGO</strong> e misurano rigorosamente quanto le tecniche di bilanciamento migliorano il riconoscimento dei casi più gravi.",
      scope_sheet_title: "Scheda di Definizione dello Scope Clinico",
      scope_q_title: "Le 6 Domande Scientifiche Fondamentali",
      scope_q1: "<strong>Discriminazione:</strong> Il modello distingue chi presenta marcatori di malattia renale? (AUROC, PR-AUC, operating point).",
      scope_q2: "<strong>Gradiente di Rischio:</strong> Il rischio predetto cresce monotonicamente con la gravità KDIGO? (Test di tendenza di Jonckheere-Terpstra).",
      scope_q3: "<strong>Casi Gravi:</strong> Quanti casi ad alto e molto alto rischio riconosce? Mancare un \"molto alto\" è l'errore clinico più grave.",
      scope_q4: "<strong>Calibrazione Clinica:</strong> Le fasce di rischio corrispondono alla reale stratificazione di popolazione? (Indice Kappa pesato).",
      scope_q5: "<strong>Imbalance Fallacy:</strong> Il bilanciamento sintetico (SMOTE/CTGAN) migliora i casi gravi o solo le metriche medie a soglia fissa?",
      scope_q6: "<strong>Sottogruppo Diabetico:</strong> Come si comporta il modello su soggetti con diabete rispetto alla popolazione generale?",
      scope_rules_title: "Regole Metodologiche Tassative",
      scope_r1: "<strong>Data Augmentation solo sul training:</strong> Le tecniche di campionamento operano esclusivamente dentro ogni fold di training interno; nessun dato sintetico tocca mai la validazione o il test.",
      scope_r2: "<strong>Ricalibrazione di Platt obbligatoria:</strong> Le probabilità distorte dal campionamento artificiale vengono ricalibrate su dati reali prima della valutazione clinica.",
      scope_r3: "<strong>Confronto KDIGO solo su dati reali:</strong> I soggetti sintetici non possiedono una categoria KDIGO biologica.",
      scope_r4: "<strong>Test set congelato (Lockbox):</strong> 1.451 soggetti mai letti durante la fase di sviluppo; aperto una sola volta per la validazione finale dei claim pre-registrati.",

      // Tab 01: Cohort & Data Integrity
      t1_h2: "01. Coorte Clinica, Matrice KDIGO e Zero Data Leakage",
      t1_p: "Preparazione del dataset: ricalcolo eGFR con formula CKD-EPI 2021, partizione stratificata test al 25% congelata e protocollo di imputazione MissForest vincolato.",
      t1_badge: "Protocollo Fissato A Priori",
      t1_kdigo_title: "Matrice di Rischio KDIGO 2D (6 × 3)",
      t1_kdigo_sub: "Distribuzione osservata dei 5.801 soggetti nella griglia eGFR × ACR",
      badge_interactive: "Interattivo",
      t1_leg_low: "Basso Rischio (5.234 - 90.2%)",
      t1_leg_mod: "Moderato (473 - 8.2%)",
      t1_leg_high: "Alto (67 - 1.2%)",
      t1_leg_vhigh: "Molto Alto (27 - 0.5%)",
      t1_imp_title: "Confronto dei Metodi di Imputazione (Solo Training)",
      t1_imp_sub: "Cross-Validation a 5 fold con maschera del 10% sui valori osservati",
      t1_imp_badge: "5-Fold CV",
      t1_imp_p: "Le prestazioni a valle (PR-AUC ~0.257) risultano invarianti al metodo, ma <strong>MissForest vincolato</strong> (50 alberi, min_samples_leaf=5, max_iter=5) riduce l'errore di ricostruzione quadratico (RMSE 0.7337) del <strong>29.3%</strong> rispetto alla mediana.",
      t1_th_imp_method: "Metodo",
      t1_th_imp_rmse: "RMSE (10% mask)",
      t1_th_imp_mae: "MAE",
      t1_th_imp_prauc: "PR-AUC",
      t1_th_imp_auroc: "AUROC",
      t1_th_imp_time: "Tempo",
      t1_leak_title: "Protocollo Rigido di Esclusione Variabili (Anti-Leakage Guard)",
      t1_leak_sub: "85 colonne escluse analiticamente per garantire la validità scientifica",
      t1_leak_badge: "Zero Leakage",
      t1_th_fam: "Famiglia di Esclusione",
      t1_th_vars: "Variabili",
      t1_th_ex: "Esempi di Variabili Escluse",
      t1_th_reas: "Motivazione Metodologica Formale",

      // Tab 02: Phase A
      t2_h2: "02. Fase A: Modelli di Screening & Spiegabilità (SHAP)",
      t2_p: "Valutazione dei cinque modelli candidati addestrati in Cross-Validation Annidata con tuning Optuna su PR-AUC. Analisi di sensibilità no_consequence e importanza esatta TreeSHAP.",
      t2_btn_export: "<span>📥</span> Esporta SVG",
      t2_roc_title: "Curve ROC di Discriminazione sul Test Set",
      t2_roc_sub: "Confronto diretto tra modelli reali e Controllo Positivo (con UmALB)",
      t2_leg_rf: "Random Forest (AUC 0.743)",
      t2_leg_xgb: "XGBoost (AUC 0.738)",
      t2_leg_lr: "LR Penalized (AUC 0.735)",
      t2_leg_pos: "Controllo Positivo (AUC 0.933)",
      t2_leg_scored: "LR SCORED (AUC 0.723)",
      t2_leg_chance: "Caso Casuale (AUC 0.500)",
      t2_feat_title: "Top 10 Feature per Importanza Clinica (SHAP / Odds Ratio)",
      t2_feat_sub: "Confronto tra coefficienti ElasticNet e valori medi assoluti TreeSHAP",
      t2_feat_badge: "Spiegabilità",
      t2_th_rank: "Rango",
      t2_th_lr: "LR ElasticNet (|β|)",
      t2_th_rf: "Random Forest (TreeSHAP)",
      t2_th_xgb: "XGBoost (TreeSHAP)",
      t2_perf_title: "Tabella Completa delle Prestazioni (Fase A)",
      t2_perf_sub: "Metriche Out-Of-Fold (5-fold annidata) e Conferma Finale sul Test Set",
      t2_perf_badge: "Soglia Operativa Fissata a Sensibilità 0.90",
      t2_th_model: "Modello",
      t2_th_oof_auc: "OOF AUROC [IC 95%]",
      t2_th_oof_prauc: "OOF PR-AUC",
      t2_th_test_auc: "Test Set AUROC [IC 95%]",
      t2_th_spec: "Specificità (Sens 0.90)",
      t2_th_alert: "Quota Allerta",
      t2_th_feats: "Feature Usate",

      // Tab 03: Phase B
      t3_h2: "03. Fase B: Smascherare l'Illusione del Bilanciamento delle Classi",
      t3_p: "Confronto metodologico tra valutazione ingenua a soglia fissa 0.50 e valutazione clinica a soglia operativa calibrata con correzione di Holm.",
      t3_badge: "Risultato Metodologico Chiave",
      t3_callout: "<strong>L'Illusione del Bilanciamento (*Class Imbalance Fallacy*):</strong> In letteratura si applica spesso SMOTE e si valuta il recall a soglia 0.50, dichiarando aumenti di sensitività (dal 4.4% al 64.6%). Questo studio dimostra che ciò è una pura illusione dovuta allo spostamento della distribuzione dei punteggi: non appena si adotta una soglia operativa a sensibilità clinica fissata (0.90) o si ricalibrano le probabilità con Platt scaling, <strong>nessuna tecnica di bilanciamento individua un singolo caso grave in più</strong> (p di Holm = 1.000).",
      t3_comp_title: "Confronto Sperimentale: Valutazione Ingenua (0.50) vs Recupero Casi Gravi Reale",
      t3_comp_sub: "Test set da 1.451 soggetti con 23 casi KDIGO gravi (17 ad alto rischio + 6 a rischio molto alto)",
      t3_comp_badge: "Test di Holm Multiplo",
      t3_th_tech: "Tecnica di Bilanciamento",
      t3_th_naive_grp: "Valutazione Ingenua (Soglia Grezza 0.50)",
      t3_th_real_grp: "Soglia Operativa Reale (Sensibilità 0.90)",
      t3_th_rec: "Recall Apparente",
      t3_th_prec: "Precisione",
      t3_th_spec: "Specificità",
      t3_th_sev: "Casi Gravi Trovati",
      t3_th_diff: "Δ vs Baseline",
      t3_th_holm: "p-value Holm",
      t3_ctgan_title: "CTGAN & Metodi Generativi Esplorativi",
      t3_ctgan_p: "L'adozione di reti generative avversarie (CTGAN a 300 epoche) per sintetizzare righe tabulari complete peggiora significativamente la discriminazione rispetto alla baseline (<span style=\"color: var(--accent-rose); font-family: var(--font-mono); font-weight: 600;\">Δ AUROC = -0.117, p = 0.006</span> su LR Penalized). Inoltre, perde 2 casi gravi rispetto al modello naturale.",
      t3_platt_title: "Ricalibrazione Annidata di Platt",
      t3_platt_p: "La ricalibrazione logistica su logit(p) stimata sui fold interni riallinea le probabilità previste alla prevalenza reale della popolazione. Questo riallineamento fa evaporare il vantaggio fittizio delle tecniche di sovracampionamento, riconducendo le curve di decisione esattamente sulla baseline naturale.",

      // Tab 04: Phase D
      t4_h2: "04. Fase D: Diagnostica del Tetto di Prestazione (Information Bottleneck)",
      t4_p: "Esplorazione sistematica di architetture avanzate (TabPFN, CatBoost, LightGBM, EBM, Decomposizione del Target) e identificazione formale del limite biologico rispetto al selection bias.",
      t4_forest_title: "Forest Plot: Valutazione Statistica del Tetto Informativo",
      t4_forest_sub: "Intervalli di confidenza corretti Nadeau-Bengio (5x5 fold) rispetto alla baseline Random Forest",
      t4_card_title: "Perché i Modelli Complessi non Superano la Baseline?",
      t4_ins1_title: "Convergenza Algoritmica (Plateau 0.70-0.74)",
      t4_ins1_badge: "Nadeau-Bengio",
      t4_ins1_p: "Architetture basate su Prior-Data Fitted Networks (TabPFN), Explainable Boosting Machines (EBM) e GBDT avanzati (CatBoost, LightGBM) convergono tutte nello stesso intervallo di prestazione (AUROC 0.692–0.710). Nessuna supera la regola di decisione a priori (Δ ≥ 0.01 con IC 95% > 0).",
      t4_ins2_title: "Decomposizione del Target (eGFR vs Albuminuria)",
      t4_ins2_badge: "Biomarker Split",
      t4_ins2_p: "Separando il target composito, emerge che la componente di filtrazione (eGFR < 60) è altamente prevedibile (AUROC 0.8172), mentre la componente di albuminuria (uACR ≥ 30) presenta un tetto biologico a 0.6865 con sole feature ematochimiche di routine.",
      t4_ins3_title: "Risoluzione del Falso Leakage (Top 21 Features)",
      t4_ins3_p: "Un'analisi preliminare informale otteneva AUROC 0.717 su un ensemble a 21 feature. Ricostruendo rigorosamente il codice, è stato provato che quel risultato era affetto da <strong>data leakage da feature selection</strong> (le 21 feature erano state scelte guardando l'intero dataset, dando AUROC 0.7155 OOF). Applicando la selezione dentro ciascun fold (Top 21 Nested), l'AUROC scende a <strong>0.7031</strong>, perfettamente in linea con i modelli singoli.",
      t4_cand_title: "Candidati Esplorati alla Ricerca del Tetto (Fase D)",
      t4_cand_sub: "Regola di decisione: miglioramento se Δ AUROC ≥ 0.01 e limite inferiore IC 95% Nadeau-Bengio > 0",
      t4_cand_badge: "9 Architetture Valutate",
      t4_th_cand: "Modello Candidato",
      t4_th_fam: "Famiglia Algoritmica",
      t4_th_oof: "AUROC OOF",
      t4_th_diff: "Differenza vs RF",
      t4_th_ci: "IC 95% Nadeau-Bengio",
      t4_th_p: "p-value Holm",
      t4_th_outcome: "Esito Regola",

      // Tab 05: Final Test & DCA
      t5_h2: "05. Lockbox Test Set & Decision Curve Analysis (DCA)",
      t5_p: "Validazione formale dei Claim scientifici su 1.451 soggetti non visti e simulazione dell'impatto clinico ed economico con curve di beneficio netto.",
      t5_btn_export: "<span>📥</span> Esporta DCA SVG",
      t5_lockbox_title: "Lockbox Confermato & Dati Sigillati",
      t5_lockbox_hash: "SHA-256 Lockbox Test Set: 7a8f3b... Congelato prima di qualsiasi calcolo",
      t5_lockbox_badge: "1.451 Soggetti Isolati (20%)",
      t5_dca_title: "Curve di Decisione Clinica (Decision Curve Analysis)",
      t5_dca_sub: "Beneficio Netto standardizzato per soglia di preferenza clinica (Pt) rispetto alle strategie estreme",
      t5_leg_rf: "Random Forest (K-Risk)",
      t5_leg_xgb: "XGBoost",
      t5_leg_all: "Testare Tutti",
      t5_leg_none: "Non Testare Nessuno (Zero)",
      t5_sim_title: "Simulatore della Soglia di Rischio Clinico",
      t5_sim_badge: "Interattivo",
      t5_sim_p: "Modifica la probabilità di soglia (P<sub>t</sub>) per osservare come varia l'accettazione del trade-off tra falsi positivi (esami urinari specialistici inutili) e falsi negativi (casi renali persi).",
      t5_sim_slider: "Soglia Decisionale (P<sub>t</sub>):",
      t5_sim_cost: "* Costo medio stimato di riferimento per test urinario ACR: <strong>49 €</strong> (intervallo 36–64 €, studio Cusick et al. 2023).",

      // Tab 06: Results & Positioning
      t6_h2: "06. Risultati Ottenuti & Posizionamento Scientifico di K-Risk",
      t6_p: "Sintesi delle evidenze empiriche, collocazione di K-Risk nel continuum diagnostico e confronto con i modelli dello stato dell'arte.",
      t6_flow_title: "Collocazione di K-Risk nel Continuum Diagnostico della Malattia Renale",
      t6_flow_sub: "Architettura di triage a due livelli: dal medico di medicina generale allo specialista nefrologo",
      t6_sota_title: "Matrice Comparativa sullo Stato dell'Arte Internazionale",
      t6_sota_sub: "Confronto metodologico tra K-Risk e i modelli di riferimento per bersaglio, predittori ed esami renali",
      t6_sota_badge: "Benchmark",
      t6_th_mod: "Modello / Studio",
      t6_th_target: "Bersaglio Clinico",
      t6_th_exams: "Esami Renali Usati come Input",
      t6_th_auc: "AUROC Validata",
      t6_th_role: "Collocazione & Note Metodologiche",
      t6_acc1_title: "1. Utilità Decisionale Dimostrata",
      t6_acc1_p: "Attraverso la Decision Curve Analysis, alla soglia del 7% K-Risk evita <strong>43.2 esami urinari inutili ogni 100 persone</strong> rispetto a \"testare tutti\", garantendo un risparmio stimato di <strong>2.116 € per 100 soggetti</strong> senza compromettere la diagnosi dei casi gravi.",
      t6_acc2_title: "2. Superiorità sul Benchmark SCORED",
      t6_acc2_p: "K-Risk supera coerentemente il riferimento storico SCORED (Bang et al.) su AUROC (0.743 vs 0.723 sul test), PR-AUC e Net Benefit, pur non richiedendo l'esame della proteinuria nelle urine.",
      t6_acc3_title: "3. Tetto Informativo Spiegato con i Dati",
      t6_acc3_p: "Il plateau ad AUROC ~0.70–0.74 è documentato empiricamente: l'eGFR si predice con alta precisione (AUROC > 0.80), mentre l'albuminuria isolata ha un limite biologico di separabilità con soli esami ematochimici. Il controllo positivo (0.933) certifica l'efficacia algoritmica.",

      // Footer
      footer_left: "<strong>K-Risk</strong> · Piattaforma Scientifica di Screening Non-Invasivo della Malattia Renale Cronica",
      footer_arch: "Architettura Certificata & Protocollo Congelato",
      t1_leg_low: "<span style=\"width: 10px; height: 10px; background: #0284c7; border-radius: 2px;\"></span> Basso Rischio (5.234 - 90.2%)",
      t1_leg_mod: "<span style=\"width: 10px; height: 10px; background: #d97706; border-radius: 2px;\"></span> Moderato (473 - 8.2%)",
      t1_leg_high: "<span style=\"width: 10px; height: 10px; background: #ea580c; border-radius: 2px;\"></span> Alto (67 - 1.2%)",
      t1_leg_vhigh: "<span style=\"width: 10px; height: 10px; background: #dc2626; border-radius: 2px;\"></span> Molto Alto (27 - 0.5%)",
      t2_leg_pos: "<span class=\"legend-line\" style=\"background: #10b981;\"></span> Controllo Positivo (AUC 0.933)",
      t2_leg_rf: "<span class=\"legend-line\" style=\"background: #2563eb;\"></span> Random Forest (AUC 0.743)",
      t2_leg_xgb: "<span class=\"legend-line\" style=\"background: #06b6d4;\"></span> XGBoost (AUC 0.725)",
      t2_leg_scored: "<span class=\"legend-line\" style=\"background: #f59e0b;\"></span> LR SCORED (AUC 0.723)",
      t2_leg_chance: "<span class=\"legend-line\" style=\"background: #64748b; border-style: dashed;\"></span> Caso Casuale (AUC 0.500)",
      t3_th_rec: "Recall Apparente",
      t3_th_prec: "Precisione",
      t3_th_spec: "Specificit\u00e0",
      t3_th_sev: "Casi Gravi Trovati",
      t3_th_diff: "Delta vs Baseline",
      t3_th_holm: "p-value di Holm",
      t4_forest_title: "Forest Plot: Test Corretto di Nadeau-Bengio sui Fold",
      t4_forest_sub: "Differenza di AUROC rispetto al riferimento (Random Forest) con IC 95%",
      t4_card_title: "Le Due Diagnostiche Fondamentali",
      t4_card_sub: "Verifica della capacit\u00e0 di apprendimento vs smascheramento del leakage",
      t4_card_badge: "Evidenze Empiriche",
      t4_diag1_title: "1. Controllo Positivo (+ UmALB)",
      t4_diag1_p: "Aggiungendo la concentrazione di albumina urinaria tra le feature, la pipeline raggiunge istantaneamente un'AUROC di <strong>0.933</strong> (PR-AUC 0.809). Ci\u00f2 certifica che il plateau a ~0.70 non \u00e8 dovuto a limitazioni dell'architettura di apprendimento, bens\u00ec alla separabilit\u00e0 intrinseca delle sole variabili ematochimiche generali.",
      t4_diag2_title: "2. Dimostrazione del Selection Bias (Ambroise & McLachlan)",
      t4_diag2_p: "Un'analisi preliminare informale su 21 feature selezionate prima della CV (procedura scorretta) produceva un'AUROC fittizia di <strong>0.718</strong> contro lo <strong>0.703</strong> reale di Random Forest. Questo ha confermato empiricamente la necessit\u00e0 della nested CV.",
      t5_h2: "05. Conferma Finale sul Test Set & Utilit\u00e0 Clinica (DCA)",
      t5_p: "Esecuzione autorizzata a protocollo congelato (Lockbox) sui 1.451 soggetti di test. Verifica dei 5 Claim scientifici e Decision Curve Analysis di Vickers.",
      t5_lockbox_title: "<span>\ud83d\udd12</span> Test Set Lockbox Ufficiale: Esecuzione Conclusa",
      t5_lockbox_hash: "SHA256: 9d3fc900a6e8b621fe8e91e6fa8f7cf0c7b3d277825b948b2c45044cbcbf5a75 | Data: 22/09/2026 | Run ID: 1",
      t5_claims_badge: "Tutti i 5 Claim Convalidati",
      t5_dca_title: "Decision Curve Analysis (DCA) & Simulatore di Utilit\u00e0 Economico-Sanitaria",
      t5_dca_desc: "Valutazione del Net Benefit clinico reale (Vickers & Elkin) rispetto all'approccio indiscriminato ('Testare Tutti') e all'inazione ('Nessuno').",
      t5_curves_title: "Curve di Net Benefit (Decision Curves)",
      t5_curves_sub: "Soglie decisionali cliniche da 0.02 a 0.20 (passo 0.005)",
      t5_leg_rf: "<span class=\"legend-line\" style=\"background: #2563eb;\"></span> Random Forest",
      t5_leg_xgb: "<span class=\"legend-line\" style=\"background: #06b6d4;\"></span> XGBoost",
      t5_leg_lr: "<span class=\"legend-line\" style=\"background: #f59e0b;\"></span> LR Penalized",
      t5_leg_all: "<span class=\"legend-line\" style=\"background: #64748b; border-style: dashed;\"></span> Testare Tutti",
      t5_leg_none: "<span class=\"legend-line\" style=\"background: #94a3b8;\"></span> Non Testare Nessuno (Zero)",
      t6_h2: "06. Risultati Ottenuti & Posizionamento Scientifico di K-Risk",
      t6_p: "Mappatura di K-Risk lungo il percorso diagnostico-terapeutico e confronto oggettivo con i principali modelli della letteratura internazionale.",
      t6_badge: "Valore Scientifico",
      t6_callout: "<strong>Dove si colloca K-Risk:</strong> K-Risk non compete con i modelli prognostici a valle (come il celebre KFRE) che richiedono esami specialistici gi\u00e0 eseguiti, n\u00e9 sostituisce i test di laboratorio. Si posiziona <strong>a monte</strong>, agendo come filtro intelligente a zero costo diagnostico aggiuntivo per selezionare i pazienti a cui prescrivere l'approfondimento nefrologico.",
      t6_flow_title: "Schema del Continuum Clinico: Il Ruolo Esatto di K-Risk",
      t6_flow_sub: "Collocazione nel percorso di cura dalla popolazione sana alla nefrologia specialistica",
      t6_flow_badge: "Pipeline Clinica",
      footer_left: "<strong>K-Risk</strong> · Piattaforma Scientifica di Screening Non-Invasivo della Malattia Renale Cronica",
      footer_arch: "Architettura Certificata & Protocollo Congelato",
    },

    en: {
      // Document Metadata
      page_title: "K-Risk — Scientific Non-Invasive Screening Platform for Chronic Kidney Disease",
      page_description: "K-Risk scientific platform: first-level clinical triage for chronic kidney disease risk estimation without specialist laboratory tests. Decision Curve Analysis and performance ceiling diagnostics.",
      
      // Topbar & Navigation
      git_chip_title: "Commit hash registered at protocol freeze",
      theme_btn_title: "Toggle theme (Light / Dark)",
      nav_scope: "Scope",
      nav_cohort: "Cohort",
      nav_phase_a: "Phase A",
      nav_phase_b: "Phase B",
      nav_phase_d: "Phase D",
      nav_final_test: "Lockbox & DCA",
      nav_results: "Results",

      // Hero Bento Header
      hero_title: "Welcome in, K-Risk",
      hero_subtitle: "Non-Invasive Screening for Chronic Kidney Disease · 1st-Level Clinical Triage & Decision Curve Analysis",
      prev_label_1: "KDIGO Low (90.2%)",
      prev_label_2: "Moderate (8.2%)",
      prev_label_3: "High (1.2%)",
      prev_label_4: "Very High (0.5%)",
      stat_cohort_label: "Total Cohort",
      stat_positives_label: "KDIGO Positives",
      stat_renal_exams_label: "Renal Exams",
      stat_auroc_label: "Test AUROC",

      // Tab 00: Scope
      scope_h2: "00. K-Risk Formal Project Scope",
      scope_p: "Definition of clinical goals, operational scope, target population, architectural constraints, and scientific foundations.",
      scope_badge: "Registered Protocol",
      scope_callout: "<strong>Clinical & Operational Summary:</strong> K-Risk trains and validates classification models that, <strong>without any renal tests</strong>, estimate who exhibits chronic kidney disease (CKD) markers, benchmark predictions against <strong>KDIGO</strong> risk stratification, and rigorously measure whether balancing techniques improve detection of severe cases.",
      scope_sheet_title: "Clinical Scope Definition Matrix",
      scope_q_title: "The 6 Core Scientific Questions",
      scope_q1: "<strong>Discrimination:</strong> Can the model distinguish subjects with renal disease markers? (AUROC, PR-AUC, operating point).",
      scope_q2: "<strong>Risk Gradient:</strong> Does predicted risk grow monotonically with KDIGO severity? (Jonckheere-Terpstra trend test).",
      scope_q3: "<strong>Severe Cases:</strong> How many high and very high risk cases are caught? Missing a 'very high' is the costliest clinical error.",
      scope_q4: "<strong>Clinical Calibration:</strong> Do risk strata correspond to real population stratification? (Weighted Kappa index).",
      scope_q5: "<strong>Imbalance Fallacy:</strong> Does synthetic balancing (SMOTE/CTGAN) improve severe cases or only mean fixed-threshold metrics?",
      scope_q6: "<strong>Diabetic Subgroup:</strong> How does the model perform in diabetic individuals compared to the general population?",
      scope_rules_title: "Strict Methodological Principles",
      scope_r1: "<strong>Data Augmentation on training only:</strong> Resampling techniques operate exclusively inside each inner training fold; zero synthetic data touches validation or test.",
      scope_r2: "<strong>Mandatory Platt Scaling:</strong> Probabilities distorted by synthetic oversampling are recalibrated on genuine data prior to clinical evaluation.",
      scope_r3: "<strong>KDIGO Benchmarking on real data only:</strong> Synthetic records do not possess genuine biological KDIGO risk categories.",
      scope_r4: "<strong>Frozen Test Set (Lockbox):</strong> 1,451 subjects never accessed during development; unsealed only once for final pre-registered claim validation.",

      // Tab 01: Cohort & Data Integrity
      t1_h2: "01. Clinical Cohort, KDIGO Matrix & Zero Data Leakage",
      t1_p: "Dataset preparation: eGFR recalculation via CKD-EPI 2021 equation, 25% stratified frozen test split, and constrained MissForest imputation protocol.",
      t1_badge: "A Priori Fixed Protocol",
      t1_kdigo_title: "2D KDIGO Risk Matrix (6 × 3)",
      t1_kdigo_sub: "Observed distribution of 5,801 subjects across eGFR × ACR grid",
      badge_interactive: "Interactive",
      t1_leg_low: "Low Risk (5,234 - 90.2%)",
      t1_leg_mod: "Moderate (473 - 8.2%)",
      t1_leg_high: "High (67 - 1.2%)",
      t1_leg_vhigh: "Very High (27 - 0.5%)",
      t1_imp_title: "Imputation Method Benchmark (Training Only)",
      t1_imp_sub: "5-fold Cross-Validation with 10% masking on observed values",
      t1_imp_badge: "5-Fold CV",
      t1_imp_p: "Downstream discrimination (PR-AUC ~0.257) remains invariant to method, but <strong>constrained MissForest</strong> (50 trees, min_samples_leaf=5, max_iter=5) reduces root mean squared error (RMSE 0.7337) by <strong>29.3%</strong> compared to median.",
      t1_th_imp_method: "Method",
      t1_th_imp_rmse: "RMSE (10% mask)",
      t1_th_imp_mae: "MAE",
      t1_th_imp_prauc: "PR-AUC",
      t1_th_imp_auroc: "AUROC",
      t1_th_imp_time: "Time",
      t1_leak_title: "Strict Feature Exclusion Protocol (Anti-Leakage Guard)",
      t1_leak_sub: "85 columns analytically excluded to guarantee scientific validity",
      t1_leak_badge: "Zero Leakage",
      t1_th_fam: "Exclusion Family",
      t1_th_vars: "Features",
      t1_th_ex: "Examples of Excluded Features",
      t1_th_reas: "Formal Methodological Justification",

      // Tab 02: Phase A
      t2_h2: "02. Phase A: Screening Models & Explainability (SHAP)",
      t2_p: "Evaluation of five candidate models trained via Nested Cross-Validation with Optuna tuning on PR-AUC. Sensitivity analysis and exact TreeSHAP importance.",
      t2_btn_export: "<span>📥</span> Export SVG",
      t2_roc_title: "Test Set Discrimination ROC Curves",
      t2_roc_sub: "Direct comparison between real models and Positive Control (with UmALB)",
      t2_leg_rf: "Random Forest (AUC 0.743)",
      t2_leg_xgb: "XGBoost (AUC 0.738)",
      t2_leg_lr: "LR Penalized (AUC 0.735)",
      t2_leg_pos: "Positive Control (AUC 0.933)",
      t2_leg_scored: "LR SCORED (AUC 0.723)",
      t2_leg_chance: "Random Chance (AUC 0.500)",
      t2_feat_title: "Top 10 Features by Clinical Importance (SHAP / Odds Ratio)",
      t2_feat_sub: "Comparison between ElasticNet coefficients and mean absolute TreeSHAP values",
      t2_feat_badge: "Explainability",
      t2_th_rank: "Rank",
      t2_th_lr: "LR ElasticNet (|β|)",
      t2_th_rf: "Random Forest (TreeSHAP)",
      t2_th_xgb: "XGBoost (TreeSHAP)",
      t2_perf_title: "Comprehensive Performance Table (Phase A)",
      t2_perf_sub: "Out-Of-Fold metrics (5-fold nested) and final Lockbox Test Set Confirmation",
      t2_perf_badge: "Operating Threshold Fixed at 0.90 Sensitivity",
      t2_th_model: "Model",
      t2_th_oof_auc: "OOF AUROC [95% CI]",
      t2_th_oof_prauc: "OOF PR-AUC",
      t2_th_test_auc: "Test Set AUROC [95% CI]",
      t2_th_spec: "Specificity (Sens 0.90)",
      t2_th_alert: "Alert Rate",
      t2_th_feats: "Features Used",

      // Tab 03: Phase B
      t3_h2: "03. Phase B: Unmasking the Class Imbalance Fallacy",
      t3_p: "Methodological comparison between naive evaluation at fixed 0.50 threshold and clinical evaluation at calibrated operating threshold with Holm correction.",
      t3_badge: "Key Methodological Result",
      t3_callout: "<strong>The Class Imbalance Fallacy:</strong> Literature often applies SMOTE and evaluates recall at an arbitrary 0.50 threshold, claiming sensitivity increases (from 4.4% to 64.6%). This study proves this is a pure mathematical illusion caused by the shift in score distribution: as soon as an operating threshold at fixed clinical sensitivity (0.90) is adopted or probabilities are recalibrated via Platt scaling, <strong>no balancing technique identifies a single additional severe case</strong> (Holm p = 1.000).",
      t3_comp_title: "Experimental Comparison: Naive Evaluation (0.50) vs Real Severe Case Detection",
      t3_comp_sub: "Test set of 1,451 subjects with 23 severe KDIGO cases (17 high risk + 6 very high risk)",
      t3_comp_badge: "Multiple Holm Test",
      t3_th_tech: "Balancing Technique",
      t3_th_naive_grp: "Naive Evaluation (Raw 0.50 Threshold)",
      t3_th_real_grp: "Real Operating Threshold (Sensitivity 0.90)",
      t3_th_rec: "Apparent Recall",
      t3_th_prec: "Precision",
      t3_th_spec: "Specificity",
      t3_th_sev: "Severe Cases Caught",
      t3_th_diff: "Δ vs Baseline",
      t3_th_holm: "Holm p-value",
      t3_ctgan_title: "CTGAN & Exploratory Generative Methods",
      t3_ctgan_p: "Adopting generative adversarial networks (CTGAN at 300 epochs) to synthesize complete tabular rows significantly impairs discrimination compared to baseline (<span style=\"color: var(--accent-rose); font-family: var(--font-mono); font-weight: 600;\">Δ AUROC = -0.117, p = 0.006</span> on LR Penalized). Furthermore, it misses 2 severe cases compared to the natural model.",
      t3_platt_title: "Nested Platt Scaling Recalibration",
      t3_platt_p: "Logistic recalibration on logit(p) estimated across inner folds realigns predicted probabilities with genuine population prevalence. This realignment eliminates the artificial benefit of oversampling techniques, returning decision curves exactly to the natural baseline.",

      // Tab 04: Phase D
      t4_h2: "04. Phase D: Performance Ceiling Diagnostics (Information Bottleneck)",
      t4_p: "Systematic exploration of advanced architectures (TabPFN, CatBoost, LightGBM, EBM, Target Decomposition) and formal identification of the biological ceiling vs selection bias.",
      t4_forest_title: "Forest Plot: Statistical Assessment of the Information Bottleneck",
      t4_forest_sub: "Corrected Nadeau-Bengio confidence intervals (5x5 fold) relative to Random Forest baseline",
      t4_card_title: "Why Don't Complex Models Outperform the Baseline?",
      t4_ins1_title: "Algorithmic Convergence (0.70-0.74 Plateau)",
      t4_ins1_badge: "Nadeau-Bengio",
      t4_ins1_p: "Architectures based on Prior-Data Fitted Networks (TabPFN), Explainable Boosting Machines (EBM), and advanced GBDTs (CatBoost, LightGBM) all converge within the identical performance range (AUROC 0.692–0.710). None surpasses the a priori decision rule (Δ ≥ 0.01 with 95% CI > 0).",
      t4_ins2_title: "Target Decomposition (eGFR vs Albuminuria)",
      t4_ins2_badge: "Biomarker Split",
      t4_ins2_p: "Decomposing the composite target reveals that filtration impairment (eGFR < 60) is highly predictable (AUROC 0.8172), whereas albuminuria (uACR ≥ 30) exhibits a biological ceiling at 0.6865 using routine blood chemistry alone.",
      t4_ins3_title: "Resolution of Spurious Leakage (Top 21 Features)",
      t4_ins3_p: "A preliminary informal analysis obtained AUROC 0.717 on a 21-feature ensemble. Rebuilding the code rigorously proved that result suffered from <strong>feature selection leakage</strong> (the 21 features were selected looking at the whole dataset, yielding 0.7155 OOF AUROC). When selection is strictly nested within each fold (Top 21 Nested), AUROC drops to <strong>0.7031</strong>, aligning perfectly with single models.",
      t4_cand_title: "Candidates Evaluated for Ceiling Diagnostics (Phase D)",
      t4_cand_sub: "Decision rule: improvement if Δ AUROC ≥ 0.01 and lower bound 95% CI Nadeau-Bengio > 0",
      t4_cand_badge: "9 Architectures Evaluated",
      t4_th_cand: "Candidate Model",
      t4_th_fam: "Model Family",
      t4_th_oof: "OOF AUROC",
      t4_th_diff: "Difference vs RF",
      t4_th_ci: "95% CI Nadeau-Bengio",
      t4_th_p: "Holm p-value",
      t4_th_outcome: "Rule Outcome",

      // Tab 05: Final Test & DCA
      t5_h2: "05. Lockbox Test Set & Decision Curve Analysis (DCA)",
      t5_p: "Formal validation of scientific claims on 1,451 unseen subjects and clinical/economic impact simulation via net benefit curves.",
      t5_btn_export: "<span>📥</span> Export DCA SVG",
      t5_lockbox_title: "Lockbox Confirmed & Sealed Data",
      t5_lockbox_hash: "Lockbox Test Set SHA-256: 7a8f3b... Frozen prior to any model computation",
      t5_lockbox_badge: "1,451 Isolated Subjects (20%)",
      t5_dca_title: "Clinical Decision Curve Analysis (DCA)",
      t5_dca_sub: "Standardized Net Benefit across clinical preference thresholds (Pt) vs extreme strategies",
      t5_leg_rf: "Random Forest (K-Risk)",
      t5_leg_xgb: "XGBoost",
      t5_leg_all: "Test All",
      t5_leg_none: "Treat None (Zero)",
      t5_sim_title: "Clinical Risk Threshold Simulator",
      t5_sim_badge: "Interactive",
      t5_sim_p: "Adjust the threshold probability (P<sub>t</sub>) to examine the trade-off between false positives (unnecessary specialist urine tests) and false negatives (missed renal cases).",
      t5_sim_slider: "Decision Threshold (P<sub>t</sub>):",
      t5_sim_cost: "* Estimated benchmark cost per uACR specialist test: <strong>€49</strong> (range €36–€64, Cusick et al. 2023 study).",

      // Tab 06: Results & Positioning
      t6_h2: "06. Project Results & Scientific Positioning of K-Risk",
      t6_p: "Synthesis of empirical findings, K-Risk's role within the diagnostic continuum, and comparison with state-of-the-art models.",
      t6_flow_title: "K-Risk in the Chronic Kidney Disease Diagnostic Continuum",
      t6_flow_sub: "Two-tiered triage architecture: from primary care to nephrologist referral",
      t6_sota_title: "State of the Art International Benchmark Matrix",
      t6_sota_sub: "Methodological benchmark between K-Risk and established models by target, predictors, and renal test dependency",
      t6_sota_badge: "Benchmark",
      t6_th_mod: "Model / Study",
      t6_th_target: "Clinical Target",
      t6_th_exams: "Renal Exams Used as Input",
      t6_th_auc: "Validated AUROC",
      t6_th_role: "Role & Methodological Notes",
      t6_acc1_title: "1. Demonstrated Clinical Decision Utility",
      t6_acc1_p: "Through Decision Curve Analysis, at the 7% threshold K-Risk spares <strong>43.2 unnecessary urine tests per 100 patients</strong> compared to \"testing everyone\", providing an estimated savings of <strong>€2,116 per 100 subjects</strong> without missing severe cases.",
      t6_acc2_title: "2. Superiority Over SCORED Benchmark",
      t6_acc2_p: "K-Risk consistently outperforms the historical SCORED benchmark (Bang et al.) in AUROC (0.743 vs 0.723 on test set), PR-AUC, and Net Benefit, without requiring urine proteinuria self-assessment.",
      t6_acc3_title: "3. Performance Ceiling Explained by Data",
      t6_acc3_p: "The ~0.70–0.74 AUROC plateau is empirically documented: eGFR is predicted with high accuracy (AUROC > 0.80), whereas isolated albuminuria has a biological separability limit using routine blood tests alone. The positive control (0.933) proves algorithmic competence.",

      // Footer
      footer_left: "<strong>K-Risk</strong> · Scientific Non-Invasive Screening Platform for Chronic Kidney Disease",
      footer_arch: "Certified Architecture & Frozen Protocol",
      t1_leg_low: "<span style=\"width: 10px; height: 10px; background: #0284c7; border-radius: 2px;\"></span> Low Risk (5,234 - 90.2%)",
      t1_leg_mod: "<span style=\"width: 10px; height: 10px; background: #d97706; border-radius: 2px;\"></span> Moderate (473 - 8.2%)",
      t1_leg_high: "<span style=\"width: 10px; height: 10px; background: #ea580c; border-radius: 2px;\"></span> High (67 - 1.2%)",
      t1_leg_vhigh: "<span style=\"width: 10px; height: 10px; background: #dc2626; border-radius: 2px;\"></span> Very High (27 - 0.5%)",
      t2_leg_pos: "<span class=\"legend-line\" style=\"background: #10b981;\"></span> Positive Control (AUC 0.933)",
      t2_leg_rf: "<span class=\"legend-line\" style=\"background: #2563eb;\"></span> Random Forest (AUC 0.743)",
      t2_leg_xgb: "<span class=\"legend-line\" style=\"background: #06b6d4;\"></span> XGBoost (AUC 0.725)",
      t2_leg_scored: "<span class=\"legend-line\" style=\"background: #f59e0b;\"></span> LR SCORED (AUC 0.723)",
      t2_leg_chance: "<span class=\"legend-line\" style=\"background: #64748b; border-style: dashed;\"></span> Random Chance (AUC 0.500)",
      t3_th_rec: "Apparent Recall",
      t3_th_prec: "Precision",
      t3_th_spec: "Specificity",
      t3_th_sev: "Severe Cases Caught",
      t3_th_diff: "Delta vs Baseline",
      t3_th_holm: "Holm p-value",
      t4_forest_title: "Forest Plot: Nadeau-Bengio Adjusted Fold Test",
      t4_forest_sub: "AUROC difference relative to baseline (Random Forest) with 95% CI",
      t4_card_title: "The Two Core Diagnostic Inquiries",
      t4_card_sub: "Assessment of learning capacity vs leakage unmasking",
      t4_card_badge: "Empirical Evidence",
      t4_diag1_title: "1. Positive Control (+ UmALB)",
      t4_diag1_p: "Adding urinary albumin concentration to predictors immediately yields an AUROC of <strong>0.933</strong> (PR-AUC 0.809). This rigorously proves that the ~0.70 plateau is not caused by algorithmic learning capacity limits, but by the intrinsic biological separability of general blood chemistry.",
      t4_diag2_title: "2. Demonstration of Selection Bias (Ambroise & McLachlan)",
      t4_diag2_p: "An informal preliminary analysis on 21 features selected prior to CV (flawed protocol) produced an artificial AUROC of <strong>0.718</strong> versus the true <strong>0.703</strong> of Random Forest. This empirically demonstrated the absolute necessity of nested CV.",
      t5_h2: "05. Final Lockbox Test Set & Clinical Utility (DCA)",
      t5_p: "Authorized execution under frozen protocol on 1,451 held-out test subjects. Verification of the 5 scientific claims and Vickers Decision Curve Analysis.",
      t5_lockbox_title: "<span>\ud83d\udd12</span> Official Lockbox Test Set: Execution Complete",
      t5_lockbox_hash: "SHA256: 9d3fc900a6e8b621fe8e91e6fa8f7cf0c7b3d277825b948b2c45044cbcbf5a75 | Date: 22/09/2026 | Run ID: 1",
      t5_claims_badge: "All 5 Claims Confirmed",
      t5_dca_title: "Decision Curve Analysis (DCA) & Health-Economic Utility Simulator",
      t5_dca_desc: "Assessment of true clinical Net Benefit (Vickers & Elkin) compared against the indiscriminate strategy ('Test All') and inaction ('None').",
      t5_curves_title: "Net Benefit Curves (Decision Curves)",
      t5_curves_sub: "Clinical decision thresholds from 0.02 to 0.20 (0.005 step)",
      t5_leg_rf: "<span class=\"legend-line\" style=\"background: #2563eb;\"></span> Random Forest",
      t5_leg_xgb: "<span class=\"legend-line\" style=\"background: #06b6d4;\"></span> XGBoost",
      t5_leg_lr: "<span class=\"legend-line\" style=\"background: #f59e0b;\"></span> LR Penalized",
      t5_leg_all: "<span class=\"legend-line\" style=\"background: #64748b; border-style: dashed;\"></span> Test All",
      t5_leg_none: "<span class=\"legend-line\" style=\"background: #94a3b8;\"></span> Test None (Zero)",
      t6_h2: "06. Project Results & Scientific Positioning of K-Risk",
      t6_p: "Mapping K-Risk along the diagnostic continuum and objective comparison against primary literature benchmarks.",
      t6_badge: "Scientific Value",
      t6_callout: "<strong>Where K-Risk stands:</strong> K-Risk does not compete with downstream prognostic tools (such as KFRE) which require already completed specialist testing, nor does it replace laboratory testing. It sits <strong>upstream</strong> as an intelligent zero-extra-cost filter to triage patients who require nephrology workup.",
      t6_flow_title: "Clinical Continuum Architecture: The Exact Role of K-Risk",
      t6_flow_sub: "Placement across the care pathway from healthy population to specialist nephrology",
      t6_flow_badge: "Clinical Pipeline",
    }
  },

  // Bilingual Dynamic Datasets
  data: {
    it: {
      scope_cards: [
        {
          id: "problem",
          num: "01",
          title: "Problema Clinico",
          summary: "La Malattia Renale Cronica (MRC) è asintomatica fino agli stadi terminali (ESRD) e colpisce oltre il 10% della popolazione.",
          detail: "Lo screening tradizionale richiede il dosaggio di creatinina ed esami urinari (uACR) che spesso non vengono prescritti in medicina generale fino alla comparsa di complicanze severe. L'intervento precoce con farmaci organo-protettori (SGLT2-inibitori, RAS-bloccanti) rallenta drasticamente il declino dell'eGFR se somministrato tempestivamente."
        },
        {
          id: "target",
          num: "02",
          title: "Definizione Rigorosa del Target",
          summary: "Costrutto KDIGO 2024 composito: eGFR < 60 mL/min/1.73m² (G3a-G5) OPPURE uACR ≥ 30 mg/g (A2-A3).",
          detail: "A differenza dei modelli focalizzati sul solo eGFR (che mancano i pazienti diabetici con iniziale danno glomerulare a filtrazione conservata), K-Risk cattura l'intero spettro di danno renale manifesto, stratificato secondo le linee guida internazionali."
        },
        {
          id: "inputs",
          num: "03",
          title: "Predittori Ammessi (21 Feature)",
          summary: "Esami ematochimici di routine di 1° livello, parametri demografici e antropometrici standard.",
          detail: "Età, Sesso, BMI, Pressione Sistolica/Diastolica, Enzimi Epatici (AST, ALT, Fosfatasi Alcalina, GGT), Indice di Fibrosi Epatica FIB-4, Emoglobina Glicata (HbA1c), Peptide C, Trigliceridi, Colesterolo HDL/Totale, Acido Urico, Emocromo (Piasrine, Globuli Rossi, Leucociti). Nessun marcatore specialistico."
        },
        {
          id: "constraints",
          num: "04",
          title: "Vincolo Metodologico Rigido",
          summary: "Totale e permanente esclusione di Creatinina Sierica, Cistatina C e qualsiasi analisi delle urine.",
          detail: "L'uso di tali esami costituirebbe target leakage circolare (predire una formula conoscendone i coefficienti interni). K-Risk è progettato espressamente per operare a monte della prescrizione specialistica, non a valle."
        },
        {
          id: "role",
          num: "05",
          title: "Ruolo Operativo nel Percorso Clinico",
          summary: "Filtro di triage opportunistico pre-laboratorio per la Medicina Generale e la Medicina del Lavoro.",
          detail: "Non emette una diagnosi formale, ma genera un indice di priorità clinica che ottimizza la richiesta mirata di esami di secondo livello (uACR e creatinina), evitando esami inutili nel 94% della popolazione a basso rischio."
        },
        {
          id: "not",
          num: "06",
          title: "Cosa NON è K-Risk",
          summary: "Non è un surrogato della nefrologia, né un calcolatore di dialisi a medio-lungo termine.",
          detail: "Non stima la progressione terminale del rene già malato (ruolo proprio del KFRE), non calcola la clearance renale e non sostituisce l'esame obiettivo del medico. È uno strumento di prioritarizzazione per soggetti apparentemente asintomatici."
        }
      ],
      positioning: {
        state_of_art_matrix: [
          {
            model: "K-Risk (Questo Studio / Coppola)",
            target: "Stadio KDIGO Moderato/Grave (eGFR < 60 o uACR ≥ 30)",
            renal_exams_used: "Nessuno (Zero Creatinina, Zero Urine)",
            auroc: "0.743",
            role: "Triage di 1° livello pre-laboratorio. Dimostrata utilità clinica con DCA.",
            status: "krisk"
          },
          {
            model: "SCORED (Bang et al. 2008)",
            target: "eGFR < 60 mL/min/1.73m²",
            renal_exams_used: "Autocertificazione proteinuria urinaria",
            auroc: "0.723",
            role: "Score clinico a punti. Superato da K-Risk su AUROC, PR-AUC e Net Benefit.",
            status: "benchmark"
          },
          {
            model: "Bragg-Gresham et al. 2018",
            target: "CKD non diagnosticata",
            renal_exams_used: "Richiede Creatinina Sierica o eGFR",
            auroc: "0.840",
            role: "Non applicabile a monte del laboratorio perché presuppone già l'esame renale.",
            status: "reference"
          },
          {
            model: "KFRE (Tangri et al. 2011/2016)",
            target: "Progressione a Dialisi/ESRD a 2-5 anni",
            renal_exams_used: "Creatinina Sierica + Albuminuria (uACR)",
            auroc: "0.850",
            role: "Modello prognostico di 2° livello: non fa screening ma stima la progressione terminale.",
            status: "reference"
          },
          {
            model: "Nelson et al. 2019",
            target: "Declino eGFR e Mortalità",
            renal_exams_used: "eGFR basale + uACR basale",
            auroc: "0.880",
            role: "Valutazione del rischio a lungo termine in popolazioni con CKD già nota.",
            status: "reference"
          },
          {
            model: "Echouffo-Tcheugui & Kengne 2012",
            target: "Incidenza di CKD a 5-10 anni",
            renal_exams_used: "Creatinina e/o Proteinuria",
            auroc: "0.760",
            role: "Revisione sistematica: la quasi totalità dei modelli richiede biomarcatori specifici.",
            status: "reference"
          }
        ],
        continuum_flow: [
          {
            step: "1",
            title: "Cittadino / Medico di Base",
            context: "Dati anagrafici, pressione arteriosa, esami del sangue di routine (AST, ALT, ALP, emocromo, glicemia, colesterolo). Nessun esame urinario specialistico.",
            action: "Raccolta esami standard già disponibili",
            highlight: false
          },
          {
            step: "2",
            title: "K-Risk (Algoritmo di Screening)",
            context: "Calcolo del rischio composito KDIGO. Se il punteggio supera la soglia operativa (Pt = 0.07), scatta l'allerta clinica e si raccomanda la conferma diagnostica.",
            action: "Evita il 43% di prescrizioni urinarie inutili",
            highlight: true
          },
          {
            step: "3",
            title: "Laboratorio di Analisi / uACR",
            context: "Esecuzione mirata del dosaggio urinario del rapporto albumina/creatinina (uACR) e creatinina sierica solo per i soggetti con allerta positiva K-Risk.",
            action: "Diagnosi formale KDIGO confermata",
            highlight: false
          },
          {
            step: "4",
            title: "Stadiazione & Terapia Nefrologica",
            context: "Nei soggetti confermati si applicano modelli di prognosi come KFRE e si avviano terapie organo-protettive precoci (SGLT2i, RAS-bloccanti, controllo pressorio).",
            action: "Intervento precoce salva-rene",
            highlight: false
          }
        ]
      },
      leakage_guard: {
        excluded_categories: [
          {
            name: "Componenti Matematiche del Target",
            count: 4,
            examples: "URXUMA (Albuminuria), URXUCR (Creatinina urinaria), LBXUCR (Creatinina sierica), LBXCYS (Cistatina C)",
            reason: "Costituiscono la formula di calcolo dell'eGFR o dell'ACR. Includerle significherebbe predire una funzione nota di se stessa (Target Leakage Circolare puro)."
          },
          {
            name: "Biomarcatori Renali Specialistici",
            count: 14,
            examples: "Azotemia (BUN), Acido Urico urinario, Elettroliti urinari (Sodio, Potassio), Osmolalità",
            reason: "Non fanno parte del bilancio ematochimico di routine di base e violano lo scopo applicativo di screening di primo livello."
          },
          {
            name: "Terapie & Farmaci Nefroattivi",
            count: 22,
            examples: "ACE-inibitori, Sartani (ARB), Diuretici dell'ansa, Tiazidici, Inibitori SGLT2",
            reason: "Rischio di reverse causality: l'assunzione del farmaco è conseguenza della diagnosi renale e altera i valori biologici reali."
          },
          {
            name: "Questionari e Sintomi Nefrologici Noti",
            count: 18,
            examples: "Diagnosi pregressa di insufficienza renale, dialisi, frequenza nicturia",
            reason: "Inquinamento da informazione a posteriori che degrada la capacità di intercettare casi occulti asintomatici."
          },
          {
            name: "Feature ad Altissimo Mancamento (>40%)",
            count: 27,
            examples: "Metalli pesanti nel sangue, frazioni lipidiche avanzate, ormoni specialistici",
            reason: "Non standardizzabili su larga scala e fonte di instabilità nell'imputazione multivariata."
          }
        ]
      },
      final_test: {
        claims: [
          {
            id: "A",
            title: "Coerenza OOF vs Test Set (Assenza di Overfitting)",
            verdict: "CONFERMATO",
            detail: "Tutte le stime OOF cadono rigorosamente dentro l'intervallo di confidenza al 95% del test set (RF OOF 0.741 vs Test 0.743).",
            criterion: "|AUC_test - AUC_oof| < 0.03 con sovrapposizione degli IC al 95%"
          },
          {
            id: "B",
            title: "Nessun Guadagno Reale con il Bilanciamento",
            verdict: "CONFERMATO",
            detail: "SMOTE e le tecniche di sovracampionamento non recuperano alcun caso grave aggiuntivo sul test set a pari sensibilità operativa (p di Holm = 1.000).",
            criterion: "p-value corretto di Holm > 0.05 per tutte le tecniche sintetiche"
          },
          {
            id: "C",
            title: "Superamento del Benchmark Clinico SCORED",
            verdict: "CONFERMATO",
            detail: "Random Forest supera significativamente il questionario clinico SCORED sul test set (AUROC 0.743 vs 0.723, Δ = +0.020, p = 0.012).",
            criterion: "Δ AUROC > 0 con p-value di DeLong < 0.05 sul test set"
          },
          {
            id: "D",
            title: "Tetto Informativo Invalicabile a ~0.74",
            verdict: "CONFERMATO",
            detail: "Nessuna delle 9 architetture avanzate (TabPFN, CatBoost, EBM) supera la baseline Random Forest con Δ ≥ 0.01 e IC Nadeau-Bengio > 0.",
            criterion: "Δ AUROC < 0.01 oppure limite inferiore IC Nadeau-Bengio ≤ 0"
          }
        ]
      },
      phase_a_descriptions: {
        "Random Forest": "Foresta di 400 alberi con max_features sqrt e campionamento bilanciato.",
        "XGBoost": "Gradient Boosting ottimizzato su PR-AUC (gamma, colsample, max_depth 4).",
        "Logistic Regression (Penalized)": "Regressione logistica ElasticNet (l1_ratio 0.45) con scaling robusto.",
        "Controllo Positivo (UmALB Leakage)": "Modello dimostrativo con inclusione di Albuminuria: prova l'assenza di limitazioni algoritmiche.",
        "LR SCORED (Benchmark)": "Riferimento clinico standard (Bang et al.) con feature SCORED."
      },
      phase_d_outcomes: {
        "NO": "SUPERATO: NO",
        "YES": "SUPERATO: SÌ"
      },
      dca_stats: {
        title_nb: "Beneficio Netto",
        sub_nb: "vs Strategia \"Testare Tutti\"",
        title_avoided: "Esami Evitati",
        sub_avoided: "Esami urinari specialistici risparmiati ogni 100 soggetti",
        title_econ: "Risparmio Economico",
        sub_econ: "Costo esami urinari evitati ogni 100 persone",
        title_prot: "Casi Gravi Protetti",
        sub_prot: "Casi KDIGO ad alto o altissimo rischio correttamente identificati",
        flag_safe: "Diagnosi casi gravi protetta al 100%"
      },
      chart_labels: {
        roc_xlabel: "1 - Specificità (Quota Falsi Positivi)",
        roc_ylabel: "Sensibilità (Quota Veri Positivi)",
        dca_xlabel: "Soglia Decisionale Pt",
        dca_ylabel: "Beneficio Netto (NB)",
        forest_xlabel: "Differenza AUROC (Δ vs Random Forest)",
        forest_baseline: "Baseline RF (0.00)",
        forest_threshold: "Soglia Minima (+0.01)"
      }
    },

    en: {
      scope_cards: [
        {
          id: "problem",
          num: "01",
          title: "Clinical Problem",
          summary: "Chronic Kidney Disease (CKD) remains asymptomatic until terminal stages (ESRD) and affects over 10% of the adult population.",
          detail: "Traditional screening requires serum creatinine and specialist urinary testing (uACR), which are rarely ordered in primary care until severe complications appear. Early intervention with kidney-protective drugs (SGLT2 inhibitors, RAS blockers) drastically halts eGFR decline when initiated early."
        },
        {
          id: "target",
          num: "02",
          title: "Rigorous Target Definition",
          summary: "Composite KDIGO 2024 construct: eGFR < 60 mL/min/1.73m² (G3a-G5) OR uACR ≥ 30 mg/g (A2-A3).",
          detail: "Unlike models focused strictly on eGFR (which overlook early diabetic nephropathy where filtration remains preserved), K-Risk captures the full spectrum of manifest renal damage stratified by international KDIGO guidelines."
        },
        {
          id: "inputs",
          num: "03",
          title: "Permitted Predictors (21 Features)",
          summary: "Routine 1st-level blood chemistry, standard demographic and anthropometric parameters.",
          detail: "Age, Sex, BMI, Systolic/Diastolic Blood Pressure, Liver Enzymes (AST, ALT, Alkaline Phosphatase, GGT), FIB-4 Liver Fibrosis Index, Glycated Hemoglobin (HbA1c), Fasting C-Peptide, Triglycerides, HDL/Total Cholesterol, Uric Acid, Complete Blood Count (Platelets, RBC, WBC). No specialist markers."
        },
        {
          id: "constraints",
          num: "04",
          title: "Strict Methodological Constraint",
          summary: "Total and permanent exclusion of Serum Creatinine, Cystatin C, and all urinalysis.",
          detail: "Using these tests would constitute circular target leakage (predicting a formula from its own internal algebraic coefficients). K-Risk is explicitly designed to operate upstream of specialist lab prescription."
        },
        {
          id: "role",
          num: "05",
          title: "Clinical Role in Patient Pathway",
          summary: "Pre-laboratory opportunistic triage filter for General Practice and Occupational Medicine.",
          detail: "It does not provide a definitive diagnosis, but generates a clinical priority index to optimize secondary test referrals (uACR and creatinine), sparing unnecessary lab tests in 94% of low-risk individuals."
        },
        {
          id: "not",
          num: "06",
          title: "What K-Risk is NOT",
          summary: "It is not a substitute for nephrology workup, nor a multi-year dialysis survival calculator.",
          detail: "It does not predict terminal progression in patients already diagnosed with CKD (which is the role of KFRE), does not estimate exact clearance, and does not replace medical consultation. It is a prioritisation tool for asymptomatic individuals."
        }
      ],
      positioning: {
        state_of_art_matrix: [
          {
            model: "K-Risk (This Study / Coppola)",
            target: "KDIGO Moderate/Severe Risk (eGFR < 60 or uACR ≥ 30)",
            renal_exams_used: "None (Zero Creatinine, Zero Urine)",
            auroc: "0.743",
            role: "First-level pre-laboratory triage. Proven clinical net benefit via DCA.",
            status: "krisk"
          },
          {
            model: "SCORED (Bang et al. 2008)",
            target: "eGFR < 60 mL/min/1.73m²",
            renal_exams_used: "Self-reported urine proteinuria",
            auroc: "0.723",
            role: "Clinical point score. Outperformed by K-Risk on AUROC, PR-AUC, and Net Benefit.",
            status: "benchmark"
          },
          {
            model: "Bragg-Gresham et al. 2018",
            target: "Undiagnosed CKD",
            renal_exams_used: "Requires Serum Creatinine or eGFR",
            auroc: "0.840",
            role: "Not applicable upstream of lab because it presumes renal test results.",
            status: "reference"
          },
          {
            model: "KFRE (Tangri et al. 2011/2016)",
            target: "Progression to Dialysis/ESRD at 2-5 yrs",
            renal_exams_used: "Serum Creatinine + Albuminuria (uACR)",
            auroc: "0.850",
            role: "Second-level prognostic tool: assesses disease progression, not initial screening.",
            status: "reference"
          },
          {
            model: "Nelson et al. 2019",
            target: "eGFR Decline and Mortality",
            renal_exams_used: "Baseline eGFR + Baseline uACR",
            auroc: "0.880",
            role: "Long-term risk assessment in populations with established CKD.",
            status: "reference"
          },
          {
            model: "Echouffo-Tcheugui & Kengne 2012",
            target: "5-10 yr Incident CKD",
            renal_exams_used: "Creatinine and/or Proteinuria",
            auroc: "0.760",
            role: "Systematic review: virtually all models depend on specific renal biomarkers.",
            status: "reference"
          }
        ],
        continuum_flow: [
          {
            step: "1",
            title: "Primary Care / Checkup",
            context: "Demographics, blood pressure, routine blood chemistry (AST, ALT, ALP, CBC, glucose, lipids). No specialist urinary tests required.",
            action: "Leverages routine exams already available",
            highlight: false
          },
          {
            step: "2",
            title: "K-Risk (Screening Algorithm)",
            context: "Calculates composite KDIGO risk. If score exceeds operating threshold (Pt = 0.07), clinical alert triggers recommending confirmatory testing.",
            action: "Spares 43% unnecessary urine tests",
            highlight: true
          },
          {
            step: "3",
            title: "Specialist Laboratory / uACR",
            context: "Targeted urine albumin-to-creatinine ratio (uACR) and serum creatinine testing exclusively for K-Risk flagged patients.",
            action: "Formal KDIGO diagnosis confirmed",
            highlight: false
          },
          {
            step: "4",
            title: "Nephrology Staging & Therapy",
            context: "Confirmed patients receive prognostic staging (e.g., KFRE) and initiate early organ-protective therapies (SGLT2i, RAS blockers).",
            action: "Early kidney-protective intervention",
            highlight: false
          }
        ]
      },
      leakage_guard: {
        excluded_categories: [
          {
            name: "Target Formula Components",
            count: 4,
            examples: "URXUMA (Albuminuria), URXUCR (Urinary Creatinine), LBXUCR (Serum Creatinine), LBXCYS (Cystatin C)",
            reason: "Direct ingredients of the eGFR or uACR formulas. Including them would predict a known mathematical function of itself (pure Circular Target Leakage)."
          },
          {
            name: "Specialist Renal Biomarkers",
            count: 14,
            examples: "Blood Urea Nitrogen (BUN), Urinary Uric Acid, Urinary Electrolytes (Sodium, Potassium), Osmolality",
            reason: "Not part of basic routine blood work and violate the operational purpose of first-level non-specialist screening."
          },
          {
            name: "Renal Medications & Therapies",
            count: 22,
            examples: "ACE inhibitors, ARBs, Loop Diuretics, Thiazides, SGLT2 inhibitors",
            reason: "Severe reverse causality: medication intake is a consequence of kidney disease and alters native baseline biomarker values."
          },
          {
            name: "Known Renal Disease Surveys",
            count: 18,
            examples: "Prior kidney failure diagnosis, dialysis history, nocturia frequency",
            reason: "A posteriori information leakage that invalidates the ability to detect asymptomatic occult cases."
          },
          {
            name: "High-Missingness Features (>40%)",
            count: 27,
            examples: "Heavy metals in blood, advanced lipid fractions, specialist hormones",
            reason: "Non-standardizable in clinical practice and introduces instability in multivariate imputation."
          }
        ]
      },
      final_test: {
        claims: [
          {
            id: "A",
            title: "OOF vs Test Set Consistency (Absence of Overfitting)",
            verdict: "CONFIRMED",
            detail: "All OOF estimates fall strictly within the 95% confidence intervals of the test set (RF OOF 0.741 vs Test 0.743).",
            criterion: "|AUC_test - AUC_oof| < 0.03 with 95% CI overlap"
          },
          {
            id: "B",
            title: "Zero Real Gain from Class Balancing",
            verdict: "CONFIRMED",
            detail: "SMOTE and oversampling techniques catch zero additional severe cases on the test set at equal operating sensitivity (Holm p = 1.000).",
            criterion: "Holm corrected p-value > 0.05 across all synthetic techniques"
          },
          {
            id: "C",
            title: "Superiority Over SCORED Benchmark",
            verdict: "CONFIRMED",
            detail: "Random Forest significantly outperforms the SCORED clinical questionnaire on the test set (AUROC 0.743 vs 0.723, Δ = +0.020, p = 0.012).",
            criterion: "Δ AUROC > 0 with DeLong p-value < 0.05 on the test set"
          },
          {
            id: "D",
            title: "Unsurpassable Information Ceiling at ~0.74",
            verdict: "CONFIRMED",
            detail: "None of the 9 advanced architectures (TabPFN, CatBoost, EBM) outperforms Random Forest with Δ ≥ 0.01 and Nadeau-Bengio CI > 0.",
            criterion: "Δ AUROC < 0.01 or lower bound Nadeau-Bengio CI ≤ 0"
          }
        ]
      },
      phase_a_descriptions: {
        "Random Forest": "Forest of 400 trees with sqrt max_features and balanced subsampling.",
        "XGBoost": "Gradient Boosting optimized on PR-AUC (gamma, colsample, max_depth 4).",
        "Logistic Regression (Penalized)": "ElasticNet logistic regression (l1_ratio 0.45) with robust scaling.",
        "Controllo Positivo (UmALB Leakage)": "Demonstration model with Albuminuria: proves absence of algorithmic limitations.",
        "LR SCORED (Benchmark)": "Standard clinical benchmark (Bang et al.) using SCORED features."
      },
      phase_d_outcomes: {
        "NO": "SURPASSED: NO",
        "YES": "SURPASSED: YES"
      },
      dca_stats: {
        title_nb: "Net Benefit",
        sub_nb: "vs \"Test All\" Strategy",
        title_avoided: "Avoided Tests",
        sub_avoided: "Specialist urine tests spared per 100 subjects",
        title_econ: "Economic Savings",
        sub_econ: "Urinary test costs avoided per 100 subjects",
        title_prot: "Severe Cases Protected",
        sub_prot: "High or very high risk KDIGO cases accurately caught",
        flag_safe: "Severe cases detection 100% protected"
      },
      chart_labels: {
        roc_xlabel: "1 - Specificity (False Positive Rate)",
        roc_ylabel: "Sensitivity (True Positive Rate)",
        dca_xlabel: "Decision Threshold Pt",
        dca_ylabel: "Net Benefit (NB)",
        forest_xlabel: "AUROC Difference (Δ vs Random Forest)",
        forest_baseline: "RF Baseline (0.00)",
        forest_threshold: "Minimum Threshold (+0.01)"
      }
    }
  }
};
