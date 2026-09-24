# Preparazione all'incontro con i relatori

Documento di studio per l'incontro richiesto il 24/09/2026: "tutto quello che hai fatto, dal giorno 0 a ora, e tutto quello che hai ottenuto, con tutti i dettagli tecnici: nomi, approcci, modelli". L'incontro è orale, senza slide, e la durata non è nota: il documento ha quindi una versione breve e una completa.

Ogni numero viene dalle tabelle in `analytics/`, dal `Notepad.md` o dalla storia git; i numeri principali sono stati ricontrollati sulle tabelle il 24/09/2026. Il test set non è stato riaperto.

**Come usarlo**
- **§0**: le due domande del messaggio, con cui probabilmente si apre l'incontro.
- **§1**: il progetto in 2 minuti.
- **§2**: il racconto completo, fase per fase, con nomi, parametri e risultati.
- **§3**: le domande probabili, con risposta e numeri. Le domande segnate **(scomoda)** toccano i punti deboli: vanno anticipate, non subite.
- **§4**: le domande da fare tu ai relatori.
- **§5**: le incongruenze della repo che i relatori potrebbero notare.
- **§6**: il glossario dei metodi. **§7**: i numeri da sapere a memoria.

---

## 0. Le due domande del messaggio

### "Finito anche le altre parti? Il KDIGO ecc.?"
Sì, la parte sperimentale è chiusa. Il KDIGO entra nel progetto in tre punti:
1. **definisce il bersaglio**: marcatori di malattia renale cronica se eGFR < 60 ml/min/1,73 m², ricalcolato con CKD-EPI 2021, oppure ACR ≥ 30 mg/g;
2. **stratifica**: i 4 livelli di rischio della heatmap KDIGO (basso, moderato, alto, molto alto) guidano lo split train/test, i fold della cross-validation e due tecniche di bilanciamento (pesi per livello, SMOTE-NC e CTGAN per livello);
3. **misura il modello**: il rischio stimato cresce con il livello? Quanti casi alto e molto alto riconosce? Le fasce del modello coincidono con i livelli?

Dopo il KDIGO sono chiuse anche:
- le Fasi A (modelli), B (bilanciamento), C (diabetici) e D (tetto di prestazione);
- l'utilità clinica (decision curve, calibrazione, costi);
- la conferma sul test set, aperta una sola volta il 22/09;
- un audit sul tetto dei dati, il 23/09.

Mancano due cose:
- la scrittura della tesi: ho iniziato l'introduzione;
- il prototipo dimostrativo: c'è un sito dei risultati in `docs/`, ma non ancora uno strumento che calcola il rischio di un paziente.

### "Come ti sembra la pipeline, corposa o minimal?"
Risposta consigliata: **corposa soprattutto nella validazione; i modelli invece sono volutamente classici.**
- **passaggi**: dal file grezzo al test set sono 10, ciascuno con il suo modulo e i suoi test:
  1. bersaglio KDIGO
  2. split
  3. selezione delle feature
  4. imputazione
  5. CV annidata con Optuna
  6. 5 modelli
  7. 8 tecniche di bilanciamento
  8. ricalibrazione
  9. valutazione per livello e per sottogruppo
  10. conferma sul test
- **modelli provati**:
  - 5 nelle Fasi A e B;
  - 11 candidati in Fase D, fra cui CatBoost, LightGBM, EBM, TabPFN v2 ed ensemble;
  - 2 modelli profondi: CTGAN come generatore di dati sintetici e TabPFN, un transformer pre-addestrato.
- **dimensioni**:
  - 13 pull request fra l'8 e il 23 settembre;
  - 28 moduli Python (circa 5.600 righe) e 177 test automatici;
  - 45 figure;
  - un registro delle decisioni di 2.100 righe.
- **cosa la rende corposa**: non il numero di modelli, ma i controlli:
  - protocolli scritti prima dei risultati;
  - cross-validation annidata;
  - intervalli di confidenza per ogni stima;
  - controlli di coerenza e di leakage;
  - test set aperto una volta sola.
- **il rischio**: per una triennale forse è anche troppo. Nel testo si può tenere un filo solo (Fasi A, B, C e test) e mettere il resto (Fase D, utilità clinica, audit) in capitoli di supporto o in appendice. È una delle domande da fare ai relatori (§4).

---

## 1. Il progetto in 2 minuti

> K-Risk stima, **senza esami renali**, chi ha marcatori di malattia renale cronica secondo KDIGO: l'obiettivo è decidere a chi prescrivere creatinina e ACR, non sostituirli.
>
> **Dati.** Un dataset pubblico: Li et al. 2026, *Scientific Data*, 5.922 soggetti di un reparto di diabetologia di Shanghai, 2012, 190 variabili. 5.801 hanno creatinina e ACR, necessari per il bersaglio.
>
> **Bersaglio e input.** Il bersaglio è binario, eGFR < 60 (ricalcolato con CKD-EPI 2021) oppure ACR ≥ 30, con il 9,8% di positivi. Tutte le variabili renali sono escluse dall'input: restano 74 variabili di routine (anagrafica, pressione, glicemia, lipidi, fegato, emocromo, stili di vita).
>
> **Metodo.** Ho confrontato 5 modelli in cross-validation annidata 5 × 5 con Optuna: classificatore di maggioranza, logistica con i predittori del punteggio clinico SCORED, logistica penalizzata, Random Forest e XGBoost. Poi ho provato 8 tecniche di bilanciamento, dai pesi di classe a SMOTE-NC e CTGAN.
>
> **Risultati.**
> - La discriminazione è modesta: AUROC circa 0,70 out-of-fold e 0,71–0,74 sul test. Nessun modello è migliore degli altri in modo dimostrabile.
> - Il bilanciamento non fa riconoscere più casi gravi. Alla soglia 0,5 sembra portare il recall dal 2% al 60%, ma è un effetto della soglia.
> - Il tetto sta nei dati: l'eGFR ridotto si riconosce (AUROC 0,80–0,86), l'albuminuria no (0,67–0,69).
> - Il modello però serve: alla soglia del 7% evita 9–13 esami inutili ogni 100 persone rispetto a "testare tutti". Vale fra i non diabetici; fra i diabetici hanno ragione le linee guida, che prescrivono l'esame a tutti.
>
> Le conclusioni sono state confermate sul test set, aperto una sola volta con un protocollo scritto prima.

---

## 2. Il racconto completo, dal giorno 0 a oggi

| data | cosa | PR |
|---|---|---|
| 08/09 | giorno 0: repo, licenza, citazione del dataset | — |
| 17/09 | primo split con il bersaglio `DN`; poi bersaglio KDIGO e nuovo split | #1, #2 |
| 18/09 | Scope; selezione delle feature e confronto delle imputazioni | #2, #3 |
| 19/09 | Fase A: 5 modelli in CV annidata, valutazione, interpretazione; bozza LaTeX | #4 |
| 19–20/09 | Fase B: bilanciamento (calcolo notturno 16:23 → 01:50) | #5 |
| 20/09 | Fase C: diabetici; preparazione del test set | #6 |
| 21–22/09 | Fase D: tetto di prestazione (post-hoc) | #7 |
| 22/09 | utilità clinica, tri-ensemble, stato dell'arte, provenienza del dataset | #8 |
| 22/09 | conclusioni, protocollo del test, **esecuzione unica sul test** (18:20–18:37) | #9 |
| 22–23/09 | sito dei risultati su GitHub Pages, in italiano e inglese | #10, #11 |
| 23/09 | audit del tetto dei dati; inizio della scrittura | #12, #13 |

Il lavoro sperimentale si concentra fra il 17 e il 23/09; i calcoli lunghi (Fase A, Fase B) sono girati di notte.

### 2.1 Giorno 0 (08/09): la repo e i dati
- **Cosa**: repo GitHub, nata come *MetaRisk* e poi rinominata K-Risk; struttura delle cartelle, README, licenza MIT per il codice, `CITATION.cff`, requirements.
- **Dati**: *A bimodal dataset for diabetes research* (Li J. et al., *Scientific Data* 13, 2026; doi 10.1038/s41597-026-06923-y; archivio Zenodo).
  - licenza CC BY 4.0, incluso senza modifiche in `data/raw/` con impronta SHA-256;
  - dizionario delle variabili nel `Supplementary Information.pdf`, letto con `pypdf`;
  - 5.922 righe × 190 colonne, una misura per soggetto (dati trasversali).

### 2.2 17/09: il primo bersaglio (`DN`) e perché l'ho abbandonato
- **Prima impostazione**:
  - bersaglio `DN > 0` contro `DN = 0`: `DN` è la colonna "Diabetic nephropathy staging" del dataset;
  - 456 positivi su 5.802 (7,86%), split stratificato su `DN` × diabete.
- **Verifica**: `DN > 0` coincide con `UmALB ≥ 30` in 455 casi su 456. È una soglia sulla **concentrazione** di albumina nelle urine, non sull'ACR, e ignora il filtrato glomerulare.
- **Tre motivi per abbandonarlo**:
  - non è la definizione KDIGO;
  - non è "nefropatia diabetica": l'82,5% dei positivi non era diabetico;
  - non permette un confronto pulito con KDIGO.
- **Effetto collaterale scoperto**: con lo split su `DN` i casi "molto alto", che dipendono soprattutto dall'eGFR, finivano 12 nel training e 15 nel test (il 56% nel test).
- **Cosa resta**: il metodo dello split. Cambia la definizione del bersaglio.

### 2.3 17–18/09: il bersaglio KDIGO e lo split
**eGFR**
- ricalcolato con **CKD-EPI 2021** senza coefficiente etnico (Inker et al. 2021), da creatinina, età e sesso; la creatinina passa da µmol/L a mg/dL dividendo per 88,4;
- la colonna `GFR` del dataset non coincide con nessuna equazione standard: provate CKD-EPI 2009 e 2021, MDRD, MDRD cinese e Cockcroft-Gault, e arriva fino a 562.

**Bersaglio**
- `y = 1` se ACR ≥ 30 mg/g oppure eGFR < 60;
- 567 positivi su 5.801 (9,77%); 121 soggetti esclusi perché senza creatinina o ACR;
- nel training i positivi sono 362 per la sola albuminuria, 25 per entrambi i criteri, 38 per il solo eGFR < 60: **il bersaglio è al 91% albuminuria**;
- binario e non a 4 classi: i "molto alto" sono solo 27 in tutta la coorte;
- classificazione e non regressione sull'ACR: l'ACR è molto asimmetrica e KDIGO è già una classificazione.

**4 livelli KDIGO** (incrocio delle categorie G dell'eGFR e A dell'ACR)
- basso 5.234, moderato 473, alto 67, molto alto 27;
- servono a stratificare, bilanciare e valutare; **mai ad allenare**.

**Split**
- 75/25, seed 42, stratificato su **livello KDIGO × diabete**: 8 strati, il più piccolo (diabetici "molto alto") con 6 soggetti;
- training 4.350 soggetti (425 positivi);
- test 1.451 (142 positivi; 17 alto e 6 molto alto; 88 diabetici, di cui 23 positivi).

**Perché così**
- 75/25: almeno 100 eventi nel test (Collins et al. 2016; Riley et al. 2021) e casi gravi in entrambi gli insiemi. 80/20 lasciava 5 "molto alto" nel test; 70/30 toglieva 28 positivi al training;
- stratificazione sul livello: su 1.000 split simulati, i "molto alto" nel test andavano da 0 a 15 senza stratificazione e da 1 a 14 stratificando sul solo bersaglio; con la stratificazione sul livello sono sempre 6;
- verifica dell'equilibrio: differenza media standardizzata (SMD) sotto 0,1 su 17 variabili (massimo 0,059, pressione sistolica).

**Dove**: `src/data/kidney.py`, `src/data/split.py`, `analytics/split/`, `Scope.md`.

### 2.4 18/09: preprocessing, da 190 a 74 variabili, e imputazione
**Mappa completa delle colonne** (`configs/config.yaml`, sezione `features`)
- ogni colonna sta in **esattamente un gruppo**: uno dei 12 motivi di esclusione, oppure numerica o categorica;
- un test fallisce se una colonna non è assegnata, così una colonna dimenticata non entra nel modello in silenzio.

**Esclusioni principali**
- leakage del bersaglio: `SCRE`, `UMAUCR`, `UmALB`, `UCRE`, `GFR`, `DN` e 10 indicatori derivati. Con queste colonne l'AUROC sarebbe 0,96–0,998;
- esami o diagnosi renali: `BUN`, `RF`;
- colonne amministrative (`NO`, `Data`…) e non documentate (`VAR00001`, `ALT.1`, che è √ALT generato da SPSS);
- colonne con oltre il 50% di mancanti; il questionario di conoscenza del diabete;
- circa 35 derivati a soglia (`BMI25`, `Anemia`…) e due combinazioni lineari esatte (`IDBIL` = `TBIL` − `DBIL`, `GLO` = `TP` − `ALB`);
- comorbidità con codice 9 = "sconosciuto": dopo la ricodifica superano il 15% di mancanti, senza segnale;
- `Homaβ`: la formula ha al denominatore `FPG` − 3,5 ed esplode, con valori oltre 1.000;
- oltre il 15% di mancanti sul training: `leg`, `LC`, `ALY`…

**Risultato**
- **74 feature: 60 numeriche + 14 categoriche**;
- set di sensibilità `no_consequence` (67 feature): senza le 7 variabili che la malattia renale altera (HGB, RBC, HCT, SUA, ALB, TP, GA).

**Controllo anti-leakage**: l'AUROC univariata massima delle 74 feature è 0,653 (età); un test automatico fallisce sopra 0,75.

**Codifiche**
- nessun one-hot: `Gender` 0/1; fumo, alcol e tè ordinali 0/1/2; `HypertenHis` vuoto = 0;
- nessun logaritmo: guadagno di PR-AUC +0,008 ± 0,005, non significativo;
- valori estremi tenuti: sono clinicamente possibili.

**Imputazione**
- il 38% dei soggetti del training ha almeno un valore mancante: eliminarli (analisi dei casi completi) è escluso;
- 4 metodi confrontati in CV a 5 fold con due criteri: ricostruzione di un 10% di valori nascosti (RMSE), e PR-AUC a valle con una logistica fissa;

| metodo | RMSE ↓ | PR-AUC ↑ | tempo |
|---|---|---|---|
| mediana | 1,038 | 0,252 | 1 s |
| KNN (k = 5) | 0,793 | 0,259 | 11 s |
| MICE (`IterativeImputer` + BayesianRidge) | 1,252 | 0,257 | 70 s |
| **MissForest** (`IterativeImputer` + ExtraTrees) | **0,734** | 0,257 | 305 s |

- a valle i metodi sono indistinguibili. La regola di 1 errore standard, in due fasi, resta al margine fra KNN e MissForest. Il confronto appaiato (Nadeau & Bengio 2003) è netto: MissForest ricostruisce meglio in 5 fold su 5, in entrambi i set;
- **scelto MissForest**, dichiarando che la regola è stata rivista dopo i primi risultati;
- parametri: 50 alberi, 5 iterazioni, `max_features="sqrt"`, `min_samples_leaf=5`;
- ordine: standardizzazione, poi moda per le categoriche, poi l'imputer su tutte le 74 colonne (le categoriche fanno da predittori, per esempio `DM` per stimare `HbA1c`);
- stimato solo sul training di ogni fold, senza mai vedere y. Nessun indicatore di mancanza: i mancanti dipendono dalla giornata di raccolta.

**Dove**: `src/data/preprocess.py`, `src/data/imputation.py`, `analytics/preprocessing/`, Notepad Passi 1–10.

### 2.5 19/09: Fase A, cinque modelli sui dati originali
**I modelli, uno per ruolo**
1. `DummyClassifier(strategy="prior")`: controllo di coerenza; deve dare AUROC 0,5 e PR-AUC uguale alla prevalenza;
2. **logistica SCORED**: i 5 predittori disponibili del punteggio clinico SCORED (Bang et al. 2007), cioè età, sesso, emoglobina, pressione sistolica e diabete. Senza penalizzazione (circa 85 eventi per variabile), coefficienti ristimati sui nostri dati;
3. **logistica penalizzata** (elastic net, solver `saga`) su tutte le 74 variabili. La penalizzazione serve perché gli eventi per variabile sono 5,7, sotto 10;
4. **Random Forest**: ensemble di alberi, bagging;
5. **XGBoost**: ensemble di alberi, gradient boosting (`tree_method="hist"`).

**Protocollo**, scritto in config prima di addestrare
- **CV annidata**: 5 fold esterni per le previsioni out-of-fold, 5 interni per gli iperparametri. Fold stratificati sul livello KDIGO, seed 42;
- **Optuna**: campionatore TPE, 100 tentativi per modello e per fold, `MedianPruner`, metrica PR-AUC;
- nessun peso di classe: appartiene alla Fase B;
- il set `no_consequence` riusa gli iperparametri di `main`;
- modello finale: iperparametri scelti sui 5 fold esterni, poi riaddestramento su tutto il training;
- i fold imputati sono stimati una volta sola e salvati (circa 1 ora): l'imputer non vede y, quindi riusarli è lecito.

**Spazi di ricerca** (Brima & Atemkeng 2026)
- logistica: `C` da 10⁻⁴ a 10 in scala logaritmica, `l1_ratio` da 0 a 1;
- Random Forest: alberi 100–1.000, profondità 3–25, `min_samples_split` 2–50, `min_samples_leaf` 1–20, `max_features` sqrt/log2/0,25/0,5/0,75, criterio gini o entropy;
- XGBoost: alberi 200–1.200, learning rate 0,01–0,3 in scala logaritmica, profondità 3–12, `subsample` 0,6–1, `colsample_bytree` 0,5–1, `reg_alpha` e `reg_lambda` 0–5.

**Iperparametri dei modelli finali**
- logistica penalizzata: C = 1,53, `l1_ratio` = 0,85; 6 coefficienti azzerati su 74;
- Random Forest: 885 alberi, profondità 24, `min_samples_split` 42, `min_samples_leaf` 3, `max_features` log2, entropy;
- XGBoost: 392 alberi, learning rate 0,025, **profondità 3**, cioè il limite inferiore, scelto in 6 ottimizzazioni su 6;
- fra i fold gli iperparametri cambiano molto (logistica: C da 0,005 a 0,52; Random Forest: profondità da 3 a 25). È il segno di un ottimo piatto: le prestazioni cambiano poco.

**Analisi di sensibilità**, decisa dopo i risultati e dichiarata
- XGBoost con profondità 1–12 sceglie **profondità 1** in 5 casi su 6. Alberi con un solo nodo equivalgono a una somma di effetti delle singole variabili: **il segnale è additivo**, senza interazioni;
- AUROC 0,696 contro 0,699: nessuna differenza. Lo spazio 1–12 diventa quello della Fase B.

**Dove**: `src/models/zoo.py`, `src/models/phase_a.py`, `src/data/folds.py`, `src/data/imputed.py`.

### 2.6 19/09: valutazione (domande 1–4) e interpretazione
**Scelte fissate prima di calcolare le metriche**
- **soglia**: la più alta con sensibilità ≥ 0,90 sulle previsioni out-of-fold;
- **fasce**: 4, con le stesse proporzioni dei livelli KDIGO (90,2 / 8,1 / 1,1 / 0,5%);
- **concordanza fasce/livelli**: kappa pesato lineare;
- **intervalli di confidenza**: DeLong per l'AUROC, logit di Boyd per la PR-AUC, Wilson per le proporzioni, bootstrap stratificato con 2.000 campioni per kappa, medie per livello e concordanza;
- **confronti fra modelli**: t corretto di Nadeau-Bengio con correzione di Holm.

**Risultati out-of-fold** (4.350 soggetti, 425 positivi)

| modello | AUROC (IC 95%) | PR-AUC | specificità a sens. 0,90 | da esaminare | "molto alto" riconosciuti | kappa |
|---|---|---|---|---|---|---|
| logistica SCORED | 0,675 (0,646–0,704) | 0,224 | 0,209 | 80,2% | 19 su 21 | 0,197 |
| logistica penalizzata | 0,697 (0,669–0,725) | 0,242 | 0,223 | 78,9% | 20 su 21 | 0,207 |
| Random Forest | 0,703 (0,676–0,731) | 0,246 | 0,232 | 78,1% | 18 su 21 | 0,209 |
| XGBoost | 0,699 (0,671–0,726) | 0,251 | 0,245 | 76,9% | 18 su 21 | 0,228 |

- **domanda 1**:
  - discriminazione modesta; la PR-AUC è 2,3–2,6 volte la prevalenza;
  - scegliendo a caso, per trovare il 90% dei positivi andrebbe esaminato il 90% dei soggetti; con il modello il 77–80%.
- **domanda 2**:
  - la probabilità media cresce a ogni livello (XGBoost: 0,086 → 0,142 → 0,194 → 0,284);
  - concordanza di Jonckheere-Terpstra 0,674–0,702: circa 7 coppie su 10 sono ordinate come KDIGO.
- **domanda 3**: alla soglia scelta 46–47 "alto" su 50 e 18–20 "molto alto" su 21. I casi gravi però non sono riconosciuti più dei moderati.
- **domanda 4**:
  - kappa circa 0,2. L'accordo osservato è alto (0,85) solo perché il 90% dei soggetti è "basso": è il paradosso della prevalenza;
  - fino a metà dei "molto alto" finisce nella fascia 1;
  - il modello riconosce la **presenza** dei marcatori, non il **grado**.
- **confronti**: nessuna differenza significativa. SCORED perde 0,021–0,029 di AUROC contro gli altri modelli, p di Holm 0,44; sulla PR-AUC p = 1,00.
- **`no_consequence`**: l'AUROC cambia fra −0,012 e +0,001, mai in modo significativo. Il modello non si regge sulle conseguenze della malattia.

**Interpretazione**
- **logistica SCORED**, odds ratio con IC di Wald:
  - diabete 2,58 (1,89–3,52);
  - età 1,47 ogni 13,8 anni;
  - pressione sistolica 1,34 ogni 16,5 mmHg;
  - emoglobina e sesso non significativi.
- **SHAP** esatto (TreeSHAP) per Random Forest e XGBoost:
  - l'età è prima in entrambi; poi ALP, FIB-4, peptide C, albumina glicata, glicemie, lipidi, acido urico;
  - 11 variabili in comune fra le prime 15 dei due modelli;
  - importanza diffusa: le prime 5 variabili fanno solo il 23–26% del totale.
- **Revisione indipendente del codice**, in sola lettura: nessun errore critico.

**Dove**: `src/models/evaluation.py`, `src/models/interpretation.py`, `analytics/phase_a/`.

### 2.7 19–20/09: Fase B, bilanciamento e data augmentation
**Domanda 5**: il bilanciamento migliora il riconoscimento dei casi gravi, o solo la metrica media?

**Tecniche**, applicate solo alla parte di training di ogni fold
- 6 principali:
  - pesi di classe (inverso della prevalenza);
  - pesi per livello KDIGO 1:2:3 sui positivi;
  - random undersampling 1:1;
  - random oversampling 1:1;
  - SMOTE-NC 1:1 (k = 5);
  - SMOTE-NC per livello: i vicini si cercano solo fra i positivi dello stesso livello.
- 2 esplorative: **CTGAN** condizionato su y e condizionato sul livello KDIGO (300 epoche, batch 500).

**Protocollo**
- stessi 5 modelli; XGBoost con profondità 1–12;
- **30 tentativi** di Optuna con **avvio caldo**: il primo tentativo è l'ottimo della Fase A;
- **ricalibrazione di Platt annidata**, stimata sulle previsioni reali dei fold interni;
- **esito primario**: casi gravi (alto + molto alto, 71) riconosciuti alla soglia con sensibilità complessiva 0,90;
- test di **McNemar esatto** e IC di **Newcombe** contro "nessuna correzione"; correzione di Holm sulle 6 tecniche.

**Esecuzione**: dal 19/09 alle 16:23 al 20/09 alle 01:50.

**Risultati**
- **nessun guadagno**: il massimo guadagno netto è di 3 casi su 71, e tutti i p di Holm valgono 1,00;
- **discriminazione** (AUROC media dei 4 modelli):

| tecnica | AUROC |
|---|---|
| nessuna correzione | 0,693 |
| pesi | 0,690–0,692 |
| undersampling | 0,689 |
| oversampling | 0,680 |
| SMOTE-NC | 0,672–0,673 |
| CTGAN | 0,616–0,622 |

- **calibrazione**: con le 6 tecniche principali l'intercetta scende da 0 a valori fra −1,4 e −2,2, cioè probabilità gonfiate; Platt la riporta a −0,01;
- **miglioramento apparente**:
  - alla soglia 0,5 il recall medio passa dal 2% (nessuna correzione) al 60% (undersampling);
  - la Random Forest passa da 0 a 266 positivi trovati su 425, e da 0 a 51 casi gravi su 71;
  - è la soglia che si sposta (Elkan 2001), non il modello che migliora.
- **CTGAN per livello**: con 17 "molto alto" per fold, quel livello compare solo nello 0,5–1,2% dei campioni generati; è stato necessario aumentare i tentativi di campionamento.

**Dove**: `src/data/augmented.py`, `src/models/phase_b.py`, `src/models/evaluation_b.py`, `analytics/phase_b/`.

### 2.8 20/09: Fase C, sottogruppo diabetico
- **Come**:
  - nessun riaddestramento: le stesse previsioni filtrate sui 263 diabetici del training (68 positivi, 16 casi gravi);
  - soglia e fasce globali, mai ristimate sul sottogruppo;
  - confronto con la sola differenza di AUROC: la PR-AUC dipende dalla prevalenza (Matos et al. 2026).
- **Risultati**:
  - AUROC diabetici − non diabetici da −0,038 a +0,042; tutti gli intervalli comprendono lo zero;
  - PR-AUC 0,44 contro 0,19, ma rapportata alla prevalenza è più bassa (1,47–1,70 volte contro 2,12–2,44);
  - alla soglia globale il modello segnala il **97–100% dei diabetici** (specificità 0,000–0,036): equivale a "testare tutti";
  - per avere sensibilità 0,90 fra i soli diabetici servirebbe una soglia 2,5–3,2 volte quella globale.
- **Difetto trovato dal controllo di coerenza**: il classificatore di maggioranza dava AUROC 0,454 per residui in virgola mobile dell'ordine di 3·10⁻¹⁶. Corretto; nessun risultato dei modelli reali era toccato.
- **Dove**: `src/models/phase_c.py`, `analytics/phase_c/`.

### 2.9 21–22/09: Fase D, il tetto di prestazione (post-hoc)
**Domanda**: si poteva fare meglio?

**Regola scritta prima dei calcoli**: un candidato migliora se, contro la Random Forest, la differenza di AUROC è almeno 0,01, l'IC di Nadeau-Bengio sta sopra zero e il p di Holm è sotto 0,05.

**11 candidati**
- media dei 3 riferimenti (logistica penalizzata, Random Forest, XGBoost);
- XGBoost con i mancanti gestiti in modo nativo; XGBoost con spazio allargato; XGBoost senza pruner;
- bersaglio scomposto: un XGBoost per l'albuminuria e uno per l'eGFR < 60, combinati con p = 1 − (1 − p_A)(1 − p_G);
- CatBoost; LightGBM; EBM; TabPFN v2;
- logistica con tutte le colonne standardizzate;
- tri-ensemble su 21 variabili.

**Risultato**: nessuno supera la regola. Le AUROC vanno da 0,692 (EBM) a 0,710 (TabPFN: +0,009, IC da −0,014 a +0,032).

**Diagnostiche**
- **controllo positivo**: con l'albumina urinaria fra le feature l'AUROC sale a 0,933. La pipeline impara quando l'informazione c'è;
- **curva di apprendimento**: dal 60% al 100% del training l'AUROC guadagna solo 0,006–0,008;
- **per componente del bersaglio**: eGFR < 60 si riconosce con AUROC 0,80–0,86, la sola albuminuria con 0,67–0,69. **Il limite è l'albuminuria**;
- **qualità dell'etichetta**: nelle giornate con creatinina urinaria bassa la quota di ACR ≥ 30 sale (rho −0,70), ma l'AUROC non peggiora.

**Tri-ensemble**
- con le 21 variabili scelte dentro ogni fold: 0,703;
- scegliendole su tutto il training: 0,716. È **distorsione da selezione** (Ambroise & McLachlan 2002), il quarto "risultato apparente";
- valore reale, la parsimonia: 21 variabili invece di 74, specificità più alta a sensibilità 0,90 (0,267), un nucleo stabile di 9 variabili.

**Dove**: `src/models/phase_d.py`, `analytics/phase_d/`.

### 2.10 22/09: qualità e utilità clinica (post-hoc)
**Decision curve** (Vickers & Elkin 2006)
- soglie dal 2% al 20%, fissate prima;
- misure: net benefit ed esami inutili evitati rispetto a "testare tutti";
- sotto il 5% il modello equivale a testare tutti;
- **al 7% evita 9–13 esami inutili ogni 100 persone, al 10% 25–28**;
- fra i diabetici, nessun vantaggio fino al 10%.

**Calibrazione**
- in media buona: intercetta 0, rapporto osservati/attesi (O:E) 1;
- Random Forest ha pendenza 1,31: schiaccia le probabilità;
- XGBoost ha pendenza 0,86: le esaspera;
- fra i diabetici la Random Forest sottostima il rischio di circa un quinto (O:E 1,26).

**Costi** (test ACR a $49, Cusick et al. 2023)
- testare tutti costa $502 per caso trovato;
- al 7% il costo scende a $336–383, ma si mancano 1,6–2,8 casi ogni 100 persone;
- non è un'analisi costo-efficacia.

**MCC e F1**: 0,08–0,16 e 0,20–0,24. Sono misure improprie alla soglia clinica, quindi solo descrittive.

**Confronto con Bragg-Gresham et al.**: per trovare gli stessi casi servono 12–27 persone esaminate in più ogni 100. Il loro modello usa l'eGFR.

**Deviazione dichiarata**: per la calibrazione gli IC vengono dal bootstrap semplice. Quello stratificato sul livello fissa il numero di eventi e restringe gli intervalli in modo artificiale.

**Dove**: `src/models/clinical_utility.py`, `analytics/quality/`.

### 2.11 22/09: la conferma sul test set, una volta sola
**Protocollo scritto prima del codice**
- 5 affermazioni da confermare o smentire (A–E), ciascuna con il suo criterio;
- modelli finali già salvati; soglie e fasce out-of-fold; nessun riaddestramento.

**Sicurezza**
- il codice legge `test.csv` solo con `final_test.authorized: true`, cambiato in un commit dedicato;
- prima di aprire il file controlla che l'albero git sia pulito, che il preprocessore sia identico bit a bit alla cache e che i 50 modelli si carichino;
- `analytics/test/RUN.json` registra l'impronta SHA-256 del file e rifiuta una seconda esecuzione senza un motivo dichiarato;
- revisione indipendente del codice prima dell'esecuzione.

**Esecuzione**: 22/09, dalle 18:20 alle 18:37, dal commit `b7d5f61`; test richiuso subito dopo (`5ffadf6`).

**Esito: nessuna affermazione contraddetta**

| affermazione | esito sul test |
|---|---|
| (A) discriminazione | AUROC 0,714–0,743; l'AUROC out-of-fold cade nell'IC per 3 modelli su 4 (SCORED fa meglio sul test) |
| (B) bilanciamento | differenze da −2 a +2 casi gravi su 23, p di Holm 1,00 |
| (C) diabetici | il modello segnala il 97,7–100% dei diabetici |
| (D) utilità clinica | 13–19 esami inutili evitati ogni 100 al 7%, 29–33 al 10% |
| (E) soglia 0,5 | recall 4,4% senza correzione contro 40–65% con le tecniche |

- le soglie si trasferiscono bene: recall 0,887–0,937 contro 0,90 atteso;
- casi gravi riconosciuti: 20–22 su 23;
- **non confermato**: la buona calibrazione della logistica penalizzata, che sul test ha pendenza 0,79.

**Dove**: `src/models/final_test.py`, `analytics/test/`.

### 2.12 22–23/09: stato dell'arte, sito dei risultati, audit del tetto
**Stato dell'arte** (`valorizzazione_tesi.md`, `docs/verifica_stato_arte.md`)
- bibliografia verificata su Crossref e PubMed;
- confronto con SCORED, KFRE, MERWACS, Muntner, Tanner, Bragg-Gresham e Wu et al. 2017, un punteggio sviluppato nello stesso ospedale (AUROC 0,70–0,72).

**Provenienza del dataset** (verificata sul testo completo di Li et al.)
- è una coorte ospedaliera, non uno screening di popolazione;
- nel training `DM` = 1 solo nel 6% dei soggetti.

**Sito su GitHub Pages** (`docs/`)
- presenta i risultati in italiano e inglese, con una decision curve interattiva;
- non calcola il rischio di un paziente.

**Audit del 23/09**: 3 diagnostiche registrate prima di eseguirle (commit `9fd7939`)
- controllo semi-sintetico: aggiungendo una variabile che da sola vale 0,75 o 0,80, l'AUROC sale a 0,795 e 0,831;
- rumore dell'etichetta vicino alla soglia: vale al massimo +0,009;
- unità dell'ACR: il fattore 176,8 non è standard; con il fattore 100 l'AUROC cambia di +0,003/+0,016.

**Verdetto**: il tetto è dei dati. Con una sola partizione, però, la regola della Fase D rileva solo differenze di 0,02–0,04.

### 2.13 Cosa manca
- **la tesi** (`docs/thesis/main.tex`, template UNISA):
  - ci sono frontespizio e introduzione in corso; gli altri capitoli sono vuoti;
  - la bibliografia (`bib.bib`, 30 voci) è avviata;
  - `valorizzazione_tesi.md` §9bis elenca 5 punti da inserire;
- **il prototipo dimostrativo**;
- **da decidere con i relatori**: cosa entra nel testo e cosa va in appendice (§4).

---

## 3. Domande probabili, con risposte

### A. Dati e bersaglio

**Che dataset è?**
Li J. et al., *A bimodal dataset for diabetes research*, *Scientific Data* 13 (2026), licenza CC BY 4.0, archiviato su Zenodo. Viene dal reparto di Diabetologia ed Endocrinologia dello Shanghai Sixth People's Hospital, febbraio–aprile 2012: 5.922 record e 190 variabili, una sola misura per soggetto. Nella repo è incluso senza modifiche, con l'impronta SHA-256.

**(scomoda) È uno screening di popolazione?**
No, e l'ho corretto io il 22/09 leggendo il testo completo dell'articolo: è una coorte ospedaliera. Gli autori parlano di pazienti con diabete, ma i dati dicono altro:
- nel training `DM` = 1 riguarda solo il 6% dei soggetti;
- fra gli altri, solo l'1,8% supera una soglia diagnostica del diabete (glicemia a digiuno ≥ 7 mmol/L, HbA1c ≥ 6,5% o glicemia a 2 ore ≥ 11,1 mmol/L).

Sono quindi persone in gran parte senza diabete, valutate da un centro per il diabete, con un reclutamento non documentato. Per questo il modello va validato in popolazioni di screening.

**Perché avete ricalcolato l'eGFR?**
La colonna `GFR` non corrisponde a nessuna equazione standard:
- ho provato CKD-EPI 2009 e 2021, MDRD, MDRD cinese e Cockcroft-Gault: differenze mediane da 6 a 49 unità, nessuna corrispondenza esatta;
- arriva a 562, mentre le equazioni non superano circa 150–160.

KDIGO 2024 raccomanda CKD-EPI 2021 senza coefficiente etnico. Effetto del ricalcolo: i soggetti con eGFR < 60 passano da 119 a 91, i positivi da 586 a 567.

**Come si calcola CKD-EPI 2021?**
eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^−1,200 × 0,9938^età × 1,012 [se donna], con la creatinina (Scr) in mg/dL.
- κ = 0,7 per le donne e 0,9 per gli uomini;
- α = −0,241 per le donne e −0,302 per gli uomini.

Codice in `src/data/kidney.py`.

**Cosa sono i 4 livelli KDIGO?**
È la heatmap che incrocia le categorie dell'eGFR (G1–G5) e dell'ACR (A1–A3):
- **basso**: G1–G2 con A1;
- **moderato**: G1–G2 con A2, oppure G3a con A1;
- **alto**: G1–G2 con A3, G3a con A2, oppure G3b con A1;
- **molto alto**: G3a con A3, G3b con A2–A3, oppure G4–G5.

Nella coorte: 5.234, 473, 67 e 27 soggetti.

**Perché un bersaglio binario e non i 4 livelli?**
Ci sono 27 "molto alto" in tutta la coorte, 21 nel training: un modello a 4 classi, e un bilanciamento per classe, sarebbero instabili. Il modello impara il binario; i 4 livelli servono a valutarlo.

**Perché non una regressione sull'ACR?**
- l'ACR è molto asimmetrica: quasi tutti i valori stanno vicino a 0, con una coda lunghissima;
- un errore vicino a 30 cambia la classe clinica;
- le tecniche di bilanciamento sono pensate per la classificazione;
- KDIGO e la decisione clinica (fare o non fare l'esame) sono soglie.

**(scomoda) Il bersaglio è al 91% albuminuria: state predicendo solo quella?**
Nella sostanza sì. Nel training i positivi sono 362 per la sola albuminuria, 25 per entrambi i criteri e 38 per il solo eGFR < 60. È il quadro atteso fuori dalla malattia avanzata: in Cina l'albuminuria riguarda il 6,7% degli adulti, l'eGFR ridotto il 2,2% (Wang et al. 2023). Ed è proprio questo che fissa il tetto: dagli esami del sangue l'albuminuria si riconosce male (0,67–0,69), l'eGFR ridotto bene (0,80–0,86).

**Perché "marcatori" e non "diagnosi" di malattia renale cronica?**
KDIGO chiede alterazioni persistenti per più di 3 mesi e la conferma dell'albuminuria su più campioni; qui c'è una sola misura. Con una misura sola un danno acuto non si distingue da uno cronico.

**(scomoda) L'unità dell'ACR è certa?**
No.
- **il problema**: il dataset non documenta le unità di `UmALB` e `UCRE`. Nel training `UMAUCR` = 176,8 × `UmALB`/`UCRE` su ogni riga, un fattore che non corrisponde a nessuna conversione standard (100 con la creatinina in mg/dL, 8.840 in µmol/L);
- **l'ipotesi**: la mediana di `UCRE` (185) è plausibile solo in mg/dL. In quel caso la soglia 30 corrisponderebbe a circa 17 mg/g;
- **cosa ho fatto**: ho tenuto l'etichetta degli autori, perché la loro colonna `HighACR` coincide con `UMAUCR` ≥ 30 su ogni riga, e ho misurato l'effetto dell'ipotesi. Con la soglia alternativa l'AUROC cambia fra +0,003 e +0,016: le conclusioni non cambiano.

**Perché lo split 75/25 stratificato su livello e diabete?**
Per avere almeno 100 eventi nel test (sono 142) e i casi gravi in entrambi gli insiemi: sempre 6 "molto alto" nel test, contro 0–15 con uno split non stratificato. Il diabete entra nella stratificazione perché il sottogruppo è centrale per la domanda 6 (23 diabetici positivi nel test). Train e test risultano equilibrati: SMD sotto 0,1 su 17 variabili.

**Perché un test set fisso e non solo la cross-validation?**
Serve un metro unico, mai toccato, per confermare alla fine le conclusioni su tutti i modelli e tutte le tecniche. La CV annidata sul training serve a scegliere e stimare; il test solo a confermare.

### B. Preprocessing

**Come avete scelto le 74 variabili?**
Con una mappa completa delle 190 colonne in config: ogni colonna sta in un solo gruppo, e un test fallisce se una colonna non è assegnata. Si esclude per un motivo esplicito:
- leakage del bersaglio;
- esami renali;
- colonne amministrative o non documentate;
- colonne vuote;
- derivati a soglia e combinazioni lineari esatte;
- codice 9;
- indici instabili;
- oltre il 15% di mancanti.

Il dettaglio è nel §2.4.

**(scomoda) Come sai di non avere leakage?**
Cinque controlli:
1. le colonne renali sono vietate da un'asserzione nel codice, verificata da un test;
2. nessuna variabile da sola supera AUROC 0,75 (la massima è l'età, 0,653);
3. imputazione, standardizzazione e bilanciamento sono stimati solo sul training di ogni fold;
4. il test set è stato letto una volta sola;
5. i controlli positivi: con l'albumina urinaria si arriva a 0,933, con una variabile semi-sintetica a 0,795 e 0,831. Se ci fosse leakage nascosto, le prestazioni senza esami renali non sarebbero ferme a 0,70.

**Perché tenere emoglobina, acido urico e albumina, che la malattia renale altera?**
Non sono leakage di definizione, ma possibili conseguenze della malattia: le ho tenute e ho fatto l'analisi di sensibilità senza le 7 variabili-conseguenza (`no_consequence`). L'AUROC cambia fra −0,012 e +0,001, mai in modo significativo: il modello non si regge su di loro.

**Perché MissForest?**
- a valle i 4 metodi sono equivalenti;
- in ricostruzione MissForest è il migliore in tutti i fold nel confronto appaiato (RMSE 0,734 contro 0,793 di KNN);
- serve alla Fase B: SMOTE interpola fra vicini e CTGAN impara la distribuzione congiunta, quindi i valori imputati devono essere fedeli;
- coincide con la letteratura (Stekhoven & Bühlmann 2012; Waljee et al. 2013; Tiwaskar et al. 2025).

Da dire con onestà: la regola di scelta in due fasi è stata introdotta dopo i primi risultati, e sul set principale darebbe KNN per un margine piccolo. Ho deciso con il confronto appaiato e l'ho dichiarato.

**Perché non eliminare le righe incomplete?**
Il 38% del training ha almeno un mancante: perderei più di un terzo del campione, e introdurrei una distorsione se i dati non mancano completamente a caso. Qui infatti i mancanti si concentrano in alcune giornate di raccolta.

**Perché niente one-hot?**
Le categoriche sono tutte binarie o ordinali (no < occasionale < regolare). Con il one-hot, SMOTE e CTGAN potrebbero generare combinazioni impossibili, come "non fuma" e "fuma regolarmente" entrambe a 1. Gli alberi tagliano bene le ordinali.

**Perché niente logaritmo e niente rimozione dei valori estremi?**
- **logaritmo**: provato sulle 25 variabili con asimmetria > 2, vale +0,008 ± 0,005 di PR-AUC, non significativo; agli alberi una trasformazione monotona non cambia nulla; e la scala clinica rende più leggibili odds ratio e SHAP;
- **valori estremi**: sono clinicamente possibili, e toglierli solo dal training renderebbe il modello impreparato sul test.

**Cos'è il "codice 9"?**
Nel dizionario il 9 significa "sconosciuto" in 26 colonne (comorbidità, questionario sul diabete). Dopo la ricodifica queste colonne superano il 15% di mancanti e non hanno segnale, quindi sono escluse. Attenzione a non ricodificare il 9 su tutto il dataset: ALT, TBIL e MPV hanno valori reali pari a 9.

### C. Modelli e addestramento

**Perché questi 5 modelli?**
Uno per ruolo, dal più semplice al più flessibile:
- la maggioranza fissa il minimo e controlla la pipeline;
- SCORED è il confronto con la clinica: è il punteggio di rischio per la malattia renale cronica più validato esternamente (Echouffo-Tcheugui & Kengne 2012);
- la logistica penalizzata è il riferimento lineare: nei modelli clinici la logistica non è inferiore al machine learning (Christodoulou et al. 2019);
- Random Forest e XGBoost sono i due ensemble di alberi, uno per famiglia (bagging e boosting).

Sono gli stessi modelli in Fase A e in Fase B, così il confronto misura solo l'effetto del bilanciamento.

**Perché non reti neurali o deep learning?**
Sui dati tabellari non hanno un vantaggio sistematico sugli ensemble di alberi (Grinsztajn et al. 2022) e sono meno interpretabili. Li ho però provati dove aveva senso:
- **CTGAN**, una rete generativa avversaria, in Fase B: peggiora (AUROC 0,62);
- **TabPFN v2**, un transformer pre-addestrato, in Fase D: 0,710, la stima più alta ma non significativamente migliore.

**Cos'è la CV annidata e perché serve?**
- **fold esterni**: 5, e ciascuno produce le previsioni sui soggetti che non ha visto;
- **fold interni**: dentro ogni fold esterno, altri 5 servono a scegliere gli iperparametri;
- **perché**: scegliere e valutare sugli stessi fold darebbe stime ottimistiche (Varma & Simon 2006; Cawley & Talbot 2010);
- **dettagli**: fold stratificati sul livello KDIGO, seed 42, gli stessi per imputazione, Fase A, Fase B e Fase D.

**Come funziona Optuna?**
- **TPE**: propone gli iperparametri modellando dove stanno i tentativi buoni e dove quelli cattivi;
- **`MedianPruner`**: interrompe un tentativo se, dopo un fold interno, è sotto la mediana dei tentativi precedenti;
- **budget**: 100 tentativi per modello e fold in Fase A; 30 con avvio caldo in Fase B.

La Fase D ha anche controllato l'effetto del pruner: senza pruner non cambia nulla (0,695).

**Perché la PR-AUC come metrica di ottimizzazione?**
Con il 9,8% di positivi guarda dove sta il problema, cioè i positivi. Il suo valore di base è la prevalenza (0,098), non 0,5 (Saito & Rehmsmeier 2015). L'ho dichiarata prima di vedere i risultati.

**Che iperparametri avete ottenuto?**
Sono nel §2.5. Due cose da notare:
- XGBoost preferisce alberi di profondità 1: il segnale è additivo;
- fra i fold gli iperparametri oscillano molto, segno di un ottimo piatto.

**Perché SCORED ha 5 predittori e non 9?**
Nel dataset mancano quattro predittori:
- la proteinuria, che è un esame renale;
- la storia cardiovascolare, lo scompenso e l'arteriopatia periferica, escluse per il codice 9.

Due predittori sono sostituiti: l'anemia dall'emoglobina continua, l'ipertensione dalla pressione sistolica misurata. I coefficienti sono ristimati sui nostri dati: **non è una validazione esterna di SCORED**, e ristimarlo lo avvantaggia.

**Perché nessun peso di classe in Fase A?**
Perché è una tecnica di bilanciamento: appartiene alla Fase B. La Fase A usa i dati originali.

**(scomoda) Alla fine quale modello scegli?**
Nessuno è migliore in modo dimostrabile: out-of-fold i p di Holm sono ≥ 0,44, e sul test gli intervalli si sovrappongono.
- la **Random Forest** ha le stime puntuali più alte (0,703 out-of-fold, 0,743 sul test), ma comprime le probabilità (pendenza 1,31 e 1,27) e va ricalibrata;
- la **logistica penalizzata** è interpretabile, ma sul test la sua calibrazione non ha retto (pendenza 0,79);
- il **tri-ensemble** discrimina come la Random Forest con 21 variabili invece di 74.

Per il prototipo è una scelta pratica, da dichiarare come tale: vorrei concordarla con voi.

**Quanto costa addestrare?**
- **cache**: i fold imputati si stimano una volta, in circa 1 ora;
- **Fase A**: diverse ore, circa 3,5 stimate per ciascuno dei 3 modelli ottimizzati;
- **Fase B**: circa 9 ore e mezza di notte (7,5 per le 6 tecniche principali, circa 1 per CTGAN, 23 minuti per la ricalibrazione di Platt);
- **test finale**: 17 minuti.

Tutto su CPU, con Python 3.12 su Windows 11.

### D. Valutazione e statistica

**Perché la soglia a sensibilità 0,90, e non 0,5 o Youden?**
- nello screening un falso negativo è un caso mancato, mentre un falso positivo costa un esame delle urine: i costi non sono uguali (Wynants et al. 2019);
- l'indice di Youden assume costi uguali, e 0,5 con una prevalenza del 9,8% non segnala quasi nessuno;
- 0,90 è vicino al punto operativo di SCORED (sensibilità 92%);
- dipende solo dall'ordinamento, quindi resta confrontabile fra le tecniche di bilanciamento, che spostano le probabilità.

**Perché fasce con le proporzioni dei livelli e non fasce fisse (per esempio 30% e 70%)?**
Con una prevalenza del 9,8% solo l'1–3% dei soggetti supera 0,30, e al massimo lo 0,2% supera 0,70: per 3 modelli su 4 la fascia alta sarebbe vuota. Con le stesse proporzioni dei livelli, il kappa non è penalizzato da distribuzioni diverse.

**Cos'è la concordanza di Jonckheere-Terpstra?**
È la quota di coppie di soggetti di livelli diversi in cui il soggetto del livello più grave ha la probabilità più alta; i pareggi valgono 1/2 e 0,5 significa nessuna tendenza. Qui vale circa 0,70. Il test dà p < 10⁻³², ma con 4.350 soggetti conta la dimensione dell'effetto, non il p.

**Perché il kappa pesato lineare e non quadratico?**
I livelli KDIGO sono ordinali. Il kappa quadratico equivale a un coefficiente di correlazione intraclasse, che tratta la scala come a intervalli (Fleiss & Cohen 1973).

**Perché questi intervalli di confidenza?**
Formule analitiche dove funzionano anche con pochi casi, perché il bootstrap copre meno del 95% quando i positivi sono pochi (Boyd et al. 2013):
- DeLong per l'AUROC;
- Wilson per le proporzioni, raccomandato con n ≤ 40, come i 21 "molto alto";
- logit di Boyd per la PR-AUC.

Il bootstrap stratificato resta per kappa e medie per livello.

**Perché il t di Nadeau-Bengio e non un t normale?**
Nella CV i fold condividono gran parte del training, quindi le differenze fra fold non sono indipendenti: il t normale sottostima la varianza e trova differenze "significative" che non lo sono. La correzione moltiplica la varianza per (1/k + n_test/n_train), cioè 0,45 invece di 0,2.

**Perché Holm?**
I confronti sono molti: 10 coppie di modelli, 6 tecniche, 11 candidati. Holm controlla l'errore complessivo come Bonferroni, ma è più potente.

**(scomoda) Con 21 "molto alto" che conclusioni puoi trarre?**
Solo differenze grandi: gli intervalli vanno da circa 0,65 a 0,99. Per questo l'analisi per livello si fa sulle previsioni out-of-fold del training (50 + 21 casi) e non sul test (17 + 6). Per lo stesso motivo le conclusioni dicono "i casi gravi non sono riconosciuti più dei moderati", non "sono riconosciuti peggio".

**Perché AUROC aggregata e non media dei fold?**
Serve l'aggregata per l'analisi per livello e per la soglia. È più prudente: penalizza i modelli con calibrazione diversa fra i fold (Forman & Scholz 2010) e vale fra 0,003 e 0,006 in meno della media dei fold.

### E. Bilanciamento

**Perché il bilanciamento non aiuta?**
- con modelli forti non migliora la discriminazione e peggiora la calibrazione (van den Goorbergh et al. 2022; Carriero et al. 2025);
- sulla classificazione, pesare le classi equivale a spostare la soglia (Elkan 2001), e la nostra soglia a sensibilità fissata lo neutralizza già;
- un guadagno reale richiederebbe un ordinamento diverso dei soggetti, e non c'è.

**Allora perché in letteratura sembra funzionare?**
Perché lo si valuta alla soglia 0,5: lì il recall passa dal 2% al 60%. È il risultato didatticamente più forte della tesi, e si è riprodotto identico sul test (dal 4,4% al 40–65%).

**Perché SMOTE-NC e non SMOTE?**
SMOTE interpola anche le categoriche come se fossero numeri, e produce valori come `Smoking` = 1,4. SMOTE-NC dà alle categoriche il valore più frequente fra i vicini.

**Perché CTGAN è andato così male?**
- **dati**: ci sono circa 340 positivi per fold (270 nei fold interni), mentre CTGAN è stato sviluppato su insiemi da 1.000 a 23.000 righe; con pochi dati genera campioni poco vari;
- **letteratura**: con circa 300 positivi SMOTE eguaglia o batte i generatori profondi (Kotelnikov et al. 2023);
- **risultato**: AUROC 0,62 contro 0,69, e pendenza di calibrazione 0,42, cioè probabilità non solo gonfiate ma ordinate male. Per questo era esplorativo fin dal protocollo.

**Cos'è la ricalibrazione di Platt annidata?**
È una regressione logistica su logit(p). Viene stimata sulle previsioni che il modello, riaddestrato sui fold interni bilanciati, fa sulle righe **reali** dei fold interni, e poi si applica al fold esterno. Porta l'intercetta da circa −2 a −0,01.

**Perché i pesi per livello 1:2:3?**
Sono l'unico modo diretto di dare più importanza ai casi gravi: è l'apprendimento sensibile al costo (Zadrozny et al. 2003), ed Elkan 2001 cita proprio la gravità di una malattia come costo. Lo schema è stato fissato prima e ce n'è uno solo, un limite dichiarato. Non ha aiutato nemmeno fra i diabetici, dove i casi gravi sono quasi 5 volte più densi.

**Perché 30 tentativi in Fase B e non 100?**
Nei log della Fase A, a 30 tentativi la PR-AUC interna dista in mediana meno di 0,005 dal valore a 100; in più c'è l'avvio caldo dall'ottimo della Fase A. Il budget ridotto è dichiarato.

### F. Diabetici

**Come va il modello sui diabetici?**
Discrimina come sugli altri: differenze di AUROC da −0,038 a +0,042, con intervalli che comprendono lo zero. Ma alla soglia globale li segnala quasi tutti (97–100%), e la decision curve non mostra vantaggi fino al 10%. Fra i diabetici hanno ragione le linee guida, che prescrivono l'esame ogni anno a tutti: il modello serve fra i non diabetici.

**Perché non riaddestrare il modello sui soli diabetici?**
Nel training sono 68 positivi, sotto i 100 eventi minimi (Riley et al. 2024). E l'uso previsto del modello riguarda i non diabetici.

**Perché non confrontate la PR-AUC fra diabetici e non diabetici?**
Perché dipende dalla prevalenza. Vale 0,44 contro 0,19, ma rapportata alla prevalenza (25,9% contro 8,7%) è più bassa. È il secondo "risultato apparente".

### G. Tetto, Fase D e audit

**(scomoda) AUROC 0,70 non è poco?**
È moderata, ma è il tetto di questo bersaglio con questi dati, ed è misurato:
- la componente eGFR si riconosce (0,80–0,86), l'albuminuria no (0,67–0,69);
- 11 strategie diverse restano fra 0,692 e 0,710;
- in letteratura i modelli per l'albuminuria senza esami delle urine stanno fra 0,58 e 0,76;
- il punteggio sviluppato nello stesso ospedale (Wu et al. 2017) arriva a 0,70–0,72.

**Come sai che il limite è nei dati e non nel metodo?**
Cinque prove:
1. il limite è l'albuminuria, non l'eGFR;
2. strategie molto diverse danno lo stesso risultato;
3. più dati aiuterebbero poco: dal 60% al 100% del training, +0,006/+0,008;
4. la pipeline trova i segnali quando ci sono: 0,933 con l'albumina urinaria, 0,795 e 0,831 con le variabili semi-sintetiche;
5. l'etichetta è rumorosa: solo il 43,5% degli ACR ≥ 30 misurati su urina casuale si conferma sulla prima urina del mattino (Saydah et al. 2013).

**(scomoda) "Nessun candidato migliora" non è solo mancanza di potenza?**
In parte sì, ed è dichiarato:
- con una sola partizione la regola rileva solo differenze da 0,02 (un confronto) a 0,04 (Holm su 11 candidati);
- la CV ripetuta prevista dalla regola non è mai stata eseguita: l'ha scoperto l'audit del 23/09, e ho deciso di non farla (circa 16 ore di calcolo per differenze attese di 0,01–0,02).

Il lato forte: l'estremo superiore degli intervalli contro la Random Forest è al massimo +0,039, quindi un guadagno oltre 0,04 è escluso per tutti i candidati.

**(scomoda) Nella configurazione c'è un tri-ensemble a 0,717: perché ora 0,70?**
Lo 0,717 veniva da un'analisi preliminare non registrata e non riproducibile. L'ho registrata come candidato della Fase D:
- scegliendo le 21 variabili dentro ogni fold, l'AUROC è 0,703;
- scegliendole su tutto il training, 0,716: è la distorsione da selezione, perché la scelta delle variabili ha visto anche i soggetti di validazione (Ambroise & McLachlan 2002).

Lo 0,717 quindi non va citato come risultato. Contro SCORED la versione distorta sembra significativa (p = 0,03), quella corretta no (p = 0,07). Il valore reale del tri-ensemble è la parsimonia.

**(scomoda) Le analisi post-hoc non sono un modo di cercare risultati?**
- sono dichiarate post-hoc nella configurazione, con regole scritte prima dei calcoli;
- da dire con onestà: git certifica che la regola precede il calcolo solo per il test set e per le diagnostiche dell'audit. Per la Fase D e il blocco qualità, protocollo e risultati sono in commit dello stesso minuto;
- nessuna analisi post-hoc ha cambiato una conclusione o scelto un modello, e il test non è stato riaperto.

**Cos'è TabPFN? E EBM?**
- **TabPFN v2** è un transformer pre-addestrato su milioni di dataset tabellari sintetici. Non si addestra sui nostri dati: fa la previsione "in contesto", in un solo passaggio, con il training come esempio;
- **EBM** (Explainable Boosting Machine) è un modello additivo generalizzato imparato con il boosting: interpretabile come una logistica, ma con curve non lineari per ogni variabile.

### H. Utilità clinica

**Cos'è la decision curve e cosa dice?**
Il net benefit è NB(t) = VP/n − FP/n × t/(1 − t). La soglia t dice quanti esami inutili vale un caso trovato: al 7%, (1 − t)/t = 13,3. Il modello si confronta con "testare tutti" e "non testare nessuno":
- sotto il 5% equivale a testare tutti;
- dal 5–6% in su batte entrambe le strategie, per tutti e 4 i modelli;
- al 7% evita 9–13 esami inutili ogni 100 persone, al 10% 25–28;
- sul test: 13–19 al 7%, 29–33 al 10%.

**Perché le soglie dal 2% al 20%, e proprio il 5% e il 7%?**
- 5% e 7% sono i punti operativi usati da Bragg-Gresham et al. per la stessa decisione: mandare all'esame dell'albuminuria un adulto senza diabete;
- 2% significa accettare 49 esami inutili per ogni caso trovato; 20% significa accettarne 4;
- l'intervallo è stato fissato prima di guardare la curva, e la curva non serve a scegliere la soglia (Vickers et al. 2019).

**(scomoda) Con MCC 0,08 e F1 0,20 il modello non è inutile?**
Quelle misure pesano allo stesso modo un caso mancato e un esame inutile, mentre al 7% un caso mancato vale 13 esami. Alla soglia clinica sono improprie (Van Calster et al. 2025): le riporto solo in una tabella descrittiva. La decision curve invece dice che il modello serve.

**Il modello fa risparmiare?**
Abbassa il costo per caso trovato: al 7%, da $502 a $336–383. Però trova meno casi: recuperare ogni caso mancato testando tutti costerebbe circa $880–1.100. Non è un'analisi costo-efficacia: non ci sono esiti, trattamenti né costi a valle.

**Com'è la calibrazione?**

| modello | pendenza out-of-fold | pendenza sul test | lettura |
|---|---|---|---|
| logistica SCORED | 0,97 | 1,11 | calibrata |
| logistica penalizzata | 0,94 | 0,79 | calibrata out-of-fold, troppo estrema sul test |
| Random Forest | 1,31 | 1,27 | probabilità troppo schiacciate, in entrambi i campioni |
| XGBoost | 0,86 | 0,95 | troppo estrema out-of-fold, calibrata sul test |

In media la calibrazione è buona (intercetta 0, O:E 1). La ricalibrazione di Platt riporta tutte le pendenze fra 1,01 e 1,15 sul test. La "pendenza media 1,02" della Fase B nascondeva due errori opposti: è il terzo "risultato apparente".

### I. Test set

**Come garantisci che il test sia stato usato una volta sola?**
- il codice legge il file solo se `authorized` è `true` in config;
- la sequenza dei commit lo documenta: protocollo e codice, poi autorizzazione (`b7d5f61`), risultati, chiusura (`5ffadf6`);
- `analytics/test/RUN.json` registra un'esecuzione sola, con data, versioni delle librerie e impronta SHA-256 del file;
- i test automatici impediscono di leggere il file fuori dall'esecuzione e di ripeterla senza un motivo scritto;
- regola d'oro: se trovassi un errore, lo correggerei dichiarando che il test è stato usato due volte.

**(scomoda) Perché sul test l'AUROC è più alta che out-of-fold?**
Non è un miglioramento:
- gli intervalli sono larghi (±0,045) e comprendono la stima out-of-fold per 3 modelli su 4;
- i modelli finali sono addestrati sul 100% del training e quelli dei fold sull'80%, ma la curva di apprendimento prevede solo +0,006/+0,008;
- il resto è variabilità di campionamento.

**Che cosa non si è confermato?**
- la buona calibrazione della logistica penalizzata: pendenza 0,79 sul test;
- l'eccesso di XGBoost: sul test è calibrato (0,95);
- la sottostima della Random Forest fra i diabetici: O:E 1,08, con un intervallo da 0,73 a 1,47, cioè con 23 eventi non si può dire nulla.

**Il test è una validazione esterna?**
No: stessa coorte e stessa finestra temporale, quindi una conferma interna. La mancanza di una validazione esterna è il limite principale.

### L. Stato dell'arte, originalità e limiti

**Cosa c'è di originale?**
Non un algoritmo nuovo, ma:
1. **il bersaglio**: il composito KDIGO attuale (eGFR < 60 o ACR ≥ 30), stimato senza esami renali in una coorte cinese con e senza diabete. Non ne ho trovati in letteratura (ricerca su PubMed ed Europe PMC del 23/09), formulato con prudenza: Wen et al. 2020 e Wu et al. 2017 ci vanno vicino;
2. **un risultato metodologico**: il bilanciamento non fa riconoscere più casi gravi, misurato con protocolli scritti prima, insieme a 4 "risultati apparenti" misurati sugli stessi dati;
3. **l'utilità clinica misurata** con la decision curve: nella revisione di Haris et al. 2024 nessuno dei 12 modelli su cartelle cliniche di comunità la riporta;
4. **il tetto spiegato con i dati**, non ipotizzato.

**Come vi collocate rispetto alla letteratura?**
Dipende dal bersaglio:
- **eGFR ridotto**: 0,83–0,92 nelle stime interne, 0,71–0,89 in validazione;
- **composito o sola albuminuria**: 0,61–0,77, come MERWACS (0,68–0,73), Muntner (0,71), Tanner (0,73–0,76), Bragg-Gresham (0,734 senza eGFR, 0,752 con eGFR) e Wu 2017 (0,70–0,72);
- **K-Risk**: 0,70 out-of-fold e 0,71–0,74 sul test, dove deve stare.

**Perché non confrontarsi con il KFRE?**
Il KFRE (Tangri et al. 2016) predice l'insufficienza renale in chi ha già una malattia renale cronica diagnosticata, e usa proprio eGFR e ACR: lavora a valle. K-Risk lavora a monte: decide a chi fare quegli esami. Se li usasse come input ricopierebbe la definizione del bersaglio.

**(scomoda) C'è sovrapposizione con Wu et al. 2017?**
È lo stesso ospedale, e il loro periodo di validazione (gennaio 2011–aprile 2015) comprende la raccolta del dataset (febbraio–aprile 2012): non si può escludere che alcuni pazienti siano in comune. Toglie parte dell'originalità, ma conferma che 0,70 è il livello realistico per questo centro.

**Quali sono i limiti principali?**
- coorte ospedaliera, non uno screening di popolazione;
- una sola misura: si parla di marcatori, non di diagnosi;
- ACR da un solo campione, di tipo e unità non documentati;
- nessuna validazione esterna;
- pochi casi gravi (71 nel training, 23 nel test) e pochi diabetici positivi (68);
- SCORED ristimato con 5 predittori su 9;
- Fase D, blocco qualità e audit post-hoc, con la CV ripetuta non eseguita;
- possibile effetto della giornata di raccolta: la prevalenza va dal 4,5% al 24,5%.

### M. Metodo di lavoro e riproducibilità

**Come hai organizzato il lavoro?**
- **git**: un branch per fase e 13 pull request unite in `main`;
- **un solo file di configurazione** (`configs/config.yaml`) per percorsi, seed, soglie, gruppi di feature, protocolli e regole di decisione;
- **protocolli prima dei risultati**: ogni protocollo è scritto nel Notepad e in config prima del calcolo;
- **test**: 177 automatici con pytest;
- **esecuzioni**: riprendibili, con i fold imputati in cache e seed fissi;
- **dati e ambiente**: dati grezzi versionati con impronta, file di lock delle versioni.

Il lavoro sperimentale sta fra il 17 e il 23/09: preparati a raccontare i tempi, con i calcoli lunghi eseguiti di notte.

**Come si riproduce?**
- i comandi, in ordine, sono nel README;
- l'ambiente è Python 3.12.10 su Windows 11, con le versioni fissate in `requirements-lock.txt`;
- librerie principali: scikit-learn 1.9.1, XGBoost 3.4.1, Optuna 5.0.0, SHAP 0.52.0, imbalanced-learn 0.14.2, ctgan 0.12.1 con PyTorch 2.14.0; per la Fase D LightGBM 4.7.0, CatBoost 1.2.10, InterpretML (EBM) e TabPFN;
- per il test, il preprocessore ristimato è identico bit a bit alla cache.

**(scomoda) Chi ha fatto le "revisioni indipendenti del codice" e l'audit "a quattro ruoli" citati nel Notepad?**
Questa risposta la conosci solo tu: dalla con esattezza e per primo, perché i relatori la troveranno nel registro. Il Notepad parla di revisioni fatte "da un agente revisore separato, in sola lettura" e di un audit con quattro ruoli (esperto di codice, critico, ricercatore, valorizzatore). In ogni caso conviene dire:
- quali strumenti hai usato e per cosa (codice, revisioni, ricerca bibliografica) e che cosa hai deciso tu;
- quali garanzie non dipendono da chi ha scritto il codice:
  - protocolli scritti prima dei risultati;
  - 177 test automatici;
  - controlli di coerenza (il classificatore di maggioranza, le due formule del net benefit);
  - difetti e correzioni delle revisioni registrati nel Notepad;
  - test set protetto;
- che sai spiegare ogni scelta: è lo scopo di questo documento.

Verifica prima dell'incontro se il corso o l'ateneo hanno regole sull'uso di strumenti di intelligenza artificiale nella tesi.

### N. Tesi e prossimi passi

**A che punto è la tesi?**
- in LaTeX, con il template UNISA: frontespizio e introduzione in corso, capitoli vuoti, bibliografia avviata (30 voci);
- il contenuto dei capitoli di metodo e risultati è già scritto nel Notepad, nelle 45 figure e nelle tabelle di `analytics/`, e va trasformato in testo;
- `valorizzazione_tesi.md` §8 prepara i punti deboli da presentare alla commissione, §9bis elenca cosa inserire.

**Quali sono i prossimi passi?**
- **scrittura della tesi**;
- **prototipo dimostrativo**, per esempio un calcolatore con avvertenze chiare: non è un dispositivo medico;
- **validazione esterna** in una popolazione di screening: è il limite principale;
- **follow-up dei pazienti discordanti** (probabilità alta ma marcatori normali): con un dataset trasversale non si sa se sono falsi allarmi o casi futuri;
- **da verificare in letteratura**: le ipotesi sul peptide C, eliminato dal rene, e sull'albumina glicata (Notepad, sezione "Interpretazione: risultati").

---

## 4. Domande da fare tu ai relatori

1. **Struttura della tesi**: tutto nel testo, oppure un filo principale (Fasi A, B, C e test) con Fase D, utilità clinica e audit come capitoli di supporto o appendici? Quanto spazio dare alle parti post-hoc?
2. **Prototipo**: che cosa si aspettano? Un calcolatore web, con quale modello e con quali avvertenze? Il sito attuale mostra i risultati ma non calcola il rischio.
3. **Tempi**: scadenze di consegna e sessione di laurea (sul frontespizio: ottobre 2026).
4. **Validazione esterna**: vale la pena tentarla in questa tesi, o resta un limite dichiarato?
5. **Strumenti usati**: come vanno dichiarati nella tesi, se ci sono regole del corso.
6. **Seguito**: il lavoro può diventare un articolo breve o un poster?

---

## 5. Prima dell'incontro: incongruenze della repo da conoscere

Nessuna cambia un risultato, ma i relatori potrebbero notarle aprendo la repo.
- **README, "Stato di avanzamento"**: il prototipo dimostrativo è ancora da spuntare, mentre `docs/` contiene il sito. Chiarisci che è un sito dei risultati.
- **README, "Struttura del repository"**: elenca `papers/`, che però è escluso da git (`.gitignore`), quindi i PDF non sono nella repo. Le fonti sono in `valorizzazione_tesi.md` §10 e in `docs/thesis/bib.bib`.
- **KDIGO 2024**: `Notepad.md` ("Punti da verificare", "Fase A: stato") e `Scope.md` ("Riferimenti") la danno ancora "da procurare". Però `valorizzazione_tesi.md` la cita come letta nel testo completo.
- **`Notepad.md`, "Da fare" e "Fase A: stato"**: alcune voci non spuntate sono concluse (Fase C, test set, librerie della Fase B). Resta davvero aperta la verifica in letteratura delle ipotesi su peptide C e albumina glicata.
- **Recall alla soglia 0,5 con l'undersampling**: il README e il Notepad scrivono 61%, `valorizzazione_tesi.md` 60%. La media dei 4 modelli (`analytics/phase_b/evaluation/naive.csv`) è 0,605: il valore giusto è 60%.
- **"Screening di popolazione"**: le sezioni del Notepad scritte prima del 22/09 usano questa formula. Va letta come "coorte ospedaliera, in maggioranza senza diabete noto", come dichiara il Notepad stesso nei limiti.
- **Numero dei test**: 168 funzioni di test, che con la parametrizzazione diventano 177 casi, il numero citato in `valorizzazione_tesi.md`.

---

## 6. Glossario dei metodi

**Clinica**
- **CKD** (malattia renale cronica): alterazione della struttura o della funzione dei reni che dura più di 3 mesi.
- **KDIGO**: l'organizzazione che scrive le linee guida sulla malattia renale; definisce la CKD con eGFR e ACR, e il rischio con la heatmap G × A.
- **eGFR**: filtrato glomerulare stimato dalla creatinina; sotto 60 ml/min/1,73 m² la funzione renale è ridotta.
- **CKD-EPI 2021**: equazione dell'eGFR da creatinina, età e sesso, senza coefficiente etnico (Inker et al. 2021).
- **ACR**: rapporto albumina/creatinina nelle urine. Da 30 mg/g in su si parla di albuminuria (categoria A2); oltre 300 di albuminuria grave (A3).
- **DKD**: malattia renale cronica nel diabete; KDIGO la usa come sinonimo di "CKD nel diabete", senza presupporre la causa.
- **SCORED**: punteggio di screening per la malattia renale occulta (Bang et al. 2007), con 9 elementi; AUROC 0,71 in validazione esterna.
- **KFRE**: equazione che predice dialisi o trapianto a 2 e 5 anni nella CKD già diagnosticata (Tangri et al. 2016).

**Dati e preprocessing**
- **Leakage**: informazione sul bersaglio che entra nell'input, o nella stima, e gonfia le prestazioni.
- **SMD** (differenza media standardizzata): differenza fra le medie divisa per la deviazione standard media; sotto 0,1 è trascurabile.
- **MissForest**: imputazione iterativa. Per ogni colonna con buchi si addestra una foresta che la predice dalle altre, e si ripete fino a stabilità. Qui è realizzata con `IterativeImputer` più `ExtraTreesRegressor`.
- **ExtraTrees**: foresta con soglie di divisione casuali; è il modello *dentro* l'imputer.
- **KNNImputer**: riempie un valore con la media dei 5 soggetti più simili.
- **MICE**: equazioni concatenate, qui con regressione bayesiana e un'imputazione sola.
- **Regola di 1 errore standard**: si sceglie il metodo più semplice entro 1 errore standard dal migliore (Hastie et al. 2009).
- **Confronto appaiato**: si confrontano due metodi sugli stessi fold e sugli stessi valori, fold per fold.

**Modelli**
- **DummyClassifier (prior)**: prevede a tutti la prevalenza del training.
- **Regressione logistica**: modello lineare sul logit della probabilità.
- **Penalizzazione**: L1 (lasso, azzera dei coefficienti), L2 (ridge, li rimpicciolisce), elastic net (le mescola in proporzione `l1_ratio`). `C` è l'inverso della forza; `saga` è il solver che supporta l'elastic net.
- **Random Forest** (Breiman 2001): media di alberi profondi, ciascuno su un campione bootstrap e con variabili casuali a ogni divisione (bagging).
- **XGBoost** (Chen & Guestrin 2016): alberi poco profondi aggiunti in sequenza, ognuno corregge gli errori dei precedenti (gradient boosting).
  - `learning_rate` riduce il contributo di ogni albero;
  - `subsample` e `colsample_bytree` campionano righe e colonne;
  - `reg_alpha` e `reg_lambda` sono le penalità L1 e L2.
- **LightGBM**: boosting con crescita per foglia e istogrammi.
- **CatBoost**: boosting con gestione nativa delle categoriche.
- **EBM**: modello additivo generalizzato imparato con il boosting; interpretabile.
- **TabPFN v2**: transformer pre-addestrato su dati tabellari sintetici; prevede senza addestrarsi sui nostri dati (Hollmann et al. 2025).
- **Ensemble**: media delle probabilità di più modelli.

**Addestramento**
- **CV stratificata**: ogni fold conserva le proporzioni degli strati, qui dei livelli KDIGO.
- **CV annidata**: una CV interna sceglie gli iperparametri, una esterna stima le prestazioni.
- **Previsioni out-of-fold**: ogni soggetto è predetto da un modello che non l'ha visto in addestramento.
- **Optuna**: libreria di ottimizzazione degli iperparametri.
- **TPE**: ottimizzazione bayesiana che confronta la densità dei tentativi buoni e dei tentativi cattivi.
- **`MedianPruner`**: interrompe un tentativo che, dopo un fold interno, è sotto la mediana dei precedenti.
- **Avvio caldo**: il primo tentativo è una configurazione già buona, qui l'ottimo della Fase A.

**Bilanciamento**
- **Undersampling / oversampling casuali**: tolgono negativi, o duplicano positivi, fino a 1:1.
- **SMOTE**: crea positivi sintetici su un segmento fra un positivo e un suo vicino positivo.
- **SMOTE-NC**: SMOTE per dati misti; le categoriche prendono il valore più frequente fra i vicini.
- **CTGAN**: rete generativa avversaria per tabelle (Xu et al. 2019), con normalizzazione per modi delle continue e un vettore condizionale per generare le classi rare.
- **Pesi di classe**: inverso della prevalenza; nei fold circa 5,1 per un positivo e 0,55 per un negativo, con media 1.
- **Pesi per esempio**, o apprendimento sensibile al costo: ogni esempio pesa quanto il costo del suo errore.
- **Platt**: ricalibrazione con una logistica su logit(p), stimata su previsioni non viste in addestramento.

**Metriche**
- **AUROC**: probabilità che un positivo a caso abbia un punteggio più alto di un negativo a caso; 0,5 è il caso. Non dipende dalla prevalenza né dalla soglia.
- **PR-AUC** (average precision): area sotto la curva precision-recall; il valore di base è la prevalenza.
- **Precision**, **recall** (sensibilità), **specificità**; **quota di allerta**: la quota di soggetti segnalati, cioè da esaminare.
- **Indice di Youden**: sensibilità + specificità − 1; la soglia che lo massimizza assume costi uguali.
- **Calibrazione**:
  - intercetta: 0 se le probabilità sono giuste in media, negativa se sono gonfiate;
  - pendenza: 1 ideale, sotto 1 probabilità troppo estreme, sopra 1 troppo schiacciate;
  - O:E: eventi osservati diviso attesi, sopra 1 c'è sottostima;
  - Brier: errore quadratico medio delle probabilità.
- **Curva di calibrazione flessibile**: logistica su una spline cubica ristretta di logit(p), con 4 nodi (Harrell 2015).
- **Net benefit / decision curve**: veri positivi meno falsi positivi pesati con t/(1 − t), confrontati con "testare tutti" e "nessuno" (Vickers & Elkin 2006).
- **MCC, F1, accuratezza bilanciata**: misure a soglia che pesano allo stesso modo i due errori; improprie alla soglia clinica.

**Statistica**
- **DeLong**: varianza non parametrica dell'AUROC, per intervalli e confronti.
- **Wilson**: intervallo per una proporzione, affidabile con pochi casi.
- **Logit di Boyd**: intervallo della PR-AUC sulla scala logit, con n uguale al numero di positivi.
- **Bootstrap percentile stratificato**: 2.000 ricampionamenti dentro ogni livello; l'intervallo va dal 2,5° al 97,5° percentile.
- **Jonckheere-Terpstra**: test di tendenza fra gruppi ordinati; la "concordanza" è la sua dimensione dell'effetto.
- **Kappa di Cohen pesato**: accordo oltre il caso, con credito parziale per i disaccordi vicini.
- **Paradosso della prevalenza**: accordo osservato alto e kappa basso quando una categoria domina.
- **McNemar esatto**: test sulle coppie discordanti (casi trovati da un metodo e non dall'altro).
- **IC di Newcombe**: intervallo della differenza fra due proporzioni appaiate.
- **t corretto di Nadeau-Bengio**: t per differenze in CV, con varianza corretta per la sovrapposizione dei training.
- **Holm**: correzione per confronti multipli, a passi, più potente di Bonferroni.
- **Effetto spettro**: togliendo i casi vicini alla soglia l'AUROC sale, anche con un'etichetta perfetta.
- **Distorsione da selezione** (Ambroise & McLachlan 2002): scegliere le variabili su tutti i dati prima della CV rende la stima ottimista.

**Interpretazione e reporting**
- **Odds ratio**: exp(coefficiente), per 1 deviazione standard o per unità; intervallo di Wald exp(β ± 1,96 · SE).
- **SHAP**: contributo di ogni variabile alla singola previsione, dalla teoria dei giochi (valori di Shapley); i contributi sommati danno la previsione.
- **TreeSHAP**: calcolo esatto dei valori SHAP per gli alberi. Per XGBoost ho usato `pred_contribs` di XGBoost, perché `shap` 0.52 con XGBoost 3.4 sbaglia il valore di base (verificato con un test).
- **TRIPOD+AI**: linee guida per riportare i modelli predittivi (Collins et al. 2024).

---

## 7. Numeri da sapere a memoria

| cosa | numero |
|---|---|
| dataset | 5.922 soggetti × 190 variabili; 5.801 utilizzabili |
| positivi | 567 (9,77%), circa il 90% per albuminuria |
| livelli KDIGO | basso 5.234, moderato 473, alto 67, molto alto 27 |
| training / test | 4.350 (425 positivi) / 1.451 (142) |
| casi gravi | 71 nel training (50 + 21), 23 nel test (17 + 6) |
| diabetici | 263 nel training (68 positivi), 88 nel test (23) |
| feature | 74 (60 numeriche + 14 categoriche); 67 in `no_consequence` |
| righe con almeno un mancante | 38% |
| eventi per variabile | 5,7 |
| AUROC out-of-fold | 0,675 (SCORED) – 0,703 (Random Forest) |
| AUROC sul test | 0,714 – 0,743 |
| PR-AUC | 0,224–0,251, cioè 2,3–2,6 volte la prevalenza |
| da esaminare a sensibilità 0,90 | 77–80%, contro il 90% scegliendo a caso |
| concordanza con i livelli | 0,674–0,702 |
| kappa fasce/livelli | 0,197–0,228 |
| "molto alto" riconosciuti | 18–20 su 21 |
| bilanciamento, casi gravi | ±3 su 71, p di Holm 1,00 |
| soglia 0,5, recall | dal 2% al 60% (training); dal 4,4% al 40–65% (test) |
| CTGAN | AUROC 0,62 |
| componenti del bersaglio | eGFR < 60: 0,80–0,86; sola albuminuria: 0,67–0,69 |
| Fase D | 11 candidati fra 0,692 e 0,710; differenza minima rilevabile 0,02–0,04 |
| controlli positivi | 0,933 (albumina urinaria); 0,795 e 0,831 (semi-sintetici) |
| utilità al 7% / 10% | 9–13 / 25–28 esami inutili evitati ogni 100 (test: 13–19 / 29–33) |
| diabetici alla soglia globale | segnalati il 97–100% |
| test set | 22/09, 18:20–18:37, una volta; 5 affermazioni, nessuna contraddetta |
| codice | 28 moduli Python, circa 5.600 righe; 177 test; 13 pull request |
