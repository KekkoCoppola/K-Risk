<div align="center">

# K-Risk

### Screening dei marcatori di malattia renale cronica senza esami renali

*Pipeline di machine learning riproducibile per stimare il rischio renale da esami del sangue di routine, confrontata con la stratificazione clinica KDIGO e con tecniche di data augmentation per classi sbilanciate.*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E.svg)](https://scikit-learn.org/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](tests/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/stato-in%20sviluppo-orange.svg)](#-stato-di-avanzamento)

Progetto di tesi di laurea triennale · Francesco Coppola

</div>

---

## Indice

1. [In breve](#-in-breve)
2. [Motivazione clinica](#-motivazione-clinica)
3. [Domande di ricerca](#-domande-di-ricerca)
4. [Dati](#-dati)
5. [Definizione del target](#-definizione-del-target)
6. [Architettura della pipeline](#-architettura-della-pipeline)
7. [Metodologia](#-metodologia)
8. [Risultati principali](#-risultati-principali)
9. [Stato di avanzamento](#-stato-di-avanzamento)
10. [Riproducibilità](#-riproducibilità)
11. [Struttura del repository](#-struttura-del-repository)
12. [Limiti dichiarati](#-limiti-dichiarati)
13. [Citazione e licenza](#-citazione-e-licenza)

---

## 🔎 In breve

K-Risk allena modelli di classificazione che, **senza usare esami renali**, stimano quali soggetti presentano marcatori di malattia renale cronica (CKD) secondo i criteri **KDIGO**. Le stime vengono poi confrontate con la stratificazione clinica del rischio, e si misura quanto le tecniche di **data augmentation** (undersampling, oversampling, SMOTE, CTGAN) migliorano il riconoscimento dei casi più gravi.

| | |
|---|---|
| **Popolazione** | 5.801 soggetti di una coorte ospedaliera (reparto di Diabetologia ed Endocrinologia, Shanghai, 2012), in maggioranza senza diabete noto |
| **Input** | 74 variabili, nessun esame renale: anagrafica, antropometria, pressione, glicemia, lipidi, emocromo, anamnesi, stili di vita |
| **Escluse** | tutte le variabili renali (creatinina, ACR, albuminuria, azotemia, eGFR e derivati) |
| **Target** | binario: `ACR ≥ 30 mg/g` oppure `eGFR < 60 ml/min/1,73 m²` (eGFR ricalcolato con CKD-EPI 2021) |
| **Prevalenza** | 567 positivi su 5.801 (**9,77%**): classi sbilanciate |
| **Valutazione** | AUC, PR-AUC, sensibilità per livello KDIGO, concordanza con la stratificazione clinica |

Il contributo non è un nuovo algoritmo, ma una **pipeline completa, riproducibile e metodologicamente rigorosa**: ogni scelta è motivata, confrontata con alternative in cross-validation e documentata.

---

## 🩺 Motivazione clinica

La malattia renale cronica è spesso **asintomatica fino a stadi avanzati** e si diagnostica con due esami specifici: la creatinina sierica (da cui si stima il filtrato, eGFR) e il rapporto albumina/creatinina urinaria (ACR). Nelle persone con diabete le linee guida prescrivono questi esami ogni anno; nella **popolazione generale** invece non vengono eseguiti di routine.

La domanda di K-Risk è quindi: **con i soli esami del sangue di routine, è possibile stabilire a chi dare la priorità per gli esami renali?** Un modello di questo tipo non sostituisce gli esami: aiuta a indirizzarli.

---

## ❓ Domande di ricerca

Le prime quattro domande valgono per i modelli sui dati originali (Fase A) e, identiche, per ogni tecnica di bilanciamento (Fase B).

| # | Domanda | Come si misura |
|---|---------|----------------|
| 1 | Il modello distingue chi ha marcatori di malattia renale? | AUC, PR-AUC, precision, recall |
| 2 | Il rischio stimato cresce con la gravità KDIGO? | probabilità per livello, test di tendenza |
| 3 | Quanti casi "alto" e "molto alto" riconosce? | sensibilità per livello (mancare un "molto alto" è l'errore più grave) |
| 4 | Le fasce di rischio del modello corrispondono alla stratificazione clinica? | concordanza con i livelli KDIGO |
| 5 | L'augmentation migliora il riconoscimento dei casi gravi o solo la metrica media? | confronto fra tecniche sulle domande 1–4 |
| 6 | Come si comporta il modello nel sottogruppo diabetico (91 casi)? | risultati descrittivi |

---

## 📊 Dati

Dataset pubblico **A bimodal dataset for diabetes research** (Li et al., *Scientific Data*, 2026), incluso senza modifiche in [`data/raw/`](data/raw/) con licenza CC BY 4.0. Fonte, dizionario delle variabili e checksum sono in [`data/README.md`](data/README.md).

- **5.922 soggetti, 190 variabili**, una sola misurazione per soggetto (dati trasversali)
- **5.801 soggetti utilizzabili**: quelli con creatinina sierica e ACR misurate, necessarie per calcolare il target
- sottogruppo diabetico: **351 soggetti, 91 positivi**, in cui il target coincide con la definizione di malattia renale nel diabete (DKD)

<p align="center">
  <img src="analytics/dataset/02_target_by_diabetes.png" width="48%" alt="Prevalenza del target per stato diabetico">
  <img src="analytics/dataset/03_missing_values.png" width="48%" alt="Valori mancanti">
</p>

---

## 🎯 Definizione del target

```
y = 1   se   ACR ≥ 30 mg/g   oppure   eGFR < 60 ml/min/1,73 m²
y = 0   altrimenti
```

L'eGFR è **ricalcolato con l'equazione CKD-EPI 2021**, raccomandata da KDIGO: la colonna `GFR` del dataset non corrisponde a nessuna equazione standard (CKD-EPI 2009/2021, MDRD, MDRD cinese, Cockcroft-Gault) e contiene valori fuori scala.

Incrociando le categorie di eGFR (G1–G5) e di ACR (A1–A3) si ottiene la **mappa di rischio KDIGO** a quattro livelli. I livelli **non vengono usati per allenare** i modelli, ma per valutarli.

| Livello KDIGO | Soggetti | Target |
|---------------|---------:|:------:|
| basso | 5.234 | 0 |
| moderato | 473 | 1 |
| alto | 67 | 1 |
| molto alto | 27 | 1 |

<p align="center">
  <img src="analytics/dataset/01_kdigo_heatmap.png" width="48%" alt="Mappa di rischio KDIGO">
  <img src="analytics/dataset/06_egfr_comparison.png" width="48%" alt="Confronto eGFR del dataset e CKD-EPI 2021">
</p>

**Perché una classificazione e non una regressione sull'ACR:** l'ACR ha una distribuzione fortemente asimmetrica, le tecniche di augmentation da confrontare sono pensate per la classificazione e KDIGO è già, per costruzione, una classificazione.

---

## 🏗️ Architettura della pipeline

```mermaid
flowchart LR
    A[(Dataset grezzo<br/>5.922 × 190)] --> B[Soggetti eleggibili<br/>5.801]
    B --> C[Target KDIGO<br/>eGFR CKD-EPI 2021 + ACR]
    C --> D{Split 75/25<br/>stratificato<br/>livello KDIGO × diabete}
    D --> T[(Test set<br/>congelato)]
    D --> TR[Training set]
    TR --> E[Selezione feature<br/>190 → 74<br/>nessuna variabile renale]
    E --> F[Cross-validation stratificata]
    subgraph FOLD [Dentro ogni fold, solo sul training]
        direction TB
        G[Standardizzazione<br/>+ imputazione MissForest] --> H[Bilanciamento<br/>Fase B]
        H --> I[Modello]
        I --> L[Ricalibrazione<br/>probabilità]
    end
    F --> FOLD
    FOLD --> M[Previsioni out-of-fold<br/>confronto con KDIGO]
    M --> N[Scelta di modello,<br/>tecnica e soglia]
    N --> O[Valutazione finale<br/>una sola volta]
    T --> O
```

Il principio guida è l'**assenza di data leakage**: ogni trasformazione che impara dai dati (imputazione, scaling, bilanciamento, generatori sintetici) è stimata **solo sul training di ogni fold**, e il test set viene letto una sola volta, alla fine.

---

## 🔬 Metodologia

### 1. Split train/test

- **75/25**, seed fisso (`42`), stratificato sulla combinazione **livello KDIGO × diabete**: i pochi casi "molto alto" e i diabetici sono distribuiti in proporzione in entrambi i set
- training: 4.350 soggetti · test: 1.451 soggetti
- equilibrio verificato con la **differenza media standardizzata (SMD)** su 15 variabili cliniche e con una simulazione su 1.000 split casuali, per mostrare che la stratificazione riduce la variabilità del numero di casi gravi nel test

<p align="center">
  <img src="analytics/split/02_smd_balance.png" width="48%" alt="Bilanciamento SMD train/test">
  <img src="analytics/split/04_stratification_simulation.png" width="48%" alt="Simulazione della stratificazione">
</p>

### 2. Selezione delle feature: da 190 a 74

Ogni colonna del dataset appartiene a **esattamente un gruppo**, dichiarato in [`configs/config.yaml`](configs/config.yaml) e verificato dai test automatici. Nessuna colonna viene esclusa senza un motivo esplicito.

| Gruppo escluso | Esempi | Motivo |
|----------------|--------|--------|
| leakage del target | `SCRE`, `UMAUCR`, `GFR`, `DN` | il target è calcolato da queste colonne |
| esami renali | `BUN`, `RF` | violano il vincolo "senza esami renali" |
| amministrative / non documentate | `NO`, `Data`, `VAR00001` | nessun significato clinico |
| vuote (> 50% NA) | `CDSMS`, `GLU`, `HPLC` | l'imputazione inventerebbe la maggior parte dei valori |
| derivate a soglia | `BMI25`, `Anemia`, `FPGover7` | ricodifiche di variabili già presenti |
| comorbidità con codice 9 | `Angina`, `MI`, `HF` | dopo la ricodifica di "sconosciuto" > 15% NA e nessun segnale |
| indici instabili | `Homaβ` | formula con denominatore `FPG − 3,5`: esplode sopra 1000 |
| > 15% NA sul training | `LC`, `ALY` | soglia di mancanti fissata a priori |

**Risultato:** 74 feature (60 numeriche + 14 categoriche). Un secondo set, **`no_consequence`** (67 feature), toglie le variabili che la malattia renale altera (emoglobina, ematocrito, acido urico, albumina…) ed è usato come **analisi di sensibilità**: serve a verificare che il modello non riconosca la malattia dalle sue conseguenze.

Come controllo anti-leakage, l'AUC univariata di ogni feature finale è confrontata con quella delle variabili renali escluse: nessuna feature in input si avvicina al loro potere discriminante.

<p align="center">
  <img src="analytics/preprocessing/01_column_map.png" width="48%" alt="Mappa delle colonne">
  <img src="analytics/preprocessing/02_univariate_auc.png" width="48%" alt="AUC univariata">
</p>

### 3. Codifiche e trasformazioni

- **categoriche**: codifica senza one-hot (binarie 0/1, ordinali 0/1/2 per fumo, alcol, tè), imputazione con la **moda**
- **numeriche**: standardizzazione, poi imputazione con **MissForest** (lo stesso ordine per tutti i candidati del confronto, perché KNN e MICE lavorano su distanze e regressioni)
- **nessuna trasformazione logaritmica**: testata sulle 25 variabili con asimmetria > 2, guadagno di PR-AUC **+0,008 ± 0,005** (non significativo, t ≈ 1,5). Si mantiene la scala clinica, più leggibile per odds ratio e SHAP
- **valori estremi tenuti**: clinicamente possibili e presenti in uno screening reale; toglierli solo dal training renderebbe il modello impreparato sul test

### 4. Scelta del metodo di imputazione

Il 38% dei soggetti del training ha almeno un valore mancante: eliminare le righe incomplete distruggerebbe oltre un terzo del campione. Quattro metodi, in ordine di complessità, sono stati confrontati in **cross-validation a 5 fold** stratificata sul livello KDIGO, con due criteri:

- **(a) ricostruzione**: si nasconde il 10% dei valori osservati e si misura l'errore (RMSE in unità standardizzate)
- **(b) prestazioni a valle**: regressione logistica fissa usata come strumento di misura (PR-AUC)

| Metodo | RMSE ↓ | PR-AUC ↑ | AUC | Tempo |
|--------|-------:|---------:|----:|------:|
| Mediana | 1,038 ± 0,047 | 0,252 ± 0,030 | 0,693 | 1 s |
| KNN (k = 5) | 0,793 ± 0,057 | 0,259 ± 0,032 | 0,695 | 11 s |
| MICE | 1,252 ± 0,135 | 0,257 ± 0,032 | 0,697 | 70 s |
| **MissForest** | **0,734 ± 0,066** | **0,257 ± 0,031** | **0,696** | **305 s** |

*Set principale, media ± errore standard sui fold. Ogni imputer usa come predittori anche le variabili categoriche (es. `DM` per stimare `HbA1c`).*

**Regola di scelta in due fasi**, derivata dalla regola di 1 errore standard (Hastie, Tibshirani & Friedman 2009): si tengono i metodi con PR-AUC entro 1 SE dal migliore, poi fra questi il più semplice con RMSE entro 1 SE dal migliore.

Cosa emerge:
- **a valle i quattro metodi sono indistinguibili** (PR-AUC 0,252–0,259, errore standard ~0,03)
- **la mediana è esclusa**: non ricostruisce nulla (RMSE ~1, come prevedere la media) e assegna lo **stesso valore** a tutti i mancanti. I 397 soggetti senza `HbA1c` ricevono tutti 5,5, diabetici compresi. In Fase B questi picchi artificiali verrebbero moltiplicati da SMOTE e appresi da CTGAN come parte della distribuzione
- **MICE è escluso**: estrapolazioni estreme sulle variabili asimmetriche (RMSE instabile)
- **KNN e MissForest sono al margine della regola**: KNN rientra nella soglia sul set principale (0,793 contro 0,800) e ne esce nel set di sensibilità. La regola, basata su errori standard non appaiati, qui non separa i due metodi
- il **confronto appaiato** (stessi fold, stessi valori nascosti), che è il confronto corretto fra metodi valutati sugli stessi dati (Nadeau & Bengio 2003), è invece netto e stabile: MissForest ricostruisce meglio **in tutti i fold di entrambi i set** (RMSE +0,060 ± 0,010 per KNN), con prestazioni a valle equivalenti

**Metodo scelto: MissForest**, per entrambi i set di feature e per entrambe le fasi. Oltre al confronto appaiato, lo motivano quattro ragioni:
- la **letteratura clinica**: MissForest ha l'errore di imputazione più basso di KNN e MICE su dati misti, di laboratorio e di diabete (Stekhoven & Bühlmann 2012; Waljee et al. 2013; Tiwaskar et al. 2025)
- la **Fase B**: SMOTE interpola fra casi vicini e CTGAN impara la distribuzione congiunta, quindi servono valori imputati fedeli e correlazioni conservate
- la **coerenza con le altre scelte**: niente logaritmo e valori estremi tenuti. Gli alberi sono insensibili ad asimmetria e valori estremi, le distanze di KNN no
- l'**analisi di sensibilità**: stesso imputer per entrambi i set di feature, così i risultati differiscono solo per le variabili tolte

La scelta è stata fatta dopo aver visto i risultati ed è dichiarata come tale. Il rischio è limitato: a valle i metodi sono equivalenti, e in Fase B è prevista un'analisi di sensibilità con KNN. Il costo (circa 5 minuti) si paga una volta per fold: i dati imputati vengono salvati e riusati per tutti i modelli e le tecniche di bilanciamento. L'imputer non vede mai il target, quindi il riuso non introduce leakage. Storia completa della decisione in [`Notepad.md`](Notepad.md), Passi 4–10.

<p align="center">
  <img src="analytics/preprocessing/04_imputation_main.png" width="70%" alt="Confronto dei metodi di imputazione">
</p>

### 5. Modelli e tecniche di bilanciamento

| Fase | Contenuto |
|------|-----------|
| **A — dati originali** | cinque modelli, uno per ruolo: classificatore di maggioranza (soglia minima), regressione logistica con i predittori del punteggio clinico SCORED (Bang et al. 2007), regressione logistica penalizzata, Random Forest, XGBoost. Cross-validation annidata, stesso budget di ottimizzazione per tutti |
| **B — bilanciamento** | gli stessi cinque modelli con: nessuna correzione, pesi di classe, undersampling, oversampling, SMOTE, CTGAN (anche condizionato al livello KDIGO) |
| **C — sottogruppo diabetico** | stesse previsioni filtrate sui diabetici, soglia e fasce globali fisse |
| **D — tetto di prestazione** (post-hoc) | dieci strategie in più (ensemble, altre famiglie di modelli, TabPFN…) e il tri-ensemble su 21 variabili, con una regola di decisione fissata prima; controllo positivo, curva di apprendimento, qualità dell'etichetta |
| **Qualità e utilità clinica** (post-hoc) | decision curve, calibrazione anche nei sottogruppi, costo per caso trovato, confronto con Bragg-Gresham et al. 2025 |
| **Conclusioni e test** | risposte alle domande 1–6, poi conferma sul test set una sola volta, con protocollo fissato prima |

### 6. Regole metodologiche

- augmentation **solo sul training e dentro ogni fold**: i dati sintetici non finiscono mai in validazione o nel test
- **ricalibrazione** delle probabilità dopo il bilanciamento, che le distorce; si riportano anche metriche basate sull'ordinamento, immuni alla distorsione
- confronto con KDIGO **solo su soggetti reali**: i sintetici non hanno un livello KDIGO
- analisi per livello sulle previsioni **out-of-fold** del training (50 "alto", 21 "molto alto"), conferma sul test (17 e 6)
- **test set usato una volta sola**: scelte di modello, tecnica e soglia si fanno in cross-validation

---

## 📈 Risultati principali

> Previsioni out-of-fold sul training (4.350 soggetti, 425 positivi). La conferma sul test set, una sola volta, ha il protocollo già fissato nel [`Notepad.md`](Notepad.md) ed è in preparazione. Sintesi completa, con lo stato dell'arte verificato, in [`valorizzazione_tesi.md`](valorizzazione_tesi.md).

| # | Domanda | Risposta |
|---|---------|----------|
| 1 | Il modello distingue chi ha marcatori di malattia renale? | sì, in modo modesto: AUROC 0,675–0,703, PR-AUC 2,3–2,6 volte la prevalenza; nessun modello migliore degli altri in modo dimostrabile |
| 2 | Il rischio stimato cresce con la gravità KDIGO? | sì: circa il 70% delle coppie di soggetti è ordinato come KDIGO |
| 3 | Quanti casi "alto" e "molto alto" riconosce? | a sensibilità 0,90, 46–47 "alto" su 50 e 18–20 "molto alto" su 21, ma non più dei moderati |
| 4 | Le fasce corrispondono alla stratificazione clinica? | poco: kappa pesato circa 0,2. Il modello riconosce la presenza dei marcatori, non il grado |
| 5 | L'augmentation migliora il riconoscimento dei casi gravi? | no, a parità di sensibilità; alla soglia 0,5 però **sembra** portare il recall dal 2% al 61% |
| 6 | Come si comporta sui diabetici? | discrimina come sugli altri, ma alla soglia globale degenera in "testare tutti" |

- **tetto di prestazione** (Fase D, post-hoc): dieci strategie in più restano fra AUROC 0,692 e 0,710. Il limite è l'albuminuria: i casi con eGFR < 60 si riconoscono bene (0,80–0,86), quelli con sola albuminuria no (0,67–0,69). Con l'albumina urinaria fra le feature (controllo positivo) l'AUROC sale a 0,933
- **utilità clinica** (post-hoc): alla soglia del 7% il modello evita 9–13 esami inutili ogni 100 persone rispetto a "testare tutti", al 10% 25–28; sui diabetici nessun vantaggio fino al 10%
- **contro SCORED** (Bang et al. 2007): i modelli con gli esami del sangue di routine sono superiori su tutte le misure, ma le differenze non sono significative: replica di Christodoulou et al. 2019
- **quattro risultati apparenti**, misurati: la soglia 0,5, la PR-AUC fra gruppi con prevalenza diversa, la media delle pendenze di calibrazione, la selezione delle variabili fuori dalla validazione

<p align="center">
  <img src="analytics/phase_b/05_naive_vs_real.png" width="48%" alt="Miglioramento apparente alla soglia 0,5 contro guadagno reale">
  <img src="analytics/quality/01_decision_curve.png" width="48%" alt="Decision curve">
</p>

---

## 🗺️ Stato di avanzamento

- [x] Acquisizione del dataset e verifica di integrità (checksum)
- [x] Ricalcolo dell'eGFR (CKD-EPI 2021) e classificazione KDIGO
- [x] Split train/test stratificato e verifica di bilanciamento
- [x] Selezione delle feature con esclusione documentata di ogni colonna
- [x] Confronto dei metodi di imputazione in cross-validation (mediana, KNN, MICE, MissForest)
- [x] Scelta motivata dell'imputer (MissForest)
- [x] Test automatici su split e preprocessing
- [x] **Fase A** — modelli sui dati originali e confronto con KDIGO
- [x] **Fase B** — tecniche di bilanciamento e data augmentation (SMOTE, CTGAN)
- [x] **Fase C** — sottogruppo diabetico
- [x] Interpretabilità (SHAP, coefficienti)
- [x] **Fase D** — ricerca del tetto di prestazione (post-hoc)
- [x] Qualità e utilità clinica: decision curve, calibrazione, costi (post-hoc)
- [x] Conclusioni delle domande 1–6
- [ ] Valutazione finale sul test set (protocollo fissato; codice revisionato prima dell'esecuzione)
- [ ] Prototipo dimostrativo

---

## ⚙️ Riproducibilità

### Ambiente

```bash
git clone https://github.com/KekkoCoppola/K-Risk.git
cd K-Risk
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
```

Testato con **Python 3.12.10** su **Windows 11**. [`requirements-lock.txt`](requirements-lock.txt) fissa le versioni esatte usate per produrre i risultati; [`requirements.txt`](requirements.txt) elenca le dipendenze dirette con le versioni minime.

### Esecuzione

```bash
python -m src.data.split                      # split train/test in data/processed/
python -m src.analytics.dataset_overview      # figure in analytics/dataset/
python -m src.analytics.split_report          # figure in analytics/split/
python -m src.data.imputation                 # confronto imputazione (~10 min, MissForest è il più lento)
python -m src.analytics.preprocessing_report  # figure in analytics/preprocessing/
python -m src.data.imputed                    # fold imputati per la CV annidata (~1 ora, una volta sola)
python -m src.models.phase_a                  # Fase A: 5 modelli, CV annidata, Optuna (diverse ore; riprende da dove si è fermata)
python -m src.models.evaluation               # valutazione Fase A sulle previsioni out-of-fold (~2 min), tabelle in analytics/phase_a/evaluation/
python -m src.models.interpretation           # odds ratio delle logistiche e SHAP di Random Forest e XGBoost (~5 min)
python -m src.analytics.phase_a_report        # figure della valutazione e dell'interpretazione Fase A in analytics/phase_a/
python -m src.models.phase_a --sensitivity depth_1_12       # analisi di sensibilità: XGBoost con max_depth 1-12 (~45 min)
python -m src.models.evaluation --sensitivity depth_1_12    # sua valutazione, confrontata con i modelli primari
python -m src.data.augmented                  # Fase B: training bilanciati per fold (cache in data/augmented/)
python -m src.models.phase_b --techniques class_weight level_weight undersampling oversampling smote_nc smote_nc_level
python -m src.data.augmented --techniques ctgan ctgan_level                 # esplorative
python -m src.models.phase_b --techniques ctgan ctgan_level
python -m src.models.phase_b --calibrate none class_weight level_weight undersampling oversampling smote_nc smote_nc_level
python -m src.models.phase_b --collect        # previsioni out-of-fold in analytics/phase_b/
python -m src.models.evaluation_b             # valutazione Fase B (esito primario, confronti, calibrazione)
python -m src.analytics.phase_b_report        # figure della Fase B in analytics/phase_b/
python -m src.models.phase_c                  # Fase C: sottogruppo diabetico (~3 min)
python -m src.analytics.phase_c_report        # figure della Fase C in analytics/phase_c/
python -m src.models.phase_d --candidates <candidati> --diagnostics <diagnostiche> --evaluate   # Fase D (nomi in configs/config.yaml)
python -m src.analytics.phase_d_report        # figure della Fase D in analytics/phase_d/
python -m src.models.clinical_utility         # qualità e utilità clinica (~3 min)
python -m src.analytics.quality_report        # figure in analytics/quality/
python -m src.models.final_test --run         # conferma sul test set: una volta sola, solo con final_test.authorized = true
python -m pytest                              # test automatici
```

### Garanzie di riproducibilità

- **un solo file di configurazione** ([`configs/config.yaml`](configs/config.yaml)): percorsi, seed, soglie cliniche, gruppi di feature, parametri di imputazione
- **seed fisso** in ogni passaggio casuale
- **dati grezzi versionati** e mai modificati; tutto ciò che è derivato si rigenera dai comandi sopra
- **test automatici** che verificano, tra l'altro: assenza di sovrapposizioni fra train e test, stratificazione, assenza di colonne renali o con codice 9 in input, copertura completa delle colonne nei gruppi di feature, regola di scelta dell'imputazione
- **registro delle decisioni** in [`Notepad.md`](Notepad.md): ogni scelta con verifiche, numeri e motivazioni; lo scope del progetto è in [`Scope.md`](Scope.md)

---

## 🗂️ Struttura del repository

```
K-Risk/
├── analytics/                 figure e tabelle generate
│   ├── dataset/               esplorazione del dataset e del target
│   ├── split/                 verifica dello split train/test
│   ├── preprocessing/         selezione feature e confronto imputazione
│   ├── phase_a/               Fase A: previsioni out-of-fold, tabelle di valutazione (evaluation/), figure
│   ├── phase_b/               Fase B: risultati per tecnica, valutazione (evaluation/), figure
│   ├── phase_c/               Fase C: sottogruppo diabetico
│   ├── phase_d/               Fase D: candidati, diagnostiche, confronto con la regola
│   ├── quality/               qualità e utilità clinica: decision curve, calibrazione, costi
│   └── test/                  conferma finale sul test set (dopo l'esecuzione unica)
├── configs/
│   └── config.yaml            unica fonte di configurazione
├── data/
│   ├── raw/                   dataset originale (CC BY 4.0, non modificato)
│   ├── processed/             split train/test (non versionato)
│   └── augmented/             training set bilanciati (non versionato)
├── docs/
│   ├── thesis/                materiale della tesi
│   └── verifica_stato_arte.md verifica bibliografica dello stato dell'arte
├── papers/                    letteratura di riferimento (KDIGO, preprocessing, modelli)
├── src/
│   ├── config.py              caricamento della configurazione
│   ├── data/
│   │   ├── load.py            caricamento e soggetti eleggibili
│   │   ├── kidney.py          eGFR CKD-EPI 2021 e livelli KDIGO
│   │   ├── split.py           split stratificato e controlli di bilanciamento
│   │   ├── preprocess.py      selezione feature, codifiche, preprocessor
│   │   ├── imputation.py      confronto dei metodi di imputazione
│   │   ├── folds.py           fold della cross-validation annidata (esterni e interni)
│   │   ├── imputed.py         fold imputati salvati una volta sola
│   │   └── augmented.py       Fase B: tecniche di bilanciamento, training bilanciati per fold
│   ├── models/
│   │   ├── zoo.py             i cinque modelli e gli spazi di ricerca
│   │   ├── phase_a.py         addestramento della Fase A (CV annidata, Optuna)
│   │   ├── evaluation.py      valutazione sulle previsioni out-of-fold (domande 1-4, confronti)
│   │   ├── interpretation.py  odds ratio delle logistiche, SHAP degli alberi
│   │   ├── phase_b.py         addestramento della Fase B (tecniche, avvio caldo, ricalibrazione di Platt)
│   │   ├── evaluation_b.py    valutazione della Fase B (casi gravi, McNemar, calibrazione)
│   │   ├── phase_c.py         Fase C: sottogruppo diabetico a soglia e fasce fisse
│   │   ├── phase_d.py         Fase D: candidati, diagnostiche, regola di decisione
│   │   ├── clinical_utility.py qualità e utilità clinica
│   │   └── final_test.py      conferma finale sul test set (protetta da final_test.authorized)
│   └── analytics/             generazione delle figure
├── tests/                     test automatici (pytest)
├── Scope.md                   perimetro e domande della tesi
├── valorizzazione_tesi.md     posizionamento rispetto allo stato dell'arte e sintesi dei risultati
└── Notepad.md                 registro delle decisioni metodologiche
```

---

## ⚠️ Limiti dichiarati

- **una sola misurazione**: KDIGO richiede alterazioni persistenti per oltre 3 mesi, quindi si parla di **marcatori**, non di diagnosi. Un danno acuto non è distinguibile da uno cronico
- **confronto con KDIGO non indipendente**: target e livelli usano gli stessi marcatori. La domanda è "senza esami renali il modello ordina i soggetti come KDIGO?", non "il modello batte KDIGO"
- **livelli gravi poco numerosi** (67 "alto", 27 "molto alto"): stime per livello instabili
- **sottogruppo diabetico piccolo** (91 positivi): risultati solo descrittivi
- **dati trasversali**: il modello riconosce lo stato presente, non prevede la progressione
- **nessuna validazione esterna**: una sola popolazione, una sola finestra temporale
- **coorte ospedaliera, non screening**: gli autori del dataset descrivono dati di un reparto di diabetologia; tipo di campione urinario per l'ACR e unità della creatinina e dell'albumina urinarie non documentati. L'utilità del modello va dimostrata in popolazioni di screening
- **possibile struttura per comunità**: la prevalenza varia dal 4,5% al 24,5% fra giornate di raccolta, effetto non modellato

> **Avvertenza.** K-Risk è un progetto di ricerca accademica. Non è un dispositivo medico, non è validato clinicamente e non deve essere usato per decisioni diagnostiche o terapeutiche.

---

## 📖 Citazione e licenza

Se utilizzi questo lavoro, cita sia il repository sia il dataset originale. I metadati sono in [`CITATION.cff`](CITATION.cff) (GitHub mostra il pulsante *Cite this repository*).

**Dataset**
> Li J. et al. *A bimodal dataset for diabetes research.* Scientific Data **13** (2026). [doi:10.1038/s41597-026-06923-y](https://doi.org/10.1038/s41597-026-06923-y) · Archivio Zenodo: [10.5281/zenodo.18270337](https://doi.org/10.5281/zenodo.18270337)

**Riferimenti clinici principali**
- KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. Kidney Int 105(4S):S117–S314 (2024). [doi:10.1016/j.kint.2023.10.018](https://doi.org/10.1016/j.kint.2023.10.018)
- KDIGO 2026, linea guida su diabete e malattia renale cronica
- Inker L.A. et al. *New creatinine- and cystatin C–based equations to estimate GFR without race.* N Engl J Med 385(19):1737–1749 (2021). [doi:10.1056/NEJMoa2102953](https://doi.org/10.1056/NEJMoa2102953) — equazione CKD-EPI 2021

**Licenze**
- codice: [MIT](LICENSE) © 2026 Francesco Coppola
- dati: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), di proprietà degli autori originali

<div align="center">

**Autore:** Francesco Coppola · [ORCID 0009-0009-0758-8226](https://orcid.org/0009-0009-0758-8226)

</div>
