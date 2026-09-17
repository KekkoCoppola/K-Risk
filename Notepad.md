# Notepad

## Target
- DN > 0 vs DN = 0 (righe con DN mancante escluse)
- 5802 soggetti, 456 positivi (7,86%)

> **Figura** `analytics/dataset/01_target_stages.png`
> quanti soggetti per ogni stadio DN (0, 3, 4, 5, mancante). Scala logaritmica, altrimenti gli stadi 4 e 5 non si vedrebbero.

> **Figura** `analytics/dataset/02_target_by_diabetes.png`
> sinistra: prevalenza DN > 0 nei non diabetici (6,9%) e nei diabetici (22,8%), con la linea della coorte intera (7,86%).
> destra: da chi sono composti i 456 positivi → 376 non diabetici (82,5%), 80 diabetici (17,5%).
> → spiega perché il target è "danno renale" e non "nefropatia diabetica".

> **Figura** `analytics/dataset/05_age_by_target.png`
> distribuzione dell'età in DN = 0 (blu) e DN > 0 (arancione). I positivi sono spostati verso età più alte.

## Split train / test

### Scelta finale
- holdout 75/25, stratificato su DN binario (0 / >0) x DM (0 / 1), seed 42
- 120 righe con DN mancante escluse prima dello split (nessun target -> non valutabili)
- script: `python -m src.data.split` -> `data/processed/train.csv`, `test.csv`
- riproducibile: stesso seed -> file identici (verificato con hash su più esecuzioni)

| set   | n    | DN+ | DN+ % | DM  | DM & DN+ |
|-------|------|-----|-------|-----|----------|
| train | 4351 | 342 | 7,86  | 263 | 60       |
| test  | 1451 | 114 | 7,86  | 88  | 20       |

> **Figura** `analytics/split/01_prevalence_train_test.png`
> la tabella qui sopra in forma di grafico: % di DN > 0, diabetici e diabetici con DN > 0, train (blu) vs test (arancione). Barre praticamente identiche = split riuscito.

### Perché uno split prima di tutto
- il test deve rappresentare la popolazione reale di screening (prevalenza 7,86%)
- ogni trasformazione che "impara" dai dati (imputazione, scaling, bilanciamento, generatori sintetici) va stimata solo sul training
- bilanciare prima dello split -> test con prevalenza artificiale (es. 50%) -> precision, PPV, PR-AUC e calibrazione gonfiate, non trasferibili alla realtà
- oversampling / SMOTE / CTGAN prima dello split -> pazienti duplicati o "copie interpolate" dello stesso soggetto sia in train sia in test -> leakage
- test unico e congelato = stesso metro per tutti i modelli e tutte le tecniche -> le differenze dipendono solo da modello/tecnica

### Perché 75/25 (criterio: numero di eventi nel test, non la percentuale)
- sotto ~100 eventi le stime di performance sul test sono instabili; consigliati ≥100, meglio 200 (Collins, Ogundimu & Altman 2016)
- confronto proporzioni (stesso seed e stratificazione):

| split | test n | DN+ test | DN+ train | DM & DN+ test |
|-------|--------|----------|-----------|---------------|
| 80/20 | 1161   | 91       | 365       | 16            |
| 75/25 | 1451   | 114      | 342       | 20            |
| 70/30 | 1741   | 137      | 319       | 24            |

- 80/20 -> 91 eventi, sotto la soglia di 100
- 70/30 -> più eventi nel test ma toglie 23 positivi al training, già scarso di positivi
- 75/25 -> test più piccolo che supera la soglia, massimo training possibile
- con 456 positivi totali la soglia di 200 eventi non è raggiungibile senza svuotare il training

> **Figura** `analytics/split/04_test_size_events.png`
> casi positivi nel training (blu) e nel test (arancione) con 80/20, 75/25 e 70/30. Linea tratteggiata = minimo consigliato di 100 eventi nel test.
> → con 80/20 il test (91) sta sotto la linea, con 75/25 (114) sta sopra.

### Perché stratificato
- simulazione 1000 split casuali 75/25 NON stratificati (seed 0–999):
  - prevalenza DN+ nel test da 5,93% a 9,58% (95% centrale: 6,69%–8,96%)
  - positivi nel test da 86 a 139 (95% centrale: 97–130) -> in una parte dei casi sotto 100 eventi
- con stratificazione: prevalenza identica in train e test (7,86%) e numero di eventi fisso (114)

> **Figura** `analytics/split/03_stratification_simulation.png` — **pannello sinistro**
> istogramma grigio = quanti positivi finiscono nel test in 1000 split casuali non stratificati.
> linea arancione tratteggiata = soglia di 100 eventi; linea blu = il nostro split stratificato (114).
> → il 4,8% degli split casuali scende sotto la soglia: senza stratificazione il risultato dipende dalla fortuna.

### Perché stratificare anche su DM
- DM non è il target, ma è la variabile clinica centrale della tesi (nefropatia *diabetica*)
- DM è fortemente associato al target: prevalenza DN > 0 22,8% nei diabetici vs 6,9% nei non diabetici (~3 volte)
- i diabetici sono pochi (351 su 5802, 6%) -> uno split casuale può sovra- o sotto-rappresentarli nel test
- stratificando anche su DM, train e test hanno la stessa composizione di diabetici e di diabetici positivi -> il test resta rappresentativo proprio sulla variabile più importante
- costo nullo: non riduce i dati e non cambia la prevalenza del target
- simulazione 1000 split stratificati solo su DN: diabetici positivi nel test da 10 a 30 (95% centrale: 13–27)
- stratificando su DN x DM: sempre 20 nel test / 60 nel train -> diabetici positivi con stesse proporzioni in entrambi gli insiemi
- tutte le 4 celle hanno numerosità sufficiente (la più piccola: DM=1 & DN+ = 80)
- NON stratificato sugli stadi DN (3/4/5): stadio 4 = 20 soggetti, stadio 5 = 2 -> celle troppo piccole, stratificazione senza senso

> **Figura** `analytics/split/03_stratification_simulation.png` — **pannello destro**
> istogramma grigio = quanti diabetici positivi finiscono nel test in 1000 split stratificati solo su DN (da 10 a 30).
> linea blu = il nostro split stratificato su DN x DM (sempre 20).
> → stratificare solo sul target non basta a garantire la stessa composizione di diabetici in train e test.

### Alternative scartate
- split temporale: tecnicamente possibile (colonna `Data`: 39 giornate di raccolta, dal 10/02/2012 al 07/04/2012), ma poco significativo -> finestra di ~2 mesi, nessuna evoluzione temporale reale da validare
- split geografico / per centro (internal-external validation): nessuna variabile di centro esplicita
  - però la prevalenza cambia molto da una giornata all'altra: DN > 0 da 1,9% a 15,5%, diabetici da 0% a 19,4%
  - ipotesi (non documentata nel data dictionary): ogni giornata = comunità / luogo di screening diverso
  - nello split casuale tutte le 39 giornate compaiono sia in train sia in test -> validazione interna, non internal-external
  - possibile analisi di sensibilità futura: split per giornate intere (alcune date solo nel test)
- split non stratificato: prevalenza e numero di eventi del test dipendono dal caso (vedi simulazione)
- bilanciamento (undersampling) sull'intero dataset prima dello split: test non realistico, perdita di ~4900 negativi reali, probabilità distorte
- solo cross-validation / bootstrap senza test: più efficiente sui dati (Steyerberg et al. 2001; Steyerberg & Harrell 2016), ma serve un test fisso per confrontare le tecniche di bilanciamento sullo stesso metro

### Verifica dell'equilibrio: differenze medie standardizzate (SMD)
- formula: SMD = |media_train − media_test| / sqrt((var_train + var_test) / 2)
- indipendente dalla numerosità (a differenza dei p-value, che con migliaia di soggetti diventano significativi anche per differenze irrilevanti)
- soglia convenzionale: SMD < 0,1 = differenza trascurabile (Austin 2009)
- risultati:

| variabile | significato                     | SMD   |
|-----------|---------------------------------|-------|
| DN (0/>0) | target (stratificato)           | 0,000 |
| DM        | diabete (stratificato)          | 0,001 |
| ALT       | transaminasi                    | 0,004 |
| CHOL      | colesterolo totale              | 0,008 |
| HDL       | colesterolo HDL                 | 0,010 |
| Age       | età                             | 0,014 |
| LDL       | colesterolo LDL                 | 0,017 |
| Bpdia     | pressione diastolica            | 0,025 |
| Bpsys     | pressione sistolica             | 0,026 |
| SUA       | acido urico                     | 0,029 |
| Gender    | sesso                           | 0,031 |
| BMI       | indice di massa corporea        | 0,031 |
| HbA1c     | emoglobina glicata              | 0,033 |
| FPG       | glicemia a digiuno              | 0,042 |
| TG        | trigliceridi                    | 0,045 |
| HGB       | emoglobina                      | 0,052 |
| waist1    | circonferenza vita              | 0,093 |

- tutte sotto 0,1 -> train e test sono campioni equivalenti della stessa popolazione
- 16 su 17 sotto 0,06; unica vicina alla soglia: waist1 (0,093) -> accettabile, da citare
- DN e DM ~0 per costruzione (variabili di stratificazione)
- le 15 variabili cliniche non sono stratificate ma risultano bilanciate comunque -> lo split non ha introdotto distorsioni su caratteristiche demografiche, antropometriche, pressorie, glicemiche, lipidiche ed ematologiche

> **Figura** `analytics/split/02_smd_balance.png`
> SMD train vs test estesa a 17 variabili (DN, DM + 15 cliniche: età, sesso, BMI, circonferenza vita, pressione, glicemia, HbA1c, lipidi, acido urico, ALT, emoglobina).
> ogni pallino = una variabile; linea arancione tratteggiata = soglia 0,1.
> → tutti i pallini a sinistra della soglia. Il più vicino è waist1 (0,093): accettabile, ma da citare.

### Limiti da dichiarare
- 114 eventi nel test: sopra la soglia minima ma lontani da 200 -> intervalli di confidenza delle metriche non stretti
- un singolo split dipende dal seed -> valutare sensibilità ripetendo con seed diversi
- diabetici con DN > 0 nel test: solo 20 -> eventuali metriche calcolate sul solo sottogruppo diabetico sarebbero molto instabili (al massimo descrittive)
- soggetti raggruppati per giornata di raccolta (prevalenza variabile fra giornate) -> lo split casuale non valuta la generalizzazione a nuove comunità
- validazione solo interna: stessa popolazione, stesso periodo, nessuna validazione esterna

### Riferimenti (da verificare prima di citare)
- Collins GS, Ogundimu EO, Altman DG. Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. Stat Med 2016
- Riley RD et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. Stat Med 2021
- Steyerberg EW et al. Internal validation of predictive models: efficiency of some procedures for logistic regression analysis. J Clin Epidemiol 2001
- Steyerberg EW, Harrell FE. Prediction models need appropriate internal, internal-external, and external validation. J Clin Epidemiol 2016
- Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. Stat Med 2009
- Collins GS et al. TRIPOD+AI statement. BMJ 2024

## Bilanciamento (solo sul training, mai sul test)
- [ ] nessuna correzione (riferimento)
- [ ] undersampling
- [ ] oversampling
- [ ] SMOTE (SMOTE-NC per variabili categoriche)
- [ ] CTGAN
- [ ] ...

## Qualità dei dati (da usare nel preprocessing)

> **Figura** `analytics/dataset/03_missing_values.png`
> le 25 variabili con più valori mancanti, in %. Alcune sono quasi vuote (CDSMS 100%, EscDrinkO 99,8%, GLU 92,1%).
> → candidate a essere escluse o gestite con attenzione.

> **Figura** `analytics/dataset/04_unknown_code_9.png`
> quante volte compare il codice 9 in ogni variabile categorica.
> → DietExcise, Compl, Known, ToChilren, GlucoseLev, OPD, Check: oltre 3000 occorrenze → probabilmente domande fatte solo ai diabetici (9 = "non applicabile", non "sconosciuto"). Da verificare sul data dictionary.
> → PAD, Arrhythmia, CAP, RF, TIA, Angina, Blind: circa 900–1100 occorrenze, come riportato nella proposta.

## Indice delle figure

Rigenerare con `python -m src.analytics.dataset_overview` e `python -m src.analytics.split_report`.

| file | cosa mostra | sezione |
|------|-------------|---------|
| `analytics/dataset/01_target_stages.png` | soggetti per stadio DN | Target |
| `analytics/dataset/02_target_by_diabetes.png` | prevalenza DN in diabetici / non diabetici e composizione dei positivi | Target |
| `analytics/dataset/03_missing_values.png` | variabili con più valori mancanti | Qualità dei dati |
| `analytics/dataset/04_unknown_code_9.png` | occorrenze del codice 9 | Qualità dei dati |
| `analytics/dataset/05_age_by_target.png` | età in DN = 0 vs DN > 0 | Target |
| `analytics/split/01_prevalence_train_test.png` | proporzioni in train e test | Split → Scelta finale |
| `analytics/split/02_smd_balance.png` | SMD train vs test con soglia 0,1 | Split → SMD |
| `analytics/split/03_stratification_simulation.png` | 1000 split: non stratificati (sx) e stratificati solo su DN (dx) | Split → Perché stratificato / Perché anche DM |
| `analytics/split/04_test_size_events.png` | positivi in train e test con 80/20, 75/25, 70/30 | Split → Perché 75/25 |

## Da fare
- modelli da scegliere
- stesso test set per tutti i confronti
