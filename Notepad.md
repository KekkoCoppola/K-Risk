# Notepad

## Disegno della tesi (decisione del 17/09/2026)

### Idea in una frase
Allenare modelli di classificazione **senza esami renali** e verificare quanto le loro stime concordano con la classificazione clinica **KDIGO**, prima sui dati originali e poi con diverse tecniche di data augmentation.

### KDIGO (lato dati)
- KDIGO non è una malattia: è l'organizzazione che definisce come identificare e classificare la **malattia renale cronica (CKD)**
- usa due variabili, entrambe presenti nel dataset:

| variabile KDIGO | significato | colonna |
|-----------------|-------------|---------|
| eGFR | capacità di filtrazione dei reni (più basso = peggio) | **ricalcolata** con CKD-EPI 2021 da `SCRE`, `Age`, `Gender` (la colonna `GFR` del dataset non corrisponde a nessuna formula standard, vedi "Punti da verificare") |
| ACR | albumina nelle urine rispetto alla creatinina (più alto = peggio) | `UMAUCR` (mg/g, verificato) |

- regola: marcatori di CKD presenti se **eGFR < 60 oppure ACR ≥ 30**
- classificazione: eGFR in 6 categorie (G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29, G5 <15), ACR in 3 (A1 <30, A2 30–300, A3 >300) → incrocio = 4 livelli di rischio

| livello KDIGO | celle G × A | intera coorte | diabetici |
|---------------|-------------|---------------|-----------|
| basso | G1–G2 con A1 | 5234 | 260 |
| moderato | G1–G2 con A2; G3a con A1 | 473 | 70 |
| alto | G1–G2 con A3; G3a con A2; G3b con A1 | 67 | 15 |
| molto alto | G3a con A3; G3b con A2–A3; G4–G5 | 27 | 6 |
| **totale** | | **5801** | **351** |

- 121 soggetti senza `SCRE` o `UMAUCR` → esclusi (target non calcolabile)
- con la colonna `GFR` originale i numeri sarebbero leggermente diversi (5789 soggetti, 586 positivi): la scelta della formula cambia il target, va dichiarata

### Perché serve il machine learning
- con eGFR e ACR in input il modello copierebbe la regola KDIGO → nessun valore, leakage puro
- il modello ha senso solo **senza esami renali**: "dai normali dati di screening posso stimare chi ha un profilo KDIGO a rischio?"
- utilità: indicare a chi fare eGFR e ACR quando non è possibile farli a tutti

### Target
- **classificazione binaria supervisionata** (non regressione)
- `y = 1` se `UMAUCR >= 30` oppure `eGFR (CKD-EPI 2021) < 60` (KDIGO moderato o superiore), altrimenti `y = 0`
- 567 positivi su 5801 (9,77%); fra i diabetici 91 su 351 (25,9%)
- il target si calcola da `UMAUCR` e dall'eGFR ricalcolato, che poi vengono **rimossi dall'input** (insieme a `SCRE` e `GFR`)
- nei diabetici il target coincide con la definizione KDIGO di "CKD nel diabete" / DKD (vedi sezione linee guida)
- nota: `Age` e `Gender` entrano nella formula dell'eGFR ma restano feature legittime (sono dati anagrafici, non esami renali); il peso dell'età nel modello va interpretato tenendone conto

Perché non regressione (predire eGFR o ACR come numero):
- ACR fortemente asimmetrica (quasi tutti vicino a 0, pochi valori altissimi) → difficile da predire, errori vicino alla soglia 30 cambiano la classe
- le tecniche di augmentation da confrontare (undersampling, oversampling, SMOTE, CTGAN) sono pensate per la classificazione sbilanciata
- KDIGO è già una classificazione → confronto naturale
- nota: la regressione logistica è un modello di **classificazione** nonostante il nome

Perché non target a 4 classi (livelli KDIGO):
- "molto alto" = 27 soggetti (~20 nel training) → augmentation e modello multiclasse troppo fragili
- soluzione: il modello impara il binario, la **valutazione** usa i 4 livelli

### Input e output
- **input**: un vettore di feature per soggetto, solo dati di screening
  - anagrafica (`Age`, `Gender`), antropometria (`BMI`, `waist1`, `WHR`), pressione (`Bpsys`, `Bpdia`), glicemia (`FPG`, `HbA1c`, `DM`), lipidi (`CHOL`, `TG`, `HDL`, `LDL`), altri esami del sangue (`SUA`, `ALT`, `AST`, `HGB`…), anamnesi e stili di vita
- **esclusi (leakage)**: eGFR ricalcolato, `GFR`, `UMAUCR`, `UmALB`, `HighUmALB`, `SCRE`, `highCr`, `BUN`, `UCRE`, `HighACR`, `ACR3degree`, `GFR5lev`, `GFR6levG5`, `DN`, `EUGFRabACR012`, `EUGFR90UACR01`, `EUGFR60abACR012`, `UACRGFR`, `RF`
- **esclusi (non feature)**: `NO` (identificativo), `Data` (data di raccolta), `filter`, `GAnum`, `GAstudy`
- **output**:
  1. probabilità `p ∈ [0, 1]` di KDIGO moderato o superiore
  2. classe 0/1 (soglia scelta sul training)
  3. fascia di rischio (bassa / media / alta)

Esempio:
```
INPUT  → Age=62, Gender=1, BMI=28.4, Bpsys=152, FPG=7.8, HbA1c=7.1, DM=1, TG=2.3, SUA=410, ...
OUTPUT → p = 0.41 → classe 1 → fascia alta
KDIGO reale (da GFR e UMAUCR, nascosto al modello) → G2 / A2 → rischio moderato
```

### Domande della tesi
**Fase A — modelli sui dati originali (senza augmentation)**
1. il modello distingue chi ha marcatori di malattia renale? → AUC, PR-AUC, precision, recall
2. il rischio stimato cresce con la gravità KDIGO? → probabilità per livello (basso → molto alto), test di tendenza (Jonckheere-Terpstra)
3. quanti soggetti "alto" e "molto alto" riconosce? → sensibilità per livello KDIGO; **mancare un "molto alto" è l'errore più grave**
4. le fasce del modello corrispondono alla stratificazione clinica? → tabella fasce vs livelli KDIGO, kappa pesato

**Fase B — stesse 4 domande per ogni tecnica di augmentation**

**Fase C — conclusioni**
5. l'augmentation migliora il riconoscimento dei casi gravi o solo la metrica media?
6. stesso modello sulla CKD nel diabete / DKD (sottogruppo diabetici, 91 casi): risultati descrittivi

### Regole metodologiche
- "dati originali" = senza augmentation, ma **dopo** il preprocessing (esclusione leakage, codice 9, valori mancanti)
- split 75/25 stratificato su **livello KDIGO (4) × DM (2)** → 8 strati, il più piccolo = 6 (diabetici molto alto)
  - test atteso: ~142 positivi (sopra i 100), ~118 moderato, ~17 alto, ~7 molto alto, ~23 diabetici positivi
- augmentation **solo sul training** e **dentro ogni fold** della cross-validation
- confronto con KDIGO **solo su soggetti reali**: i sintetici non hanno un livello KDIGO vero
- **ricalibrare** le probabilità dopo l'augmentation (Platt o isotonica su fold di dati reali): il bilanciamento gonfia le probabilità
  - riportare anche metriche basate sull'ordinamento (AUC, test di tendenza), valide anche senza calibrazione
- augmentation e gravità: confrontare augmentation sul solo binario vs augmentation che mantiene le proporzioni dei livelli KDIGO fra i positivi (CTGAN condizionato)
- valutazione su due livelli, perché nel test i "molto alto" sono pochi (~7):
  1. **previsioni out-of-fold** sul training (ogni soggetto reale predetto dal modello del fold in cui non è stato usato) → analisi KDIGO principale (~20 molto alto, ~50 alto)
  2. **test set** → conferma finale

### Limiti da dichiarare
- una sola misurazione per soggetto: KDIGO richiede alterazioni persistenti > 3 mesi → parlare di "marcatori di malattia renale cronica", non di diagnosi di CKD
- confronto con KDIGO non indipendente: target e livelli KDIGO usano gli stessi marcatori → la domanda è "senza esami renali il modello ordina i soggetti come KDIGO?", non "il modello batte KDIGO"
- livelli gravi poco numerosi (alto 67, molto alto 27) → stime per livello instabili
- sottogruppo diabetico piccolo (91 positivi) → solo descrittivo
- eGFR ricalcolato da un'unica creatinina, senza cistatina C → possibili errori di classificazione vicino alla soglia 60
- nessuna validazione esterna: le linee guida raccomandano modelli di rischio validati esternamente sulla popolazione di destinazione

### Cosa dicono le linee guida KDIGO (letture in `papers/`)
Documenti letti:
- *KDIGO 2026 Clinical Practice Guideline for Diabetes Management in CKD* — **bozza di revisione pubblica, marzo 2026** (aggiornamento della linea guida 2022)
- *KDIGO 2026 Clinical Practice Guideline for AKI and AKD* — **bozza di revisione pubblica, marzo 2026**
- *KDIGO 2026 Clinical Practice Guideline for Anemia in CKD* — Kidney International 109 (Suppl 1S), S1–S99, gennaio 2026
- le bozze possono cambiare nella versione finale → citarle come "public review draft"

Conseguenze per target e variabili:

| cosa dice la linea guida | conseguenza per la tesi |
|--------------------------|-------------------------|
| CKD = anomalie di struttura o funzione renale **persistenti ≥ 3 mesi**; diagnosi basata su eGFR e UACR (diabete e CKD, cap. 1.1 e 1.4) | target coerente (eGFR < 60 o ACR ≥ 30), ma con una sola misurazione → "marcatori di CKD", non diagnosi |
| stadiazione con **entrambi** eGFR e UACR sulla heatmap KDIGO (Practice Point 1.4.1, Figura 1) | confermati i 4 livelli di rischio usati per la valutazione |
| albuminuria da confermare con **2 valori anomali su 3 campioni in 3–6 mesi** (variabilità giornaliera) | limite da dichiarare: un singolo ACR può sovrastimare i positivi |
| eGFR da **equazione validata**, non dalla sola creatinina | ricalcolare eGFR con CKD-EPI 2021 (la colonna `GFR` non corrisponde a formule standard) |
| "CKD nel diabete", "diabete e CKD" e "DKD" sono sinonimi (Practice Point 1.1.1), senza presupporre la causa | usare questa terminologia invece di "nefropatia diabetica" per il sottogruppo |
| nelle persone con diabete: **testare tutti** con UACR + eGFR, alla diagnosi di diabete di tipo 2 e poi ogni anno (Practice Point 1.3.1–1.3.4) | nei diabetici il modello **non sostituisce** gli esami: l'utilità pratica è la prioritizzazione nella popolazione generale di screening |
| distinzione fra *case-finding* (test su popolazione ad alto rischio, es. diabetici) e *screening* (popolazione generale non selezionata) | il dataset è uno screening di popolazione → posizionare il modello lì |
| HbA1c poco affidabile nella CKD avanzata (Figura 7) | `HbA1c` resta feature, ma nei casi gravi va interpretata con cautela |
| anemia: Hb < 13 g/dl nei maschi, < 12 g/dl nelle femmine; prevalenza crescente al calare dell'eGFR, > 50% in G4–G5 (linea guida anemia) | `HGB` è in parte **conseguenza** della malattia renale: non è leakage di definizione, ma nell'interpretazione (SHAP) va letta come effetto, non come causa; possibile analisi di sensibilità senza `HGB` |
| AKD = alterazioni (es. eGFR < 60) con durata **≤ 3 mesi** (linea guida AKI/AKD, Practice Point 1.3.1) | con una sola misurazione non si distingue un danno acuto (AKD) da uno cronico (CKD) → rafforza la formulazione "marcatori" |
| per il rischio si raccomandano modelli **validati esternamente** sulla popolazione di destinazione | assenza di validazione esterna = limite principale del prototipo |

## Storico: il primo target (`DN`) e perché è stato abbandonato
- prima impostazione: `DN > 0` vs `DN = 0` su 5802 soggetti, 456 positivi (7,86%), split stratificato su DN × DM
- verifica sui dati: `DN > 0` coincide con `UmALB >= 30` (455 casi su 456) → `DN` è una soglia sulla sola **concentrazione** di albumina urinaria, applicata a tutta la coorte
- problemi:
  - non è la definizione KDIGO: usa la concentrazione invece dell'ACR e ignora il filtrato glomerulare
  - non è nefropatia diabetica: l'82,5% dei positivi non era diabetico
  - rende impossibile un confronto pulito con KDIGO, perché misura una cosa diversa
- il **metodo** dello split resta valido e viene riusato: cambia la definizione del target

## Split train / test

### Scelta finale
- holdout 75/25, stratificato su **livello KDIGO (4 valori) × DM (0/1)** = 8 strati, seed 42
- popolazione: 5801 soggetti con `SCRE` e `UMAUCR` presenti (121 righe escluse: target non calcolabile)
- script: `python -m src.data.split` -> `data/processed/train.csv`, `test.csv`
- riproducibile: stesso seed -> stessi file (verificato rieseguendo lo script)

| set   | n    | positivi | positivi % | DM  | DM positivi |
|-------|------|----------|------------|-----|-------------|
| train | 4350 | 425      | 9,77       | 263 | 68          |
| test  | 1451 | 142      | 9,79       | 88  | 23          |

| livello KDIGO | train | test | % nel test |
|---------------|-------|------|------------|
| basso | 3925 | 1309 | 25,0 |
| moderato | 354 | 119 | 25,2 |
| alto | 50 | 17 | 25,4 |
| molto alto | 21 | 6 | 22,2 |

> **Figura** `analytics/split/01_prevalence_train_test.png`
> la prima tabella in forma di grafico: % di positivi, diabetici e diabetici positivi, train (blu) vs test (arancione). Barre praticamente identiche = split riuscito.

> **Figura** `analytics/split/03_kdigo_levels_train_test.png`
> quanta parte di ogni livello KDIGO è finita nel test. Linea tratteggiata = quota attesa del 25%.
> → tutti i livelli fra 22,2% e 25,4%: anche i casi gravi sono ripartiti correttamente.

### Perché lo split è stato rifatto rispetto alla prima versione
Tre motivi, in ordine di gravità.

**1. I casi gravi erano ripartiti male.** Con lo split precedente (stratificato su `DN`, che dipende solo dall'albuminuria), i casi KDIGO "molto alto" — che dipendono soprattutto dal filtrato glomerulare — finivano dove capitava:

| livello KDIGO | vecchio split: train | vecchio split: test | % nel test |
|---------------|----------------------|---------------------|------------|
| basso | 3921 | 1302 | 25% |
| moderato | 356 | 116 | 25% |
| alto | 50 | 17 | 25% |
| **molto alto** | **12** | **15** | **56%** |

→ al training restavano 12 casi gravi su 27: pochissimi da cui imparare e su cui generare dati sintetici, e metriche del test basate su una ripartizione fortunata. Con la nuova stratificazione: 21 nel training e 6 nel test (22,2%).

**2. Il target è diverso.** Prima `DN > 0` (sola concentrazione di albumina), ora "ACR ≥ 30 o eGFR < 60" con eGFR ricalcolato CKD-EPI 2021: etichette diverse, non confrontabili.

**3. La popolazione è diversa.** Prima si escludevano le righe senza `DN`, ora quelle senza `SCRE` o `UMAUCR`: 12 soggetti idonei al nuovo target erano fuori dal vecchio split e 13 soggetti del vecchio split non sono idonei.

### Perché uno split prima di tutto
- il test deve rappresentare la popolazione reale di screening (prevalenza 9,77%)
- ogni trasformazione che "impara" dai dati (imputazione, scaling, bilanciamento, generatori sintetici) va stimata solo sul training
- bilanciare prima dello split -> test con prevalenza artificiale (es. 50%) -> precision, PPV, PR-AUC e calibrazione gonfiate, non trasferibili alla realtà
- il livello KDIGO reale di ogni soggetto del test serve per il confronto con la clinica: deve restare intatto
- oversampling / SMOTE / CTGAN prima dello split -> pazienti duplicati o "copie interpolate" dello stesso soggetto sia in train sia in test -> leakage
- test unico e congelato = stesso metro per tutti i modelli e tutte le tecniche -> le differenze dipendono solo da modello/tecnica

### Perché 75/25 (criterio: numero di eventi nel test, non la percentuale)
- sotto ~100 eventi le stime di performance sul test sono instabili; consigliati ≥100, meglio 200 (Collins, Ogundimu & Altman 2016)
- confronto proporzioni (stesso seed e stratificazione):

| split | test n | positivi test | positivi train | molto alto nel test | diabetici positivi nel test |
|-------|--------|---------------|----------------|---------------------|-----------------------------|
| 80/20 | 1161   | 113           | 454            | 5                   | 18                          |
| 75/25 | 1451   | 142           | 425            | 6                   | 23                          |
| 70/30 | 1741   | 170           | 397            | 8                   | 27                          |

- con il target KDIGO tutte e tre le proporzioni superano i 100 eventi: la scelta si sposta sull'equilibrio fra i due insiemi
- 80/20 -> solo 5 casi "molto alto" nel test e 18 diabetici positivi: troppo pochi per le analisi per livello
- 70/30 -> toglie 28 positivi al training, che serve per modelli e augmentation
- 75/25 -> compromesso: 142 positivi nel test (margine sopra i 100) e 425 nel training
- 75/25 mantiene inoltre continuità con la prima versione dello split: cambia il target, non il disegno

> **Figura** `analytics/split/05_test_size_events.png`
> positivi nel training (blu) e nel test (arancione) con 80/20, 75/25 e 70/30. Linea tratteggiata = minimo consigliato di 100 eventi nel test.

### Perché stratificato
- simulazione 1000 split casuali 75/25 NON stratificati (seed 0–999):
  - positivi nel test da 111 a 170 (95% centrale: 123–161) -> nessuno sotto i 100 eventi, ma oscillazione ampia
  - casi "molto alto" nel test da **0 a 15**: in alcuni split il livello più grave sparisce del tutto dal test
- con stratificazione: prevalenza identica in train e test e numeri fissi (142 positivi, 6 casi "molto alto")
- la stratificazione qui serve meno per la prevalenza complessiva (con 567 positivi è stabile) e molto di più per **garantire i livelli gravi in entrambi gli insiemi**

> **Figura** `analytics/split/04_stratification_simulation.png` — **pannello sinistro**
> istogramma grigio = quanti positivi finiscono nel test in 1000 split casuali non stratificati.
> linea arancione tratteggiata = soglia di 100 eventi; linea blu = il nostro split stratificato (142).

### Perché stratificare sul livello KDIGO e non solo sul target
- il target binario mette nello stesso gruppo un soggetto "moderato" (473 casi) e uno "molto alto" (27 casi): stratificare solo sul target non protegge i secondi
- simulazione 1000 split stratificati **solo sul target**: casi "molto alto" nel test da **1 a 14** (95% centrale: 3–11); in 64 split su 1000 ce ne sono al massimo 3
- stratificando sul livello KDIGO: sempre 6 nel test e 21 nel training
- è la condizione che rende possibili le domande della tesi: sensibilità per livello e andamento del rischio stimato lungo la gravità

> **Figura** `analytics/split/04_stratification_simulation.png` — **pannello destro**
> istogramma grigio = quanti casi "molto alto" finiscono nel test in 1000 split stratificati solo sul target (da 1 a 14).
> linea blu = il nostro split stratificato su livello KDIGO × DM (sempre 6).

### Perché stratificare anche su DM
- DM non è il target, ma è la variabile clinica centrale della tesi (il sottogruppo diabetico = CKD nel diabete)
- DM è fortemente associato al target: prevalenza 25,9% nei diabetici contro 8,7% nei non diabetici (~3 volte)
- i diabetici sono pochi (351 su 5801, 6%) -> uno split casuale può sovra- o sotto-rappresentarli nel test
- stratificando anche su DM: 68 diabetici positivi nel training e 23 nel test, proporzioni identiche
- costo nullo: non riduce i dati e non cambia la prevalenza del target
- le 8 celle hanno tutte numerosità sufficiente; la più piccola (diabetici "molto alto") ha 6 soggetti, il minimo per poter stratificare

### Alternative scartate
- split temporale: tecnicamente possibile (colonna `Data`: 39 giornate di raccolta, dal 10/02/2012 al 07/04/2012), ma poco significativo -> finestra di ~2 mesi, nessuna evoluzione temporale reale da validare
- split geografico / per centro (internal-external validation): nessuna variabile di centro esplicita
  - però la prevalenza cambia molto da una giornata all'altra: dal 4,5% al 24,5% sulle 39 giornate
  - ipotesi (non documentata nel data dictionary): ogni giornata = comunità / luogo di screening diverso
  - nello split casuale tutte le 39 giornate compaiono sia in train sia in test -> validazione interna, non internal-external
  - possibile analisi di sensibilità futura: split per giornate intere (alcune date solo nel test)
- split non stratificato: numero di eventi e soprattutto di casi gravi nel test dipendono dal caso (vedi simulazione)
- stratificazione sul solo target binario: non protegge i livelli gravi (vedi simulazione)
- bilanciamento (undersampling) sull'intero dataset prima dello split: test non realistico, perdita di ~5000 negativi reali, probabilità distorte
- solo cross-validation / bootstrap senza test: più efficiente sui dati (Steyerberg et al. 2001; Steyerberg & Harrell 2016), ma serve un test fisso per confrontare le tecniche di bilanciamento sullo stesso metro

### Verifica dell'equilibrio: differenze medie standardizzate (SMD)
- formula: SMD = |media_train − media_test| / sqrt((var_train + var_test) / 2)
- indipendente dalla numerosità (a differenza dei p-value, che con migliaia di soggetti diventano significativi anche per differenze irrilevanti)
- soglia convenzionale: SMD < 0,1 = differenza trascurabile (Austin 2009)
- risultati:

| variabile | significato                     | SMD   |
|-----------|---------------------------------|-------|
| target    | marcatori di CKD (stratificato) | 0,001 |
| DM        | diabete (stratificato)          | 0,001 |
| SUA       | acido urico                     | 0,001 |
| TG        | trigliceridi                    | 0,009 |
| ALT       | transaminasi                    | 0,017 |
| waist1    | circonferenza vita              | 0,023 |
| CHOL      | colesterolo totale              | 0,028 |
| FPG       | glicemia a digiuno              | 0,028 |
| BMI       | indice di massa corporea        | 0,034 |
| LDL       | colesterolo LDL                 | 0,034 |
| Age       | età                             | 0,038 |
| HGB       | emoglobina                      | 0,038 |
| HbA1c     | emoglobina glicata              | 0,039 |
| Bpdia     | pressione diastolica            | 0,040 |
| HDL       | colesterolo HDL                 | 0,045 |
| Gender    | sesso                           | 0,053 |
| Bpsys     | pressione sistolica             | 0,059 |

- tutte sotto 0,1 -> train e test sono campioni equivalenti della stessa popolazione
- massimo 0,059 (pressione sistolica): nessuna variabile vicina alla soglia (nella versione precedente waist1 arrivava a 0,093)
- target e DM ~0 per costruzione (variabili di stratificazione)
- le 15 variabili cliniche non sono stratificate ma risultano bilanciate comunque -> lo split non ha introdotto distorsioni su caratteristiche demografiche, antropometriche, pressorie, glicemiche, lipidiche ed ematologiche

> **Figura** `analytics/split/02_smd_balance.png`
> SMD train vs test su 17 variabili (target, DM + 15 cliniche: età, sesso, BMI, circonferenza vita, pressione, glicemia, HbA1c, lipidi, acido urico, ALT, emoglobina).
> ogni pallino = una variabile; linea arancione tratteggiata = soglia 0,1.
> → tutti i pallini ben a sinistra della soglia.

### Limiti da dichiarare
- 142 eventi nel test: sopra la soglia minima ma sotto i 200 -> intervalli di confidenza delle metriche non stretti
- **6 casi "molto alto" e 17 "alto" nel test**: le metriche per livello calcolate sul solo test sono indicative -> analisi principale sulle previsioni out-of-fold del training (21 e 50 casi)
- un singolo split dipende dal seed -> valutare sensibilità ripetendo con seed diversi
- diabetici positivi nel test: solo 23 -> metriche sul sottogruppo diabetico molto instabili (al massimo descrittive)
- soggetti raggruppati per giornata di raccolta (prevalenza variabile fra giornate) -> lo split casuale non valuta la generalizzazione a nuove comunità
- validazione solo interna: stessa popolazione, stesso periodo, nessuna validazione esterna

### Codice
- `src/data/kidney.py`: eGFR (CKD-EPI 2021), categorie G e A, livello KDIGO, target
- `src/data/load.py`: caricamento e selezione dei soggetti con target calcolabile
- `src/data/split.py`: strati (anche da variabili calcolate), split, salvataggio, riepiloghi e SMD
- `configs/config.yaml`: colonne, soglie del target, `stratify: [kdigo_level, DM]`, seed, test_size
- `tests/test_split.py`: numerosità, plausibilità dell'eGFR, conteggi per livello, riproducibilità, insiemi disgiunti, proporzioni degli strati, casi gravi e diabetici positivi nel test

### Riferimenti (da verificare prima di citare)
- Collins GS, Ogundimu EO, Altman DG. Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. Stat Med 2016
- Riley RD et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. Stat Med 2021
- Steyerberg EW et al. Internal validation of predictive models: efficiency of some procedures for logistic regression analysis. J Clin Epidemiol 2001
- Steyerberg EW, Harrell FE. Prediction models need appropriate internal, internal-external, and external validation. J Clin Epidemiol 2016
- Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. Stat Med 2009
- Collins GS et al. TRIPOD+AI statement. BMJ 2024

## Bilanciamento (solo sul training, mai sul test)
- [ ] nessuna correzione (riferimento, Fase A)
- [ ] undersampling
- [ ] oversampling
- [ ] SMOTE (SMOTE-NC per variabili categoriche)
- [ ] CTGAN (anche nella variante condizionata al livello KDIGO)
- [ ] ...

Per ogni tecnica: stesse 4 domande della Fase A, probabilità ricalibrate, confronto KDIGO solo su soggetti reali.

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
| `analytics/dataset/01_kdigo_heatmap.png` | soggetti in ogni cella della heatmap KDIGO (G × A), colorate per livello di rischio | Disegno della tesi → KDIGO |
| `analytics/dataset/02_target_by_diabetes.png` | prevalenza del target in diabetici / non diabetici e composizione dei positivi | Disegno della tesi → Target |
| `analytics/dataset/03_missing_values.png` | variabili con più valori mancanti | Qualità dei dati |
| `analytics/dataset/04_unknown_code_9.png` | occorrenze del codice 9 | Qualità dei dati |
| `analytics/dataset/05_age_by_target.png` | età dei positivi vs negativi | Disegno della tesi → Target |
| `analytics/dataset/06_egfr_comparison.png` | colonna `GFR` del dataset vs eGFR ricalcolato CKD-EPI 2021 | Punti da verificare → formula eGFR |
| `analytics/split/01_prevalence_train_test.png` | proporzioni in train e test | Split → Scelta finale |
| `analytics/split/02_smd_balance.png` | SMD train vs test con soglia 0,1 | Split → SMD |
| `analytics/split/03_kdigo_levels_train_test.png` | quota di ogni livello KDIGO finita nel test | Split → Scelta finale |
| `analytics/split/04_stratification_simulation.png` | 1000 split: non stratificati (sx) e stratificati solo sul target (dx) | Split → Perché stratificato / livello KDIGO |
| `analytics/split/05_test_size_events.png` | positivi in train e test con 80/20, 75/25, 70/30 | Split → Perché 75/25 |

## Punti da verificare
- [x] **formula dell'eGFR** — verificato: la colonna `GFR` **non coincide** con nessuna formula standard
  - provate: CKD-EPI 2009, CKD-EPI 2021, MDRD, MDRD cinese, Cockcroft-Gault → differenza mediana da 6 a 49 unità, 0% di corrispondenze esatte
  - valori poco plausibili: `GFR` massimo 562 (le equazioni per eGFR non superano ~150–160)
  - decisione: **ricalcolare eGFR con CKD-EPI 2021** (senza coefficiente etnico) da `SCRE` (µmol/L → mg/dl dividendo per 88,4), `Age`, `Gender`
  - effetto: eGFR < 60 passa da 119 a 91 soggetti; positivi da 586 a 567; 29 soggetti < 60 con `GFR` ma ≥ 60 con CKD-EPI, 1 nel verso opposto
- [x] **codifica di `Gender`** — dedotta dai dati (il dizionario non è leggibile): 1 = maschio, 2 = femmina (creatinina mediana 76,7 vs 54,9 µmol/L; emoglobina mediana 155 vs 135 g/L)
- [x] **unità di `UMAUCR`** — compatibile con mg/g: A2 (30–300) 459 soggetti, A3 (> 300) 57; in mg/mmol la soglia 30 corrisponderebbe a 300 mg/g e i positivi sarebbero implausibilmente molti
  - nota: `UMAUCR` = `UmALB` / `UCRE` × costante fissa (rapporto costante in tutto il dataset) → calcolata dagli autori in modo coerente; l'unità di `UCRE` non è documentata
- [ ] **codice 9**: per ogni variabile decidere se significa "sconosciuto" o "non applicabile" (DietExcise, Compl, Known… > 4000 occorrenze)
- [ ] **definizione di DM**: 109 soggetti con DM = 0 hanno HbA1c ≥ 6,5% → probabilmente il diabete è definito senza HbA1c (anamnesi, glicemia a digiuno, glicemia post-carico); da citare
- [ ] **dizionario vs dati**: DN stadio 4 dichiarato "con GFR < 10", ma nei dati GFR va da 12 a 175 → condizione non applicata
- [x] linee guida KDIGO 2026 lette (diabete e CKD, AKI/AKD, anemia) → conseguenze nella sezione "Cosa dicono le linee guida KDIGO"
- [ ] della linea guida KDIGO 2022 (diabete e CKD, Kidney International 102, suppl. 5S) in `papers/` ci sono solo le pagine introduttive → la bozza 2026 la aggiorna, basta citare quella (verificando se nel frattempo è uscita la versione finale)
- [ ] manca la linea guida KDIGO 2024 sulla **valutazione e gestione della CKD** (Kidney International 105, suppl. 4S): è il riferimento primario per definizione di CKD, equazioni eGFR (CKD-EPI 2021) e categorie G/A → procurarla e citarla

## Da fare
- [x] aggiornare config, split, test e figure al target KDIGO (stratificazione livello KDIGO × DM)
- [ ] mappa di esclusione leakage in config
- [ ] preprocessing (codice 9, valori mancanti, tipi di variabile) come pipeline stimata solo sul training
- [ ] modelli da scegliere (baseline, regressione logistica regolarizzata, Random Forest, Gradient Boosting)
- [ ] stesso test set per tutti i confronti
