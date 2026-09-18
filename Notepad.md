# Notepad

## Disegno della tesi (decisione del 17/09/2026)

### Idea in una frase
Allenare modelli di classificazione **senza esami renali** e verificare quanto le loro stime concordano con la classificazione clinica **KDIGO**, prima sui dati originali e poi con diverse tecniche di data augmentation.

### KDIGO (lato dati)
- KDIGO non è una malattia: è l'organizzazione che definisce come identificare e classificare la **malattia renale cronica (CKD)**
- usa due variabili, entrambe presenti nel dataset:

| variabile KDIGO | significato | colonna |
|-----------------|-------------|---------|
| eGFR | capacità di filtrazione dei reni (più basso = peggio) | **ricalcolata** con CKD-EPI 2021 (Inker et al. 2021) da `SCRE`, `Age`, `Gender` (la colonna `GFR` del dataset non corrisponde a nessuna formula standard, vedi "Punti da verificare") |
| ACR | albumina nelle urine rispetto alla creatinina (più alto = peggio) | `UMAUCR` (mg/g, verificato) |

- regola: marcatori di CKD presenti se **eGFR < 60 oppure ACR ≥ 30** (KDIGO 2024)
- classificazione: eGFR in 6 categorie (G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29, G5 <15), ACR in 3 (A1 <30, A2 30–300, A3 >300) → incrocio = 4 livelli di rischio (heatmap KDIGO: KDIGO 2024; KDIGO 2026a, Figura 1)

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
- `y = 1` se `UMAUCR >= 30` (ACR ≥ 30 mg/g) oppure `eGFR (CKD-EPI 2021) < 60 ml/min/1,73 m²` (KDIGO moderato o superiore), altrimenti `y = 0`
- 567 positivi su 5.801 soggetti eleggibili (9,77%); fra i diabetici 91 su 351 (25,9%)
- **composizione clinica dei positivi (insight fondamentale):**
  - **509 su 567 (89,8%)** sono positivi per la sola albuminuria (`UMAUCR >= 30`).
  - solo **58 su 567 (10,2%)** presentano `eGFR < 60` isolato senza albuminuria.
  - *conseguenza clinica per la tesi:* in uno screening di popolazione generale non selezionata, il modello intercetta primariamente il **danno renale precoce (perdita di selettività della barriera di filtrazione)** e non l'insufficienza renale avanzata/terminale. Questo spiega perché i predittori chiave attesi siano pressione arteriosa, età e compenso glicemico, e dà grande peso al limite metodologico di KDIGO sulla variabilità biologica di una singola misurazione di ACR (KDIGO 2024; KDIGO 2026a).
- il target si calcola da `UMAUCR` e dall'eGFR ricalcolato, che poi vengono **rimossi dall'input** (insieme a `SCRE` e `GFR`)
- nei diabetici il target coincide formalmente con la definizione KDIGO di "CKD nel diabete" / DKD (vedi sezione linee guida)
- nota: `Age` e `Gender` entrano nella formula dell'eGFR ma restano feature legittime di input (dati anagrafici primari, non esami specialistici renali); il peso dell'età andrà interpretato tenendone conto

Perché non regressione (predire eGFR o ACR come numero):
- ACR fortemente asimmetrica (quasi tutti vicino a 0, coda lunghissima con rari picchi enormi) → errori vicino alla soglia 30 cambiano la classe clinica
- le tecniche di augmentation (undersampling, oversampling, SMOTE, CTGAN) sono concepite per la classificazione sbilanciata (Chawla et al. 2002; Xu et al. 2019)
- KDIGO è già una classificazione e le decisioni cliniche di screening sono soglie d'azione (fare o non fare approfondimenti specialistici)
- nota: la regressione logistica è un modello di **classificazione** nonostante il nome

Perché non target a 4 classi (livelli KDIGO):
- "molto alto" = 27 soggetti in tutta la coorte (~21 nel training, 6 nel test) → augmentation e modello multiclasse statisticamente fragili e soggetti a forte overfitting
- soluzione metodologica: il modello impara il binario (0/1), la **valutazione clinica** usa i 4 livelli KDIGO (calibrazione per livello, sensibilità sui livelli gravi, test di tendenza)

### Input e output
- **input**: vettore di feature cliniche di screening di primo livello (demografia, antropometria, parametri vitali, assetto glicemico, lipidogramma, funzionalità epatica, emocromo completo, complicanze microvascolari non renali)
- **esclusione rigorosa (leakage, non-feature, collinearità, missing > 15%)**: vedi sezione dettagliata "Preprocessing e selezione delle feature"
- **output**:
  1. probabilità `p ∈ [0, 1]` di marcatori di malattia renale (KDIGO moderato o superiore)
  2. classe 0/1 (soglia ottimizzata sul training)
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
2. il rischio stimato cresce con la gravità KDIGO? → probabilità per livello (basso → molto alto), test di tendenza (Jonckheere-Terpstra: Terpstra 1952; Jonckheere 1954)
3. quanti soggetti "alto" e "molto alto" riconosce? → sensibilità per livello KDIGO; **mancare un "molto alto" è l'errore più grave**
4. le fasce del modello corrispondono alla stratificazione clinica? → tabella fasce vs livelli KDIGO, kappa pesato (Cohen 1968)

**Fase B — stesse 4 domande per ogni tecnica di augmentation**

**Fase C — conclusioni**
5. l'augmentation migliora il riconoscimento dei casi gravi o solo la metrica media?
6. stesso modello sulla CKD nel diabete / DKD (sottogruppo diabetici, 91 casi): risultati descrittivi

### Regole metodologiche
- "dati originali" = senza augmentation, ma **dopo** il preprocessing (esclusione leakage, codice 9, valori mancanti)
- split 75/25 stratificato su **livello KDIGO (4) × DM (2)** → 8 strati, il più piccolo = 6 (diabetici molto alto)
  - test atteso: ~142 positivi (sopra i 100), ~118 moderato, ~17 alto, ~7 molto alto, ~23 diabetici positivi
- augmentation **solo sul training** e **dentro ogni fold** della cross-validation (Santos et al. 2018)
- confronto con KDIGO **solo su soggetti reali**: i sintetici non hanno un livello KDIGO vero
- **ricalibrare** le probabilità dopo l'augmentation (Platt 1999 o isotonica, Zadrozny & Elkan 2002, su fold di dati reali): il bilanciamento gonfia le probabilità (van den Goorbergh et al. 2022)
  - riportare anche metriche basate sull'ordinamento (AUC, test di tendenza), valide anche senza calibrazione
- augmentation e gravità: confrontare augmentation sul solo binario vs augmentation che mantiene le proporzioni dei livelli KDIGO fra i positivi (CTGAN condizionato, Xu et al. 2019)
- valutazione su due livelli, perché nel test i "molto alto" sono pochi (~7):
  1. **previsioni out-of-fold** sul training (ogni soggetto reale predetto dal modello del fold in cui non è stato usato) → analisi KDIGO principale (~20 molto alto, ~50 alto)
  2. **test set** → conferma finale

### Limiti da dichiarare
- una sola misurazione per soggetto: KDIGO richiede alterazioni persistenti > 3 mesi → parlare di "marcatori di malattia renale cronica", non di diagnosi di CKD
- confronto con KDIGO non indipendente: target e livelli KDIGO usano gli stessi marcatori → la domanda è "senza esami renali il modello ordina i soggetti come KDIGO?", non "il modello batte KDIGO"
- livelli gravi poco numerosi (alto 67, molto alto 27) → stime per livello instabili
- sottogruppo diabetico piccolo (91 positivi) → solo descrittivo
- eGFR ricalcolato da un'unica creatinina, senza cistatina C → possibili errori di classificazione vicino alla soglia 60 (l'equazione combinata creatinina-cistatina C è più accurata: Inker et al. 2021)
- nessuna validazione esterna: le linee guida raccomandano modelli di rischio validati esternamente sulla popolazione di destinazione (KDIGO 2026a; Steyerberg & Harrell 2016; Collins et al. 2024)

### Cosa dicono le linee guida KDIGO (letture in `papers/`)
Documenti letti:
- *KDIGO 2026 Clinical Practice Guideline for Diabetes Management in CKD* — **bozza di revisione pubblica, marzo 2026** (aggiornamento della linea guida 2022) → citata come **KDIGO 2026a**
- *KDIGO 2026 Clinical Practice Guideline for AKI and AKD* — **bozza di revisione pubblica, marzo 2026** → **KDIGO 2026b**
- *KDIGO 2026 Clinical Practice Guideline for Anemia in CKD* — Kidney International 109 (Suppl 1S), S1–S99, gennaio 2026 → **KDIGO 2026c**
- le bozze possono cambiare nella versione finale → citarle come "public review draft"

Conseguenze per target e variabili:

| cosa dice la linea guida | conseguenza per la tesi |
|--------------------------|-------------------------|
| CKD = anomalie di struttura o funzione renale **persistenti ≥ 3 mesi**; diagnosi basata su eGFR e UACR (KDIGO 2026a, cap. 1.1 e 1.4) | target coerente (eGFR < 60 o ACR ≥ 30), ma con una sola misurazione → "marcatori di CKD", non diagnosi |
| stadiazione con **entrambi** eGFR e UACR sulla heatmap KDIGO (Practice Point 1.4.1, Figura 1) | confermati i 4 livelli di rischio usati per la valutazione |
| albuminuria da confermare con **2 valori anomali su 3 campioni in 3–6 mesi** (variabilità giornaliera) | limite da dichiarare: un singolo ACR può sovrastimare i positivi |
| eGFR da **equazione validata**, non dalla sola creatinina | ricalcolare eGFR con CKD-EPI 2021 (la colonna `GFR` non corrisponde a formule standard) |
| "CKD nel diabete", "diabete e CKD" e "DKD" sono sinonimi (Practice Point 1.1.1), senza presupporre la causa | usare questa terminologia invece di "nefropatia diabetica" per il sottogruppo |
| nelle persone con diabete: **testare tutti** con UACR + eGFR, alla diagnosi di diabete di tipo 2 e poi ogni anno (Practice Point 1.3.1–1.3.4) | nei diabetici il modello **non sostituisce** gli esami: l'utilità pratica è la prioritizzazione nella popolazione generale di screening |
| distinzione fra *case-finding* (test su popolazione ad alto rischio, es. diabetici) e *screening* (popolazione generale non selezionata) | il dataset è uno screening di popolazione → posizionare il modello lì |
| HbA1c poco affidabile nella CKD avanzata (Figura 7) | `HbA1c` resta feature, ma nei casi gravi va interpretata con cautela |
| anemia: Hb < 13 g/dl nei maschi, < 12 g/dl nelle femmine; prevalenza crescente al calare dell'eGFR, > 50% in G4–G5 (KDIGO 2026c) | `HGB` è in parte **conseguenza** della malattia renale: non è leakage di definizione, ma nell'interpretazione (SHAP) va letta come effetto, non come causa; possibile analisi di sensibilità senza `HGB` |
| AKD = alterazioni (es. eGFR < 60) con durata **≤ 3 mesi** (KDIGO 2026b, Practice Point 1.3.1) | con una sola misurazione non si distingue un danno acuto (AKD) da uno cronico (CKD) → rafforza la formulazione "marcatori" |
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
- ogni trasformazione che "impara" dai dati (imputazione, scaling, bilanciamento, generatori sintetici) va stimata solo sul training (Kaufman et al. 2012; Kapoor & Narayanan 2023)
- bilanciare prima dello split -> test con prevalenza artificiale (es. 50%) -> precision, PPV, PR-AUC e calibrazione gonfiate, non trasferibili alla realtà (van den Goorbergh et al. 2022)
- il livello KDIGO reale di ogni soggetto del test serve per il confronto con la clinica: deve restare intatto
- oversampling / SMOTE / CTGAN prima dello split -> pazienti duplicati o "copie interpolate" dello stesso soggetto sia in train sia in test -> leakage (Santos et al. 2018)
- test unico e congelato = stesso metro per tutti i modelli e tutte le tecniche -> le differenze dipendono solo da modello/tecnica

### Perché 75/25 (criterio: numero di eventi nel test, non la percentuale)
- sotto ~100 eventi le stime di performance sul test sono instabili; consigliati ≥100, meglio 200 (Collins, Ogundimu & Altman 2016; Riley et al. 2021)
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
  - possibile analisi di sensibilità futura: split per giornate intere (alcune date solo nel test), cioè validazione internal-external (Steyerberg & Harrell 2016); in alternativa modelli che tengono conto della struttura gerarchica dei dati (Shahbazi & Azadeh-Fard 2025)
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

### Riferimenti
Riferimenti completi nella sezione **Bibliografia** in fondo al file (Collins, Ogundimu & Altman 2016; Riley et al. 2021; Steyerberg et al. 2001; Steyerberg & Harrell 2016; Austin 2009; Collins et al. 2024).

## Bilanciamento (solo sul training, mai sul test)
- [ ] nessuna correzione (riferimento, Fase A)
- [ ] undersampling
- [ ] oversampling
- [ ] SMOTE (SMOTE-NC per variabili categoriche; Chawla et al. 2002)
- [ ] CTGAN (anche nella variante condizionata al livello KDIGO; Xu et al. 2019)
- [ ] analisi di sensibilità dell'imputer: una tecnica (es. SMOTE) ripetuta con KNN al posto di MissForest (Passo 10)
- [ ] ...

Per ogni tecnica: stesse 4 domande della Fase A, probabilità ricalibrate (van den Goorbergh et al. 2022), confronto KDIGO solo su soggetti reali.

## Preprocessing e selezione delle feature

### Analisi globale delle 190 variabili
Tutte le 190 colonne del dataset originale sono state verificate incrociando i dati grezzi con il dizionario estratto dal file `Supplementary Information.pdf` (Li et al. 2026, 7 pagine, letto con `pypdf`) e calcolando l'AUC univariato sul solo training set rispetto al target KDIGO.
- **AUC univariato massimo riscontrato:** 0,66 (corrispondente ad `Age`). Nessuna variabile non renale possiede una capacità predittiva anomala o sproporzionata: **nessun leaker forte nascosto**.
- **Comportamento delle righe incomplete:** sulle 75 feature finali, il **38,0% dei soggetti del training** presenta almeno un valore mancante (dovuto principalmente a `Tea` 13,1%, `HbA1c` 9,1%, `Drinking` 8,2%, blocco dell'emocromo differenziale 7,2%, `Smoking` 6,9%; le circonferenze sono solo all'1,7%). *Nota: una prima stima del 34% era calcolata su un sottoinsieme di 16 colonne.* Pertanto la *Complete Case Analysis* (eliminazione delle righe) è inaccettabile poiché distruggerebbe oltre un terzo del campione e, se i dati non mancano completamente a caso, introdurrebbe distorsione (Sterne et al. 2009; van Buuren 2018); l'imputazione è un passaggio metodologico obbligatorio.

### Tassonomia delle esclusioni (Perché toglierle e cosa succede se restano)

| Categoria | Colonne escluse | Perché toglierle | Cosa succederebbe se NON si togliessero |
| :--- | :--- | :--- | :--- |
| **1. Leakage di definizione e calcolo del target** | `SCRE`, `UMAUCR`, `UmALB`, `UCRE`, `GFR`, e i flag derivati `highCr`, `HighUmALB`, `HighACR`, `ACR3degree`, `GFR5lev`, `GFR6levG5`, `EUGFRabACR012`, `EUGFR90UACR01`, `EUGFR60abACR012`, `UACRGFR`, `DN` | Il target è calcolato matematicamente da queste variabili (`UMAUCR` per ACR, `SCRE` per eGFR). `UCRE` è il denominatore dell'ACR. `EUGFR90UACR01` è letteralmente il target calcolato con il vecchio GFR. | **AUC 0,96–0,998 (Cheating totale).** Il modello impara una singola regola banale (`UMAUCR >= 30` o `EUGFR90UACR01 == 1`), memorizza la scorciatoia e annulla l'intera utilità del machine learning. Nella clinica reale, se avessimo già questi esami non servirebbe alcun modello predittivo. |
| **2. Esami e diagnosi renali** | `BUN` (azotemia), `RF` (Renal Failure: anamnesi di insufficienza renale) | Violano il presupposto fondante della tesi: "stimare il rischio renale **senza esami renali**". `RF` è la diagnosi del target già conclamata e riferita dal paziente. | `BUN` ha AUC univariato modesto (0,57) ma la sua inclusione verrebbe stroncata da qualsiasi revisore per incoerenza metodologica. Con `RF`, il modello copierebbe l'anamnesi già nota invece di intercettare il danno asintomatico non diagnosticato. |
| **3. Metadati amministrativi e artefatti di studio** | `NO` (ID), `Data` (giorno di screening), `filter` (formula del paper), `GAnum`, `GAstudy` | Non descrivono la biologia o lo stato clinico del soggetto. `Data`: la prevalenza oscilla dal 4,5% al 24,5% a seconda della giornata di screening (raggruppamento per comunità). `filter`: contiene la formula logica `UMAUCR >= 0 & GFR5lev >= 0`, legata alla disponibilità degli esami renali. | **Overfitting su correlazioni spurie.** Gli algoritmi ad albero memorizzerebbero che certe date o blocchi di ID avevano casualmente più eventi, imparando l'effetto-centro/comunità invece del profilo metabolico del paziente. |
| **4. Colonne non documentate o artefatti software** | `VAR00001` (valori 1/9, nessuna etichetta), `ALT.1` (verificato: è $\sqrt{\text{ALT}}$ con zeri al posto dei NaN, generata da SPSS, r = 0,95 con ALT), `DF` (dizionario dichiara "2 = sì", nei dati compaiono solo 1 e NaN) | Mancanza di documentazione affidabile o artefatti di trasformazione software. | Variabili prive di significato clinico sono indifendibili davanti alla commissione di laurea. `VAR00001` presenta peraltro un segnale spurio (positivi 13% vs 9%) che creerebbe distorsione. |
| **5. Colonne vuote o strutturali (> 50% missing)** | `CDSMS` (100% NA), `GLU` (93%), `HPLC` (73%), `PLCR` (78%), `DMnew3class`, `Cancer`, `EyeDia`, `DR` (costante), blocco fumo/alcol di dettaglio (`SmokingRange`, `DrinkYears`, `EscDrink*`...) | Più del 50% di valori mancanti. Il blocco fumo/alcol di dettaglio è compilato solo da chi fuma o beve. | L'algoritmo di imputazione dovrebbe inventare dal 70% al 99% dei valori: il modello finirebbe per imparare la funzione di imputazione e non la realtà. `SmokingRange` ha AUC apparentemente alto (0,66) solo perché gli anni di fumo crescono banalmente con l'età. |
| **6. Questionario di conoscenza del diabete** | `Known`, `OPD`, `ToChilren`, `DietExcise`, `Compl`, `Check`, `GlucoseLev` | Misurano la consapevolezza/conoscenza soggettiva del diabete e non la salute del paziente. Circa il 75% dei record ha codice 9. La percentuale di positivi è piatta (8–11%) per qualsiasi risposta. | 7 feature di puro rumore che appesantiscono la dimensionalità e degradano la generazione di dati sintetici con SMOTE e CTGAN. *Nota bene:* `Compl` misura la consapevolezza delle complicanze, non la loro presenza clinica oggettiva. |
| **7. Derivati a soglia e binarizzazioni ridondanti (~35 colonne)** | `AgeGroup`, `Age3Group`, `BMI25`, `obesity`, `overweight`, `WGOCobBMI`, `BMI4Class`, `WC*`, `BP13085`, `WHO1999hbp`, `Anemia`, `Hb4per`, `HighUA`, `FPGover7`, `P2PGover11`, `PGclass*`, `FIB4n`, `TGdis`, `DysHDL*`, `Smoking2`, `Drinking2`... | Sono categorizzazioni arbitrarie di grandezze continue già presenti nel dataset (es. `Anemia` corrisponde esattamente a una soglia su `HGB`; `BMI25` a una soglia su `BMI`). | **(1) Multicollinearità severa:** i coefficienti della regressione logistica diventano instabili e l'importanza (SHAP/alberi) si disperde tra copie clonate. **(2) Generazione di soggetti fisiologicamente impossibili con SMOTE/CTGAN:** l'interpolazione sintetica crea incoerenze surreali, come un paziente con `BMI = 26,3` ma con `BMI25 = 0,4` (sotto soglia). Mantenere la sola variabile quantitativa continua originale evita il problema alla radice. |
| **8. Combinazioni lineari esatte** | `IDBIL` (= `TBIL` − `DBIL`), `GLO` (= `TP` − `ALB`) | Sono combinazioni algebriche esatte calcolabili direttamente da altre colonne presenti. | La matrice di covarianza diventa singolare (collinearità perfetta), impedendo la convergenza o rendendo instabili i modelli lineari. |

### Il conflitto del Composito CVD e la risoluzione del Codice 9

1. **Il test sul composito cardiovascolare (`cvd_history`) e su `WeightLoss`:**
   * Si era ipotizzato di aggregare le patologie cardiovascolari sparse (`Angina`, `MI`, `HF`, `TIA`, `CerebralIn`, `CerebralHe`, `PAD`, `CAP`) in un unico composito binario (`CVD_history = 1` se almeno una presente).
   * **Verifica empirica sui dati:** dopo la ricodifica del codice 9 in `NaN`, `cvd_history` presenta il **25,4% di dati mancanti** (superando la soglia massima del 15%), con soli 69 casi "sì" e una prevalenza di positivi piatta: **10,1% tra i CVD positivi vs 9,1% tra i CVD negativi**. Il segnale predittivo è nullo ($p \approx 0,7$).
   * Analogamente, `WeightLoss` presenta il **32,7% di NA** e una percentuale di positivi piatta.
2. **Decisione metodologica:**
   * In piena coerenza con la regola oggettiva del 15% di missingness e l'assenza di segnale univariato, **l'intero blocco delle comorbidità anamnestiche (`Angina`, `MI`, `HF`, `TIA`, `PAD`, `CAP`, `CerebralIn`, `CerebralHe`, `Arrhythmia`, `Hyper16`, `Hyper13`, `Hypo14`, `Blind`, `Cancer`, `Amputation`) e `WeightLoss` viene ESCLUSO.**
3. **L'effetto collaterale decisivo sul Codice 9:**
   * Escludendo le comorbidità e il questionario diabete (entrambi con > 20% di missing dopo ricodifica), **nessuna colonna contenente il codice 9 rimane tra le feature in input**.
   * Il problema del codice 9 si risolve in modo pulito e automatico: non serve alcuna complessa ingegneria di ricodifica per i modelli di machine learning.
   * **Avvertenza fondamentale per il codice:** la ricodifica di 9 a `NaN` non deve **mai** essere eseguita con un `.replace(9, np.nan)` globale sull'intero dataframe: variabili biologiche continue come `ALT` (58 soggetti con valore reale = 9 U/L), `TBIL` (58 soggetti con 9 µmol/L) e `MPV` (216 soggetti con 9 fL) contengono valori fisiologici reali pari a 9 che verrebbero cancellati.

### Feature legittime conservate nel modello

Le feature ammesse appartengono a grandezze cliniche e di laboratorio di screening generale di primo livello:
* **Dati demografici e antropometrici:** `Age`, `Gender`, `Height`, `Weight`, `BMI`, `waist1` (circonferenza vita), `hip` (fianchi), `WHR` (rapporto vita/fianchi). *(Nota: `leg` ha il 20,9% di NA e viene esclusa superando la soglia del 15%)*.
* **Parametri vitali ed emodinamici:** `Bpsys` (sistolica), `Bpdia` (diastolica), `HR` (frequenza cardiaca).
* **Assetto glicemico e metabolico:** `FPG` (glicemia a digiuno), `HbA1c` (emoglobina glicata: 9,2% NA, mantenuta e imputata), `DM` (stato diabetico), `DMhis` (anamnesi diabete), `PG2h` (glicemia post-carico a 2 ore), `FINS` (insulina a digiuno), `FCP` (peptide C a digiuno), `HomaIR`.
* **Lipidogramma:** `CHOL` (colesterolo totale), `TG` (trigliceridi), `HDL`, `LDL`.
* **Funzionalità epatica e metabolismo sistemico:** `ALT`, `AST`, `ASTALT` (rapporto AST/ALT), `GGT`, `ALP`, `TBIL` (bilirubina totale), `DBIL` (bilirubina diretta), `TP` (proteine totali), `ALB` (albumina sierica), `AG` (rapporto albumina/globuline).
* **Emocromo completo:** `WBC`, `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC`, `PLT`, `PDW`, `PCT`, `RDWCV`, `RDWSD`, `MONO`, `MONOcount`, `NEUT`, `NEUTcount`, `BASO`, `BASOcount`, `EO`, `EOcount`, `LYMPH`, `LYMPHcount`. *(Nota: `LC`, `LCcount`, `ALY`, `ALYcount` escluse perché al 19,4% di NA)*.
* **Complicanze microvascolari non renali:** `DRyd` (retinopatia da fundus fotografico, 0% NA), `cataract` (0% NA), `MADPN` e `TFDPN` (neuropatia periferica da monofilamento e diapason, 3,9% NA). Mantenute perché riflettono la compromissione microangiopatica sistemica (triade microvascolare) con tassi di missing trascurabili.
* **Stili di vita:** `Smoking`, `Drinking`, `Tea`.
* **Aggiunte dopo la verifica sul codice** (erano in input ma mancavano da questo elenco):
  * composizione corporea: `Fat` (% di grasso corporeo)
  * assetto glicemico: `INS2h`, `CP2h` (insulina e peptide C a 2 ore), `ISIGutt` (indice di sensibilità insulinica di Gutt et al. 2000; `Homaβ` esclusa al Passo 7); `GA` è fra le variabili-conseguenza
  * funzionalità epatica: `FIB4` (indice di fibrosi da età, AST, ALT, piastrine; Sterling et al. 2006)
  * sierologia: `HbsAg` (epatite B), `HCVAb` (epatite C: l'infezione da HCV è associata alla malattia renale cronica; KDIGO 2022b)
  * anamnesi: `HypertenHis` (vuoto → 0, vedi Passo 2), `DMfamilyHistory` (familiarità per diabete; le colonne per singolo parente `FatherDM`, `MotherDM`, `BSDM`, `ChildDM` sono escluse perché riassunte da questa)
* **Totale: 74 feature** (60 numeriche + 14 categoriche; `Homaβ` esclusa al Passo 7). Elenco definitivo in `configs/config.yaml`, sezione `features`.

### Variabili-conseguenza della malattia renale (Tenute + Analisi di Sensibilità)
* **Quali sono:** `HGB`, `RBC`, `HCT` (serie rossa ed ematocrito), `SUA` (acido urico), `ALB`, `TP` (albumina e proteine sieriche), `GA` (albumina glicata). Valori mancanti bassi sul training: `HGB`/`RBC`/`HCT` 1,0%, `SUA` 1,4%, `GA` 0,3%, `ALB`/`TP` 0%.
* **Natura clinica:** non sono leaker di definizione, ma parametri che si alterano come *effetto* (conseguenza fisiopatologica) della ridotta funzione renale (l'anemia da ridotta eritropoietina, KDIGO 2026c; l'iperuricemia da calo dell'escrezione renale, KDIGO 2024; l'ipoalbuminemia da grave proteinuria).
* **Gestione metodologica per la tesi:**
  1. Vengono incluse nella pipeline principale poiché arricchiscono la capacità di triage del modello.
  2. Verrà documentata un'**analisi di sensibilità** (prestazioni del modello escludendo **tutte e 7** le variabili-conseguenza: `HGB`, `RBC`, `HCT`, `SUA`, `ALB`, `TP`, `GA` → set di feature `no_consequence`, 67 feature) per dimostrare alla commissione che il modello si regge solidamente sui fattori di rischio causali a monte (pressione, glicemia, BMI, lipidi, età) e non esclusivamente sulla causalità inversa.
  3. Nell'interpretazione dell'importanza delle feature (SHAP/coefficienti) verranno discusse esplicitamente come "marcatori di danno sistemico" e non come "cause eziologiche".

### Strategia di Imputazione (Soglia 15% e CV con 1-SE rule)
* **Soglia di ammissibilità:** variabili con $> 15\%$ di dati mancanti vengono eliminate; variabili con $\le 15\%$ vengono ammesse e imputate.
* **Mini-confronto in Cross-Validation (solo sul train set congelato):**
  Si confrontano 4 strategie di imputazione basate sulla letteratura specifica (`papers/preprocessing/`: Tiwaskar et al. 2025; Alnowaiser 2024; Hameed & Ali 2025):
  1. **Mediana / Moda** (baseline univariato semplice e robusto)
  2. **KNNImputer** (imputazione basata su vicinato clinico; Troyanskaya et al. 2001; efficace su dati di diabete in Alnowaiser 2024)
  3. **MICE / IterativeImputer** (imputazione multivariata con equazioni concatenate; van Buuren & Groothuis-Oudshoorn 2011)
  4. **MissForest** (Random Forest iterativo per missing data complessi; Stekhoven & Bühlmann 2012; il migliore dei tre metodi ML su dati di diabete in Tiwaskar et al. 2025)
* **Regola decisionale: "1-Standard-Error Rule"** (Hastie, Tibshirani & Friedman 2009, §7.10): si calcola la metrica target (AUC / PR-AUC) in cross-validation out-of-fold. Si sceglie il metodo computazionalmente più semplice e parsimonioso (es. Mediana o KNN) purché le sue prestazioni rientrino entro 1 errore standard (1 SE) dal metodo con il punteggio assoluto più alto.
* **Esito (vedi log, Passi 5–10):** a valle i 4 metodi sono indistinguibili; la regola è stata rivista in due fasi (PR-AUC, poi RMSE). Fra KNN e MissForest la regola resta al margine (Passi 6, 8, 10): il confronto appaiato sui fold mostra che MissForest ricostruisce meglio in modo sistematico, ed è il metodo adottato (Passi 9–10).
* **Rigore anti-leakage:** l'imputer viene addestrato **esclusivamente sui fold di training** (all'interno di ciascun fold di CV e infine sull'intero train set da 4.350 soggetti) e applicato in sola trasformazione (`transform`) sul test set congelato (1.451 soggetti), in piena aderenza alle linee guida TRIPOD+AI (Collins et al. 2024).

### Implementazione del preprocessing (log passo per passo)

**Passo 1 — mappa delle colonne in `configs/config.yaml` (sezione `features`)**
- ogni colonna del dataset sta in **esattamente un gruppo**: escluso per un motivo (11 gruppi) oppure in input come numerica o categorica. Verificato: 190 colonne, 190 assegnate, nessun duplicato, nessuna dimenticata
- input finale: **61 numeriche + 14 categoriche = 75 feature** *(poi 60 + 14 = 74, dopo l'esclusione di `Homaβ` al Passo 7)*
- perché una mappa completa invece di una lista di esclusioni: con una lista di sole esclusioni, una colonna dimenticata entrerebbe nel modello in silenzio. Con la mappa completa un test fallisce se una colonna non è assegnata o se ne compare una nuova
- due colonne che non erano ancora state assegnate a nessun gruppo:
  - `Waist2` (circonferenza ombelicale) → **esclusa** nel gruppo `duplicate_measures`: seconda misura della vita, r = 0,86 con `waist1`; `WHR` è esattamente `waist1` / `hip`, quindi la misura di riferimento è `waist1`
  - `ThalussemiaAND` / `ThalussemiaOR` → **escluse come derivati**: sono flag di screening calcolati dall'emocromo (`ThalussemiaOR` coincide con MCV < 80 oppure MCH < 27 nel 99% dei casi), e `MCV` e `MCH` sono già in input
- `Cancer` compare sia fra le comorbidità sia fra le colonne vuote (93% NA): assegnata a `sparse`, perché ogni colonna ha un solo motivo
- regola sui derivati, resa esplicita:
  - **esclusi**: soglie e classificazioni (`BMI25`, `Anemia`…) e combinazioni lineari esatte (`IDBIL`, `GLO`)
  - **tenuti**: indici clinici continui non lineari, che non sono ricavabili con una somma dalle altre colonne e hanno un significato clinico proprio: `BMI`, `WHR`, `HomaIR` (= FPG · FINS / 22,5, verificato), `Homaβ`, `ISIGutt`, `FIB4`, `ASTALT`
- `unknown_columns`: le 26 colonne in cui il dizionario usa 9 = sconosciuto. Non servono a ricodificare (nessuna resta in input) ma a **verificarlo**: se una finisse fra le feature, il codice si ferma
- `consequence`: le 7 variabili-conseguenza (`HGB, RBC, HCT, SUA, ALB, TP, GA`) che escono nel set di sensibilità `no_consequence`

**Passo 2 — `src/data/preprocess.py`: selezione, codifiche e preprocessor**
- divisione netta fra due tipi di trasformazione:
  - **senza stato** (riga per riga, non imparano nulla dai dati) → `select_features`, `encode`: si possono applicare prima della cross-validation senza leakage
  - **con stato** (imparano dai dati: mediane, vicini, regressioni) → `build_preprocessor`: va stimato **solo sul training di ogni fold**, e non riceve mai y
- `select_features(df, feature_set)`: set `main` (75 feature) o `no_consequence` (68) *(poi 74 e 67, Passo 7)*. Si ferma con un errore se fra le feature compare una colonna di leakage, un esame renale o una colonna con codice 9
- `encode(df)`, codifiche (nessun one-hot in tutta la pipeline):
  - `HypertenHis`: nei dati solo 1 o vuoto → vuoto = 0 (assunzione da dichiarare: "nessuna anamnesi di ipertensione riferita"). Risultato: 147 sì, 4203 no nel training
  - `Gender`: 1/2 → **0 = maschio, 1 = femmina**
  - `Smoking`, `Drinking`, `Tea`: 1/2/3 → **0/1/2 ordinali** (no < occasionale < regolare). Perché ordinali e non one-hot: l'ordine esiste davvero, resta 1 colonna invece di 3, gli alberi la tagliano in modo naturale, e con SMOTE/CTGAN il one-hot rischierebbe combinazioni impossibili ("no" e "regolare" entrambi a 1). In augmentation vanno dichiarate **discrete** (SMOTE-NC, Chawla et al. 2002; colonne discrete di CTGAN, Xu et al. 2019), altrimenti SMOTE genera valori come `Smoking = 1,4`
- `build_preprocessor(imputer, colonne)`:
  - numeriche: standardizzazione (che ignora i NaN) → imputer scelto. La standardizzazione viene prima perché KNN e MICE lavorano su distanze e regressioni, che senza scala sarebbero dominate dalle variabili con valori grandi (es. `PLT` ~250 contro `HbA1c` ~6)
  - categoriche: sempre **moda** (imputazione selettiva). Motivo: KNN e MICE applicati a una binaria producono valori senza senso come `DM = 0,37`. Limite: la moda ignora le correlazioni (es. fumo e sesso), effetto piccolo con NA ≤ 13%
  - nessun indicatore di mancanza (`add_indicator`): i NaN sono concentrati in alcune giornate di raccolta → l'indicatore imparerebbe la giornata, non la persona. Gli indicatori aiutano solo se il meccanismo di mancanza resta uguale fra sviluppo e uso del modello (Sisk et al. 2023)
- verifiche sul training (numeri reali dal codice):
  - X: 4350 × 75; NA massimo **13,1%** (`Tea`), nessuna colonna oltre il 15%; 55 colonne con almeno un NA
  - righe con almeno un NA: **38,0%** → analisi dei soli casi completi esclusa
  - AUC univariato massimo sulle feature finali: **0,653** (`Age`), poi `ALP` 0,631, `FIB4` 0,623, `Bpsys` 0,610 → nessun leaker (soglia di allarme 0,75)

**Passo 3 — `tests/test_preprocess.py`: 9 test, tutti verdi (17 in totale con quelli dello split)** *(numeri di questo passo: 75 feature; dopo il Passo 7 il test verifica 74, e al Passo 10 i test totali sono 21)*
Ogni test protegge da un errore preciso:

| test | cosa verifica | errore che impedisce |
|---|---|---|
| `test_every_column_has_exactly_one_group` | le 190 colonne del file raw sono tutte assegnate, una sola volta | una colonna dimenticata (o aggiunta in una nuova versione del dataset) che entra nel modello senza che nessuno l'abbia valutata |
| `test_no_forbidden_columns` (×2 set) | nessuna colonna di leakage, esame renale o con codice 9 fra le feature | il ritorno accidentale di `UMAUCR` & co. dopo una modifica alla config |
| `test_feature_sets` | `main` = 75 feature; `no_consequence` = `main` meno esattamente le 7 variabili-conseguenza | un'analisi di sensibilità che toglie colonne diverse da quelle dichiarate |
| `test_missing_threshold` | nessuna feature oltre il 15% di NA; le 5 escluse per NA sono davvero oltre | una soglia dichiarata ma non rispettata |
| `test_encoding` | `Gender` 0/1, ordinali in {0, 1, 2}, `HypertenHis` senza vuoti | codifiche sbagliate (es. ordinali ancora 1/2/3) |
| `test_no_hidden_leakage` | AUC univariato < 0,75 per ogni feature, sul training | un leaker non ancora noto |
| `test_preprocessor_fills_everything` | preprocessor stimato su una parte del training, applicato a righe mai viste: nessun NaN in uscita, stesse colonne | un preprocessor che funziona solo sui dati su cui è stato stimato |
| `test_one_se_rule_prefers_simplest` | con tre metodi a 0,30 / 0,31 / 0,33 ± 0,02 la regola sceglie il secondo (il più semplice entro 1 errore standard, non il migliore assoluto) | una regola di scelta implementata male |

- i test usano **solo il training**: il test set resta congelato anche per i controlli automatici
- il file raw viene letto solo nell'intestazione (`nrows=0`) per confrontare i nomi delle colonne: 15 secondi in meno

**Passo 4 — `src/data/imputation.py`: confronto dei metodi di imputazione (solo training)**
- **cosa si confronta**: solo le 61 variabili **numeriche**. Le categoriche sono sempre imputate con la moda (imputazione selettiva, vedi Passo 2)
- **candidati**, in ordine di semplicità (conta per la regola di 1 errore standard): mediana < KNN (k = 5) < MICE (`IterativeImputer` + regressione bayesiana) < MissForest (`IterativeImputer` + ExtraTrees)
  - precisazione terminologica: **MissForest** è una tecnica di imputazione (Stekhoven & Bühlmann 2012: per ogni colonna con buchi allena una foresta che la predice dalle altre, e ripete fino a stabilizzazione). **ExtraTrees** non lo è: è il modello usato *dentro* l'imputer. Quella implementata è quindi una variante "tipo MissForest", non l'implementazione originale in R
  - "MICE" qui è **una sola imputazione** (la versione con più dataset imputati e risultati combinati serve per l'inferenza statistica, non per un modello predittivo: van Buuren 2018; Sisk et al. 2023)
- **imputare è già allenare modelli**: KNN memorizza le righe del training, MICE allena una regressione per colonna, MissForest una foresta per colonna, e anche la mediana "impara" un numero. Per questo l'imputer si stima solo sul training di ogni fold. La differenza rispetto al classificatore: l'imputer predice **una feature dalle altre feature** e non vede mai y
- **due criteri**, nello stesso fold (cross-validation a 5 fold stratificata sul livello KDIGO, seed 42):
  - (a) **errore di ricostruzione**: nel fold di validazione si nasconde il 10% dei valori osservati, l'imputer stimato sul fold di training li ricostruisce, si misurano RMSE e MAE in unità standardizzate. Si nascondono valori **solo nelle colonne che hanno NA nel training**, cioè quelle che l'imputer deve davvero imparare a riempire (protocollo di Tiwaskar 2025 e Hameed & Ali 2025)
  - (b) **prestazioni a valle**: una regressione logistica fissa (`class_weight="balanced"`) usata come **strumento di misura**, non come modello della tesi → PR-AUC (metrica principale, adatta a classi sbilanciate: Saito & Rehmsmeier 2015) e AUC
- **regola di scelta**: il metodo più semplice con PR-AUC entro 1 errore standard dal migliore (Hastie, Tibshirani & Friedman 2009, §7.10). Motivo: con differenze dentro il rumore della CV non ha senso pagare la complessità
- **controllo anti-leakage**: se una qualsiasi AUC supera 0,9 il programma lo segnala (con sole feature non renali sarebbe implausibile)
- **"MissForest con freni"** — primo tentativo interrotto dopo oltre 20 minuti senza risultati. Causa: 49 colonne numeriche hanno almeno un NA, quindi `skip_complete` risparmia poco, e ogni fold stimava l'imputer due volte (una per criterio). Correzioni:
  1. `max_features="sqrt"`: ogni albero considera √p colonne per split. È anche **più fedele** al MissForest originale (mtry = √p di default; Stekhoven & Bühlmann 2012)
  2. 50 alberi, `min_samples_leaf=5`, 5 iterazioni, `skip_complete=True` (le colonne senza NA fanno solo da predittori)
  3. **un solo fit per fold**: l'imputer stimato dentro la pipeline della logistica serve anche al criterio (a)
  4. `n_jobs=1`: con foreste così piccole il parallelismo costa più di quanto rende (e generava migliaia di avvisi di joblib)
  - risultato: circa 70 secondi per fold invece di diversi minuti
- **primo fold (controllo di plausibilità)**: AUC 0,67–0,68 per tutti i metodi → nessun segno di leakage, in linea con l'AUC univariato massimo (0,65)
- **anomalia di MICE (primo fold)**: MAE più basso di tutti (0,33) ma RMSE più alto (1,45). Significa che MICE ricostruisce bene la maggior parte dei valori ma ne sbaglia alcuni di molto: la regressione lineare **estrapola** su variabili molto asimmetriche (trigliceridi, GGT, insulina) producendo valori estremi. Mediana, KNN e MissForest non possono uscire dall'intervallo dei valori osservati

**Passo 5 — risultati del confronto (5 fold sul training, media ± errore standard)**

Set `main` (75 feature):

| metodo | RMSE ↓ | MAE ↓ | PR-AUC ↑ | AUC | tempo totale |
|---|---|---|---|---|---|
| mediana | 1,031 ± 0,058 | 0,667 | 0,249 ± 0,027 | 0,691 | 1 s |
| KNN | 0,774 ± 0,053 | 0,499 | 0,253 ± 0,029 | 0,693 | 10 s |
| MICE | 0,870 ± 0,168 | **0,298** | 0,254 ± 0,031 | 0,695 | 65 s |
| MissForest | **0,724** ± 0,065 | 0,433 | 0,253 ± 0,028 | 0,695 | 344 s |

Set `no_consequence` (68 feature):

| metodo | RMSE ↓ | MAE ↓ | PR-AUC ↑ | AUC | tempo totale |
|---|---|---|---|---|---|
| mediana | 0,991 ± 0,031 | 0,655 | 0,241 ± 0,025 | 0,688 | 1 s |
| KNN | 0,742 ± 0,026 | 0,490 | 0,245 ± 0,025 | 0,690 | 9 s |
| MICE | 1,138 ± 0,167 | **0,327** | 0,242 ± 0,026 | 0,691 | 52 s |
| MissForest | **0,664** ± 0,033 | 0,414 | 0,244 ± 0,025 | 0,689 | 279 s |

Lettura:
- **a valle i quattro metodi sono indistinguibili**: PR-AUC 0,249–0,254 (errore standard ~0,03), AUC 0,69–0,70. Per un classificatore addestrato sui dati originali, il metodo di imputazione è irrilevante
- **nessun leakage**: AUC ~0,69, lontanissima dalla soglia di allarme 0,9 e coerente con l'AUC univariato massimo (0,65). Riferimento: la PR-AUC di un classificatore casuale è pari alla prevalenza (Saito & Rehmsmeier 2015), 9,8%, quindi una PR-AUC di 0,25 è circa 2,5 volte il caso
- **nella ricostruzione dei valori le differenze sono nette**: la mediana è la peggiore (RMSE ~1,0 in unità standardizzate, cioè come prevedere la media); MissForest e KNN i migliori. MICE conferma il comportamento del primo fold: MAE migliore ma RMSE instabile (errore standard 0,17) per le estrapolazioni sulle variabili asimmetriche
- togliere le 7 variabili-conseguenza abbassa la PR-AUC di circa 0,01 (0,253 → 0,245 con KNN), dentro l'errore standard: il modello non si regge su di loro
- regola dichiarata (il più semplice entro 1 errore standard sulla **sola PR-AUC**) → **mediana** in entrambi i set
- **decisione aperta**: la regola guarda solo il criterio (b) e ignora il criterio (a), che invece distingue nettamente i metodi. Il problema riguarda la Fase B: con la mediana, 397 soggetti senza `HbA1c` (28 diabetici) ricevono tutti 5,5, mentre i diabetici con il valore misurato hanno in media 6,57. KNN assegna in media 6,07 ai diabetici e 5,45 ai non diabetici. I picchi artificiali creati dalla mediana verrebbero moltiplicati da SMOTE e imparati da CTGAN come parte della distribuzione
- file: `analytics/preprocessing/imputation_comparison_*.csv` (medie), `imputation_folds_*.csv` (singoli fold), figure `04_imputation_*.png`
- nota tecnica: un'esecuzione intermedia è terminata senza output né file (causa non identificata, probabilmente l'interruzione del processo in background); la ripetizione con output non bufferizzato ha dato exit code 0 e i risultati riportati sopra

**Passo 6 — decisione: KNN (k = 5), fissato in `configs/config.yaml` (`imputation.method`)** — *superata dal Passo 9 (MissForest)*

Regola di scelta **rivista dopo aver visto i risultati** (da dichiarare così nella tesi):
1. si tengono i metodi con PR-AUC entro 1 errore standard dal migliore (prestazioni a valle)
2. fra questi, si sceglie il più semplice con RMSE entro 1 errore standard dal migliore (ricostruzione dei valori)

Perché la regola è cambiata: quella originale guardava solo il criterio (b) e ignorava il criterio (a), che faceva già parte del protocollo (ripreso da Tiwaskar 2025) ed è l'unico che distingue i metodi. Il criterio (a) conta per la Fase B, dove SMOTE e CTGAN lavorano sulla distribuzione congiunta delle variabili.

Esito della regola in due fasi:
- set `main` → **KNN**: RMSE 0,774, entro la soglia di 0,789 (MissForest 0,724 + 0,065); PR-AUC 0,253, entro la soglia di 0,224
- set `no_consequence` → la regola sceglierebbe MissForest (KNN 0,742 supera la soglia di 0,697)
- **un solo imputer per entrambi i set, deciso sul set principale**: l'analisi di sensibilità deve differire solo per le 7 variabili tolte. Se cambiasse anche l'imputer, una differenza di risultati non sarebbe più attribuibile alle variabili

Perché non la mediana:
- RMSE ~1,0 in unità standardizzate: non ricostruisce nulla del valore mancante (equivale a prevedere la media)
- attenua varianza e correlazioni (Tiwaskar 2025) e crea masse puntiformi: i 397 soggetti senza `HbA1c` ricevono tutti 5,5, compresi 28 diabetici (i diabetici con valore misurato hanno in media 6,57)
- rischio sistematico **atteso** (non misurato) per la Fase B: SMOTE interpola fra punti identici e moltiplica i picchi, CTGAN li impara come parte della distribuzione. Evitarlo costa ~9 secondi

Perché non MissForest:
- nessun beneficio a valle (PR-AUC identica)
- vantaggio in ricostruzione piccolo nel set principale (0,724 contro 0,774, entro 1 errore standard); più netto nel set `no_consequence` (0,664 contro 0,742), che però non decide per quanto detto sopra
- circa **30 volte più lento** (344 s contro 10 s sul set principale): in Fase B il costo si moltiplica per tecniche × modelli × fold senza guadagno a valle

Perché KNN:
- stessa PR-AUC degli altri, RMSE ridotto del 25% rispetto alla mediana (0,774 contro 1,031), ~10 secondi
- il valore mancante è stimato dai 5 soggetti più simili sulle variabili numeriche standardizzate → valori spostati verso il fenotipo del soggetto: `HbA1c` imputata in media 6,07 nei diabetici e 5,45 nei non diabetici (contro 5,5 per tutti con la mediana)
- limiti da dichiarare: la media di 5 vicini **riduce comunque** la varianza (molto meno della mediana, ma non la elimina); ogni imputazione altera un po' le distanze usate da SMOTE; k = 5 è fissato a priori e non ottimizzato; i vicini sono calcolati solo sulle numeriche (le categoriche sono imputate con la moda)

Codice e verifiche:
- `src/data/imputation.py`: `two_stage_choice` (regola in due fasi), `chosen_imputer()` (l'imputer di config, da usare nelle fasi successive); l'esecuzione stampa l'esito di entrambe le regole
- `tests/test_preprocess.py`: con i numeri reali del set principale la regola a una fase sceglie la mediana e quella a due fasi KNN; l'imputer di config è un `KNNImputer` → **19 test verdi**
- figure `analytics/preprocessing/04_imputation_*.png`: RMSE (barre da zero, soglia della seconda fase) e PR-AUC (punti con barre d'errore, perché l'asse non parte da zero e delle barre esagererebbero differenze inesistenti), KNN in blu

**Passo 7 — ultimi controlli sulle 61 numeriche: asimmetria, valori estremi, collinearità**

*Asimmetria → nessuna trasformazione logaritmica (decisione)*
- 25 numeriche su 61 hanno asimmetria > 2 (le peggiori: `Homaβ` 41, `ISIGutt` 22, `ASTALT` 20, `CP2h` 14,5, `FINS` 12,6, `HomaIR` 11,6, `AST` 11,5, `GGT` 9,2, `ALP` 8,3)
- proposta valutata: `log1p` su queste 25 (senza stato, niente leakage)
- test empirico (KNN + regressione logistica, 5 fold stratificati sul livello KDIGO, training):
  - senza log: PR-AUC **0,2530 ± 0,0286** (errore standard; deviazione standard fra fold 0,064)
  - con log: PR-AUC **0,2613 ± 0,0284**
  - differenza appaiata sugli stessi fold: **+0,0083 ± 0,0054**, a favore del log in 4 fold su 5, **non significativa** (t ≈ 1,5). Il t appaiato sui fold di CV è già ottimista, perché i fold condividono gran parte del training (Nadeau & Bengio 2003): la conclusione "non significativa" è quindi prudente
  - un test indipendente dell'utente dà la stessa conclusione (+0,0045)
- **decisione: nessuna trasformazione**, si mantiene la scala clinica originale. Motivazioni:
  1. guadagno per la logistica piccolo e dentro il rumore → parsimonia
  2. per gli alberi (Random Forest, Gradient Boosting) una trasformazione monotona non cambia nulla: dividono i dati per rango (Hastie, Tibshirani & Friedman 2009, §10.7)
  3. la scala originale è più immediata da interpretare (odds ratio e grafici SHAP in unità cliniche; Lundberg & Lee 2017)
- precisazioni per non sovrastimare gli argomenti:
  - la logistica **fa parte** della Fase A (baseline e regolarizzata): il log non è irrilevante per tutti i modelli, ha un effetto piccolo e non significativo sulla logistica
  - il log non renderebbe i risultati illeggibili: in letteratura clinica si usa l'odds ratio "per raddoppio" del valore e SHAP può mostrare i valori grezzi. La scala originale è preferita per immediatezza, non perché l'alternativa sia impossibile
- limite da dichiarare: nella logistica pochi valori estremi pesano sui coefficienti; eventuale analisi di sensibilità con log sulla sola logistica

*`Homaβ` → esclusa (nuovo gruppo `unstable_indices`)*
- `Homaβ` = 20 · FINS / (FPG − 3,5) (Matthews et al. 1985): quando la glicemia a digiuno si avvicina a 3,5 mmol/L il denominatore tende a zero e l'indice esplode → 9 valori sopra 1000 nel training, asimmetria 41. Sono artefatti della formula, non valori biologici, e un logaritmo non li correggerebbe
- l'informazione non si perde: `FPG` e `FINS` restano in input
- `HomaIR` (= FPG · FINS / 22,5; Matthews et al. 1985) **resta**, nonostante r = 0,96 con `FINS`: è un indice clinico noto di insulino-resistenza e la formula è stabile (un prodotto, non una divisione)
- feature finali: **74** (60 numeriche + 14 categoriche); set `no_consequence`: **67**

*Valori estremi → tenuti*
- da 1 a 9 casi per variabile nel training: `Homaβ` > 1000 (9, esclusa), HGB < 70 g/L (5, anemia grave), GGT > 1000 (5), TG > 20 mmol/L (5), BMI < 14 (3), peso < 30 kg (3), statura < 130 cm (2), PLT > 1000 (2), PAS ≥ 250 (1), ALT > 500 (1), FINS > 200 (1)
- nessun valore impossibile (colesterolo totale minimo 2,12 mmol/L)
- **decisione: nessuna rimozione né winsorizzazione**. Sono valori clinicamente possibili, uno screening reale li contiene, e togliere gli outlier solo dal training renderebbe il modello impreparato sul test (critica di Hameed & Ali 2025 agli studi che rimuovono gli outlier)

*Collinearità → nota, nessuna azione*
- coppie con |r| > 0,9 fra le numeriche: `FINS`/`HomaIR` 0,96, `EO`/`EOcount` 0,92, `MCV`/`MCH` 0,91, `HCT`/`HGB` 0,91
- con la logistica regolarizzata e con gli alberi non è un problema per la previsione; lo è per l'**interpretazione**: l'importanza si divide fra le due variabili della coppia → da ricordare nell'analisi SHAP/coefficienti

*Conseguenza sul confronto delle imputazioni*: togliere `Homaβ` cambia il set di feature (75 → 74) e le distanze di KNN → confronto rieseguito sulle 74 feature (risultati sotto).

**Passo 8 — confronto delle imputazioni rieseguito sulle 74 feature**

| metodo | RMSE ↓ (main) | PR-AUC ↑ (main) | RMSE ↓ (no_consequence) | PR-AUC ↑ (no_consequence) | tempo (main) |
|---|---|---|---|---|---|
| mediana | 1,038 ± 0,047 | 0,252 ± 0,030 | 1,038 ± 0,046 | 0,245 ± 0,028 | 1 s |
| KNN | 0,793 ± 0,061 | 0,255 ± 0,032 | 0,782 ± 0,049 | 0,249 ± 0,028 | 7 s |
| MICE | 1,253 ± 0,136 | 0,257 ± 0,033 | 0,955 ± 0,095 | 0,246 ± 0,030 | 61 s |
| MissForest | **0,725** ± 0,066 | 0,256 ± 0,031 | **0,717** ± 0,054 | 0,248 ± 0,027 | 327 s |

- **la regola in due fasi ora sceglie MissForest in entrambi i set**: nel set principale KNN (0,7926) supera la soglia (0,7250 + 0,0663 = 0,7913) di 0,0013
- **confronto appaiato** (stessi fold e stessi valori nascosti per tutti i metodi; sul confronto fra algoritmi in cross-validation vedi Nadeau & Bengio 2003):
  - RMSE KNN − MissForest: **+0,068 ± 0,006** (main), **+0,065 ± 0,006** (no_consequence); MissForest migliore in **10 fold su 10**
  - PR-AUC KNN − MissForest: −0,0003 ± 0,0016 (main) → identici a valle
- lettura: il vantaggio di MissForest nella ricostruzione è **sistematico**, non rumore. La regola con errore standard non appaiato lo mascherava al Passo 6, perché la variabilità di difficoltà fra i fold gonfia l'errore standard
- **correzione dell'argomento sul costo (Passo 6)**: l'imputazione non dipende dalla tecnica di augmentation né dal modello → si stima una volta per fold (5 fit, ~5 minuti) e si riusa per tutte le combinazioni della Fase B, più un fit finale sul training. Il costo non si moltiplica per tecniche × modelli
- **decisione**: MissForest, vedi Passo 9

**Passo 9 — decisione finale: MissForest, per entrambi i set e per entrambe le fasi** *(decisione confermata al Passo 10; le motivazioni basate sull'esito della regola, punto 1 della prima lista e punto 5, sono state riviste lì)*

Fissato in `configs/config.yaml` (`imputation.method: MissForest`). Sostituisce la scelta del Passo 6 (KNN), che era stata presa sulle 75 feature.

*Perché la scelta del Passo 6 non regge più*
- al Passo 6 KNN si reggeva su due argomenti: (1) la regola in due fasi lo sceglieva; (2) MissForest costava circa 30 volte di più e il costo si sarebbe moltiplicato in Fase B
- sulle 74 feature definitive (Passo 8) **la stessa regola, applicata senza modifiche, sceglie MissForest** in entrambi i set
- l'argomento del costo è caduto: l'imputer non dipende né dalla tecnica di augmentation né dal modello, quindi si stima una volta per fold e si riusa (Passo 8)
- restare su KNN avrebbe richiesto una **seconda** modifica della regola dopo aver visto i risultati, proprio quando non favorisce il metodo preferito. È il rischio del "giardino dei sentieri che si biforcano": scelte analitiche che dipendono dai dati rendono i risultati meno credibili (Gelman & Loken 2014). La regola in due fasi è stata introdotta una sola volta (Passo 6) ed è stata applicata invariata al Passo 8: così va dichiarata nella tesi, come richiede la trasparenza sulla gestione dei dati mancanti (Collins et al. 2024)

*Perché MissForest è la scelta migliore per questa tesi*
1. **ricostruisce meglio, in modo sistematico e non per caso**: nel confronto appaiato (stessi fold, stessi valori nascosti) RMSE KNN − MissForest = **+0,068 ± 0,006** nel set principale e **+0,065 ± 0,006** nel set `no_consequence`, MissForest migliore in **10 fold su 10** (t ≈ 11)
2. **è coerente con la letteratura sui dati clinici**:
   - nel lavoro originale MissForest supera KNN e MICE su dati continui e misti (Stekhoven & Bühlmann 2012)
   - su dati di laboratorio clinici MissForest ha l'errore di imputazione più basso fra i metodi confrontati, KNN e MICE compresi (Waljee et al. 2013)
   - su dati di diabete MissForest è il migliore in MAE, RMSE e R² nel 100% dei casi e conserva le correlazioni fra variabili come nel dataset completo (Tiwaskar et al. 2025)
   - lo studio a favore di KNN in `papers/` (Alnowaiser 2024) lo confronta solo con l'eliminazione delle righe, non con MissForest
3. **serve la Fase B, che è il cuore della tesi (Scope, domanda 5)**:
   - SMOTE crea casi sintetici interpolando fra vicini nello spazio delle feature (Chawla et al. 2002), CTGAN impara la distribuzione congiunta delle variabili (Xu et al. 2019)
   - valori imputati più vicini al vero e correlazioni conservate significano meno artefatti moltiplicati da SMOTE o appresi da CTGAN
   - è lo stesso motivo per cui la mediana è stata esclusa al Passo 6, portato fino in fondo
4. **è coerente con le decisioni del Passo 7**: abbiamo scelto di non trasformare le 25 variabili asimmetriche e di tenere i valori estremi. KNN lavora su distanze, che pochi valori enormi dominano. MissForest usa alberi, che dividono per rango e sono insensibili ad asimmetria e valori estremi (Hastie, Tibshirani & Friedman 2009, §10.7), e cattura relazioni non lineari e interazioni fra variabili (Stekhoven & Bühlmann 2012)
5. **stesso metodo in entrambi i set**: la regola sceglie MissForest sia in `main` sia in `no_consequence`. Il problema del Passo 6 (KNN in un set, MissForest nell'altro) sparisce
6. **un solo imputer per Fase A e Fase B**: il confronto A contro B misura solo l'effetto dell'augmentation. La Fase A non ci perde nulla: a valle la PR-AUC è identica (differenza appaiata −0,0003 ± 0,0016)

*Perché non KNN*
- ricostruisce peggio in tutti i fold
- sensibile ad asimmetria e valori estremi, che abbiamo scelto di tenere
- la media di 5 vicini comprime la varianza più di MissForest (RMSE più alto); k = 5 era fissato a priori
- vantaggi reali ma secondari per questa tesi: semplicità, determinismo, velocità (~6 s)

*Costo e implementazione*
- circa 5 minuti per le 5 stime in cross-validation, più una stima finale sul training intero
- nelle Fasi A e B i fold imputati si stimano **una volta sola** e si salvano su disco, per ogni fold e per ogni set di feature. È lecito: l'imputer è stimato solo sul training del fold e non vede mai y, quindi riusarlo non introduce leakage
- `chosen_imputer()` legge `MissForest` dalla config: non si possono usare imputer diversi per errore

*Limiti da dichiarare*
- è una variante "tipo MissForest" (`IterativeImputer` di scikit-learn con ExtraTrees, 50 alberi, 5 iterazioni), non l'implementazione originale in R con Random Forest. Parametri ridotti per contenere il costo (Passo 4)
- ha una componente casuale: riproducibile con seed fisso (42)
- come ogni imputazione a valore singolo, riduce un po' la varianza (van Buuren 2018)
- il vantaggio per la Fase B è **atteso** (letteratura e ricostruzione migliore), non ancora misurato
- le prove di letteratura riguardano soprattutto mancanti completamente casuali (MCAR: Tiwaskar et al. 2025 lo dichiarano come limite). Qui i mancanti sono concentrati in alcune giornate di raccolta: nessuno dei due metodi è stato verificato in questo scenario

*Codice e verifiche*
- `configs/config.yaml`: `imputation.method: MissForest`
- `tests/test_preprocess.py`: nuovo test con i numeri del Passo 8 (la regola in due fasi sceglie MissForest); l'imputer di config è un `IterativeImputer` con `ExtraTreesRegressor` → **20 test verdi**
- figure `analytics/preprocessing/04_imputation_*.png` rigenerate, MissForest in blu

**Passo 10 — categoriche come predittori dell'imputazione e conferma di MissForest**

*Correzione della pipeline*
- fino al Passo 9 l'imputer numerico vedeva solo le 60 numeriche: `DM`, `DMhis`, `Gender`, `HypertenHis` e le altre categoriche non venivano mai usate per stimare i valori mancanti. Contraddiceva l'argomento principale (una `HbA1c` mancante va stimata sapendo se il soggetto è diabetico) e il punto di forza del MissForest originale, i dati misti (Stekhoven & Bühlmann 2012)
- ora `build_preprocessor` lavora in due passi: (1) numeriche standardizzate, categoriche imputate con la moda; (2) imputer su tutte le 74 colonne. Le categoriche, già complete, non vengono modificate (`skip_complete` per MICE e MissForest; KNN e mediana riempiono solo i NaN) ma fanno da predittori
- la modifica vale per tutti e 4 i candidati, quindi il confronto resta equo
- verifiche: nessun NaN in uscita, categoriche ancora intere, l'imputer riceve 74 colonne (test aggiornato)
- limite: le categoriche mancanti restano imputate con la moda (NA ≤ 13%, `Tea`), perché `IterativeImputer` usa un solo regressore per tutte le colonne e non può classificarle come fa il MissForest originale

*Risultati (5 fold sul training, media ± errore standard)*

| metodo | RMSE ↓ (main) | PR-AUC ↑ (main) | RMSE ↓ (no_consequence) | PR-AUC ↑ (no_consequence) | tempo (main) |
|---|---|---|---|---|---|
| mediana | 1,038 ± 0,047 | 0,252 ± 0,030 | 1,038 ± 0,046 | 0,245 ± 0,028 | 1 s |
| KNN | 0,793 ± 0,057 | 0,259 ± 0,032 | 0,788 ± 0,048 | 0,248 ± 0,029 | 11 s |
| MICE | 1,252 ± 0,135 | 0,257 ± 0,032 | 0,950 ± 0,095 | 0,245 ± 0,029 | 70 s |
| MissForest | **0,734** ± 0,066 | 0,257 ± 0,031 | **0,729** ± 0,053 | 0,246 ± 0,029 | 305 s |

*Esito della regola in due fasi*
- set `main` → **KNN**: RMSE 0,7932 entro la soglia di 0,7996 (MissForest 0,7337 + 0,0659)
- set `no_consequence` → **MissForest**: KNN 0,7875 oltre la soglia di 0,7826
- al Passo 8 KNN era fuori soglia per 0,0013, ora è dentro per 0,0064: una modifica piccola e corretta della pipeline sposta l'esito. Sul set principale la regola **non discrimina** fra KNN e MissForest: è un pareggio

*Confronto appaiato (stessi fold, stessi valori nascosti), KNN − MissForest*

| | set `main` | set `no_consequence` |
|---|---|---|
| RMSE | +0,060 ± 0,010 (t ≈ 6), MissForest migliore in 5 fold su 5 | +0,058 ± 0,005 (t ≈ 11), 5 su 5 |
| MAE | +0,064 ± 0,002 (t ≈ 44), 5 su 5 | +0,069 ± 0,004 (t ≈ 19), 5 su 5 |
| PR-AUC | +0,0014 ± 0,0015 (a favore di KNN, non significativo) | +0,0018 ± 0,0009 (a favore di KNN, al limite) |

- il risultato è **stabile** rispetto al Passo 8: MissForest ricostruisce meglio in tutti i fold di entrambi i set, in entrambe le esecuzioni
- a valle KNN ha una PR-AUC appena più alta, non significativa; il t appaiato sui fold di CV è già ottimista (Nadeau & Bengio 2003)

*Decisione: resta MissForest, con motivazione rivista*
- la regola in due fasi usa l'errore standard non appaiato e sul set principale è al margine: non separa i due metodi. Quando due metodi sono valutati sugli stessi fold il confronto corretto è quello appaiato (Nadeau & Bengio 2003), ed è stabile e netto a favore di MissForest
- restano valide le motivazioni del Passo 9: letteratura (punto 2), Fase B (punto 3), coerenza con il Passo 7 (punto 4), un solo imputer per le due fasi (punto 6)
- con MissForest i due set usano lo stesso metodo, ed è anche quello che la regola sceglie nel set `no_consequence`
- **da dichiarare nella tesi, senza ammorbidire**:
  1. la regola in due fasi è stata introdotta dopo i primi risultati (Passo 6)
  2. sul set principale finale la regola dà KNN, per un margine piccolo
  3. la decisione per MissForest si basa sul confronto appaiato e sulle esigenze della Fase B, un criterio scelto dopo aver visto i dati (Gelman & Loken 2014)
- il rischio di questa scelta è limitato: a valle i metodi sono equivalenti, quindi i risultati della Fase A non dipendono dall'imputer
- verifica prevista in Fase B: ripetere una tecnica (es. SMOTE) con KNN come analisi di sensibilità, per misurare se la scelta dell'imputer conta davvero dove ci aspettiamo che conti

*Codice e verifiche*
- `src/data/preprocess.py`: `build_preprocessor` in due passi (sopra)
- `src/data/imputation.py`: l'errore di ricostruzione si misura sulle sole numeriche dell'output; docstring aggiornata alla regola in due fasi
- `tests/test_preprocess.py`: l'imputer riceve tutte le 74 colonne; nuovo test con i numeri di questo passo (la regola dà KNN su `main` e MissForest su `no_consequence`) → **21 test verdi**
- figure `analytics/preprocessing/04_imputation_*.png` rigenerate

**Figure del preprocessing** (rigenerare con `python -m src.analytics.preprocessing_report`, dopo `python -m src.data.imputation`):

| file | cosa mostra |
|---|---|
| `analytics/preprocessing/01_column_map.png` | destinazione delle 190 colonne: 74 in input, 116 escluse, per motivo |
| `analytics/preprocessing/02_univariate_auc.png` | AUC univariato delle 12 feature migliori accanto alle colonne renali escluse, soglia di allarme 0,75 |
| `analytics/preprocessing/03_missing_values.png` | % di NA per colonna (≥ 1%), soglia del 15%, in grigio le 5 escluse |
| `analytics/preprocessing/04_imputation_main.png` / `_no_consequence.png` | confronto dei 4 imputer sui due criteri |

---

## Indice delle figure

Rigenerare con `python -m src.analytics.dataset_overview`, `python -m src.analytics.split_report` e (dopo `python -m src.data.imputation`) `python -m src.analytics.preprocessing_report`.

| file | cosa mostra | sezione |
|------|-------------|---------|
| `analytics/dataset/01_kdigo_heatmap.png` | soggetti in ogni cella della heatmap KDIGO (G × A), colorate per livello di rischio | Disegno della tesi → KDIGO |
| `analytics/dataset/02_target_by_diabetes.png` | prevalenza del target in diabetici / non diabetici e composizione dei positivi | Disegno della tesi → Target |
| `analytics/dataset/03_missing_values.png` | variabili con più valori mancanti | Preprocessing |
| `analytics/dataset/04_unknown_code_9.png` | occorrenze del codice 9 | Preprocessing |
| `analytics/dataset/05_age_by_target.png` | età dei positivi vs negativi | Disegno della tesi → Target |
| `analytics/dataset/06_egfr_comparison.png` | colonna `GFR` del dataset vs eGFR ricalcolato CKD-EPI 2021 | Punti da verificare → formula eGFR |
| `analytics/split/01_prevalence_train_test.png` | proporzioni in train e test | Split → Scelta finale |
| `analytics/split/02_smd_balance.png` | SMD train vs test con soglia 0,1 | Split → SMD |
| `analytics/split/03_kdigo_levels_train_test.png` | quota di ogni livello KDIGO finita nel test | Split → Scelta finale |
| `analytics/split/04_stratification_simulation.png` | 1000 split: non stratificati (sx) e stratificati solo sul target (dx) | Split → Perché stratificato / livello KDIGO |
| `analytics/split/05_test_size_events.png` | positivi in train e test con 80/20, 75/25, 70/30 | Split → Perché 75/25 |
| `analytics/preprocessing/01_column_map.png` | destinazione delle 190 colonne per gruppo | Preprocessing → Passo 1 |
| `analytics/preprocessing/02_univariate_auc.png` | AUC univariato: feature in input vs colonne renali | Preprocessing → Passo 2 |
| `analytics/preprocessing/03_missing_values.png` | % di NA per colonna con soglia 15% | Preprocessing → Passo 2 |
| `analytics/preprocessing/04_imputation_*.png` | confronto dei 4 imputer (RMSE e PR-AUC) | Preprocessing → Passi 5–6 |

## Punti da verificare
- [x] **formula dell'eGFR** — verificato: la colonna `GFR` **non coincide** con nessuna formula standard
  - provate: CKD-EPI 2009 (Levey et al. 2009), CKD-EPI 2021 (Inker et al. 2021), MDRD (Levey et al. 1999), MDRD cinese (Ma et al. 2006), Cockcroft-Gault (Cockcroft & Gault 1976) → differenza mediana da 6 a 49 unità, 0% di corrispondenze esatte
  - valori poco plausibili: `GFR` massimo 562 (le equazioni per eGFR non superano ~150–160)
  - decisione: **ricalcolare eGFR con CKD-EPI 2021** (senza coefficiente etnico; Inker et al. 2021, raccomandata da KDIGO 2024) da `SCRE` (µmol/L → mg/dl dividendo per 88,4), `Age`, `Gender`
  - effetto: eGFR < 60 passa da 119 a 91 soggetti; positivi da 586 a 567; 29 soggetti < 60 con `GFR` ma ≥ 60 con CKD-EPI, 1 nel verso opposto
- [x] **codifica di `Gender`** — verificata: dizionario riporta 0/1 ma i dati usano 1/2. Deduzione confermata dalla fisiologia: 1 = maschio, 2 = femmina (creatinina mediana 76,7 vs 54,9 µmol/L; emoglobina mediana 155 vs 135 g/L)
- [x] **unità di `UMAUCR`** — compatibile con mg/g: A2 (30–300) 459 soggetti, A3 (> 300) 57; in mg/mmol la soglia 30 corrisponderebbe a 300 mg/g e i positivi sarebbero implausibilmente molti
  - nota: `UMAUCR` = `UmALB` / `UCRE` × costante fissa (rapporto costante in tutto il dataset) → calcolata dagli autori in modo coerente; l'unità di `UCRE` non è documentata
- [x] **definizione esatta di DM** — risolto: `DM = 1` corrisponde esattamente alla condizione `DMhis == 1 | FPG >= 7.0 | PG2h >= 11.1` (0 discordanze su tutta la coorte): sono i criteri OMS su glicemia a digiuno e da carico (WHO 1999). La definizione usata nello studio originale non includeva l'HbA1c (criterio ≥ 6,5% introdotto da WHO 2011); questo spiega perfettamente i 109 soggetti con `HbA1c >= 6.5%` ma `DM = 0`.
- [x] **codice 9 e comorbilità** — risolto: il dizionario dichiara "9 = sconosciuto". Sia le comorbilità sia il questionario diabete superano la soglia del 15% di missingness (rispettivamente 20–25% e ~75%) con segnale nullo (10,1% vs 9,1% di prevalenza). Vengono esclusi in blocco: nessuna colonna con codice 9 entra nel modello.
- [x] **artefatto SPSS `ALT.1`** — verificato: la colonna duplicata `ALT.1` corrisponde a $\sqrt{\text{ALT}}$ con imputazione impropria a 0 dei valori mancanti; esclusa.
- [x] linee guida KDIGO 2026 lette (diabete e CKD, AKI/AKD, anemia) → conseguenze nella sezione "Cosa dicono le linee guida KDIGO"
- [ ] della linea guida KDIGO 2022 (diabete e CKD, Kidney International 102, suppl. 5S; KDIGO 2022a) in `papers/` ci sono solo le pagine introduttive → la bozza 2026 la aggiorna, basta citare quella (verificando se nel frattempo è uscita la versione finale)
- [ ] manca la linea guida KDIGO 2024 sulla **valutazione e gestione della CKD** (Kidney International 105, suppl. 4S): è il riferimento primario per definizione di CKD, equazioni eGFR (CKD-EPI 2021) e categorie G/A → procurarla e citarla

## Da fare
- [x] aggiornare config, split, test e figure al target KDIGO (stratificazione livello KDIGO × DM)
- [x] analisi di tutte le 190 variabili, dizionario PDF e mappa delle esclusioni
- [x] definizione della strategia di imputazione (confronto 4 metodi in CV con 1-SE rule)
- [x] aggiornare `configs/config.yaml` con la lista definitiva di feature ammesse ed escluse (sezione `features`)
- [x] modulo di preprocessing (`src/data/preprocess.py`): selezione, codifiche, preprocessor (imputazione + scaling) stimato solo sul train
- [x] confronto dei metodi di imputazione (`src/data/imputation.py`) e test (`tests/test_preprocess.py`)
- [x] fissare in config il metodo di imputazione scelto (MissForest, Passo 9; prima KNN, Passo 6)
- [ ] Fasi A e B: stimare l'imputer una volta per fold e salvare su disco i fold imputati, per ogni set di feature
- [ ] modelli da scegliere (baseline, regressione logistica regolarizzata, Random Forest, Gradient Boosting). Riferimenti: Random Forest (Breiman 2001), Gradient Boosting (Friedman 2001); sui dati tabellari i metodi ad albero restano competitivi o superiori al deep learning (Grinsztajn et al. 2022; Ye et al. 2025), anche in ambito clinico e con classi sbilanciate (Yildiz & Kalayci 2025; Brima & Atemkeng 2026; Moulaei et al. 2024)
- [ ] stesso test set congelato per tutti i confronti e le tecniche di data augmentation

### Da fare all'inizio della Fase A
- [ ] stimare l'imputer (MissForest) una volta per fold e salvare su disco i fold imputati, per ogni set di feature, da riusare in tutte le combinazioni di Fase A e B
- [ ] procurare il PDF della linea guida KDIGO 2024 sulla CKD (Kidney Int 105(4S), doi:10.1016/j.kint.2023.10.018) e ricontrollare le citazioni che la usano
- [ ] scegliere l'implementazione del Gradient Boosting: XGBoost / LightGBM / CatBoost, usati nei paper di `papers/best_models/` (Yildiz & Kalayci 2025; Brima & Atemkeng 2026), oppure `HistGradientBoostingClassifier` di scikit-learn
- [ ] aggiungere a `requirements.txt` e `requirements-lock.txt` le librerie necessarie (modelli; per la Fase B `imbalanced-learn` e `ctgan`)

---

## Bibliografia

Citazioni nel testo in formato autore-anno. ✅ = PDF in `papers/`, metadati letti dal file. Tutte le voci con DOI sono state verificate su Crossref il 18/09/2026 (titolo, autori, rivista, volume, fascicolo, pagine, anno). Senza DOI e quindi non verificabili su Crossref: linee guida WHO, bozze KDIGO 2026, Platt 1999 (capitolo di libro), Hastie et al. 2009 (libro), Xu et al. 2019 e Lundberg & Lee 2017 (atti NeurIPS, indicato l'identificativo arXiv).

### Dataset
- **Li et al. 2026** — Li J. et al. *A bimodal dataset for diabetes research.* Scientific Data 13 (2026). doi:10.1038/s41597-026-06923-y. Dati: Zenodo, doi:10.5281/zenodo.18270337

### Linee guida cliniche
- **KDIGO 2024** — KDIGO CKD Work Group. *KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease.* Kidney Int 105(4S):S117–S314 (2024) — *da procurare (vedi "Punti da verificare")*. doi:10.1016/j.kint.2023.10.018
- **KDIGO 2022a** — KDIGO Diabetes Work Group. *KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease.* Kidney Int 102(5S):S1–S127 (2022) — ✅ solo pagine introduttive in `papers/kdigo/`. doi:10.1016/j.kint.2022.06.008
- **KDIGO 2022b** — KDIGO Hepatitis C Work Group. *KDIGO 2022 Clinical Practice Guideline for the Prevention, Diagnosis, Evaluation, and Treatment of Hepatitis C in Chronic Kidney Disease.* Kidney Int 102(6S):S129–S205 (2022). doi:10.1016/j.kint.2022.07.013
- **KDIGO 2026a** — *KDIGO 2026 Clinical Practice Guideline for Diabetes and Chronic Kidney Disease (Chapter 1, 2 & 4 update).* Public review draft, marzo 2026 ✅
- **KDIGO 2026b** — *KDIGO 2026 Clinical Practice Guideline for Acute Kidney Injury (AKI) and Acute Kidney Disease (AKD).* Public review draft, marzo 2026 ✅
- **KDIGO 2026c** — *KDIGO 2026 Clinical Practice Guideline for the Management of Anemia in Chronic Kidney Disease.* Kidney Int 109(1S):S1–S99 (2026) ✅
- **WHO 1999** — World Health Organization. *Definition, diagnosis and classification of diabetes mellitus and its complications.* WHO/NCD/NCS/99.2 (1999)
- **WHO 2011** — World Health Organization. *Use of glycated haemoglobin (HbA1c) in the diagnosis of diabetes mellitus.* WHO/NMH/CHP/CPM/11.1 (2011)

### Equazioni e indici clinici
- **Inker et al. 2021** — Inker L.A., Eneanya N.D., Coresh J. et al. *New creatinine- and cystatin C-based equations to estimate GFR without race.* N Engl J Med 385(19):1737–1749 (2021). doi:10.1056/NEJMoa2102953
- **Levey et al. 2009** — Levey A.S., Stevens L.A., Schmid C.H. et al. *A new equation to estimate glomerular filtration rate.* Ann Intern Med 150(9):604–612 (2009). doi:10.7326/0003-4819-150-9-200905050-00006
- **Levey et al. 1999** — Levey A.S., Bosch J.P., Lewis J.B. et al. *A more accurate method to estimate glomerular filtration rate from serum creatinine: a new prediction equation.* Ann Intern Med 130(6):461–470 (1999). doi:10.7326/0003-4819-130-6-199903160-00002
- **Ma et al. 2006** — Ma Y.C., Zuo L., Chen J.H. et al. *Modified glomerular filtration rate estimating equation for Chinese patients with chronic kidney disease.* J Am Soc Nephrol 17(10):2937–2944 (2006). doi:10.1681/ASN.2006040368
- **Cockcroft & Gault 1976** — Cockcroft D.W., Gault M.H. *Prediction of creatinine clearance from serum creatinine.* Nephron 16(1):31–41 (1976). doi:10.1159/000180580
- **Matthews et al. 1985** — Matthews D.R., Hosker J.P., Rudenski A.S. et al. *Homeostasis model assessment: insulin resistance and β-cell function from fasting plasma glucose and insulin concentrations in man.* Diabetologia 28(7):412–419 (1985). doi:10.1007/BF00280883
- **Gutt et al. 2000** — Gutt M., Davis C.L., Spitzer S.B. et al. *Validation of the insulin sensitivity index (ISI0,120): comparison with other measures.* Diabetes Res Clin Pract 47(3):177–184 (2000). doi:10.1016/S0168-8227(99)00116-3
- **Sterling et al. 2006** — Sterling R.K., Lissen E., Clumeck N. et al. *Development of a simple noninvasive index to predict significant fibrosis in patients with HIV/HCV coinfection.* Hepatology 43(6):1317–1325 (2006). doi:10.1002/hep.21178

### Disegno dello studio, validazione e reporting
- **Collins, Ogundimu & Altman 2016** — Collins G.S., Ogundimu E.O., Altman D.G. *Sample size considerations for the external validation of a multivariable prognostic model: a resampling study.* Stat Med 35(2):214–226 (2016). doi:10.1002/sim.6787
- **Riley et al. 2021** — Riley R.D., Debray T.P.A., Collins G.S. et al. *Minimum sample size for external validation of a clinical prediction model with a binary outcome.* Stat Med 40(19):4230–4251 (2021). doi:10.1002/sim.9025
- **Steyerberg et al. 2001** — Steyerberg E.W., Harrell F.E., Borsboom G.J. et al. *Internal validation of predictive models: efficiency of some procedures for logistic regression analysis.* J Clin Epidemiol 54(8):774–781 (2001). doi:10.1016/S0895-4356(01)00341-9
- **Steyerberg & Harrell 2016** — Steyerberg E.W., Harrell F.E. *Prediction models need appropriate internal, internal-external, and external validation.* J Clin Epidemiol 69:245–247 (2016). doi:10.1016/j.jclinepi.2015.04.005
- **Austin 2009** — Austin P.C. *Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples.* Stat Med 28(25):3083–3107 (2009). doi:10.1002/sim.3697
- **Collins et al. 2024** — Collins G.S., Moons K.G.M., Dhiman P. et al. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods.* BMJ 385:e078378 (2024). doi:10.1136/bmj-2023-078378
- **Kaufman et al. 2012** — Kaufman S., Rosset S., Perlich C., Stitelman O. *Leakage in data mining: formulation, detection, and avoidance.* ACM Trans Knowl Discov Data 6(4):15 (2012). doi:10.1145/2382577.2382579
- **Kapoor & Narayanan 2023** — Kapoor S., Narayanan A. *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9):100804 (2023). doi:10.1016/j.patter.2023.100804
- **Hastie, Tibshirani & Friedman 2009** — Hastie T., Tibshirani R., Friedman J. *The Elements of Statistical Learning*, 2ª ed. Springer (2009). §7.10 (regola di 1 errore standard), §10.7 (proprietà degli alberi)
- **Gelman & Loken 2014** — Gelman A., Loken E. *The statistical crisis in science.* American Scientist 102(6):460–465 (2014). doi:10.1511/2014.111.460
- **Nadeau & Bengio 2003** — Nadeau C., Bengio Y. *Inference for the generalization error.* Mach Learn 52(3):239–281 (2003). doi:10.1023/A:1024068626366

### Dati mancanti e imputazione
- **Waljee et al. 2013** — Waljee A.K., Mukherjee A., Singal A.G. et al. *Comparison of imputation methods for missing laboratory data in medicine.* BMJ Open 3(8):e002847 (2013). doi:10.1136/bmjopen-2013-002847
- **Tiwaskar et al. 2025** — Tiwaskar S., Rashid M., Gokhale P. *Impact of machine learning-based imputation techniques on medical datasets: a comparative analysis.* Multimed Tools Appl 84(9):5905–5925 (2025). doi:10.1007/s11042-024-19103-0 ✅
- **Hameed & Ali 2025** — Hameed W.M., Ali N.A. *Enhancing accuracy of diabetes diagnosis system using instance-wise feature importance aware imputer and TabNet deep neural classifier.* Expert Syst Appl 270:126498 (2025) ✅. doi:10.1016/j.eswa.2025.126498
- **Alnowaiser 2024** — Alnowaiser K. *Improving healthcare prediction of diabetic patients using KNN imputed features and tri-ensemble model.* IEEE Access 12:16783–16793 (2024). doi:10.1109/ACCESS.2024.3359760 ✅
- **Troyanskaya et al. 2001** — Troyanskaya O., Cantor M., Sherlock G. et al. *Missing value estimation methods for DNA microarrays.* Bioinformatics 17(6):520–525 (2001). doi:10.1093/bioinformatics/17.6.520
- **van Buuren & Groothuis-Oudshoorn 2011** — van Buuren S., Groothuis-Oudshoorn K. *mice: Multivariate Imputation by Chained Equations in R.* J Stat Softw 45(3):1–67 (2011). doi:10.18637/jss.v045.i03
- **van Buuren 2018** — van Buuren S. *Flexible Imputation of Missing Data*, 2ª ed. CRC Press (2018). doi:10.1201/9780429492259
- **Stekhoven & Bühlmann 2012** — Stekhoven D.J., Bühlmann P. *MissForest: non-parametric missing value imputation for mixed-type data.* Bioinformatics 28(1):112–118 (2012). doi:10.1093/bioinformatics/btr597
- **Sterne et al. 2009** — Sterne J.A.C., White I.R., Carlin J.B. et al. *Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls.* BMJ 338:b2393 (2009). doi:10.1136/bmj.b2393
- **Sisk et al. 2023** — Sisk R., Sperrin M., Peek N., van Smeden M., Martin G.P. *Imputation and missing indicators for handling missing data in the development and deployment of clinical prediction models: a simulation study.* Stat Methods Med Res 32(8):1461–1477 (2023). doi:10.1177/09622802231165001

### Classi sbilanciate, augmentation e calibrazione
- **Chawla et al. 2002** — Chawla N.V., Bowyer K.W., Hall L.O., Kegelmeyer W.P. *SMOTE: synthetic minority over-sampling technique.* J Artif Intell Res 16:321–357 (2002). Include SMOTE-NC per variabili miste. doi:10.1613/jair.953
- **Xu et al. 2019** — Xu L., Skoularidou M., Cuesta-Infante A., Veeramachaneni K. *Modeling tabular data using conditional GAN.* NeurIPS 32 (2019). arXiv:1907.00503
- **Santos et al. 2018** — Santos M.S., Soares J.P., Abreu P.H. et al. *Cross-validation for imbalanced datasets: avoiding overoptimistic and overfitting approaches.* IEEE Comput Intell Mag 13(4):59–76 (2018). doi:10.1109/MCI.2018.2866730
- **van den Goorbergh et al. 2022** — van den Goorbergh R., van Smeden M., Timmerman D., Van Calster B. *The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression.* J Am Med Inform Assoc 29(9):1525–1534 (2022). doi:10.1093/jamia/ocac093
- **Platt 1999** — Platt J.C. *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.* In: Advances in Large Margin Classifiers, MIT Press, 61–74 (1999)
- **Zadrozny & Elkan 2002** — Zadrozny B., Elkan C. *Transforming classifier scores into accurate multiclass probability estimates.* Proc. KDD 2002, 694–699. doi:10.1145/775047.775151
- **Saito & Rehmsmeier 2015** — Saito T., Rehmsmeier M. *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLoS ONE 10(3):e0118432 (2015). doi:10.1371/journal.pone.0118432

### Modelli e interpretabilità
- **Breiman 2001** — Breiman L. *Random forests.* Mach Learn 45(1):5–32 (2001). doi:10.1023/A:1010933404324
- **Friedman 2001** — Friedman J.H. *Greedy function approximation: a gradient boosting machine.* Ann Stat 29(5):1189–1232 (2001). doi:10.1214/aos/1013203451
- **Grinsztajn et al. 2022** — Grinsztajn L., Oyallon E., Varoquaux G. *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 35, Datasets and Benchmarks Track (2022). doi:10.52202/068431-0037
- **Ye et al. 2025** — Ye H.-J., Liu S.-Y., Cai H.-R., Zhou Q.-L., Zhan D.-C. *A closer look at deep learning methods on tabular datasets.* arXiv:2407.00956v4 (2025) ✅
- **Yildiz & Kalayci 2025** — Yildiz A.Y., Kalayci A. *Gradient boosting decision trees on medical diagnosis over tabular data.* IEEE ICAD 2025. doi:10.1109/ICAD65464.2025.11114069 ✅
- **Brima & Atemkeng 2026** — Brima Y., Atemkeng M. *An empirical study of machine learning robustness and scalability for imbalanced tabular clinical data in emergency and critical care.* Sci Rep 16:18004 (2026). doi:10.1038/s41598-026-56413-9 ✅
- **Moulaei et al. 2024** — Moulaei K., Afshari L., Moulaei R. et al. *Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models.* Sci Rep 14:31392 (2024). doi:10.1038/s41598-024-82931-5 ✅
- **Shahbazi & Azadeh-Fard 2025** — Amiri Shahbazi M., Azadeh-Fard N. *Hierarchical data modeling: a systematic comparison of statistical, tree-based, and neural network approaches.* Mach Learn Appl 21:100688 (2025) ✅. doi:10.1016/j.mlwa.2025.100688
- **Lundberg & Lee 2017** — Lundberg S.M., Lee S.-I. *A unified approach to interpreting model predictions.* NeurIPS 30 (2017). arXiv:1705.07874

### Statistica per la valutazione clinica
- **Terpstra 1952** — Terpstra T.J. *The asymptotic normality and consistency of Kendall's test against trend, when ties are present in one ranking.* Indagationes Mathematicae 14 (Proc. KNAW, Serie A, 55):327–333 (1952). doi:10.1016/S1385-7258(52)50043-X
- **Jonckheere 1954** — Jonckheere A.R. *A distribution-free k-sample test against ordered alternatives.* Biometrika 41(1/2):133–145 (1954). doi:10.2307/2333011
- **Cohen 1968** — Cohen J. *Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit.* Psychol Bull 70(4):213–220 (1968). doi:10.1037/h0026256
