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
  2. classe 0/1 (soglia con sensibilità 0,90 sulle previsioni out-of-fold del training; vedi "Valutazione: scelte fissate prima dei risultati")
  3. fascia di rischio (fascia 1–4, con le proporzioni dei livelli KDIGO nel training; decisione del 19/09/2026, prima erano previste 3 fasce)

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
- **provenienza del dataset (verifica del 22/09/2026 sul testo completo di Li J et al. 2026, *Sci Data* 13:652)**: i dati vengono dal reparto di Diabetologia ed Endocrinologia dello Shanghai Sixth People's Hospital, febbraio-aprile 2012; gli autori non li descrivono come screening di popolazione e non documentano né il tipo di campione urinario per l'ACR né le unità di `UCRE` e `UmALB`. Nelle sezioni scritte prima di questa data, "screening di popolazione" e "popolazione generale di screening" vanno letti come "coorte ospedaliera, in maggioranza senza diabete noto": l'utilità del modello va dimostrata in popolazioni di screening. Dettaglio in `docs/verifica_stato_arte.md`

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
- [ ] pesi di classe (apprendimento sensibile al costo: gli errori sui positivi pesano circa 9 volte di più; Elkan 2001; Brima & Atemkeng 2026)
- [ ] analisi di sensibilità dell'imputer: una tecnica (es. SMOTE) ripetuta con KNN al posto di MissForest (Passo 10)

Per ogni tecnica: gli stessi 5 modelli della Fase A (sezione "Fase A — modelli e protocollo"), stesse 4 domande della Fase A, probabilità ricalibrate (van den Goorbergh et al. 2022), confronto KDIGO solo su soggetti reali.

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
  - (a) **errore di ricostruzione**: nel fold di validazione si nasconde il 10% dei valori osservati, l'imputer stimato sul fold di training li ricostruisce, si misurano RMSE e MAE in unità standardizzate. Si nascondono valori **solo nelle colonne che hanno NA nel training**, cioè quelle che l'imputer deve davvero imparare a riempire (protocollo di Tiwaskar et al. 2025 e Hameed & Ali 2025)
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

Perché la regola è cambiata: quella originale guardava solo il criterio (b) e ignorava il criterio (a), che faceva già parte del protocollo (ripreso da Tiwaskar et al. 2025) ed è l'unico che distingue i metodi. Il criterio (a) conta per la Fase B, dove SMOTE e CTGAN lavorano sulla distribuzione congiunta delle variabili.

Esito della regola in due fasi:
- set `main` → **KNN**: RMSE 0,774, entro la soglia di 0,789 (MissForest 0,724 + 0,065); PR-AUC 0,253, entro la soglia di 0,224
- set `no_consequence` → la regola sceglierebbe MissForest (KNN 0,742 supera la soglia di 0,697)
- **un solo imputer per entrambi i set, deciso sul set principale**: l'analisi di sensibilità deve differire solo per le 7 variabili tolte. Se cambiasse anche l'imputer, una differenza di risultati non sarebbe più attribuibile alle variabili

Perché non la mediana:
- RMSE ~1,0 in unità standardizzate: non ricostruisce nulla del valore mancante (equivale a prevedere la media)
- attenua varianza e correlazioni (Tiwaskar et al. 2025) e crea masse puntiformi: i 397 soggetti senza `HbA1c` ricevono tutti 5,5, compresi 28 diabetici (i diabetici con valore misurato hanno in media 6,57)
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

## Fase A — modelli e protocollo (decisione del 19/09/2026)

### I cinque modelli (gli stessi in Fase A e in Fase B)

| # | modello | famiglia e ruolo | implementazione | riferimenti |
|---|---|---|---|---|
| 1 | classificatore di maggioranza | soglia minima; controllo di coerenza della pipeline | `DummyClassifier(strategy="prior")` | PR-AUC attesa = prevalenza (Saito & Rehmsmeier 2015) |
| 2 | regressione logistica SCORED | punteggio clinico: predittori del punteggio SCORED disponibili nello screening, coefficienti ristimati | `LogisticRegression` senza penalizzazione su `Age`, `Gender`, `HGB`, `Bpsys`, `DM` | Bang et al. 2007; Echouffo-Tcheugui & Kengne 2012 |
| 3 | regressione logistica penalizzata | modello lineare di riferimento, tutte le 74 feature | `LogisticRegression`, penalizzazione L1 / L2 / elastic net e sua forza scelte in CV interna | Christodoulou et al. 2019; Pavlou et al. 2015; Tibshirani 1996; Hoerl & Kennard 1970; Zou & Hastie 2005 |
| 4 | Random Forest | ensemble ad albero, bagging | `RandomForestClassifier` | Breiman 2001; Yildiz & Kalayci 2025; Brima & Atemkeng 2026 |
| 5 | XGBoost | ensemble ad albero, gradient boosting | `XGBClassifier` | Chen & Guestrin 2016; Brima & Atemkeng 2026; Yildiz & Kalayci 2025 |

Stessi modelli nelle due fasi: il confronto fra Fase A e Fase B misura solo l'effetto delle tecniche di bilanciamento.

### Motivazione (testo per la tesi)
I modelli sono stati selezionati per famiglia teorica e per ruolo metodologico, dai modelli lineari agli ensemble ad albero, e sono gli stessi nelle due fasi sperimentali, così che il confronto misuri solo l'effetto delle tecniche di bilanciamento. Come termini di confronto si utilizzano un classificatore di maggioranza (Dummy), che fissa la soglia minima di prestazione e fa da controllo di coerenza della pipeline, e una regressione logistica con i predittori del punteggio clinico SCORED (Bang et al. 2007) disponibili nello screening, ristimata sui dati dello studio; SCORED è il modello di rischio per la malattia renale cronica più validato esternamente secondo una revisione sistematica (Echouffo-Tcheugui & Kengne 2012). Il modello lineare di riferimento è la regressione logistica penalizzata (L1, L2 o elastic net): nei modelli di predizione clinica la regressione logistica non è risultata inferiore ai metodi di machine learning (Christodoulou et al. 2019), e la penalizzazione è raccomandata quando gli eventi per variabile sono pochi (Peduzzi et al. 1996; Pavlou et al. 2015; Riley et al. 2020), come in questo studio (circa 5,7). Per valutare il contributo delle relazioni non lineari e delle interazioni si impiegano due ensemble ad albero, uno per famiglia: Random Forest per il bagging (Breiman 2001) e XGBoost per il gradient boosting (Chen & Guestrin 2016), entrambi competitivi su dati clinici tabellari sbilanciati (Brima & Atemkeng 2026). Fra le implementazioni di gradient boosting le differenze riportate sono piccole (Yildiz & Kalayci 2025); XGBoost è stato scelto perché è risultato il migliore nel contesto più vicino a quello dello studio, i dati clinici sbilanciati (Brima & Atemkeng 2026). Reti neurali e modelli tabellari pre-addestrati sono stati esclusi: sui dati tabellari non mostrano un vantaggio sistematico sugli ensemble ad albero (Grinsztajn et al. 2022; Yildiz & Kalayci 2025; Ye et al. 2025) e sono meno interpretabili.

### Regressione logistica SCORED: predittori e adattamenti
- SCORED (Bang et al. 2007) è un punteggio per individuare la malattia renale occulta nella popolazione generale. Elementi, confermati dall'abstract originale e dalla tabella della revisione di Echouffo-Tcheugui & Kengne 2012: età, sesso femminile, ipertensione, diabete, arteriopatia periferica, storia di malattia cardiovascolare, scompenso cardiaco, proteinuria, anemia
- sviluppato su 8.530 adulti della survey NHANES 1999–2002 (Stati Uniti), studio trasversale di popolazione, target eGFR < 60 (MDRD). AUC 0,88 nella validazione interna e 0,71 nella validazione esterna (coorte ARIC); con soglia ≥ 4 punti: sensibilità 92%, specificità 68%, valore predittivo positivo 18%, negativo 99% (Bang et al. 2007)
- **perché proprio SCORED**: secondo la revisione sistematica dei modelli di rischio per la malattia renale cronica è il modello più affidabile, perché è il più validato esternamente, con discriminazione ragionevole (Echouffo-Tcheugui & Kengne 2012). È anche l'unico pensato esplicitamente per lo screening della popolazione generale, come il nostro dataset
- la stessa revisione indica come predittori più frequenti nei modelli di rischio età, sesso, BMI, diabete, pressione sistolica, creatinina, proteinuria e albumina o proteine sieriche (Echouffo-Tcheugui & Kengne 2012): usare la pressione sistolica misurata al posto dell'anamnesi di ipertensione è coerente con la letteratura
- predittori usati:
  - `Age`, continua invece delle fasce
  - `Gender`
  - `HGB`, continua invece del flag di anemia (`Anemia` è esclusa come derivata a soglia)
  - `Bpsys`, la pressione misurata, invece dell'anamnesi di ipertensione (`HypertenHis` ha solo 147 "sì" nel training)
  - `DM`
- predittori non disponibili: malattia cardiovascolare, scompenso cardiaco, arteriopatia periferica (escluse per il codice 9) e proteinuria (esame renale)
- i coefficienti sono **ristimati** sui nostri dati: **non** è una validazione esterna di SCORED e non va chiamato "punteggio validato"
- SCORED è stato costruito per eGFR < 60; il nostro target è per il 90% albuminuria → differenza da dichiarare
- set `no_consequence`: `HGB` esce, restano 4 predittori
- circa 85 eventi per variabile (425 / 5): nessuna penalizzazione necessaria

### Perché la logistica penalizzata e non quella completa senza penalizzazione
- 425 eventi nel training su 74 predittori ≈ **5,7 eventi per variabile** (circa 4,6 nei fold di training della CV esterna), sotto la soglia di 10 (Peduzzi et al. 1996). Il criterio degli eventi per variabile è una semplificazione, ma anche i criteri di Riley et al. 2020 vanno nella stessa direzione: rischio di sovradattamento
- la penalizzazione riduce i coefficienti verso zero e migliora le previsioni su dati nuovi quando gli eventi sono pochi (Pavlou et al. 2015)
- tipo di penalizzazione (L1: Tibshirani 1996; L2: Hoerl & Kennard 1970; elastic net: Zou & Hastie 2005) e forza sono iperparametri scelti nella CV interna

### Modelli esclusi e perché
- **LightGBM e CatBoost**: in Yildiz & Kalayci 2025 le posizioni medie in classifica sono vicine (LightGBM 2,6, CatBoost 3,1, XGBoost 4,4, con barre d'errore sovrapposte); nel contesto più vicino al nostro vince XGBoost (Brima & Atemkeng 2026). Un solo gradient boosting, per parsimonia. Il vantaggio di CatBoost sulle categoriche non serve: le nostre sono tutte binarie o ordinali già codificate
- **KNN e SVM**: ultimi in classifica in Yildiz & Kalayci 2025; SVM non produce probabilità in modo nativo
- **reti neurali (MLP, TabNet) e modelli pre-addestrati (TabPFN, TabICL)**: nessun vantaggio sistematico sui dati tabellari (Grinsztajn et al. 2022; Yildiz & Kalayci 2025; Ye et al. 2025), meno interpretabili. TabPFN e TabICL non addestrano i parametri, quindi si combinano male con il bilanciamento della Fase B. Possibile sviluppo futuro
- **regressione logistica completa senza penalizzazione**: vedi sopra, 5,7 eventi per variabile

### Il classificatore di maggioranza in Fase B
- le metriche di **ordinamento** non cambiano: PR-AUC sempre pari alla prevalenza, AUC sempre 0,5
- le **probabilità** invece cambiano: dopo un oversampling al 50% il modello prevede 0,5
- serve da **controllo di coerenza**: se in una tecnica la sua PR-AUC si discosta dalla prevalenza, nella pipeline c'è un errore

### Protocollo di addestramento e valutazione
- **cross-validation annidata** (Varma & Simon 2006; Cawley & Talbot 2010):
  - **5 fold esterni**, gli stessi del confronto delle imputazioni (stratificati sul livello KDIGO, seed 42): producono le previsioni sui soggetti non visti, usate per l'analisi per livello KDIGO
  - **5 fold interni** dentro ogni fold esterno, per ottimizzare gli iperparametri. Ottimizzare e valutare sugli stessi fold renderebbe le previsioni ottimiste
- **ottimizzazione**: Optuna (Akiba et al. 2019) con il campionatore TPE (Bergstra et al. 2011)
  - **stesso numero di tentativi per ogni modello**, così le differenze dipendono dai modelli e non dall'ottimizzazione (Brima & Atemkeng 2026: 100 tentativi con CV a 5 fold)
  - numero di tentativi e spazi di ricerca fissati **prima** di vedere i risultati, dopo una stima dei tempi; un eventuale budget inferiore a 100, per il costo della Fase B, va dichiarato
- **metrica di ottimizzazione**: PR-AUC (Saito & Rehmsmeier 2015), dichiarata prima
- **imputazione**: MissForest stimato sul training di ogni fold, interno ed esterno (circa 30 stime per set di feature, circa 30 minuti), una volta sola; i fold imputati vengono salvati e riusati da tutti i modelli
- **nessun peso di classe in Fase A**: è una tecnica di bilanciamento, va in Fase B
- **confronto fra modelli**: appaiato sui 5 fold esterni (Nadeau & Bengio 2003), riportato comunque, senza scegliere un modello "vincitore" prima della Fase B
- **soglia di classificazione**: scelta sulle previsioni della CV, mai sul test
- **modello finale**: iperparametri scelti con CV a 5 fold sull'intero training, riaddestramento sull'intero training, **una sola valutazione** sul test set
- **interpretazione**: SHAP per Random Forest e XGBoost (Lundberg & Lee 2017; esempio di uso clinico in Moulaei et al. 2024), coefficienti per le logistiche; attenzione alle coppie molto correlate (Passo 7), fra cui l'importanza si divide

### Protocollo fissato prima dei risultati (19/09/2026, opzione A)
Scritto in `configs/config.yaml` (sezione `phase_a`) prima di lanciare qualsiasi addestramento; i tempi misurati prima (un fold interno, nessuna metrica calcolata) servivano solo per il budget.

| elemento | scelta | fonte |
|---|---|---|
| budget | **100 tentativi** di Optuna per modello e per fold | Brima & Atemkeng 2026 |
| campionatore | TPE, seed 42 | Brima & Atemkeng 2026; Bergstra et al. 2011; Akiba et al. 2019 |
| interruzione anticipata | `MedianPruner` con i valori predefiniti di Optuna: dopo ogni fold interno, un tentativo si interrompe se la sua PR-AUC media è sotto la mediana dei tentativi precedenti allo stesso passo (dal 6° tentativo) | Brima & Atemkeng 2026 |
| metrica | PR-AUC media sui fold interni | Saito & Rehmsmeier 2015 |
| set ottimizzato | solo `main`; `no_consequence` **riusa gli iperparametri** di `main` fold per fold: l'analisi di sensibilità cambia solo le variabili | scelta nostra (opzione A) |
| modello finale | iperparametri scelti con i 5 fold esterni come CV dell'intero training, riaddestramento sull'intero training | Varma & Simon 2006; Cawley & Talbot 2010 |

Spazi di ricerca:

| modello | iperparametri e intervalli | fonte |
|---|---|---|
| logistica penalizzata | `C` 10⁻⁴–10 (scala log); `l1_ratio` 0–1 (0 = L2, 1 = L1, intermedi = elastic net); solver `saga` | Zou & Hastie 2005; Friedman et al. 2010. Il limite superiore di `C` è 10: valori più alti equivalgono alla logistica senza penalizzazione, esclusa per gli eventi per variabile |
| Random Forest | alberi 100–1000; profondità 3–25; `min_samples_split` 2–50; `min_samples_leaf` 1–20; `max_features` sqrt / log2 / 0,25 / 0,5 / 0,75; criterio gini / entropy | Brima & Atemkeng 2026, App. A6. Le frazioni di `max_features` sono una scelta nostra (l'articolo dice solo "frazioni fisse") |
| XGBoost | alberi 200–1200; learning rate 0,01–0,3; profondità 3–12; `subsample` 0,6–1,0; `colsample_bytree` 0,5–1,0; `reg_alpha` e `reg_lambda` 0–5 | Brima & Atemkeng 2026, App. A6. Learning rate in scala logaritmica: scelta nostra (l'articolo non la specifica) |
| classificatore di maggioranza, logistica SCORED | nessun iperparametro | — |

Costo stimato: circa 3,5 ore per modello ottimizzato (3 modelli), ridotte dall'interruzione anticipata dei tentativi; il set `no_consequence` costa pochi minuti (nessuna ottimizzazione).

Altre regole implementate:
- nessun peso di classe e nessun bilanciamento
- la convergenza della logistica si registra per ogni modello (campo `converged`); gli avvisi di mancata convergenza durante l'ottimizzazione non vengono mostrati
- ogni risultato si salva appena pronto (l'esecuzione si può interrompere e riprendere)
- il test set non viene letto

Codice:
- `src/models/zoo.py`: i cinque modelli e gli spazi di ricerca letti dalla config
- `src/models/phase_a.py` (`python -m src.models.phase_a`): per ogni modello e fold esterno salva `analytics/phase_a/<set>/<modello>_<fold>.json` (iperparametri, PR-AUC interna, tentativi interrotti, tempi, previsioni out-of-fold), i tentativi di Optuna in `analytics/phase_a/trials/`, i modelli finali in `models/`; alla fine `analytics/phase_a/oof_predictions.csv`
- `tests/test_phase_a.py`: 11 test su dati imputati con la mediana (veloci); tra l'altro verificano che ogni soggetto abbia esattamente una previsione out-of-fold, che il set `no_consequence` riusi gli iperparametri senza ottimizzare e che l'analisi di sensibilità modifichi solo lo spazio del suo modello

### Valutazione: scelte fissate prima dei risultati (19/09/2026)
Fissate dopo l'addestramento ma **prima di calcolare qualsiasi metrica**: dei risultati era stata letta solo la struttura (60 file, 43.500 previsioni out-of-fold = 5 modelli × 2 set × 4.350 soggetti, una per soggetto, nessun valore mancante, logistiche tutte convergenti), nessuna prestazione. Scritte anche in `configs/config.yaml` (sezione `evaluation`). Valgono identiche per ogni tecnica della Fase B.

Dati: solo le previsioni out-of-fold del training (4.350 soggetti, 425 positivi: 354 moderato, 50 alto, 21 molto alto). Il test set non viene letto; soglia e fasce stimate qui si applicano **una volta sola** al test, con il modello finale.

| elemento | scelta | motivazione e fonte |
|---|---|---|
| soglia di classificazione | la più alta con **sensibilità ≥ 0,90** sulle previsioni out-of-fold aggregate | la soglia deve riflettere le conseguenze delle decisioni, non un criterio statistico (Wynants et al. 2019): nello screening un falso negativo è un caso mancato, un falso positivo costa un ACR urinario. Punto operativo vicino a quello di SCORED (sensibilità 92% con ≥ 4 punti, Bang et al. 2007). Dipende solo dall'ordinamento delle previsioni, non dalla calibrazione: resta confrontabile in Fase B, dove il bilanciamento sposta le probabilità (van den Goorbergh et al. 2022). A parità di sensibilità complessiva, la sensibilità per livello dice se i casi gravi sono riconosciuti più degli altri (domande 3 e 5) |
| altri punti operativi | sensibilità 0,80 / 0,85 / 0,90 / 0,95, solo descrittivi | riportare più soglie (Wynants et al. 2019) |
| metriche alla soglia | precision, recall, specificità, quota di soggetti da testare | le ultime due servono a leggere il costo della soglia |
| fasce di rischio | **4 fasce** ("fascia 1–4", per non confonderle con i livelli) dai quantili delle previsioni, con le stesse proporzioni dei livelli KDIGO nel training (90,2 / 8,1 / 1,1 / 0,5%) | il kappa richiede le stesse categorie e dipende da prevalenza e differenza fra le distribuzioni marginali (Feinstein & Cicchetti 1990; Byrt et al. 1993; Sim & Wright 2005): con marginali uguali la componente di disaccordo sistematico sparisce. Basate sull'ordinamento, quindi confrontabili in Fase B |
| concordanza fasce/livelli | kappa pesato con pesi **lineari** (Cohen 1968), con tabella 4 × 4 e accordo osservato | i livelli KDIGO sono ordinali: i pesi quadratici equivalgono a un coefficiente di correlazione intraclasse, che tratta la scala come a intervalli (Fleiss & Cohen 1973) |
| AUC e PR-AUC | sulle previsioni out-of-fold aggregate (principali) e come media dei 5 fold | l'aggregato serve per l'analisi per livello e per la soglia, ma penalizza i modelli non calibrati fra un fold e l'altro (Forman & Scholz 2010): la differenza fra le due fa da diagnostica. PR-AUC stimata come average precision (Boyd et al. 2013) |
| tendenza per livello | Jonckheere-Terpstra unilaterale (il rischio cresce con il livello), approssimazione normale con correzione per i pareggi (Hollander, Wolfe & Chicken 2014). Dimensione dell'effetto: **concordanza** = quota di coppie di soggetti di livelli diversi ordinate come KDIGO (pareggi 1/2; 0,5 = nessuna tendenza) | con 4.350 soggetti il p-value è piccolo per qualsiasi modello sensato: conta la dimensione dell'effetto |
| intervalli di confidenza (95%) | AUC: DeLong et al. 1988. PR-AUC: intervallo logit con n = numero di positivi (Boyd et al. 2013). Precision, recall, specificità, sensibilità per livello: Wilson (Wilson 1927; Brown, Cai & DasGupta 2001). Kappa, probabilità medie per livello, concordanza: bootstrap percentile **stratificato sul livello KDIGO**, 2000 campioni, fasce ristimate a ogni campione (Boyd et al. 2013; Carpenter & Bithell 2000) | intervalli analitici dove esistono formule valide con pochi casi: il bootstrap copre meno del 95% quando i positivi sono pochi (Boyd et al. 2013, letto dal testo); Wilson è raccomandato per n ≤ 40 (Brown, Cai & DasGupta 2001), come i 21 "molto alto". Il bootstrap stratificato conserva le numerosità dei livelli in ogni campione. Scelta proposta da noi: l'utente non ha espresso preferenze |
| confronto fra modelli | t appaiato sui 5 fold esterni con varianza corretta (Nadeau & Bengio 2003), su PR-AUC (metrica primaria) e AUC; 10 coppie per set, p-value corretti con Holm 1979 | nessun "vincitore" prima della Fase B; 4 gradi di libertà, potenza bassa |
| analisi di sensibilità | domande 1–4 anche sul set `no_consequence`; differenza no_consequence − main per ogni modello, stesso test | — |
| classificatore di maggioranza | solo controllo di coerenza: in ogni fold AUC = 0,5 e PR-AUC = prevalenza (Saito & Rehmsmeier 2015); soglia e fasce non definite (previsione costante) | — |

Alternative scartate:
- **soglia di Youden** (Youden 1950): è il solo criterio "ottimo" coerente a pesi uguali (Perkins & Schisterman 2006), ma assume che falsi negativi e falsi positivi costino uguale, uno dei tre miti di Wynants et al. 2019
- **soglia 0,90 contro 0,80**: 0,80 riduce i soggetti da testare ma non ha un ancoraggio in letteratura; resta nella griglia descrittiva
- **p ≥ prevalenza** e **fasce a soglie di probabilità fisse**: richiedono probabilità calibrate (la calibrazione della Random Forest non è garantita) e mescolano discriminazione e calibrazione; le soglie fisse sono arbitrarie
- **3 fasce contro KDIGO ridotto a 3 livelli**: più stabile, ma perde la distinzione alto / molto alto
- **intervalli solo bootstrap** (copertura bassa con pochi positivi) o **sui 5 fold** (4 gradi di libertà, impossibili per livello, copertura bassa: Bates, Hastie & Tibshirani 2024; Boyd et al. 2013)

Limiti da dichiarare:
- gli intervalli sono condizionati ai modelli addestrati: non includono la variabilità dell'addestramento, che entra solo nei confronti fra fold (Bates, Hastie & Tibshirani 2024)
- soglia e fasce sono stimate sulle stesse previsioni su cui si valutano (un parametro ciascuna): leggero ottimismo, verificato sul test
- con la soglia a sensibilità fissata il recall sulle previsioni out-of-fold è 0,90 per costruzione: diventa informativo solo sul test
- la fascia 4 contiene circa 21 soggetti: stime instabili; il kappa resta influenzato dalla prevalenza dei livelli (90% basso)
- fuori dalle domande 1–4: calibrazione (Van Calster et al. 2019) e curve di decisione (Vickers & Elkin 2006), da riconsiderare in Fase B, dove la ricalibrazione è prevista

Codice:
- `src/models/evaluation.py` (`python -m src.models.evaluation`, circa 2 minuti): tabelle in `analytics/phase_a/evaluation/` — `q1_discrimination`, `q1_operating_points`, `q2_levels`, `q2_trend`, `q3_sensitivity`, `q4_bands`, `q4_kappa`, `folds` (metriche per fold), `comparison_models`, `comparison_sets`, `cutpoints` (soglia e limiti delle fasce da applicare al test)
- `tests/test_evaluation.py`: 15 test su dati sintetici; tra l'altro il test di Jonckheere confrontato con quello di Kendall (sono lo stesso test), DeLong con il calcolo diretto, la formula dell'intervallo della PR-AUC, il bootstrap che conserva le numerosità dei livelli, nessuna lettura del test set (anche nello script delle figure)
- `tests/test_interpretation.py`: 5 test su dati sintetici (intervallo di Wald contro l'Hessiana calcolata a mano, somma dei valori SHAP uguale alla previsione, contributi di XGBoost identici a quelli del pacchetto `shap`, nessuna lettura del test set)
- suite completa: 56 test verdi (19/09/2026)

### Valutazione: risultati (19/09/2026)
Calcolati con le scelte della sezione precedente, fissate prima. Solo previsioni out-of-fold del training (4.350 soggetti, 425 positivi); il test set non è stato toccato e servirà da conferma finale. Tabelle in `analytics/phase_a/evaluation/`, figure `analytics/phase_a/01`–`08` (vedi "Indice delle figure"). Valori del set `main` salvo dove indicato; IC al 95%.

**Controllo di coerenza**: il classificatore di maggioranza ha AUC = 0,5 e PR-AUC = prevalenza (0,098) in tutti i 10 fold (5 × 2 set): la pipeline è coerente. Le prevalenze dei training dei 5 fold coincidono (0,0977; fold stratificati sul livello KDIGO), quindi le sue previsioni sono costanti anche aggregate.

**Domanda 1 — discriminazione** (figure 01, 02, 03)

| modello | AUC (IC) | AUC media fold | PR-AUC (IC) | PR-AUC media fold |
|---|---|---|---|---|
| logistica SCORED | 0,675 (0,646–0,704) | 0,679 | 0,224 (0,187–0,266) | 0,243 |
| logistica penalizzata | 0,697 (0,669–0,725) | 0,700 | 0,242 (0,204–0,285) | 0,255 |
| Random Forest | 0,703 (0,676–0,731) | 0,708 | 0,246 (0,207–0,289) | 0,261 |
| XGBoost | 0,699 (0,671–0,726) | 0,704 | 0,251 (0,212–0,295) | 0,262 |
| maggioranza | 0,500 | 0,500 | 0,098 | 0,098 |

Alla soglia con sensibilità 0,90:

| modello | soglia su p | precision (IC) | recall | specificità (IC) | soggetti da testare |
|---|---|---|---|---|---|
| logistica SCORED | 0,050 | 0,110 (0,100–0,121) | 0,901 | 0,209 (0,197–0,222) | 80,2% |
| logistica penalizzata | 0,048 | 0,112 (0,102–0,123) | 0,901 | 0,223 (0,211–0,237) | 78,9% |
| Random Forest | 0,058 | 0,113 (0,103–0,124) | 0,901 | 0,232 (0,219–0,246) | 78,1% |
| XGBoost | 0,042 | 0,114 (0,104–0,126) | 0,901 | 0,245 (0,232–0,259) | 76,9% |

Quota di soggetti da testare agli altri punti operativi (solo descrittivi): sensibilità 0,80 → 55–65% (Random Forest 55,0%, XGBoost 58,7%, logistica penalizzata 60,7%, SCORED 64,8%); 0,85 → 66–74%; 0,95 → 88–89%. Scegliendo i soggetti a caso, per trovare il 90% dei positivi bisognerebbe testarne il 90%: il modello fa risparmiare 10–13 punti percentuali.

- discriminazione **modesta**: AUC circa 0,70, PR-AUC circa 2,5 volte la prevalenza. Ordine di grandezza simile all'AUC di SCORED in validazione esterna (0,71; Bang et al. 2007), che però aveva un altro target (eGFR < 60)
- la PR-AUC aggregata è più bassa della media dei fold di 0,011–0,019 (AUC: 0,003–0,005): piccole differenze di calibrazione fra i modelli dei diversi fold (Forman & Scholz 2010). La stima aggregata è quella prudente
- il recall è 0,901 per costruzione (soglia a sensibilità fissata); il confronto fra modelli alla soglia sta nella specificità, i cui IC si sovrappongono

**Domanda 2 — il rischio stimato cresce con la gravità KDIGO?** (figura 04)

| modello | basso | moderato | alto | molto alto | concordanza (IC) |
|---|---|---|---|---|---|
| logistica SCORED | 0,092 | 0,139 | 0,178 | 0,230 | 0,674 (0,646–0,702) |
| logistica penalizzata | 0,091 | 0,152 | 0,202 | 0,247 | 0,696 (0,668–0,723) |
| Random Forest | 0,093 | 0,137 | 0,164 | 0,195 | 0,702 (0,674–0,728) |
| XGBoost | 0,086 | 0,142 | 0,194 | 0,284 | 0,698 (0,671–0,724) |

Valori = probabilità media per livello (IC bootstrap nelle tabelle). Jonckheere-Terpstra: z da 12,0 a 13,9, p < 10⁻³² per tutti i modelli.

- **sì**: in tutti i modelli la probabilità media cresce a ogni livello, e circa il 70% delle coppie di soggetti di livelli diversi è ordinato come KDIGO
- la separazione più netta è fra "basso" e gli altri livelli; fra "moderato" e "alto" le distribuzioni si sovrappongono molto (mediane 0,11–0,12 contro 0,13–0,15), e gli IC delle medie di "alto" e "molto alto" si sovrappongono in tutti i modelli
- XGBoost e logistica penalizzata separano meglio il "molto alto" (media 0,28 e 0,25); la Random Forest comprime le probabilità (media del "molto alto" 0,195)

**Domanda 3 — quanti casi gravi riconosce?** (figura 05; soglia con sensibilità complessiva 0,90)

| modello | moderato (n = 354) | alto (n = 50) | molto alto (n = 21) |
|---|---|---|---|
| logistica SCORED | 317 (0,895) | 47 (0,94) | 19 (0,905; IC 0,711–0,973) |
| logistica penalizzata | 316 (0,893) | 47 (0,94) | 20 (0,952; IC 0,773–0,992) |
| Random Forest | 319 (0,901) | 46 (0,92) | 18 (0,857; IC 0,654–0,950) |
| XGBoost | 318 (0,898) | 47 (0,94) | 18 (0,857; IC 0,654–0,950) |

- "molto alto" mancati: 1 (logistica penalizzata), 2 (SCORED), 3 (Random Forest, XGBoost)
- a parità di sensibilità complessiva i casi gravi **non** sono riconosciuti chiaramente più dei moderati: sensibilità simili, e con 21 casi gli IC vanno da circa 0,65 a 0,99. Il test set (6 "molto alto") non potrà cambiare questa conclusione: è il limite dichiarato sui livelli poco numerosi
- set `no_consequence`: differenze di al massimo 2 soggetti per livello; "molto alto" invariati salvo SCORED (18/21)

**Domanda 4 — le fasce del modello corrispondono ai livelli KDIGO?** (figura 06)

| modello | kappa pesato (IC) | accordo osservato | "molto alto" in fascia 4 | "molto alto" in fascia 1 |
|---|---|---|---|---|
| logistica SCORED | 0,197 (0,156–0,236) | 0,851 | 3 su 21 | 10 |
| logistica penalizzata | 0,207 (0,171–0,245) | 0,854 | 0 su 21 | 8 |
| Random Forest | 0,209 (0,165–0,246) | 0,854 | 3 su 21 | 9 |
| XGBoost | 0,228 (0,183–0,266) | 0,855 | 5 su 21 | 6 |

- concordanza **bassa**: kappa circa 0,2 (set `no_consequence`: 0,19–0,21). L'accordo osservato alto (0,85) dipende quasi tutto dal livello "basso", il 90% dei soggetti: è il paradosso della prevalenza (Feinstein & Cicchetti 1990)
- il 72–75% dei "moderato" e il 62–70% degli "alto" finiscono nella fascia 1, e fino a metà dei "molto alto" nella fascia 1: le fasce del modello non riproducono la stratificazione per gravità. Il modello separa soprattutto la presenza di marcatori, non il loro grado
- inizio della fascia 4: p ≥ 0,56 (logistica penalizzata), 0,46 (XGBoost), 0,43 (SCORED), 0,32 (Random Forest): conferma le probabilità compresse della Random Forest, da tenere presente per la ricalibrazione in Fase B

**Confronto fra modelli** (figura 07; Nadeau & Bengio 2003 sui 5 fold esterni, p corretti con Holm)
- PR-AUC (metrica primaria): tutte le differenze fra modelli sono ≤ 0,019 in valore assoluto, p corretto = 1,00
- AUC: la logistica SCORED perde 0,021 contro la logistica penalizzata (IC da −0,003 a 0,046), 0,029 contro la Random Forest (da −0,008 a 0,066), 0,025 contro XGBoost (da −0,004 a 0,054); p non corretti 0,07–0,10, corretti 0,44. Le altre differenze sono ≤ 0,008
- **nessuna differenza dimostrata**: una logistica con 5 predittori da screening è vicina ai modelli con 74 feature, coerente con Christodoulou et al. 2019. Con 4 gradi di libertà la potenza è bassa: "non dimostrata" non significa "assente"

**Analisi di sensibilità `no_consequence`** (figure 02 e 08): togliendo HGB, RBC, HCT, SUA, ALB, TP, GA
- AUC: da −0,012 (XGBoost) a +0,001 (SCORED); PR-AUC: da −0,021 (XGBoost) a −0,002 (SCORED); nessuna differenza significativa (p 0,23–0,64)
- le prestazioni **non dipendono** in modo rilevante dalle variabili alterate dalla malattia renale

**Sintesi per la tesi**
1. senza esami renali i modelli riconoscono i marcatori di malattia renale in modo modesto (AUC circa 0,70): con sensibilità 0,90 andrebbe testato circa il 78% della popolazione, contro il 90% di una scelta casuale
2. il rischio stimato cresce con la gravità KDIGO (concordanza circa 0,70), ma le fasce del modello concordano poco con i livelli (kappa circa 0,2): il modello riconosce la presenza dei marcatori più che il loro grado
3. i casi gravi non sono riconosciuti più dei moderati; 1–3 "molto alto" su 21 mancati alla soglia scelta
4. nessun modello è migliore degli altri in modo dimostrabile; la logistica SCORED a 5 predittori è poco distante
5. le variabili-conseguenza non spiegano le prestazioni
6. punto di partenza per la Fase B (domanda 5): le tecniche di bilanciamento migliorano il riconoscimento dei casi gravi a parità di sensibilità complessiva, o solo le metriche medie?

Da confermare sul test set, una sola volta, con i modelli finali e la soglia e le fasce di `analytics/phase_a/evaluation/cutpoints.csv`.

### Analisi di sensibilità: profondità di XGBoost (decisa dopo i risultati, 19/09/2026)
Controllo degli iperparametri scelti da Optuna (set main):
- **XGBoost**: `max_depth` = 3, cioè il **limite inferiore** dello spazio (3–12, Brima & Atemkeng 2026), in 6 ottimizzazioni su 6 (5 fold esterni e modello finale), con learning rate basso (0,010–0,025). L'ottimo è probabilmente sotto il limite: alberi molto poco profondi, poche interazioni. È coerente con il risultato "logistica ≈ XGBoost": il segnale è quasi additivo
- Random Forest: scelte instabili fra i fold (profondità da 3 a 25), tipico di un ottimo piatto; nessun limite sistematico
- logistica penalizzata: `C` nel mezzo dello spazio (scala logaritmica), `l1_ratio` spesso vicino a 0 (quasi solo L2); 0 è un estremo naturale, non un limite dello spazio

Decisione, **dichiarata come presa dopo aver visto i risultati**:
- i risultati primari restano quelli del protocollo fissato prima (spazio 3–12)
- XGBoost viene rilanciato con `max_depth` 1–12, stesso protocollo (100 tentativi, TPE, MedianPruner, stessi fold; il set `no_consequence` riusa gli iperparametri): `python -m src.models.phase_a --sensitivity depth_1_12` (config `phase_a.sensitivity`), risultati e modelli in cartelle separate (`analytics/phase_a/sensitivity/depth_1_12/`, `models/sensitivity/depth_1_12/`)
- valutazione: `python -m src.models.evaluation --sensitivity depth_1_12`, con il modello `xgboost_depth_1_12` confrontato con gli altri sugli stessi fold
- lo spazio 1–12 è quello della **Fase B**, così "nessuna correzione" e le tecniche di bilanciamento usano lo stesso spazio

Risultati (19/09/2026, circa 45 minuti; tabelle in `analytics/phase_a/sensitivity/depth_1_12/evaluation/`; file primari non modificati, verificato dalle date):
- profondità scelta: **1** in 5 ottimizzazioni su 6, **2** nell'altra. Con alberi a un solo nodo di divisione il modello è una somma di effetti delle singole variabili, senza interazioni: conferma che il segnale è additivo
- set main: AUC 0,696 (0,668–0,724) contro 0,699 del protocollo primario; PR-AUC 0,250 (0,211–0,293) contro 0,251
- differenza appaiata sui 5 fold (primario − profondità 1–12, Nadeau & Bengio): AUC +0,007 (da −0,004 a 0,017; p = 0,15), PR-AUC +0,003 (da −0,022 a 0,027; p = 0,78). Set `no_consequence`: AUC +0,001, PR-AUC +0,000
- alla soglia con sensibilità 0,90: specificità 0,256 contro 0,245, soggetti da testare 75,9% contro 76,9%; "molto alto" riconosciuti 18/21 in entrambi, "alto" 48/50 contro 47/50; kappa 0,235 contro 0,228; concordanza 0,695 contro 0,698
- **conclusione**: il limite inferiore dello spazio non ha penalizzato XGBoost. I risultati primari sono robusti, e le interazioni fra variabili non aggiungono informazione, coerentemente con "logistica ≈ XGBoost"
- nota: in questa valutazione la famiglia di Holm comprende 15 coppie (6 modelli); per il confronto fra le due versioni di XGBoost si riporta il p non corretto

### Interpretazione: scelte fissate prima del calcolo (19/09/2026)
Fissate prima di calcolare coefficienti e valori SHAP (erano stati misurati solo i tempi di calcolo).

| elemento | scelta | fonte |
|---|---|---|
| modelli interpretati | i **modelli finali** (iperparametri scelti in CV, riaddestrati sull'intero training), set `main`; `no_consequence` solo in tabella | protocollo della Fase A |
| logistica SCORED | odds ratio per 1 deviazione standard (numeriche, standardizzate nel preprocessing) o per unità (categoriche: `Gender` 1 = maschio → 2 = femmina, `DM` 0 → 1), con **intervallo di Wald al 95%**: il modello non è penalizzato | Hosmer, Lemeshow & Sturdivant 2013 |
| logistica penalizzata | odds ratio per 1 DS o per unità, **senza intervalli** (la penalizzazione riduce i coefficienti: gli errori standard usuali non valgono); numero di coefficienti azzerati; prime 15 variabili per valore assoluto del coefficiente | Tibshirani 1996; Zou & Hastie 2005 |
| Random Forest, XGBoost | valori SHAP **esatti** con TreeSHAP, su tutti i 4.350 soggetti del training (circa 5 minuti): Random Forest sulla scala della probabilità, XGBoost sulla scala logit. Per XGBoost si usa il TreeSHAP interno di XGBoost (`pred_contribs`): i contributi sono identici a quelli del pacchetto `shap` 0.52, che però con XGBoost 3.4 sbaglia il valore base di una costante (verificato su dati sintetici, test in `tests/test_interpretation.py`) | Lundberg & Lee 2017; Lundberg et al. 2020 |
| importanza globale | media del valore assoluto SHAP per variabile; prime 15; grafico a sciame (beeswarm) per il verso dell'effetto | Lundberg et al. 2020 |
| confronto fra modelli | prime 15 variabili di ciascun modello affiancate; conta quante sono comuni | — |

Cautele da dichiarare:
- SHAP e coefficienti descrivono **il modello, non la causalità**: una variabile importante può essere conseguenza o marcatore della malattia renale, non causa
- le variabili molto correlate (Passo 7) si dividono l'importanza: il peso di un gruppo va letto insieme
- le spiegazioni sono calcolate sui dati di addestramento del modello finale (descrivono cosa ha imparato, non le prestazioni)

### Interpretazione: risultati (19/09/2026)
Calcolati con `python -m src.models.interpretation` (tabelle in `analytics/phase_a/interpretation/`, figure 09–11). Modelli finali, set `main` salvo dove indicato.

**Logistica SCORED** (odds ratio, IC di Wald al 95%; DS di `Age` = 13,8 anni, di `Bpsys` = 16,5 mmHg nel training):

| predittore | odds ratio (IC) | p |
|---|---|---|
| `DM` (diabete sì contro no) | 2,58 (1,89–3,52) | < 10⁻⁸ |
| `Age` (per 13,8 anni) | 1,47 (1,32–1,64) | < 10⁻¹¹ |
| `Bpsys` (per 16,5 mmHg) | 1,34 (1,22–1,47) | < 10⁻⁸ |
| `HGB` (per 1 DS) | 0,92 (0,81–1,04) | 0,19 |
| `Gender` (femmina contro maschio) | 0,96 (0,75–1,22) | 0,73 |

- i tre fattori di rischio classici della malattia renale (diabete, età, pressione) portano il segnale; emoglobina e sesso non aggiungono nulla di dimostrabile. Set `no_consequence` (senza `HGB`): valori praticamente identici (DM 2,54; Age 1,50; Bpsys 1,34)

**Logistica penalizzata** (modello finale: `C` = 1,53, `l1_ratio` = 0,85; 6 coefficienti azzerati su 74):
- primo coefficiente: `DRyd` (retinopatia), odds ratio 3,48. Riguarda però solo 27 soggetti (0,6%; positivi il 37% contro il 9,6% degli altri): effetto forte ma raro. La retinopatia è una complicanza microvascolare come il danno renale del diabete
- blocco ematologico con segni opposti: `HGB` 0,45, `MCV` 1,92, `RBC` 1,87, `MCHC` 1,41 per DS. Sono variabili legate fra loro (emoglobina ≈ globuli rossi × volume medio × concentrazione media): i coefficienti singoli sono instabili e va letto il blocco, non la singola variabile (cautela già dichiarata)
- poi `HypertenHis` 1,74, `Gender` 1,73 (a parità di emoglobina), circonferenza vita 1,51 contro fianchi 0,66 (obesità addominale), `WBC` 1,46, `FCP` 1,37
- la classifica per coefficiente mescola unità diverse (1 DS per le numeriche, 0 → 1 per le binarie): non è confrontabile con l'importanza SHAP

**Random Forest e XGBoost** (SHAP, figure 10–11):
- prime variabili comuni ai due modelli: 11 su 15 (`Age`, `ALP`, `FIB4`, `CP2h`, `FCP`, `GA`, `FPG`, `GGT`, `SUA`, `LDL`, `TG`). La logistica penalizzata ne condivide con entrambi solo `FCP`
- **`Age` è la prima variabile in entrambi**, e pesa anche attraverso `FIB4`, che contiene l'età nella formula. `ALP` è seconda in entrambi. Coerente con le AUC univariate del Passo 2 (età 0,653, `ALP` 0,631, `FIB4` 0,623)
- gruppi: assetto glicemico e insulinico (`FCP`, `CP2h`, `FPG`, `PG2h`, `GA`, `ISIGutt`, `INS2h`), lipidi (`LDL`, `HDL`, `TG`, `CHOL`), fegato (`FIB4`, `GGT`, `AST`); `Bpsys` è settima nella Random Forest, fuori dalle prime 15 in XGBoost
- importanza **diffusa**: le prime 5 variabili spiegano solo il 23–26% dell'importanza totale (set main). Molti predittori deboli, coerente con l'AUC modesta
- versi dell'effetto (figura 11): valori alti di età, `ALP`, `FIB4`, peptide C, glicemie, `LDL`, `TG`, `SUA` e `GGT` alzano il rischio stimato; `ISIGutt` basso (resistenza all'insulina) lo alza; `GA` **basso** lo alza in entrambi i modelli; in XGBoost `INS2h` e `AST` alti lo abbassano (effetti condizionati da variabili correlate: `CP2h`, `ALT`/`FIB4`)
- variabili-conseguenza fra le prime 15: `SUA` (quinta in XGBoost) e `GA`; togliendole le prestazioni non cambiano (analisi `no_consequence`): altre variabili ne compensano il contributo

**Ipotesi da verificare in letteratura prima di scriverle nella tesi** (non sono risultati):
1. il peptide C è eliminato in gran parte dal rene: `FCP` e `CP2h` alti potrebbero riflettere in parte una ridotta clearance renale, cioè funzionare come variabili-conseguenza non dichiarate. Il segno opposto di `INS2h` (a parità di peptide C) suggerisce che il modello usi il rapporto peptide C / insulina
2. `GA` basso associato al rischio: l'albumina glicata dipende dal ricambio dell'albumina, che la perdita urinaria di albumina può alterare
3. se confermate, sono candidate a un'analisi di sensibilità aggiuntiva (da decidere e dichiarare, non ora)

### Revisione indipendente del codice (19/09/2026)
Revisione di `src/models/evaluation.py`, `src/analytics/phase_a_report.py` e delle modifiche a `src/models/phase_a.py`, fatta da un agente revisore separato (sola lettura): **nessun errore critico o grave**. Verificati numericamente DeLong, soglia a sensibilità fissata (anche pareggi e arrotondamenti), Jonckheere-Terpstra, Holm, Nadeau-Bengio, allineamento righe/etichette; nessuna lettura del test set. Osservazioni minori:
- la famiglia di Holm comprende le 10 coppie, compresi i confronti con il classificatore di maggioranza (come scritto nel protocollo: "10 coppie per set"). Verificato: escludendoli, i p corretti delle 6 coppie di modelli reali non cambiano (differenza massima 2 · 10⁻¹⁶). Protocollo mantenuto e dichiarato nella figura 07
- aggiunti controlli sui casi degeneri (`delong` senza positivi o negativi, PR-AUC = 0 o 1 → intervallo non definito); nessun effetto sui risultati
- test più stringenti: formula dell'intervallo della PR-AUC verificata numericamente; controllo "nessuna lettura del test set" esteso allo script delle figure

### Fase B — decisioni preliminari (19/09/2026)
Approvate prima di scrivere il protocollo completo della Fase B (ora nella sezione "Fase B — protocollo fissato prima dei risultati"):
- **stessi 5 modelli** della Fase A: il confronto deve misurare solo l'effetto delle tecniche di bilanciamento
- spazio di XGBoost con `max_depth` 1–12 (vedi analisi di sensibilità sopra)
- **esito primario della domanda 5**: sensibilità sui casi gravi (alto + molto alto: 71 soggetti nelle previsioni out-of-fold) alla soglia con sensibilità complessiva 0,90, stimata per ogni tecnica. Confronto appaiato con "nessuna correzione" **sugli stessi soggetti** con il test di McNemar esatto (McNemar 1947). Secondario: PR-AUC con il test di Nadeau & Bengio 2003
- da dichiarare: con 71 casi gravi si possono dimostrare solo differenze grandi
- **cautela su CTGAN**: ogni fold di training esterno ha circa 340 positivi (circa 270 nei fold interni), pochi per addestrare una rete generativa; la sua adeguatezza va verificata in letteratura prima di includerlo
- ancora da decidere, prima del lancio: elenco definitivo delle tecniche con fonti verificate; valutazione "ingenua" alla soglia 0,5 come analisi secondaria fissata in anticipo; riottimizzazione degli iperparametri per tecnica o riuso, e budget; ricalibrazione (serve per le probabilità medie della domanda 2, non per soglia e fasce, che dipendono solo dall'ordinamento)

### Fonti lette per la Fase A
- Echouffo-Tcheugui & Kengne 2012: testo completo, `papers/kdigo/file.pdf`
- Boyd et al. 2013 e Forman & Scholz 2010: testo completo (PDF d'autore), per gli intervalli della PR-AUC e per il calcolo delle metriche in cross-validation
- Bang et al. 2007: solo abstract (pagina salvata `papers/kdigo/00000779-200702260-00016~screening-for-occult-renal-disease-scored-a-simple.html`); basta per i predittori e le prestazioni, il testo completo servirebbe solo per i pesi del punteggio, che non usiamo perché i coefficienti sono ristimati

### Ambiente
- installati `xgboost` 3.4.1 e `optuna` 5.0.0 (19/09/2026); `numpy`, `pandas` e `scikit-learn` restano alle versioni fissate nel file di lock
- installato `shap` 0.52.0 (19/09/2026) per l'interpretazione, con `numba` 0.67.0, `llvmlite` 0.49.0 e `slicer` 0.0.8; `numpy` resta 2.5.3 (il timore che `numba` la cambiasse non si è verificato)

---

## Fase B — protocollo fissato prima dei risultati (19/09/2026)
Scritto in `configs/config.yaml` (sezione `phase_b`) e qui **prima di qualsiasi addestramento della Fase B**; sostituisce le "decisioni preliminari" (sezione precedente). Domanda: *il bilanciamento migliora il riconoscimento dei casi gravi (alto + molto alto) o solo la metrica media?* (domanda 5 dello Scope).

### Che cosa dice la letteratura (e perché queste tecniche)
- con modelli forti il bilanciamento **non migliora la discriminazione e peggiora la calibrazione**: van den Goorbergh et al. 2022 (logistica); Carriero et al. 2025 (simulazione con logistica, SVM, Random Forest, XGBoost: XGBoost peggiora, Random Forest migliora poco); Elor & Averbuch-Elor 2022 e Roesler et al. 2026 (preprint). L'effetto sulla classificazione equivale spesso a **spostare la soglia** (Elkan 2001), cosa che la nostra soglia a sensibilità fissata neutralizza già: un guadagno reale può venire solo da un **ordinamento diverso** dei soggetti
- i **pesi per singolo esempio** sono l'unico modo diretto per dare più importanza ai casi gravi: apprendere con pesi proporzionali al costo di ogni esempio equivale a minimizzare il costo atteso (Zadrozny, Langford & Abe 2003; Correa Bahnsen et al. 2015); Elkan 2001 cita esplicitamente "the severity of an illness" come costo
- con circa 300 positivi **SMOTE eguaglia o batte i generatori profondi** (Kotelnikov et al. 2023; Camino et al. 2020; Manousakas & Aydöre 2023); CTGAN è stato sviluppato e validato su insiemi da 1.000 a 23.000 righe (Xu et al. 2019; Zhao et al. 2024) e con pochi dati genera campioni poco vari (Seedat et al. 2024) → **solo esplorativo**. Il random oversampling è competitivo con metodi più complessi (Batista et al. 2004; Elor & Averbuch-Elor 2022)
- generare **dentro ciascun sottogruppo** ha supporto indiretto: squilibrio interno alla classe e piccoli sottogruppi (Jo & Japkowicz 2004; Fernández et al. 2018, §5.1), classi ordinate (Pérez-Ortiz et al. 2015), generazione per sottogruppo (Chakraborty et al. 2021). Evita di interpolare fra un "moderato" e un "molto alto"
- paper locali: Brima & Atemkeng 2026 (nessuno schema di pesi è il migliore ovunque; pesi a inverso della frequenza instabili con classi rarissime); Hameed & Ali 2025 (SMOTEENN, prova debole: una sola suddivisione, soglia di default)
- il bilanciamento va applicato **solo al training di ogni fold** (Santos et al. 2018; Demircioğlu 2024)

### Protocollo
| elemento | scelta |
|---|---|
| modelli | gli stessi 5 della Fase A; XGBoost con `max_depth` 1–12 |
| set | solo `main` (la Fase A ha mostrato che `no_consequence` non cambia le prestazioni) |
| "nessuna correzione" | risultati della Fase A (per XGBoost l'analisi di sensibilità `depth_1_12`) |
| tecniche principali (6) | 1. **pesi di classe**: inverso della prevalenza, media dei pesi = 1; 2. **pesi per livello KDIGO 1 : 2 : 3** sui positivi (moderato, alto, molto alto), riscalati in modo che positivi e negativi pesino uguale; 3. **random undersampling** 1:1; 4. **random oversampling** 1:1; 5. **SMOTE-NC** 1:1 (k = 5; le 14 variabili categoriche e ordinali trattate come nominali); 6. **SMOTE-NC per livello**: generazione dentro ciascun livello positivo, proporzioni dei livelli conservate, k = min(5, n − 1) |
| esplorative (2) | **CTGAN** condizionato su y e **CTGAN condizionato sul livello KDIGO**: generatore addestrato sull'intero fold di training (circa 3.480 righe, meccanismo di condizionamento di Xu et al. 2019), campionamento dei soli positivi fino a 1:1. Protocollo ridotto: nessuna ottimizzazione (iperparametri del braccio SMOTE-NC dello stesso fold), solo fold esterni e training intero, nessuna ricalibrazione, fuori dalla correzione di Holm |
| dove | solo sulla parte di training di ogni fold interno ed esterno e del training intero; validazione sempre reale; il livello KDIGO non è mai un predittore (serve solo per pesi e generazione); dati sintetici generati **una volta** per tecnica e fold e riusati da tutti i modelli |
| ottimizzazione | **30 tentativi** per fold, il primo è l'ottimo della Fase A per lo stesso fold (avvio caldo), TPE seed 42, MedianPruner, PR-AUC sulla validazione interna reale. Motivazione: nei log della Fase A, a 30 tentativi la PR-AUC interna dista in mediana 0,0004–0,005 dal valore a 100 (massimo 0,019), meno della variabilità fra fold; budget ridotto **dichiarato** |
| esito primario | sensibilità sui **71 casi gravi** (alto + molto alto) alla soglia con sensibilità complessiva 0,90, stimata per tecnica e modello sulle probabilità **non ricalibrate**; confronto appaiato con "nessuna correzione" sugli stessi soggetti con il **test di McNemar esatto** (McNemar 1947) e **IC di Newcombe** della differenza appaiata (Newcombe 1998); **Holm** sulle 6 tecniche principali, separatamente per modello |
| secondari | PR-AUC e AUC contro "nessuna correzione" (Nadeau & Bengio 2003); domande 1–4 per tecnica; calibrazione (intercetta, pendenza, Brier: Van Calster et al. 2016) grezza e ricalibrata; **valutazione "ingenua" alla soglia 0,5** sulle probabilità grezze, per mostrare il miglioramento apparente |
| ricalibrazione | **Platt annidato**: per ogni fold esterno il modello con i parametri scelti viene riaddestrato sui 5 fold interni bilanciati e predice le righe interne **reali**; la regressione di Platt su logit(p) si applica alle previsioni del fold esterno (Platt 1999; con pochi eventi Platt è preferibile all'isotonica: Niculescu-Mizil & Caruana 2005). Usata per le probabilità medie della domanda 2 e per la calibrazione; l'esito primario e le metriche di ordinamento restano sulle probabilità grezze, come nella Fase A (la ricalibrazione per fold cambierebbe leggermente l'ordinamento aggregato) |

Tecniche considerate e non incluse:
- **Borderline-SMOTE** (Han et al. 2005) e **ADASYN** (He et al. 2008): concentrano la generazione vicino al confine fra le classi; con livelli gravi di 17–40 casi per fold aumentano il rischio di generare rumore, e la letteratura non ne mostra un vantaggio sui modelli forti
- **SMOTE-ENN / SMOTE-Tomek** (Batista et al. 2004): la pulizia rimuove anche negativi reali, confondendo l'effetto della generazione
- **TVAE, TabDDPM, CTAB-GAN+**: stessi limiti di CTGAN con pochi dati; un solo generatore profondo esplorativo basta
- **EasyEnsemble, RUSBoost, Balanced Random Forest**: cambiano il modello, contro la regola "stessi 5 modelli"

Limiti da dichiarare:
- con 71 casi gravi si possono dimostrare **solo differenze grandi**
- budget di ottimizzazione ridotto (30 tentativi contro 100 della Fase A), motivato dai log
- SMOTE-NC tratta le variabili ordinali come nominali e cerca i vicini su tutte le 74 variabili, anche per la logistica SCORED che ne usa 5
- la generazione con CTGAN non è esattamente ripetibile: la cache dei dati sintetici (con hash) è la traccia di riproducibilità

### Ambiente della Fase B
- installati `imbalanced-learn` 0.14.2 e `ctgan` 0.12.1 (con `torch` 2.14.0 e `rdt` 1.22.0). `rdt` richiede **pandas < 3**: pandas è passato da 3.0.5 a **2.3.3**
- verificato che il cambio non tocca i risultati: con pandas 2.3.3 i 56 test passano e la valutazione della Fase A rilanciata produce tabelle **identiche** a quelle committate (`git diff` vuoto). Un solo ambiente, niente ambiente separato per CTGAN; file di lock rigenerato dall'ambiente reale

## Fase B — risultati (20/09/2026)
Calcolati con il protocollo fissato prima (sezione precedente). Solo previsioni out-of-fold del training (4.350 soggetti, 425 positivi, 71 casi gravi); il test set non è stato toccato. Tabelle in `analytics/phase_b/evaluation/`, figure `analytics/phase_b/01`–`06`.

Esecuzione: 19/09 16:23 → 20/09 01:50. Le 6 tecniche principali (5 modelli × 6 fold ciascuna, 30 tentativi di Optuna con avvio caldo) in 7 ore e mezza; cache e addestramento dei due bracci CTGAN circa 1 ora; ricalibrazione di Platt 23 minuti.

**Controllo di coerenza superato**: il classificatore di maggioranza ha PR-AUC uguale alla prevalenza e AUC 0,5 in tutti i fold e per tutte le tecniche.

**Modifica dichiarata durante l'esecuzione**: CTGAN condizionato sul livello si è fermato con un errore perché non riusciva a generare abbastanza casi "molto alto" (82 su 173 richiesti). Il condizionamento di CTGAN non è rigido e con 17 casi reali quel livello compare in circa lo **0,5–1,2%** dei campioni generati. Sono stati aumentati i lotti di campionamento e i tentativi (da 50 a 200) e il tasso di accettazione viene ora registrato. È un cambiamento del campionamento, non del protocollo, ed è la conferma pratica del limite previsto per i generatori profondi con pochi casi.

### Esito primario (domanda 5): casi gravi riconosciuti (su 71) a sensibilità complessiva 0,90

| tecnica | logistica SCORED | logistica penalizzata | Random Forest | XGBoost |
|---|---|---|---|---|
| nessuna correzione | 66 | 67 | 64 | 66 |
| pesi di classe | 66 | 67 | 67 | 66 |
| pesi per livello 1:2:3 | 67 | 66 | 66 | 67 |
| undersampling | 66 | 68 | 64 | 65 |
| oversampling | 67 | 68 | 67 | 63 |
| SMOTE-NC | 66 | 67 | 66 | 65 |
| SMOTE-NC per livello | 65 | 68 | 67 | 65 |
| CTGAN (esplorativo) | 64 | 67 | 67 | 64 |
| CTGAN per livello (esplorativo) | 60 | 67 | 63 | 65 |

- **nessun guadagno dimostrato**: il massimo guadagno netto è di **3 casi** su 71 (Random Forest con SMOTE-NC per livello: 5 casi guadagnati, 2 persi), la massima perdita netta di 6 (logistica SCORED con CTGAN per livello). Tutti i p corretti con Holm valgono **1,00**; il p non corretto più piccolo è 0,25
- gli intervalli di Newcombe della differenza vanno circa da −0,09 a +0,13: con 71 casi gravi si vedono solo differenze grandi, come dichiarato

### Discriminazione e domande 1–4 (medie sui 4 modelli reali)

| tecnica | AUC | PR-AUC | specificità a sensibilità 0,90 | concordanza (tendenza) | kappa pesato |
|---|---|---|---|---|---|
| nessuna correzione | 0,693 | 0,240 | 0,230 | 0,692 | 0,212 |
| pesi di classe | 0,690 | 0,232 | 0,231 | 0,689 | 0,202 |
| pesi per livello 1:2:3 | 0,692 | 0,235 | 0,238 | 0,691 | 0,198 |
| undersampling | 0,689 | 0,220 | 0,235 | 0,687 | 0,180 |
| oversampling | 0,680 | 0,223 | 0,217 | 0,679 | 0,191 |
| SMOTE-NC | 0,672 | 0,209 | 0,213 | 0,670 | 0,169 |
| SMOTE-NC per livello | 0,673 | 0,209 | 0,225 | 0,671 | 0,165 |
| CTGAN (esplorativo) | 0,616 | 0,160 | 0,178 | 0,615 | 0,104 |
| CTGAN per livello (esplorativo) | 0,622 | 0,166 | 0,154 | 0,622 | 0,112 |

- **nessuna tecnica migliora**: tutte le differenze di PR-AUC e AUC contro "nessuna correzione" sono negative o nulle, salvo scarti trascurabili dei pesi per livello (+0,001 e +0,004 per XGBoost). Nessuna è significativa (p di Holm minimo 0,35)
- i **pesi** (di classe e per livello) sono praticamente neutri: cambiano le probabilità, non l'ordinamento
- **SMOTE-NC** peggiora un po' (PR-AUC da −0,012 a −0,050), soprattutto sugli alberi; la variante per livello si comporta come quella standard
- **CTGAN peggiora molto** (AUC 0,62 contro 0,69, PR-AUC 0,16 contro 0,24; p non corretto fino a 0,002): con circa 340 positivi per fold i dati sintetici sono di qualità insufficiente, come atteso dalla letteratura
- la sensibilità per livello resta la stessa (moderato 0,89–0,90, alto 0,93–0,96, molto alto 0,83–0,91 in tutte le tecniche) e nessuna tecnica privilegia i livelli gravi

### Calibrazione (medie sui 4 modelli)

| tecnica | intercetta grezza | pendenza grezza | intercetta dopo Platt | pendenza dopo Platt |
|---|---|---|---|---|
| nessuna correzione | 0,00 | 1,02 | −0,00 | 0,99 |
| pesi di classe | −2,06 | 1,01 | −0,01 | 0,99 |
| pesi per livello | −2,04 | 1,06 | −0,01 | 0,98 |
| undersampling | −2,23 | 1,03 | −0,00 | 0,94 |
| oversampling | −2,00 | 0,96 | −0,01 | 0,96 |
| SMOTE-NC | −1,41 | 0,79 | 0,01 | 0,97 |
| SMOTE-NC per livello | −1,41 | 0,79 | 0,01 | 0,97 |
| CTGAN (esplorativo) | −1,02 | 0,42 | non ricalibrato | — |

- **il bilanciamento distrugge la calibrazione**: senza correzione l'intercetta è 0,00, con le tecniche di bilanciamento scende a −2,0 / −2,2, cioè le probabilità sono sistematicamente gonfiate (van den Goorbergh et al. 2022; Carriero et al. 2025)
- la **ricalibrazione di Platt annidata rimette a posto l'intercetta** (−0,01) e lascia la pendenza vicino a 1 (0,94–0,99). Con CTGAN la pendenza grezza è 0,42: le probabilità non sono solo gonfiate, sono mal ordinate

### Miglioramento apparente contro miglioramento reale (figura 05)
Recall alla soglia "di default" 0,5, sulle probabilità grezze, e positivi trovati su 425:

| tecnica | recall medio a 0,5 | positivi trovati (Random Forest) | sensibilità sui casi gravi a 0,5 |
|---|---|---|---|
| nessuna correzione | **0,02** | **0 su 425** | 0,00–0,11 |
| undersampling | **0,61** | 266 | 0,66–0,73 |
| pesi di classe | 0,52 | 178 | 0,54–0,68 |
| pesi per livello | 0,52 | 178 | 0,54–0,72 |
| oversampling | 0,48 | 118 | 0,38–0,66 |
| SMOTE-NC | 0,36 | 76 | 0,24–0,68 |

- alla soglia 0,5 il bilanciamento sembra **trasformare il modello**: il recall passa dal 2% al 61%, e i casi gravi riconosciuti da 0 su 71 a circa 50 su 71
- **è un effetto della soglia, non del modello**: a parità di sensibilità complessiva (0,90) le stesse tecniche non guadagnano nulla e la discriminazione peggiora. È esattamente quanto previsto da Elkan 2001 (pesare le classi equivale a spostare la soglia) e da van den Goorbergh et al. 2022

### Sintesi per la tesi
1. **Nessuna tecnica di bilanciamento migliora il riconoscimento dei casi gravi** a parità di sensibilità complessiva: differenze entro ±3 casi su 71, tutte non significative
2. **Alcune peggiorano**: SMOTE-NC perde fino a 0,05 di PR-AUC; CTGAN, con circa 340 positivi per fold, perde 0,08 di AUC ed è il peggiore su tutte le domande
3. **Le due tecniche con i pesi sono neutre** sull'ordinamento e rovinano la calibrazione, che la ricalibrazione di Platt recupera
4. **Il miglioramento apparente è grande e ingannevole**: alla soglia 0,5 il recall passa dal 2% al 61%. È il risultato didatticamente più forte della tesi
5. anche **i pesi per livello KDIGO**, la tecnica costruita apposta per i casi gravi, non li fanno riconoscere di più
6. risposta alla **domanda 5 dello Scope**: con questi dati il bilanciamento migliora solo la metrica media quando la si misura male; il riconoscimento dei casi gravi non cambia

### Revisione indipendente del codice della Fase B (20/09/2026)
Revisione di `src/data/augmented.py`, `src/models/phase_b.py`, `src/models/evaluation_b.py`, `src/analytics/phase_b_report.py` e delle modifiche a `src/models/phase_a.py`, fatta da un agente revisore separato (sola lettura): **nessun difetto critico**. Verificati: nessuna riga sintetica nella validazione, allineamento fra righe, etichette e pesi, SMOTE per livello che non interpola fra livelli diversi, ricalibrazione mai stimata su righe viste in addestramento, riferimento "nessuna correzione" preso dalla cartella giusta (per XGBoost dall'analisi `depth_1_12`), formule di McNemar e Newcombe ricavate a mano, famiglia di Holm limitata alle 6 tecniche principali.
- **corretto (problema maggiore)**: `pooled()` non controllava che ogni soggetto avesse la sua previsione. Con l'esecuzione riprendibile, una tecnica incompleta avrebbe prodotto NaN trattati come "non rilevato", cioè risultati sbagliati in silenzio. Ora è un errore esplicito; i NaN restano ammessi solo dove previsti (probabilità ricalibrate dei bracci CTGAN). Rilanciata la valutazione: le 16 tabelle sono identiche, quindi i risultati erano già calcolati su dati completi
- **corretto (minore)**: guardia sulle quote per livello, che ora non possono essere negative
- **da dichiarare (minori)**: la generazione con CTGAN non è garantita identica fra ambienti diversi (la cache con l'impronta sha256 è la traccia di riproducibilità); se l'esecuzione viene interrotta fra il salvataggio di un fold e la scrittura del log, quel fold manca nel log ma non nei dati

### Limiti
- 71 casi gravi: rilevabili solo differenze grandi (gli intervalli coprono circa ±0,1 di sensibilità)
- budget di ottimizzazione ridotto a 30 tentativi con avvio caldo; alla luce dei risultati (differenze molto minori del rumore fra fold) un budget maggiore non avrebbe cambiato le conclusioni
- un solo rapporto di bilanciamento (1:1) e un solo schema di pesi per livello (1:2:3)
- CTGAN valutato con protocollo ridotto e senza ricalibrazione: resta un braccio esplorativo
- conclusioni sulle previsioni out-of-fold del training; conferma finale sul test set a fine progetto

---

## Fase C — protocollo fissato prima dei risultati (20/09/2026)
Risponde alla **domanda 6** dello Scope: come si comporta lo stesso modello sui diabetici. Scritto **prima di calcolare qualsiasi numero sul sottogruppo**, come richiesto dalle regole metodologiche. Nessun riaddestramento e nessuna ottimizzazione: si filtrano le previsioni out-of-fold già salvate. Il test set non viene letto.

### Numeri del sottogruppo (verificati su `data/processed/train.csv` il 20/09/2026)

| gruppo | n | positivi | prevalenza | basso | moderato | alto | molto alto | casi gravi |
|---|---|---|---|---|---|---|---|---|
| diabetici (`DM` = 1) | 263 | 68 | 25,9% | 195 | 52 | 11 | 5 | 16 |
| non diabetici | 4.087 | 357 | 8,7% | 3.730 | 302 | 39 | 16 | 55 |
| training completo | 4.350 | 425 | 9,8% | 3.925 | 354 | 50 | 21 | 71 |

Il sottogruppo è **arricchito di casi gravi**: 16 su 263 (6,1%) contro 55 su 4.087 (1,3%). È questo, più della prevalenza, il motivo per cui il sottogruppo interessa alla tesi.

### Che cosa dice la letteratura (e perché queste scelte)
Sei fonti, DOI verificati su Crossref il 20/09/2026, elencate in Bibliografia → "Fase C". Tre punti decidono il protocollo.

1. **La PR-AUC non si confronta fra sottogruppi con prevalenza diversa.** Matos et al. 2026 (revisione sistematica delle metriche di equità, Lancet Digital Health): *"AUROC Parity is recommended for quantifying discrimination disparities, whereas AUPRC Parity is inadvisable due to its semi-proper scoring nature, lack of focus (mixing discrimination and clinical utility), and being a discriminatory metric that favours higher-prevalence subgroups"*. Stessa conclusione in Van Calster et al. 2025: *"The AUPRC has no clear interpretation and depends on the prevalence, which goes beyond assessing discrimination"*; la dimostrazione formale è in McDermott et al. 2024. Con 25,9% contro 8,7% la PR-AUC dei diabetici risulterebbe più alta **anche a parità di modello**: differenziarla produrrebbe un risultato inventato.
2. **La soglia è una scelta medica, non statistica.** Van Calster et al. 2025, Box 1: *"the decision threshold should be defined based on medical rather than statistical arguments"*. Quindi la soglia globale non si ristima sul sottogruppo. Lo stesso paper però osserva che una regola come "almeno 90% di sensibilità" *"Depending on specificity and prevalence this could require very different decision thresholds"*, e Matos et al. 2026 raccomandano di *"prioritise probability-based fairness metrics (that are not dependent on thresholds)"* proprio perché le metriche a soglia fissa valutano *"at specific and arbitrarily chosen decision cut-offs"*. Da qui: confronto fra sottogruppi **senza soglia** (AUROC), tabelle a soglia globale fissa come secondarie, e una tabella descrittiva che misura **quanto si sposterebbe** la soglia fra i diabetici.
3. **Con 68 positivi non si fanno test.** Riley et al. 2024 (parte 3): *"at least 100 events and 100 non-events are needed to estimate measures such as the c statistic ... and calibration slope"*. Ne abbiamo 68, e 16 casi gravi. TRIPOD+AI (Collins et al. 2024) item 23a chiede *"Report model performance estimates with confidence intervals, including for any key subgroups"*: stime con intervalli, non test.

### Protocollo
1. **Bracci: `none` e `level_weight`**, entrambi letti da `analytics/phase_b/oof_predictions.csv`.
   - `none` ("nessuna correzione") è l'**analisi principale**.
   - `level_weight` (pesi per livello KDIGO 1:2:3) è l'**unico** braccio di confronto, **descrittivo**, mai il titolo di un risultato. Motivo fissato ora: è l'unica tecnica progettata per i livelli gravi, e il sottogruppo diabetico è quello dove i casi gravi sono più densi (6,1% contro 1,3%). La domanda pre-registrata è: la tecnica costruita per i casi gravi si comporta diversamente proprio dove i casi gravi abbondano?
   - Le altre tecniche di campionamento e i due bracci CTGAN **non entrano** in Fase C: in Fase B non hanno mostrato guadagno e aggiungerebbero colonne senza potenza.
2. **Soglia e fasce: globali fisse**, lette da `analytics/phase_b/evaluation/cutpoints.csv`, riga per tecnica e modello. Non si ricalcolano sul sottogruppo.
3. **Metriche dentro ciascun gruppo** (diabetici, non diabetici, e training completo come riferimento), per i 4 modelli reali più il classificatore di maggioranza:
   - domanda 1: AUC (IC di DeLong) e PR-AUC (IC logit di Boyd), **sempre riportata accanto alla prevalenza del proprio gruppo**; precision, recall, specificità e quota di allerta alla soglia globale fissa (IC di Wilson);
   - domanda 2: probabilità media per livello KDIGO e concordanza di Jonckheere-Terpstra, con bootstrap stratificato sul livello (2.000 campioni, seed 42), come in Fase A e B;
   - domanda 3: sensibilità per livello alla soglia globale fissa, con riconosciuti/totale e IC di Wilson. **È il risultato principale: 16 casi gravi fra i diabetici**;
   - domanda 4: fasce globali contro livelli KDIGO e kappa pesato.
4. **Confronto diabetici − non diabetici: solo ΔAUROC.** I due gruppi sono disgiunti, quindi SE(Δ) = radice di (SE₁² + SE₂²) con gli errori standard di DeLong già implementati in `src/models/evaluation.py`. **La ΔPR-AUC non viene calcolata né riportata** (punto 1 della letteratura). La PR-AUC resta nelle tabelle solo dentro ciascun gruppo, accanto alla sua prevalenza.
5. **Nessun test formale, nessuna correzione per confronti multipli**: non ci sono ipotesi da rifiutare. Solo stime con intervalli, dichiarate descrittive.
6. **Controllo di coerenza**: fra i diabetici il classificatore di maggioranza deve dare AUC 0,5 e PR-AUC pari alla prevalenza del sottogruppo (0,259). Se non succede, c'è un errore di filtraggio.
7. **Soglia descrittiva**: in una tabella separata, la soglia che darebbe sensibilità 0,90 fra i **soli** diabetici, con la sensibilità sui casi gravi e la quota di allerta che ne seguirebbero. Serve a misurare quanto la politica globale sia arbitraria su questo sottogruppo. Etichettata "descrittiva"; **non viene usata in nessun'altra tabella**.

### Tabelle e figure
Tabelle in `analytics/phase_c/evaluation/`: `q1_discrimination.csv`, `q1_operating_points.csv`, `q2_levels.csv`, `q2_trend.csv`, `q3_sensitivity.csv`, `q4_bands.csv`, `q4_kappa.csv`, `subgroup_comparison.csv` (solo ΔAUROC), `threshold_descriptive.csv`.

Figure in `analytics/phase_c/`: `01_discrimination_diabetici.png` e `02_severe_cases.png`, descritte nell'Indice delle figure.

### Codice
Modulo nuovo `src/models/phase_c.py` più `tests/test_phase_c.py`. **Nessuna modifica ai moduli esistenti.**

Verifica fatta il 20/09/2026: **`evaluation.evaluate()` non è riusabile così com'è** per la Fase C, perché rifitterebbe soglia e fasce sul sottogruppo:
- `evaluate_model()` chiama `threshold(y, p)`, che ricalcola la soglia a sensibilità 0,90 sui soli soggetti passati;
- `evaluate()` calcola `proportions` dalla distribuzione dei livelli dei soggetti passati, quindi le fasce diventerebbero i quantili del sottogruppo.

Entrambe violerebbero il punto 2. `phase_c.py` inietta quindi soglia e fasce fisse e riusa gli helper di basso livello di `evaluation.py`: `delong`, `pr_auc`, `wilson`, `operating_point`, `jonckheere`, `assign_bands`, `stratified_indices`, `percentile_interval`.

### Correzioni alla sezione "Da fare per concludere il progetto"
Quella sezione è stata scritta prima di queste verifiche. Due punti della sua lista "Fase C" sono superati:
- **punto 2** indicava `analytics/phase_a/evaluation/cutpoints.csv`. Si usa invece `analytics/phase_b/evaluation/cutpoints.csv`. Motivo verificato il 20/09/2026: Fase A (set `main`) e Fase B (tecnica `none`) danno previsioni **identiche** per logistica SCORED, logistica penalizzata, Random Forest e classificatore di maggioranza, ma **diverse per XGBoost** (differenza massima 0,38, correlazione 0,94), perché il riferimento della Fase B prende XGBoost dall'analisi `depth_1_12` — è la convenzione già scritta in `configs/config.yaml`, voce `phase_b.baseline.overrides`. Le due soglie globali di XGBoost differiscono (0,04250 contro 0,04189). Usando il file della Fase B, `none` e `level_weight` vengono da un'unica fonte coerente e la Fase C resta confrontabile con la Fase B;
- **punto 7** dava per scontato che bastasse richiamare `evaluation.evaluate`. Non basta, per il motivo spiegato qui sopra in "Codice".

### Limiti, dichiarati in anticipo
- 68 positivi e 16 casi gravi: sotto la soglia minima di 100 eventi indicata da Riley et al. 2024. Gli intervalli saranno molto ampi e i risultati **non possono** sostenere una conclusione comparativa;
- il livello "molto alto" ha **5 soggetti** fra i diabetici: il bootstrap stratificato sul livello ricampiona 5 persone, quindi l'intervallo della probabilità media di quel livello è degenere. Va letto come indicazione, non come stima;
- il sottogruppo è definito dalla colonna `DM`, che è **anche una variabile in input** ai modelli (ed è uno dei predittori della logistica SCORED): non è un sottogruppo indipendente dal modello;
- `level_weight` è descrittivo: con 16 casi gravi nessuna differenza fra bracci è interpretabile;
- vale il limite generale già dichiarato nello Scope: sottogruppo diabetico piccolo, solo descrittivo.

---

## Fase C — risultati (20/09/2026)
Calcolati con il protocollo fissato prima (sezione precedente), sulle previsioni out-of-fold della Fase B. Nessun riaddestramento, nessuna ottimizzazione, test set non letto. Tabelle in `analytics/phase_c/evaluation/`, figure `analytics/phase_c/01`–`02`. Esecuzione: 2 minuti e 31 secondi.

**Controllo di coerenza superato**: il classificatore di maggioranza dà AUC 0,500 esatta e PR-AUC esattamente pari alla prevalenza in tutti e tre i gruppi.

### Difetto trovato dal controllo di coerenza (20/09/2026)
Alla prima esecuzione il controllo è **fallito**: fra i diabetici il classificatore di maggioranza del braccio `level_weight` dava AUC 0,454 invece di 0,500.

Causa, verificata: con i pesi per livello quel classificatore prevede 0,5, ma il valore salvato è `0,4999999999999938`–`0,4999999999999941`, cioè **tre valori distinti che differiscono di 3·10⁻¹⁶**, costanti dentro ogni fold. Aggregando i fold quel residuo in virgola mobile crea un **ordinamento fittizio** fra soggetti, che sposta l'AUC di 0,046. La Fase B non se ne era accorta perché dichiarava "AUC 0,5 in tutti i fold": dentro un singolo fold i valori sono identici e il problema non compare.

Correzione: `phase_c.denoise_constant` riporta alla media le previsioni che sono costanti dentro ogni fold (`evaluation.is_constant`) e il cui intervallo complessivo è sotto 10⁻¹². Sui quattro modelli reali la condizione non si verifica mai, quindi nessun risultato ne è toccato. È un difetto di **presentazione dei risultati del controllo**, non delle previsioni: nessuna tabella della Fase A o della Fase B va rifatta.

### Domanda 1: discriminazione (braccio "nessuna correzione", 4 modelli reali)

| gruppo | n | positivi | prevalenza | AUC | PR-AUC | PR-AUC / prevalenza |
|---|---|---|---|---|---|---|
| diabetici | 263 | 68 | 25,9% | 0,608–0,719 | 0,379–0,440 | **1,47–1,70** |
| non diabetici | 4.087 | 357 | 8,7% | 0,647–0,677 | 0,185–0,213 | **2,12–2,44** |

- **la PR-AUC dei diabetici sembra il doppio** (0,44 contro 0,19 per Random Forest) **ma è un artefatto della prevalenza**: rapportata alla prevalenza del proprio gruppo è più bassa, cioè il modello ordina i diabetici **peggio**, non meglio. È la conferma empirica di Matos et al. 2026 e McDermott et al. 2024 sui nostri dati, ed è il motivo per cui la differenza di PR-AUC non viene riportata
- **l'AUC non cambia**: differenze diabetici − non diabetici da **−0,038 a +0,042**, con intervalli che vanno da −0,124 a +0,117 e **coprono lo zero per tutti e quattro i modelli**. Con 68 positivi non si poteva vedere altro (Riley et al. 2024)

### Il risultato principale: alla soglia globale il modello segnala quasi tutti i diabetici

| gruppo | recall | specificità | quota di allerta | precision | casi gravi riconosciuti |
|---|---|---|---|---|---|
| diabetici | 0,985–1,000 | **0,000–0,036** | **0,970–1,000** | 0,259–0,263 | **16 su 16** |
| non diabetici | 0,882–0,885 | 0,220–0,268 | 0,746–0,789 | ~0,10 | 48–51 su 55 |

- **tutti e 16 i casi gravi diabetici sono riconosciuti da ogni modello e da entrambi i bracci**, ma la sensibilità è massima **solo perché il modello segnala il 97–100% dei diabetici**: la specificità è praticamente zero. Alla soglia globale, dentro questo sottogruppo, il modello **non aggiunge nulla rispetto a "testare tutti"**
- è la verifica empirica di quanto lo Scope già affermava per via clinica: *"nelle persone con diabete le linee guida prescrivono ACR ed eGFR ogni anno. L'utilità del modello è stabilire la priorità degli esami nella popolazione generale di screening"*. Ora non è più solo un'argomentazione: è un numero

### Soglia descrittiva (non usata altrove)
Per avere sensibilità 0,90 **fra i soli diabetici** servirebbe una soglia da **2,5 a 3,2 volte** quella globale (da 0,042–0,058 a 0,109–0,162). A quella soglia la quota di allerta scenderebbe a 0,707–0,829 e i casi gravi riconosciuti sarebbero 15 o 16 su 16.

È la misura di quanto la politica globale sia arbitraria su questo sottogruppo, e conferma alla lettera Van Calster et al. 2025: *"Depending on specificity and prevalence this could require very different decision thresholds"*.

### Domande 2 e 4
- **tendenza** (concordanza di Jonckheere-Terpstra): diabetici 0,619–0,712, non diabetici 0,646–0,676. Il rischio stimato cresce con la gravità KDIGO anche nel sottogruppo, con intervalli molto più larghi
- **fasce contro livelli** (kappa pesato): diabetici 0,124–0,219, non diabetici 0,144–0,197. Concordanza osservata molto più bassa fra i diabetici (0,35–0,55 contro 0,88): le fasce globali, applicate a un gruppo che il modello colloca quasi tutto in alto, si sbilanciano verso le fasce superiori. È l'informazione cercata, non un difetto

### Braccio descrittivo: pesi per livello KDIGO
Nessun beneficio nel sottogruppo dove i casi gravi sono più densi. Fra i diabetici l'AUC scende (0,607–0,650 contro 0,608–0,719 senza correzione) e la differenza contro i non diabetici resta negativa per tutti i modelli (da −0,040 a −0,029). I casi gravi riconosciuti restano 16 su 16, per lo stesso motivo di prima. Con 16 casi gravi nessuna di queste differenze è interpretabile: è materiale descrittivo, come fissato nel protocollo.

### Sintesi per la tesi (risposta alla domanda 6)
1. **La discriminazione non cambia** fra diabetici e non diabetici: differenze di AUC entro ±0,04, tutti gli intervalli coprono lo zero
2. **La PR-AUC sembra molto migliore fra i diabetici e non lo è**: 0,44 contro 0,19 è l'effetto della prevalenza 25,9% contro 8,7%. Rapportata alla propria prevalenza è peggiore. È il secondo "miglioramento apparente" della tesi, dopo quello della soglia 0,5 in Fase B, e questa volta riguarda la scelta della metrica
3. **Alla soglia globale il modello degenera in "testare tutti" sui diabetici**: specificità 0,000–0,036, quota di allerta 0,97–1,00. Riconosce tutti e 16 i casi gravi senza fornire informazione
4. **La soglia che servirebbe davvero sui diabetici è 2,5–3,2 volte quella globale**: la stessa regola clinica ("almeno 90% di sensibilità") implica soglie molto diverse in popolazioni con prevalenza diversa
5. **i pesi per livello KDIGO non aiutano nemmeno qui**, dove i casi gravi sono quasi cinque volte più densi (6,1% contro 1,3%)
6. conclusione operativa: il modello ha senso **nella popolazione generale di screening**, non nei diabetici, dove le linee guida prescrivono già gli esami ogni anno

### Limiti
Quelli dichiarati in anticipo nel protocollo, tutti confermati dai numeri: 68 positivi e 16 casi gravi (sotto i 100 eventi minimi di Riley et al. 2024), 5 soggetti "molto alto" con bootstrap degenere, sottogruppo definito da `DM` che è anche una variabile in input, braccio `level_weight` solo descrittivo.

---

### Revisione indipendente del codice della Fase C (20/09/2026)
Revisione di `src/models/phase_c.py`, `src/analytics/phase_c_report.py`, `tests/test_phase_c.py` e del blocco `phase_c` di `configs/config.yaml`, fatta da un agente revisore separato in sola lettura, con il protocollo pre-registrato come riferimento: **nessun difetto critico**.

Verificati e confermati: soglia e fasce mai ristimate sul sottogruppo (`phase_c.py` non chiama mai `evaluate`, `evaluate_model`, `threshold` o `band_cutpoints`, solo gli helper di basso livello dichiarati); nessuna differenza di PR-AUC calcolata o esposta; nessuna colonna di p in nessuna delle 9 tabelle (`trend_row` scarta esplicitamente il `p_value` di Jonckheere); test set mai letto, né direttamente né tramite i moduli importati; bracci limitati a `none` e `level_weight`; somma in quadratura degli errori standard legittima perché i due gruppi sono disgiunti per costruzione; nessun NaN e nessuna riga persa nei `groupby` (cardinalità esatta 2 × 5 × 3 e 2 × 4 × 3); il livello "molto alto" con 5 soggetti non degenera in NaN; `denoise_constant` con soglia 10⁻¹² non può attivarsi su un modello reale (scarto tipico ≥ 10⁻²). Confermato anche che `test_threshold_is_the_given_one_not_refitted`, `test_bands_are_the_given_ones_not_quantiles` e `test_delong_se_matches_evaluation` non sono tautologici: fallirebbero davvero se il difetto che coprono fosse presente.

- **corretto (problema maggiore)**: se un modello **vero** fosse assente da `cutpoints.csv` (rigenerazione parziale della Fase B, refuso nel nome, disallineamento fra i due file), veniva **escluso in silenzio** da `q3_sensitivity`, cioè dal risultato principale della Fase C, senza errore né avviso. È la stessa classe di difetto di `pooled()` in Fase B. Ora la soglia può mancare solo se le previsioni sono costanti dentro ogni fold (`evaluation.is_constant`), cioè solo per il classificatore di maggioranza; in ogni altro caso è un `RuntimeError` esplicito. Test aggiunto: `test_missing_cutpoints_for_a_real_model_is_an_error`
- **corretto (minore)**: la chiave `phase_c.output` in `configs/config.yaml` non era letta da nessun modulo ed è stata tolta; la chiave `phase_c.primary` non era letta e la figura ricavava il braccio principale da `TECHNIQUES[0]`, cioè da una seconda fonte di verità che poteva divergere. Ora `phase_c.PRIMARY` e `phase_c.SECONDARY` vengono dalla config, con un controllo all'import che i bracci siano due e che `primary` sia fra loro
- **non corretto, con motivazione**: `phase_c.delong_se` duplica la formula di `evaluation.delong`. Il revisore la giudica giustificata dal vincolo "nessuna modifica ai moduli esistenti" e dal fatto che l'intervallo di `evaluation.delong` è troncato a [0, 1], quindi non sempre permette di risalire all'errore standard. Il rischio di disallineamento futuro è coperto da `test_delong_se_matches_evaluation`

Dopo le correzioni: 25 test della Fase C superati; tabelle e figure rigenerate; controllo di coerenza del classificatore di maggioranza ancora superato (AUC 0,500 esatta, PR-AUC uguale alla prevalenza).

---

## Qualità e utilità clinica — protocollo post-hoc (22/09/2026)
**Dichiarazione**: questo blocco è stato deciso **dopo** aver visto i risultati delle Fasi A, B, C e D. Non era previsto dallo Scope. Nasce da una domanda legittima: le misure usate finora (AUC, PR-AUC, sensibilità per livello) dicono quanto il modello ordina bene i soggetti, non se **serva a qualcosa**. Essendo post-hoc, va letto come analisi esplorativa, non come verifica di un'ipotesi pre-registrata. Numeri e regole sono comunque scritti qui **prima di qualsiasi calcolo del blocco**.

Tutto sulle previsioni out-of-fold della Fase B, braccio "nessuna correzione" (`analytics/phase_b/oof_predictions.csv`; per XGBoost è l'analisi `depth_1_12`, come in Fase C, così previsioni e soglie vengono da un'unica fonte). Quattro modelli reali (logistica SCORED, logistica penalizzata, Random Forest, XGBoost) più il classificatore di maggioranza come controllo di coerenza. I candidati della Fase D non entrano: nessuno ha superato la regola fissata. **Il test set non viene letto.**

### Perché queste misure (letteratura)
Fonti in Bibliografia → "Utilità clinica e misure di qualità (post-hoc)" e → "Fase C". DOI verificati su Crossref, citazioni verificate sul testo completo il 22/09/2026.

1. **Il set minimo raccomandato comprende utilità clinica e calibrazione, non solo l'AUROC.** Van Calster et al. 2025: *"We recommend the following measures and plots as essential to report: AUROC, calibration plot, a clinical utility measure such as net benefit with decision curve analysis, and a plot with probability distributions per outcome category."* Finora la tesi riporta la prima voce e i numeri della calibrazione: questo blocco completa l'elenco.
2. **MCC, F1 e accuratezza bilanciata sono improprie alla soglia clinica.** Van Calster et al. 2025: *"For a given decision threshold t, all classification measures are improper [...] This is because a decision threshold implies specific misclassification costs, but these are not used in classification measures."* Matos et al. 2026: *"MCC Parity is also highly questionable due to its highly complex formulation that hampers interpretability."* Si riportano perché i revisori le chiedono, ma solo in una tabella descrittiva, come ammette la stessa fonte: *"Although improper, the combination of PPV and NPV and/or the combination of sensitivity and specificity may be reported descriptively if desired, but always as an addition to the core set."*
3. **Il net benefit risponde alla domanda "serve?"** (Vickers & Elkin 2006): `NB(t) = VP/n − FP/n × t/(1−t) = sensibilità × prevalenza − (1 − specificità) × (1 − prevalenza) × t/(1−t)`, confrontato con "testare tutti" (`prevalenza − (1 − prevalenza) × t/(1−t)`) e "non testare nessuno" (0). La soglia `t` fissa il cambio: un caso trovato vale `(1−t)/t` esami inutili. Unità: casi trovati netti per persona (×100 = ogni 100 persone).
4. **L'intervallo di soglie si fissa prima, e la curva non serve a sceglierne una.** Vickers, van Calster & Steyerberg 2019: *"Investigators should first work out a clinically reasonable range of threshold probabilities [...] They should then determine whether the net benefit of their model or test is better than alternatives across this range"*; usare la curva per scegliere la soglia è *"a frequent and fundamental misunderstanding"*.
5. **Quando il riferimento è "testare tutti", il net benefit si esprime anche come esami evitati.** Stessa fonte: *"Expressing net benefit in terms of avoided unnecessary diagnostic procedures or avoided unnecessary treatments is recommended if the reference strategy is 'intervention for all.'"* Formula: `esami inutili evitati ogni 100 = 100 × (NB_modello − NB_tutti) / (t/(1−t))`. Cambia l'unità, non le conclusioni.
6. **Niente test né intervalli sul net benefit.** Vickers et al. 2023: *"Null hypothesis testing or simple consideration of confidence intervals are of questionable value for decision curve analysis"*; Van Calster et al. 2025 chiedono intervalli *"where possible, with the exception of clinical utility measures"*. Gli intervalli restano per calibrazione e proporzioni.
7. **Net benefit nei sottogruppi.** Matos et al. 2026 indicano come unica misura di utilità clinica per i sottogruppi il *subgroup net benefit* di Benitez-Aurioles et al. 2024 (loro riferimento 88), che *"allows each subgroup to have its own prevalence term"*. **Definizione verificata sul testo di Benitez-Aurioles et al. 2024 (equazione 5)**: `sNB_g(t) = 1 − π_g + λ × NB_g(t)`, dove `NB_g` è il net benefit ordinario calcolato dentro il gruppo (sua numerosità, sua prevalenza `π_g`) e `λ` è la riduzione relativa del rischio del trattamento che segue la decisione. Dentro un gruppo `1 − π_g` e `λ` sono gli stessi per tutte le strategie, quindi **la classifica fra modello, "testare tutti" e "non testare nessuno" dipende solo da `NB_g`, per qualunque λ > 0**. Si riporta quindi `NB_g`; λ non viene scelto, perché richiederebbe l'effetto del trattamento dopo la diagnosi, fuori dal perimetro della tesi. Il net benefit **non si confronta fra gruppi**: dipende dalla prevalenza, e con prevalenza più bassa ci si può aspettare solo un net benefit più basso (Matos et al. 2026). *Correzione rispetto agli appunti del 20/09/2026*: lì "subgroup net benefit" era usato come sinonimo di "net benefit con la prevalenza del gruppo", prima di aver letto la definizione formale, che aggiunge il termine `1 − π_g` e il peso λ.
8. **Calibrazione.** Van Calster et al. 2019: la calibrazione moderata *"is assessed with a flexible calibration curve [...] for example, using loess or spline functions"*; per una curva precisa *"a minimum of 200 patients with and 200 patients without the event has been suggested"*; la calibrazione forte, cioè per ogni combinazione dei predittori, *"is a utopic goal"*; il test di Hosmer-Lemeshow è sconsigliato. Quindi la calibrazione dentro i sottogruppi è una **condizione necessaria** della calibrazione forte, non la sua verifica. TRIPOD+AI (item 23a) chiede le stime nei sottogruppi con intervalli di confidenza. Misure: intercetta e pendenza (Van Calster et al. 2016, già implementate in `evaluation_b.calibration`), rapporto O:E (Van Calster et al. 2025: *"An O:E ratio > 1 indicates underestimation"*), Brier.

### Intervallo di soglie: 0,02–0,20, fissato prima di calcolare
- **5% e 7%** sono le due soglie che Bragg-Gresham et al. 2024 usano come esempi di punto operativo per questa stessa decisione (mandare un adulto senza diabete a fare l'esame dell'albuminuria): *"employing a cut-point of 0.05 for screening would require just under half of the non-diabetic population to be screened, but would detect 85% of individuals with albuminuria"*; al 7% *"we could detect 73.2% of individuals with albuminuria by screening 37.7% of the non-diabetic population"*. La soglia già in uso nel progetto (0,042–0,058, sensibilità 0,90) cade in quel range
- **0,02** = 49 esami inutili accettati per caso trovato, circa $2.400 a caso ai $49 per test ACR di Cusick et al. 2023. Difendibile perché lo stesso lavoro trova lo screening di popolazione già costo-efficace ($86.300 per QALY, una tantum a 55 anni)
- **0,20** = 4 esami inutili per caso. Oltre, si starebbe assumendo che un esame da $49 pesi quanto un intervento
- griglia: da 0,020 a 0,200 con passo 0,005 (37 soglie), in `configs/config.yaml`, blocco `quality`

### Che cosa si calcola
1. **Decision curve** dei 4 modelli contro "testare tutti" e "non testare nessuno", su tutto il training. Positivo = `p ≥ t`, come nel resto del progetto. Per ogni modello: le soglie in cui `NB_modello > max(NB_tutti, 0)` ed esami inutili evitati ogni 100
2. **Net benefit dentro i sottogruppi** diabetici (263, prevalenza 25,9%) e non diabetici (4.087, 8,7%), ciascuno contro "testare tutti" e "non testare nessuno" **dello stesso gruppo**. Domanda: fra i diabetici il modello batte "testare tutti", cioè quello che le linee guida già prescrivono?
3. **Curva di calibrazione**, tutto il training: curva flessibile = regressione logistica di y su una spline cubica ristretta del logit di p, 4 nodi ai quantili 5, 35, 65 e 95% di logit(p) (i valori di Harrell 2015), con banda puntuale al 95% dall'errore standard del modello (metodo delta); punti ai decili di p con intervallo di Wilson; sotto, la distribuzione delle probabilità per esito (il *"plot with probability distributions per outcome category"* del punto 1). Intercetta, pendenza, O:E e Brier con IC da bootstrap stratificato sul livello KDIGO (2.000 campioni, seed 42, come nelle altre fasi)
4. **MCC, F1, accuratezza bilanciata** (più sensibilità, specificità, VPP e VPN) alla soglia del progetto (`analytics/phase_b/evaluation/cutpoints.csv`, sensibilità 0,90) e alle soglie 0,05 e 0,07. Tabella descrittiva, dichiarata impropria nel titolo; mai in una figura, mai come risultato
5. **Confronto con Bragg-Gresham et al. 2024**:
   - (a) quota di soggetti da esaminare per trovare l'85% dei positivi (soglia a sensibilità 0,85), bersaglio della tesi, tutto il training;
   - (b) versione più vicina al loro disegno: **solo non diabetici, bersaglio sola albuminuria (ACR ≥ 30 mg/g)**, stesse previsioni: quota esaminata e quota di casi trovati a 0,05 e 0,07, e quota da esaminare a sensibilità 0,85. I modelli non sono addestrati per quel bersaglio, quindi la stima è prudente;
   - caveat da scrivere sempre accanto ai numeri: popolazione (adulti USA, NHANES 1999–2020, contro screening cinese); definizione di diabete (loro anche HbA1c ≥ 6,5%, qui la colonna `DM`); **predittori: il loro modello usa eGFR < 60, cioè la creatinina, che la tesi esclude per costruzione** perché è metà del bersaglio; c-statistic 0,752 in validazione. È un ordine di grandezza, non un testa a testa
6. **Punti operativi a 0,05 e 0,07 esatti**, per gruppo: esami ogni 100 persone, casi trovati e mancati ogni 100, casi gravi (alto + molto alto) riconosciuti con IC di Wilson, net benefit, esami evitati
7. **Costo per caso trovato** a ogni soglia della griglia: esami per caso trovato (`esami / veri positivi`) × $49, con l'intervallo $36–$64 di Cusick et al. 2023; "testare tutti" come riferimento (`1 / prevalenza` esami per caso). In più il **costo incrementale per caso aggiuntivo** passando dal modello a "testare tutti": `$49 × (n − esami_modello) / (VP_tutti − VP_modello)`, cioè quanto costa ciascun caso che il modello lascia indietro e che "testare tutti" recupera. **Non è un'analisi costo-efficacia**: niente QALY, niente costi a valle, niente effetto del trattamento. Si conta solo il test ACR (la creatinina per l'eGFR non è conteggiata, quindi il costo è sottostimato); gli esami di routine che alimentano il modello si assumono già disponibili
8. **Calibrazione nei sottogruppi** (diabetici, non diabetici): stesse misure e stessa curva del punto 3 dentro ciascun gruppo, bootstrap stratificato sul livello dentro il gruppo. Fra i diabetici gli eventi sono 68, sotto i 200 suggeriti da Van Calster et al. 2019: la curva sarà imprecisa, dichiarato ora

### Controlli di coerenza, fissati prima
- "non testare nessuno" vale 0 a ogni soglia; il classificatore di maggioranza (prevede 0,0977 per tutti) deve coincidere con "testare tutti" per `t ≤ 0,0977` e con "non testare nessuno" sopra
- le due formule del net benefit (dai conteggi; da sensibilità, specificità e prevalenza) devono coincidere
- il net benefit è collassabile: `n_diabetici × NB_diabetici + n_non_diabetici × NB_non_diabetici = n × NB_tutti` a ogni soglia e per ogni strategia
- intercetta, pendenza e Brier su tutto il training devono coincidere con `analytics/phase_b/evaluation/calibration.csv` (tecnica `none`, probabilità grezze)

### Impegni presi prima di calcolare
- il risultato viene riportato **qualunque sia**, anche se il modello non batte "testare tutti" in nessuna parte dell'intervallo
- intervallo e griglia di soglie non vengono ritoccati dopo aver visto la curva
- nessun test formale, nessun intervallo sul net benefit
- nessuna soglia "raccomandata" ricavata dalla curva (Vickers et al. 2019)
- nessuna conclusione di costo-efficacia dai numeri del punto 7

### Codice
Modulo nuovo `src/models/clinical_utility.py` più `tests/test_clinical_utility.py`; tabelle in `analytics/quality/evaluation/`, figure in `analytics/quality/` generate da `src/analytics/quality_report.py`; blocco `quality` in `configs/config.yaml`. Nessuna modifica ai moduli esistenti: si riusano `evaluation` (`operating_point`, `wilson`, `threshold_at_sensitivity`, `stratified_indices`, `percentile_interval`), `evaluation_b` (`pooled`, `calibration`) e `phase_c` (`groups`, `denoise_constant`).

### Limiti, dichiarati in anticipo
- post-hoc ed esplorativo;
- previsioni out-of-fold di un solo training, nessuna validazione esterna: la decision curve vale per una popolazione con prevalenza 9,8%;
- 68 positivi fra i diabetici: net benefit e calibrazione del sottogruppo sono instabili;
- un solo costo, in dollari USA, da un'analisi statunitense: la traduzione economica è illustrativa;
- il confronto con Bragg-Gresham è fra studi con popolazione, bersaglio e predittori diversi.

---

## Qualità e utilità clinica — risultati (22/09/2026)
Calcolati con il protocollo della sezione precedente, sulle previsioni out-of-fold della Fase B. Nessun riaddestramento, test set non letto. Tabelle in `analytics/quality/evaluation/` (11 CSV), figure `analytics/quality/01`–`06`. Esecuzione: 2 minuti e 30 secondi; due esecuzioni successive danno tabelle identiche byte per byte.

**Controlli di coerenza tutti superati**: le due formule del net benefit coincidono (tolleranza 10⁻¹²); il classificatore di maggioranza coincide con "testare tutti" fino a 0,095 e con "non testare nessuno" da 0,100, in tutti e tre i gruppi; il net benefit è collassabile (diabetici + non diabetici = totale, a ogni soglia e per ogni modello); intercetta, pendenza e Brier su tutto il training coincidono con `analytics/phase_b/evaluation/calibration.csv`.

### Deviazione dal protocollo, dichiarata: intervalli della calibrazione
Il protocollo fissava intervalli da bootstrap stratificato sul livello KDIGO, come nelle altre fasi. **Per la calibrazione quella scelta è sbagliata**, e lo si è visto dai numeri: ogni livello sopra "basso" è positivo per definizione, quindi stratificando sul livello **ogni campione ha esattamente lo stesso numero di eventi** (425 su tutto il training, 68 fra i diabetici). L'intercetta e il rapporto O:E misurano proprio lo scarto fra eventi osservati e attesi, e con il numero di eventi fisso la loro variabilità principale sparisce: fra i diabetici l'O:E della logistica SCORED risultava 0,957–1,045, mentre la sola variabilità binomiale di 68 eventi vale circa ±20%. Sulle misure di discriminazione delle fasi precedenti (AUC, sensibilità) stratificare sull'esito è corretto; qui no.

Correzione: accanto agli intervalli pre-registrati (colonne `*_low`, `*_high` di `calibration.csv`, conservate) sono calcolati intervalli da **bootstrap semplice** (colonne `*_low_simple`, `*_high_simple`, stessi 2.000 campioni e seme 42). **Nel testo e nelle figure si usano quelli semplici.** Le pendenze cambiano poco; intercetta e O:E si allargano di 4–5 volte. Test che documenta il meccanismo: `test_level_stratified_bootstrap_fixes_the_number_of_events`. Nessun'altra tabella è toccata.

### 1. Decision curve su tutto il training (figura 01)
Net benefit ogni 100 persone ed esami inutili evitati ogni 100 rispetto a "testare tutti":

| soglia | "testare tutti" | modelli (NB ogni 100) | esami inutili evitati ogni 100 |
|---|---|---|---|
| 2% | 7,93 | 7,93–7,94 | 0,0 a +0,7 |
| 5% | 5,02 | 5,03–5,14 | +0,1 a +2,3 |
| 7% | 2,98 | 3,67–3,97 | **+9,2 a +13,1** |
| 10% | −0,26 | 2,51–2,83 | +24,9 a +27,8 |
| 15% | −6,15 | 1,46–1,94 | +43,1 a +45,8 |
| 20% | −12,79 | 0,94–1,41 | +54,9 a +56,8 |

- **Sotto il 5% il modello è indistinguibile da "testare tutti"**: le differenze oscillano fra −4 e +2 esami ogni 100, di segno alterno fra soglie vicine. È atteso: a soglie così basse il modello segnala quasi tutti
- **dal 5–6% in su il modello batte entrambe le strategie in modo continuo**, per tutti e quattro i modelli (logistica penalizzata e Random Forest da 0,050, XGBoost da 0,045, logistica SCORED da 0,060). Il vantaggio cresce con la soglia: 9–13 esami inutili evitati ogni 100 persone al 7%, 25–28 al 10%
- sopra il 9,8% (la prevalenza) "testare tutti" ha net benefit negativo, cioè fa più danno che utile; il modello resta positivo fino al 20%
- **nessun modello domina**: le quattro curve si intrecciano entro 0,5 casi ogni 100 (mediana 0,3 dal 5% in su). Coerente con la Fase A (nessuna differenza significativa di AUC) e con Christodoulou et al. 2019

### 2. Net benefit dentro i sottogruppi (figura 02)
- **diabetici** (prevalenza 25,9%): "testare tutti" ha net benefit positivo su tutta la griglia (da 24,3 a 7,3 ogni 100). **Fino al 10% nessun modello fa meglio di "testare tutti"**: esami evitati fra −15,8 e +3,4 ogni 100, cioè zero entro il rumore; XGBoost è sotto "testare tutti" quasi ovunque fino al 15%. Solo sopra il 12% Random Forest e logistica penalizzata guadagnano qualcosa (+5,7 e +4,6 esami evitati al 12%, +16 e +13 al 20%), soglie che nessuno userebbe per un esame da $49 in un gruppo a rischio così alto
- **non diabetici** (prevalenza 8,7%): stesso quadro del training completo, con "testare tutti" che diventa negativo già all'8,7%; al 7% il modello evita 10–14 esami inutili ogni 100
- è la **terza conferma**, con una terza misura, del risultato della Fase C: sui diabetici il modello non aggiunge nulla a quello che le linee guida già prescrivono. In Fase C lo diceva la specificità (0,000–0,036), qui lo dice il net benefit, che tiene conto anche della prevalenza del gruppo

### 3 e 8. Calibrazione, complessiva e nei sottogruppi (figure 04 e 05)
Intervalli da bootstrap semplice (vedi la deviazione sopra):

| gruppo | modello | intercetta | pendenza | O:E |
|---|---|---|---|---|
| tutti | logistica SCORED | 0,00 (−0,11; 0,10) | 0,97 (0,82; 1,11) | 1,00 (0,91; 1,09) |
| tutti | logistica penalizzata | −0,01 (−0,12; 0,10) | 0,94 (0,80; 1,07) | 0,99 (0,91; 1,08) |
| tutti | Random Forest | 0,00 (−0,11; 0,09) | **1,31 (1,13; 1,48)** | 1,00 (0,91; 1,09) |
| tutti | XGBoost | 0,01 (−0,10; 0,11) | **0,86 (0,75; 0,97)** | 1,01 (0,92; 1,10) |
| diabetici | logistica SCORED | 0,00 (−0,30; 0,28) | 0,82 (0,30; 1,39) | 1,00 (0,80; 1,21) |
| diabetici | logistica penalizzata | 0,18 (−0,15; 0,48) | 0,69 (0,38; 1,09) | 1,12 (0,90; 1,37) |
| diabetici | Random Forest | **0,31 (0,02; 0,58)** | **1,79 (1,10; 2,64)** | **1,26 (1,01; 1,52)** |
| diabetici | XGBoost | 0,18 (−0,14; 0,49) | 0,68 (0,37; 1,07) | 1,13 (0,91; 1,37) |
| non diabetici | tutti e quattro | −0,05 a 0,00 | 0,83–1,23 | 0,96–1,00 |

- **in media la calibrazione è buona** (intercetta 0, O:E 1 per tutti i modelli), **ma la "pendenza 1,02" riportata in Fase B è una media che nasconde due errori opposti**: Random Forest ha pendenza 1,31 (probabilità troppo schiacciate verso la media: sottostima i soggetti ad alto rischio, visibile nella curva sopra il 15%) e XGBoost 0,86 (probabilità troppo estreme). Le due logistiche sono calibrate. Il numero della Fase B era corretto come media; da solo era fuorviante
- la curva flessibile resta vicina alla diagonale dove sta la grande maggioranza dei soggetti (probabilità sotto il 15%, istogrammi della figura 04), cioè proprio nell'intervallo delle soglie della decision curve
- **nei sottogruppi (controllo di equità, TRIPOD+AI item 23a)**: fra i non diabetici nessun problema. **Fra i diabetici Random Forest sottostima il rischio di circa un quinto** (probabilità media 20,6% contro 25,9% osservato; O:E 1,26, intervallo che esclude 1). Logistica penalizzata e XGBoost vanno nella stessa direzione (O:E 1,12–1,13) con intervalli che includono 1; la logistica SCORED, che ha il diabete fra i suoi 5 predittori, è calibrata anche lì. Meccanismo plausibile, coerente con la pendenza: un modello che schiaccia le probabilità verso la media penalizza il gruppo più a rischio
- conseguenza pratica limitata: sui diabetici le linee guida prescrivono comunque l'esame (punto 2), quindi la sottostima non cambia nessuna decisione. Va dichiarata come limite di equità, non nascosta
- 68 eventi fra i diabetici, sotto i 200 suggeriti da Van Calster et al. 2019: le curve di quel gruppo sono larghe, come dichiarato nel protocollo

### 4. MCC, F1, accuratezza bilanciata — misure improprie, solo descrittive
Tutto il training, `classification_descriptive.csv`:

| soglia | MCC | F1 | accuratezza bilanciata | sensibilità | specificità | VPP | VPN |
|---|---|---|---|---|---|---|---|
| del progetto (sensibilità 0,90) | 0,08–0,11 | 0,20–0,21 | 0,56–0,58 | 0,90 | 0,21–0,26 | 0,11–0,12 | 0,95–0,96 |
| 5% | 0,07–0,12 | 0,19–0,22 | 0,54–0,60 | 0,84–0,94 | 0,14–0,36 | 0,11–0,12 | 0,95–0,96 |
| 7% | 0,13–0,16 | 0,22–0,24 | 0,61–0,63 | 0,71–0,83 | 0,39–0,55 | 0,13–0,15 | 0,94–0,96 |

Con MCC 0,08 e F1 0,20 il modello sembrerebbe inutile; la decision curve dice che al 7% evita 9–13 esami inutili ogni 100 persone. Le due letture divergono perché MCC e F1 pesano allo stesso modo un caso mancato e un esame inutile, mentre alla soglia del 7% un caso mancato vale 13 esami inutili: è esattamente l'argomento di Van Calster et al. 2025. Queste misure restano in tabella perché i revisori le chiedono, mai come risultato.

### 5. Confronto con Bragg-Gresham et al. 2024 (figura 06)

| disegno | quota esaminata a sensibilità 0,85 | alla soglia 5%: esaminati / trovati | alla soglia 7%: esaminati / trovati |
|---|---|---|---|
| tesi, tutto il training, bersaglio composito | 67,8–73,8% | 65,7–86,7% / 83,5–94,4% | 47,7–63,5% / 71,3–83,3% |
| non diabetici, sola albuminuria (più vicino a loro) | 68,7–76,9% | 63,8–85,8% / 80,4–93,5% | 45,1–61,3% / 65,8–79,2% |
| **Bragg-Gresham et al. 2024** | — | "poco meno di metà" / 85% | 37,7% / 73,2% |

- **a parità di casi trovati i nostri modelli devono esaminare più persone**, anche nel disegno più vicino al loro (non diabetici, sola albuminuria): per trovare il 73% dei casi 49,5–60,1% contro 37,7%, cioè **12–22 persone in più ogni 100**; per trovarne l'85% 68,7–76,9% contro "poco meno di metà", cioè **almeno 19–27 in più**. Avvicinare il disegno al loro non cambia l'ordine di grandezza del divario
- spiegazione principale, già nel protocollo: il loro modello usa **eGFR < 60 fra i predittori**, cioè la creatinina, che la tesi esclude per costruzione; più una coorte NHANES di 44.322 adulti contro 4.350 soggetti. La c-statistic 0,752 in validazione contro 0,67–0,69 dei nostri modelli sulla sola albuminuria (Fase D) è coerente col divario
- la quota di XGBoost a sensibilità 0,85 (67,8%) differisce da quella annotata il 20/09 (66,2%) perché qui XGBoost viene dall'analisi `depth_1_12`, come in tutte le analisi dalla Fase B in poi

### 6. Punti operativi a 0,05 e 0,07 (`operating_points.csv`)

| gruppo | soglia | esami ogni 100 | casi trovati ogni 100 | casi mancati ogni 100 | casi gravi riconosciuti |
|---|---|---|---|---|---|
| tutti | 5% | 65,7–86,7 | 8,2–9,2 (su 9,8) | 0,6–1,6 | 62–67 su 71 |
| tutti | 7% | 47,7–63,5 | 7,0–8,1 | 1,6–2,8 | 54–63 su 71 |
| diabetici | 5% | 94,7–100 | 25,5–25,9 (su 25,9) | 0,0–0,4 | 16 su 16 |
| diabetici | 7% | 88,6–100 | 24,0–25,9 | 0,0–1,9 | 15–16 su 16 |

Al 7% il modello manda in laboratorio circa metà della popolazione e riconosce 54–63 dei 71 casi gravi: fra i casi mancati ci sono 8–17 casi alto o molto alto. Il costo clinico del risparmio va scritto accanto al risparmio.

### 7. Costo per caso trovato (figura 03, `costs.csv`)
"Testare tutti" costa **$502 per caso trovato** (10,2 esami per caso, $49 ciascuno).

| soglia | costo per caso trovato | risparmio ogni 100 persone | casi mancati ogni 100 | costo di ogni caso in più trovato testando tutti |
|---|---|---|---|---|
| 5% | $394–461 | $653–1.683 | 0,6–1,6 | $987–1.184 |
| 7% | $336–383 | $1.787–2.562 | 1,6–2,8 | $880–1.095 |
| 10% | $272–285 | $3.163–3.382 | 3,7–4,4 | $766–860 |

Con l'intervallo di Cusick et al. 2023 ($36–$64) tutti i valori scalano del −27% / +31%.

- il modello **abbassa il costo per caso trovato** (dell'8–21% al 5%, del 24–33% al 7%) **ma lo fa trovando meno casi**. Il numero che conta è l'ultima colonna: ogni caso che il modello lascia indietro si recupererebbe testando tutti a circa **$900–1.200**
- lettura coerente con la decision curve, senza nessuna analisi in più: la soglia `t` dichiara quanto vale un caso trovato, `(1−t)/t` esami. Al 5% un caso vale 19 esami, cioè $931, e recuperarlo testando tutti costa $987–1.184: quasi pari, infatti al 5% il net benefit del modello e quello di "testare tutti" quasi coincidono. Al 7% un caso vale 13,3 esami ($651) e recuperarlo costa $880–1.095: il modello conviene, e la decision curve lo mostra
- **non è una conclusione di costo-efficacia** (impegno del protocollo): Cusick et al. 2023 trovano costo-efficace lo screening di tutta la popolazione a $86.300 per QALY, un'analisi con esiti, trattamenti e costi a valle che qui non c'è. Da questi numeri non si può dire che il modello "fa risparmiare" il sistema sanitario: si può dire quanto costa, in esami, ogni caso che si sceglie di non cercare

### Sintesi per la tesi
1. **Il modello ha utilità clinica nella popolazione del dataset (prevalenza 9,8%, in maggioranza senza diabete noto), ma solo da soglie del 5–6% in su.** Al 7% evita 9–13 esami inutili ogni 100 persone rispetto a "testare tutti", a parità di beneficio; al 10% 25–28. Sotto il 5% equivale a testare tutti
2. **alle due soglie di Bragg-Gresham il quadro è diviso**: al 5% il guadagno è quasi nullo (0–2 esami ogni 100), al 7% è moderato. Quale delle due sia giusta è una scelta clinica, non statistica, e la curva non la decide (Vickers et al. 2019)
3. **sui diabetici nessun modello batte "testare tutti" fino al 10%**: terza conferma, con una terza misura, che lì hanno ragione le linee guida
4. **la calibrazione è buona in media ma non per tutti i modelli**: Random Forest schiaccia le probabilità (pendenza 1,31) e sottostima di un quinto il rischio dei diabetici; XGBoost le estremizza (0,86). La "pendenza media 1,02" della Fase B nascondeva due errori opposti: è il **terzo "risultato apparente"** della tesi, dopo la soglia 0,5 (Fase B) e la PR-AUC dei diabetici (Fase C), e riguarda di nuovo un numero riassuntivo letto da solo
5. **il risparmio è reale ma ha un prezzo**: al 7%, $1.800–2.600 in meno ogni 100 persone, 1,6–2,8 casi mancati ogni 100 (8–17 casi gravi su 71 nel training). Ogni caso mancato costerebbe $900–1.100 da recuperare. Nessuna conclusione di costo-efficacia
6. **rispetto a Bragg-Gresham servono 12–27 persone esaminate in più ogni 100** per trovare gli stessi casi: il prezzo di non usare la creatinina, coerente con il tetto trovato in Fase D
7. **MCC 0,08–0,16 e F1 0,20–0,24 farebbero giudicare il modello inutile**: la decision curve mostra perché queste misure, improprie alla soglia clinica, non rispondono alla domanda

### Limiti
Quelli dichiarati nel protocollo, più due emersi dai risultati:
- la deviazione sugli intervalli della calibrazione, dichiarata sopra;
- **provenienza del dataset (verifica del 22/09/2026 sul testo completo di Li J et al. 2026)**: i dati vengono dal reparto di Diabetologia ed Endocrinologia dello Shanghai Sixth People's Hospital, febbraio-aprile 2012; gli autori non li descrivono come screening di popolazione, e non documentano né il tipo di campione urinario per l'ACR né le unità di `UCRE` e `UmALB`. La decision curve vale per una coorte ospedaliera con prevalenza 9,8%; la formula "popolazione generale di screening" usata nello Scope e nelle fasi precedenti va rivista;
- il net benefit ha una precisione limitata dagli stessi 425 positivi del training: le differenze fra modelli (≤ 0,5 casi ogni 100) non sono interpretabili, quelle contro "testare tutti" oltre il 6% sì, perché crescono fino a decine di esami ogni 100. Nessun intervallo per scelta (Vickers et al. 2023).

---

## Fase D — ricerca del tetto di prestazione (21/09/2026)
**Dichiarazione**: analisi post-hoc, decisa dopo aver visto le Fasi A–C per rispondere alla domanda "si poteva fare meglio?"; va letta come esplorativa. Regola di decisione, candidati e diagnostiche sono stati scritti in `configs/config.yaml` (blocco `phase_d`, commit `b51f00a`) **prima di qualsiasi calcolo**. Stessi fold esterni della Fase A; il test set non viene letto. Prima dei candidati è stato fatto un audit riga per riga della pipeline, controllato da un revisore indipendente: nessun leakage, imputazione e bilanciamento stimati solo dentro i fold, test set mai letto dal codice dei modelli.

### Protocollo
- **riferimenti**: logistica penalizzata, Random Forest e XGBoost con profondità 1–12 (Fase A)
- **regola**: un candidato migliora se, contro il riferimento con AUROC media sui fold più alta (Random Forest, 0,708), la differenza di AUROC è ≥ 0,01, il limite inferiore dell'IC 95% di Nadeau-Bengio è > 0 e il p di Holm fra i candidati è < 0,05. Chi la supera va ripetuto su 3 partizioni diverse prima di essere adottato
- **metriche**: AUROC, PR-AUC, specificità a sensibilità 0,90
- **candidati** (10, più il tri-ensemble aggiunto il 22/09/2026, sezione successiva): media dei tre riferimenti; XGBoost sui dati non imputati (valori mancanti gestiti dall'algoritmo); XGBoost con spazio allargato (`min_child_weight`, `gamma`, 100 tentativi); bersaglio scomposto (un XGBoost per l'albuminuria e uno per l'eGFR < 60, probabilità combinate); CatBoost; LightGBM; EBM; TabPFN v2; XGBoost senza pruner; logistica penalizzata con tutte le colonne standardizzate
- **diagnostiche**, non candidati: controllo positivo (XGBoost con l'albumina urinaria fra le feature), curva di apprendimento (dal 20% al 100% del training di ogni fold), qualità dell'etichetta (creatinina urinaria per giornata di raccolta; AUROC per componente del bersaglio)

### Risultati (`analytics/phase_d/`, figure 01–04)

| modello | AUROC (IC 95%) | PR-AUC | specificità a sens. 0,90 | Δ AUROC contro RF (IC 95%) |
|---|---|---|---|---|
| Random Forest (riferimento) | 0,703 (0,676–0,731) | 0,246 | 0,232 | — |
| logistica penalizzata (riferimento) | 0,697 (0,669–0,725) | 0,242 | 0,223 | — |
| XGBoost profondità 1–12 (riferimento) | 0,696 (0,668–0,724) | 0,250 | 0,256 | — |
| TabPFN v2 | 0,710 (0,682–0,737) | 0,253 | 0,231 | +0,009 (−0,014; +0,032) |
| ensemble (media dei 3) | 0,708 (0,680–0,735) | 0,258 | 0,253 | +0,002 (−0,009; +0,013) |
| tri-ensemble, 21 variabili | 0,703 (0,676–0,731) | 0,252 | 0,267 | −0,002 (−0,025; +0,020) |
| bersaglio scomposto | 0,702 (0,674–0,729) | 0,260 | 0,243 | −0,001 (−0,015; +0,013) |
| XGBoost, NaN nativi | 0,701 (0,673–0,729) | 0,250 | 0,256 | −0,001 (−0,015; +0,012) |
| LightGBM | 0,701 (0,674–0,728) | 0,246 | 0,247 | −0,001 (−0,023; +0,022) |
| logistica, tutto standardizzato | 0,696 (0,669–0,724) | 0,250 | 0,248 | −0,011 (−0,060; +0,039) |
| CatBoost | 0,695 (0,667–0,723) | 0,240 | 0,240 | −0,008 (−0,030; +0,014) |
| XGBoost senza pruner | 0,695 (0,667–0,723) | 0,249 | 0,239 | −0,011 (−0,030; +0,009) |
| XGBoost, spazio allargato | 0,693 (0,665–0,721) | 0,244 | 0,233 | −0,009 (−0,027; +0,010) |
| EBM | 0,692 (0,664–0,720) | 0,257 | 0,233 | −0,009 (−0,020; +0,002) |

- **nessun candidato supera la regola** (p di Holm fra 0,99 e 1,00): strategie molto diverse restano fra 0,692 e 0,710. TabPFN ha la stima più alta ma resta sotto la soglia di rilevanza, con un intervallo che comprende lo zero

**Diagnostiche**
- **controllo positivo**: aggiungendo l'albumina urinaria l'AUROC sale a 0,933 (0,917–0,949). La pipeline impara quando l'informazione c'è
- **curva di apprendimento**: dal 60% al 100% del training l'AUROC sale da 0,694 a 0,700 (logistica) e da 0,690 a 0,698 (XGBoost). Più soggetti aiuterebbero poco
- **per componente del bersaglio**: i positivi per eGFR < 60 (63 nel training) si distinguono dai negativi con AUROC 0,81–0,84 nei riferimenti (0,80–0,86 su tutti i modelli); i positivi per sola albuminuria (362) con 0,67–0,69. **Il limite è l'albuminuria**, che è presente nel 91% dei positivi
- **qualità dell'etichetta**: per giornata di raccolta, dove la creatinina urinaria mediana è bassa la quota di ACR ≥ 30 sale (rho di Spearman −0,70, p = 6·10⁻⁷), mentre l'albumina urinaria non segue (rho +0,16): l'ACR sale perché scende il denominatore. Nelle giornate con creatinina urinaria bassa l'AUROC non è peggiore (0,71–0,73 contro 0,69), quindi l'anomalia non spiega le prestazioni; va dichiarata insieme alle unità di `UCRE` e `UmALB`, non documentate

**Letteratura** (verificata il 22/09/2026, dettaglio in `docs/verifica_stato_arte.md`)
- modelli per la sola albuminuria senza esami: C 0,709–0,714 (Muntner et al. 2011) e 0,728–0,761 (Tanner et al. 2015); con eGFR, HbA1c, HDL e acido urico fra i predittori 0,752 (Bragg-Gresham et al. 2025); nel diabete tipo 2, anche con la creatinina sierica, 0,61–0,67 (Khitan et al. 2021)
- un ACR su campione singolo è un'etichetta rumorosa: solo il 43,5% degli ACR ≥ 30 su urina casuale viene confermato sulla prima urina del mattino (Saydah et al. 2013); variabilità intra-individuale del 48,8% nel diabete tipo 2 (Rasaratnam et al. 2024); con etichette rumorose l'AUROC misurata si schiaccia verso 0,5 anche per un modello perfetto (Menon et al. 2015)

**Sintesi**: con esami del sangue di routine e un ACR da campione singolo il tetto è intorno a 0,70–0,71. Il limite sta nell'informazione (l'albuminuria), non nel metodo.

---

## Fase D — candidato aggiunto: tri-ensemble su 21 variabili (22/09/2026)
**Perché.** `valorizzazione_tesi.md` citava un "tri-ensemble sulle 21 variabili più importanti" con AUROC 0,717 e PR-AUC 0,273, stimato in un'analisi preliminare di cui nel repository non c'erano né codice né tabelle. Su richiesta dell'utente è stato registrato come candidato della Fase D, **in `configs/config.yaml` prima di scrivere il codice e di calcolarlo**, con la stessa regola di decisione degli altri candidati.

**Definizione (fissata prima).** `tri_ensemble_top21`: media delle probabilità dei tre riferimenti (logistica penalizzata, Random Forest, XGBoost con profondità 1–12) addestrati sulle prime 21 variabili, con gli iperparametri della Fase A di ogni fold (nessuna nuova ottimizzazione, quindi nessuna informazione dal fold esterno). Le 21 variabili si scelgono **dentro ogni fold esterno, sul solo training**, per rango medio fra i tre modelli, col criterio di `interpretation.top_features`: |coefficiente| per la logistica, SHAP medio assoluto per gli alberi (per la Random Forest su 500 righe estratte per classe: TreeSHAP su 954 alberi profondi 25 sarebbe durato ore). Diagnostica `tri_ensemble_leaky`: stessa pipeline con le variabili scelte **sull'intero training** (classifiche dei modelli finali della Fase A), per misurare la distorsione da selezione (Ambroise & McLachlan 2002); non è un candidato.

**Codice.** `src/models/phase_d.py`: `consensus_top`, `ranking`, `fold_rankings`, `full_training_rankings`, `run_tri_ensemble`. Test in `tests/test_phase_d.py`, fra cui `test_nested_tri_ensemble_selects_on_training_rows_only`, che fallirebbe se la classifica vedesse le righe del fold esterno. Esecuzione: 1 minuto e mezzo.

**Risultati** (`analytics/phase_d/discrimination.csv`, `comparison.csv`):

| versione | AUROC (IC 95%) | PR-AUC | specificità a sensibilità 0,90 | contro RF |
|---|---|---|---|---|
| selezione nel fold (candidato) | 0,703 (0,676–0,731) | 0,252 | 0,267 | −0,002 (−0,025; +0,020), p di Holm 1,00 |
| selezione su tutto il training (diagnostica) | 0,716 (0,689–0,742) | 0,264 | 0,272 | — |

- **non supera la regola** (serviva ΔAUROC ≥ +0,01 con IC sopra zero e Holm < 0,05)
- **lo 0,717 dell'analisi preliminare è distorsione da selezione**: lo riproduce solo la versione con le variabili scelte su tutti i dati (+0,013 di AUROC, +0,012 di PR-AUC). Non va citato come risultato
- contro SCORED (confronto post-hoc, sui 5 fold, non corretto): selezione nel fold +0,027 (−0,004; +0,057), p = 0,07; selezione su tutto il training +0,039 (+0,007; +0,071), p = 0,03. **La distorsione basta a rendere "significativo" un vantaggio che non lo è**: quarto "risultato apparente" della tesi
- **valore reale: parsimonia.** Con 21 variabili invece di 74 discrimina come la Random Forest ed è il modello con la specificità più alta a sensibilità 0,90 (0,267). Contro SCORED: 7,8 esami in meno ogni 100 persone per trovare l'85% dei casi (66,0 contro 73,8), 5,2 in meno per il 90% (75,0 contro 80,2), net benefit superiore a 31 soglie su 31 fra 5% e 20%, 5,6 esami inutili evitati in più ogni 100 al 7%
- **stabilità della selezione**: 34 variabili diverse nei 5 fold; 9 scelte sempre (età, pressione sistolica, glicemia a digiuno e a 2 ore, peptide C a digiuno, acido urico, ALP, FIB-4, LDL), altre 6 in 4 fold su 5

**Effetto sugli altri candidati.** La famiglia di Holm passa da 10 a 11 candidati: cambiano solo i p corretti (EBM da 0,90 a 0,99; gli altri restano 1,00). Nessuna conclusione della Fase D cambia. La figura `analytics/phase_d/01_candidates.png` è rigenerata con il nuovo candidato.

**Limiti.** Iperparametri scelti in Fase A su 74 variabili, non riottimizzati per 21; il numero 21 viene dall'analisi preliminare e non è stato ottimizzato (ottimizzarlo ora sarebbe un'altra scelta dopo i risultati).

---

## Conclusioni delle domande 1–6 (22/09/2026, previsioni out-of-fold)
Risposte alle domande dello Scope, sulle previsioni out-of-fold del training (4.350 soggetti, 425 positivi, prevalenza 9,8%). Sono le affermazioni che il test set dovrà confermare o smentire (sezione "Conferma finale sul test set — protocollo"). Popolazione: coorte ospedaliera di Shanghai, 2012, in maggioranza senza diabete noto.

### Domanda 1 — il modello distingue chi ha marcatori di malattia renale?
**Sì, in modo modesto.** AUROC da 0,675 (logistica SCORED) a 0,703 (Random Forest); PR-AUC 0,224–0,251, cioè 2,3–2,6 volte la prevalenza. Alla soglia con sensibilità 0,90 va esaminato il 77–80% dei soggetti (specificità 0,21–0,26).
- **nessun modello è migliore degli altri in modo dimostrabile** (p di Holm ≥ 0,44). I modelli con 74 variabili superano la logistica SCORED a 5 predittori su tutte le misure (AUROC +0,021 / +0,029, meno esami a parità di sensibilità, net benefit più alto), ma senza significatività: è la replica di Christodoulou et al. 2019
- **il tetto è nell'informazione**: dieci strategie in più (Fase D) restano fra 0,692 e 0,710. L'eGFR < 60 si riconosce bene (AUROC 0,80–0,86), l'albuminuria no (0,67–0,69); la letteratura con lo stesso tipo di bersaglio riporta 0,68–0,76
- **serve?** Sì, da soglie del 5–6% in su: al 7% evita 9–13 esami inutili ogni 100 persone rispetto a "testare tutti", al 10% 25–28 (decision curve, blocco qualità). Sotto il 5% equivale a testare tutti
- **calibrazione**: buona in media (intercetta 0, rapporto O:E 1) e per le due logistiche (pendenze 0,97 e 0,94); Random Forest schiaccia le probabilità (pendenza 1,31), XGBoost le esaspera (0,86)

### Domanda 2 — il rischio stimato cresce con la gravità KDIGO?
**Sì.** In tutti i modelli la probabilità media cresce a ogni livello (XGBoost: 0,086 basso, 0,142 moderato, 0,194 alto, 0,284 molto alto) e circa il 70% delle coppie di soggetti di livelli diversi è ordinato come KDIGO (concordanza di Jonckheere-Terpstra 0,674–0,702). La separazione netta è fra "basso" e gli altri livelli; "alto" e "molto alto" si sovrappongono.

### Domanda 3 — quanti casi "alto" e "molto alto" riconosce?
**Alla soglia con sensibilità 0,90, 46–47 "alto" su 50 e 18–20 "molto alto" su 21** (1–3 "molto alto" mancati). I casi gravi non sono riconosciuti più dei moderati: sensibilità simili, con intervalli larghi per i 21 "molto alto".

### Domanda 4 — le fasce del modello corrispondono alla stratificazione clinica?
**Poco.** Kappa pesato 0,197–0,228; l'accordo osservato alto (0,85) viene quasi tutto dal livello "basso", il 90% dei soggetti (paradosso della prevalenza). Fino a metà dei "molto alto" finisce nella fascia di rischio più bassa. Con la domanda 3: il modello riconosce la **presenza** dei marcatori, non il loro **grado**.

### Domanda 5 — l'augmentation migliora il riconoscimento dei casi gravi o solo la metrica media?
**Nessuna delle due, se misurata bene.** A parità di sensibilità complessiva (0,90) nessuna tecnica riconosce più casi gravi di "nessuna correzione" (differenze entro ±3 casi su 71, p di Holm 1,00); SMOTE-NC e CTGAN peggiorano la discriminazione; pesatura e campionamento distruggono la calibrazione (intercetta fino a −2,2), che la ricalibrazione di Platt recupera. Nemmeno i pesi per livello KDIGO, costruiti apposta per i casi gravi, li fanno riconoscere di più.

**Ma alla soglia 0,5 sembrano trasformare il modello**: il recall passa dal 2% al 61% e i casi gravi riconosciuti da 0 a circa 50 su 71. È un effetto della soglia, non del modello (Elkan 2001; van den Goorbergh et al. 2022).

### Domanda 6 — come si comporta lo stesso modello sui diabetici?
**Discrimina come sugli altri** (differenze di AUROC da −0,038 a +0,042, tutti gli intervalli coprono lo zero), **ma alla soglia globale degenera in "testare tutti"**: segnala il 97–100% dei diabetici (specificità 0,000–0,036). La decision curve lo conferma: fino al 10% nessun modello batte "testare tutti" fra i diabetici. La PR-AUC dei diabetici sembra doppia (0,44 contro 0,19) solo per la prevalenza (25,9% contro 8,7%). Random Forest sottostima di circa un quinto il rischio dei diabetici (O:E 1,26). **Il modello ha senso fra i soggetti senza diabete; fra i diabetici hanno ragione le linee guida**, che prescrivono l'esame ogni anno.

### Il filo della tesi
Esiste una convinzione diffusa: bilanciare le classi aiuta a trovare i casi rari. La tesi l'ha messa alla prova con protocolli scritti prima dei risultati, su dati clinici reali e senza esami renali. **Non regge**: nessuna tecnica fa riconoscere più casi gravi. In compenso **sembra** reggere, e la tesi misura quattro modi in cui un numero, letto senza controllare come è stato ottenuto, inganna:
1. la soglia 0,5 (Fase B): recall dal 2% al 61% senza che il modello migliori;
2. la PR-AUC fra gruppi con prevalenza diversa (Fase C): 0,44 contro 0,19 con un ordinamento peggiore;
3. la media delle pendenze di calibrazione (blocco qualità): 1,02 come media di 1,31 e 0,86;
4. la selezione delle variabili fuori dalla validazione (Fase D): AUROC da 0,703 a 0,716 e un vantaggio su SCORED che diventa "significativo".

Accanto a questo, un risultato positivo e misurato: con i soli esami del sangue di routine il modello raggiunge il tetto di prestazione di questo bersaglio e, a soglie cliniche ragionevoli, evita esami inutili nei soggetti senza diabete.

### Limiti delle conclusioni
- previsioni out-of-fold di un solo dataset: conferma sul test set in sospeso, nessuna validazione esterna
- coorte ospedaliera, non screening di popolazione; ACR da campione singolo, con tipo di campione e unità non documentati
- 71 casi gravi nel training (23 nel test) e 68 positivi fra i diabetici: stime per livello e per sottogruppo instabili
- Fase D e blocco qualità sono post-hoc ed esplorativi

---

## Conferma finale sul test set — preparazione (20/09/2026)
Il test set **non è ancora stato letto**. Qui si annota solo quanto verificato per prepararne la trasformazione.

### Il preprocessore ristimato riproduce la cache, bit a bit
Il problema noto era che la cache dei fold (`src/data/imputed.py`) salva solo le matrici e non l'oggetto che imputa e standardizza, quindi per trasformare il test serve ristimarlo sull'intero training. Restava da dimostrare che la ristima dia **esattamente** quello che la Fase A e la Fase B hanno usato.

Verificato il 20/09/2026, **senza leggere il test set**: ristimando `build_preprocessor(clone(chosen_imputer()), X.columns).fit(X)` sull'intero training e trasformando il training stesso, la matrice coincide con `data/processed/imputed/<set>/full.npz`:

| set di feature | forma | colonne e ordine | differenza massima | identiche bit a bit | tempo |
|---|---|---|---|---|---|
| `main` | 4.350 × 74 | uguali | 0,000e+00 | sì | 101 s |
| `no_consequence` | 4.350 × 67 | uguali | 0,000e+00 | sì | 75 s |

Il fold `full` della cache è già stimato su tutto il training (`train_idx = arange(n)` in `imputed.all_folds`) e il seme è fisso dentro `imputation.imputers()` (`random_state=42` su `IterativeImputer` e `ExtraTreesRegressor`), quindi la ristima è deterministica. Conseguenza: **l'unica operazione che toccherà il test set sarà una singola `.transform`**, e questo confronto resta come test di riproducibilità permanente, eseguibile senza toccare il test.

(`IterativeImputer` emette `ConvergenceWarning: Early stopping criterion not reached`: atteso con `forest_max_iter: 5`, il "MissForest con freni" scelto nei Passi 8–10. La cache era stata costruita con lo stesso avviso, tanto che le matrici coincidono.)

---

## Da fare per concludere il progetto (scritto il 20/09/2026)
Restano due blocchi: la **Fase C** (sottogruppo diabetico e conclusioni) e la **conferma finale sul test set**. Qui c'è esattamente cosa fare, nell'ordine consigliato.

### Fase C — sottogruppo diabetico (domanda 6) e conclusioni
Numeri reali nel training: **263 diabetici**, di cui **68 positivi** e **16 casi gravi** (11 "alto", 5 "molto alto"); per livello: basso 195, moderato 52, alto 11, molto alto 5.

1. **Nessun nuovo addestramento.** I modelli restano quelli già addestrati: si filtrano i soggetti diabetici nelle previsioni out-of-fold già salvate (`analytics/phase_a/oof_predictions.csv` e `analytics/phase_b/oof_predictions.csv`). Riaddestrare sul solo sottogruppo non ha senso con 68 positivi.
2. **Soglia e fasce**: quelle globali, già fissate (`analytics/phase_a/evaluation/cutpoints.csv`); non vanno ricalcolate sul sottogruppo. Come analisi descrittiva si può riportare anche la soglia che darebbe sensibilità 0,90 fra i soli diabetici, dichiarandola come descrittiva.
3. **Cosa calcolare**: le domande 1–4 ristrette ai diabetici (AUC, PR-AUC, precision, recall, specificità; probabilità per livello e tendenza; sensibilità per livello; fasce contro livelli), con intervalli di confidenza, e il confronto descrittivo fra diabetici e non diabetici (differenze di AUC e PR-AUC).
4. **Niente test formali**: con 68 positivi e 16 casi gravi la potenza è nulla. Solo stime con intervalli, dichiarando che sono descrittive (già previsto dallo Scope: "sottogruppo diabetico piccolo, solo descrittivo").
5. **Quali bracci**: "nessuna correzione" come analisi principale; al massimo una tecnica della Fase B come confronto descrittivo (da fissare prima, non dopo aver visto i numeri).
6. **Attenzione alla prevalenza**: fra i diabetici è il 25,9% contro il 9,8% complessivo. La PR-AUC va sempre confrontata con la prevalenza del sottogruppo, non con quella generale, altrimenti sembra migliore senza esserlo.
7. **Codice**: un modulo nuovo (per esempio `src/models/phase_c.py`) che filtra le previsioni sui diabetici e richiama `evaluation.evaluate`, più 1–2 figure in `analytics/phase_c/`. Nessuna modifica ai moduli esistenti.
8. **Conclusioni della tesi** (punto 5 dello Scope): sintesi delle domande 1–6, cioè Fase A, Fase B e sottogruppo, con i limiti già elencati.

Tempo stimato: mezza giornata, nessun calcolo pesante.

### Conferma finale sul test set (una sola volta, alla fine)
Il test set (1.451 soggetti, circa 142 positivi, 17 "alto" e 6 "molto alto") non è mai stato letto. Prima di toccarlo va preparato tutto, perché **si esegue una volta sola**.

1. **Da fissare prima, per iscritto, in questa sezione**:
   - quali modelli portare al test: proposta, i 5 modelli finali della Fase A (set `main`) più i modelli finali delle tecniche di Fase B, tutti valutati una volta sola, senza scegliere il "migliore" in base al test;
   - soglia e fasce: quelle stimate sulle previsioni out-of-fold (`cutpoints.csv` della Fase A e della Fase B), applicate così come sono;
   - ricalibrazione: per le tecniche della Fase B si usano i parametri di Platt del modello finale, già salvati in `analytics/phase_b/<tecnica>/calibration/main/<modello>_full.json`;
   - metriche: le stesse domande 1–4 più l'esito primario sui casi gravi del test (23 soggetti), con intervalli; dichiarare che con 23 casi gravi gli intervalli sono amplissimi.
2. **Problema tecnico da risolvere prima** (verificato il 20/09/2026): **il codice attuale non prepara il test set**. La cache dei fold (`src/data/imputed.py`) salva solo le matrici, non l'oggetto che imputa e standardizza, e `test.csv` viene letto solo da `src/data/split.py`. Serve quindi:
   - ristimare il preprocessore (standardizzazione più MissForest) **sull'intero training**, esattamente come per il fold "full", e usarlo per trasformare il test (qualche minuto di calcolo);
   - verificare che la trasformazione sia riproducibile (stesso seed, stesse colonne, stesso ordine) e che il test non entri mai nella stima;
   - salvare il test trasformato in `data/processed/imputed/<set>/test.npz` (non versionato).
3. **Codice**: un modulo nuovo (per esempio `src/models/final_test.py`) che trasforma il test, carica i modelli finali (`models/phase_a_*.joblib`, `models/phase_b/<tecnica>/*.joblib`), applica soglia, fasce e ricalibrazione **fissate prima**, calcola le tabelle e scrive in `analytics/test/`. Test automatici su dati sintetici come per le altre fasi.
4. **Ordine di esecuzione**: preparare il codice, farlo rivedere, lanciare i test automatici, **poi** eseguire una sola volta sul test set.
5. **Regola d'oro**: se dopo l'esecuzione si scopre un errore, si corregge e si dichiara apertamente che il test è stato usato due volte. Non si ritocca la soglia né si cambiano i modelli dopo aver visto i risultati del test.
6. **Dopo il test**: risultati nel Notepad, aggiornamento di Scope e README, e release `v1.0.0`.

Tempo stimato: un giorno di lavoro, più qualche minuto di calcolo.

### Ordine consigliato
1. Fase C (usa solo dati già calcolati).
2. Preparazione del codice per il test set e sua revisione.
3. Esecuzione unica sul test set.
4. Scrittura della tesi: i capitoli di metodi e risultati sono già coperti da questo Notepad, dalle 17 figure e dalle tabelle in `analytics/`.

---

## Indice delle figure

Rigenerare con `python -m src.analytics.dataset_overview`, `python -m src.analytics.split_report`, (dopo `python -m src.data.imputation`) `python -m src.analytics.preprocessing_report` e (dopo `python -m src.models.evaluation` e `python -m src.models.interpretation`) `python -m src.analytics.phase_a_report`. Le figure della Fase B: `python -m src.analytics.phase_b_report` (dopo `python -m src.models.evaluation_b`). Le figure della Fase C: `python -m src.analytics.phase_c_report` (dopo `python -m src.models.phase_c`). Le figure della Fase D: `python -m src.analytics.phase_d_report` (dopo `python -m src.models.phase_d --evaluate`). Le figure del blocco qualità e utilità clinica: `python -m src.analytics.quality_report` (dopo `python -m src.models.clinical_utility`).

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
| `analytics/phase_a/01_roc_pr.png` | curve ROC e precision-recall dei 4 modelli (set main, previsioni out-of-fold), maggioranza come riferimento | Fase A → Valutazione: risultati, domanda 1 |
| `analytics/phase_a/02_discrimination_ci.png` | AUC e PR-AUC con IC 95%, set main (pieno) e no_consequence (vuoto) | domanda 1 e analisi di sensibilità |
| `analytics/phase_a/03_operating_points.png` | sensibilità contro % di soggetti da testare, confronto con la scelta casuale | domanda 1 (costo della soglia) |
| `analytics/phase_a/04_probability_by_level.png` | probabilità stimata per livello KDIGO (scatole, media con IC, singoli soggetti gravi), concordanza di Jonckheere-Terpstra | domanda 2 |
| `analytics/phase_a/05_sensitivity_by_level.png` | sensibilità per livello KDIGO alla soglia scelta, IC di Wilson, riconosciuti/totale | domanda 3 |
| `analytics/phase_a/06_bands_vs_levels.png` | tabella 4 × 4 fasce del modello contro livelli KDIGO, kappa pesato | domanda 4 |
| `analytics/phase_a/07_model_comparison.png` | differenze fra modelli (PR-AUC e AUC) con IC di Nadeau-Bengio e p di Holm | confronto fra modelli |
| `analytics/phase_a/08_no_consequence.png` | differenze no_consequence − main per modello | analisi di sensibilità |
| `analytics/phase_a/09_odds_ratios.png` | odds ratio della logistica SCORED (IC di Wald, main e no_consequence) e prime 15 della logistica penalizzata | Interpretazione: risultati |
| `analytics/phase_a/10_shap_importance.png` | importanza SHAP media, prime 15 variabili, Random Forest e XGBoost | Interpretazione: risultati |
| `analytics/phase_a/11_shap_beeswarm.png` | valori SHAP per soggetto (posizione) e valore della variabile (colore), prime 15 | Interpretazione: risultati |
| `analytics/phase_b/01_primary_endpoint.png` | esito primario: differenza di casi gravi riconosciuti contro nessuna correzione (IC di Newcombe) | Fase B — risultati |
| `analytics/phase_b/02_discrimination_vs_none.png` | differenze di PR-AUC e AUC contro nessuna correzione (IC di Nadeau-Bengio) | Fase B — risultati |
| `analytics/phase_b/03_sensitivity_by_level.png` | sensibilità per livello KDIGO, per tecnica e modello | Fase B — risultati |
| `analytics/phase_b/04_calibration.png` | calibrazione (intercetta e pendenza) grezza e dopo Platt | Fase B — risultati |
| `analytics/phase_b/05_naive_vs_real.png` | miglioramento apparente alla soglia 0,5 contro guadagno reale a parità di sensibilità | Fase B — risultati |
| `analytics/phase_b/06_kappa.png` | kappa pesato fasce/livelli per tecnica e modello | Fase B — risultati |
| `analytics/phase_c/01_discrimination_diabetici.png` | AUC e PR-AUC con IC: diabetici, non diabetici e training completo; per la PR-AUC la prevalenza di ciascun gruppo è la linea di riferimento | Fase C — protocollo |
| `analytics/phase_c/02_severe_cases.png` | casi gravi riconosciuti su 16 fra i diabetici alla soglia globale fissa, per modello e braccio, con IC di Wilson | Fase C — protocollo |
| `analytics/phase_d/01_candidates.png` | AUROC dei candidati della Fase D con IC di DeLong e differenza contro Random Forest con IC di Nadeau-Bengio e soglia di rilevanza | Fase D — ricerca del tetto |
| `analytics/phase_d/02_learning_curve.png` | curva di apprendimento di logistica penalizzata e XGBoost (20–100% del training di ogni fold) | Fase D — diagnostiche |
| `analytics/phase_d/03_urine_creatinine_by_day.png` | per giornata di raccolta: creatinina urinaria mediana contro quota di ACR ≥ 30 e contro albumina urinaria mediana | Fase D — qualità dell'etichetta |
| `analytics/phase_d/04_label_quality.png` | AUROC per componente del bersaglio (eGFR < 60, sola albuminuria) e per gruppo di giornate | Fase D — diagnostiche |
| `analytics/quality/01_decision_curve.png` | decision curve su tutto il training (soglie 2–20%) contro "testare tutti" e "non testare nessuno"; a destra esami inutili evitati ogni 100 | Qualità e utilità clinica — risultati, punto 1 |
| `analytics/quality/02_decision_curve_subgroups.png` | decision curve dentro diabetici e non diabetici, ciascuno contro il proprio "testare tutti" (scale diverse di proposito) | punto 2 |
| `analytics/quality/03_costs.png` | costo del test ACR per caso trovato e costo di ogni caso in più trovato testando tutti ($49 a test) | punto 7 |
| `analytics/quality/04_calibration.png` | curva di calibrazione flessibile con banda, decili con IC di Wilson e distribuzione delle probabilità per esito, 4 modelli | punto 3 |
| `analytics/quality/05_calibration_subgroups.png` | curve di calibrazione dentro diabetici e non diabetici, con O:E e IC | punto 8 |
| `analytics/quality/06_bragg_gresham.png` | quota esaminata contro quota di casi trovati, bersaglio della tesi e disegno vicino a Bragg-Gresham et al. 2024, con i loro punti operativi | punto 5 |

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
- [x] Fasi A e B: stimare l'imputer una volta per fold e salvare su disco i fold imputati, per ogni set di feature (`src/data/imputed.py`)
- [x] modelli scelti: Dummy, logistica SCORED, logistica penalizzata, Random Forest, XGBoost, con protocollo e riferimenti (sezione "Fase A — modelli e protocollo"). Gradient Boosting in generale: Friedman 2001
- [ ] stesso test set congelato per tutti i confronti e le tecniche di data augmentation

### Fase A: stato
- [ ] procurare il PDF della linea guida KDIGO 2024 sulla CKD (Kidney Int 105(4S), doi:10.1016/j.kint.2023.10.018) e ricontrollare le citazioni che la usano
- [x] scegliere l'implementazione del Gradient Boosting → XGBoost (sezione "Fase A — modelli e protocollo")
- [x] aggiungere a `requirements.txt` e `requirements-lock.txt` le librerie della Fase A: `xgboost`, `optuna`
- [x] stimare l'imputer una volta per fold e salvare i fold imputati: `src/data/folds.py`, `src/data/imputed.py` (`python -m src.data.imputed`), test in `tests/test_folds.py`
- [x] `shap` 0.52.0 aggiunto (19/09/2026; porta `numba` 0.67.0, `llvmlite` 0.49.0, `slicer` 0.0.8; `numpy` resta 2.5.3)
- [ ] librerie ancora da aggiungere per la Fase B: `imbalanced-learn`, `ctgan`
- [x] interpretazione della Fase A (odds ratio, SHAP) e revisione indipendente del codice
- [x] analisi di sensibilità XGBoost `max_depth` 1–12: nessuna differenza con il protocollo primario (sezione "Analisi di sensibilità: profondità di XGBoost")
- [ ] verificare in letteratura le ipotesi su peptide C e albumina glicata (sezione "Interpretazione: risultati")
- [x] Fase B: protocollo fissato, codice (`src/data/augmented.py`, `src/models/phase_b.py`, `src/models/evaluation_b.py`, `src/analytics/phase_b_report.py`), esecuzione e risultati (sezione "Fase B — risultati")
- [ ] Fase C: sottogruppo diabetico (domanda 6) e conclusioni — passi dettagliati nella sezione "Da fare per concludere il progetto"
- [ ] conferma finale sul test set (una sola volta) — passi dettagliati nella stessa sezione; da risolvere prima: il codice non prepara ancora il test set
- [x] fissare spazi di ricerca e numero di tentativi di Optuna dopo una stima dei tempi, prima di vedere i risultati (sezione "Protocollo fissato prima dei risultati")
- [x] pipeline di addestramento della Fase A: `src/models/zoo.py`, `src/models/phase_a.py`, `tests/test_phase_a.py`
- [x] lanciare la Fase A (19/09/2026, notte): 5 modelli × 6 fold × 2 set, previsioni out-of-fold in `analytics/phase_a/oof_predictions.csv`
- [x] fissare le scelte della valutazione prima dei risultati (soglia, fasce, intervalli: sezione "Valutazione: scelte fissate prima dei risultati") e scrivere il modulo (`src/models/evaluation.py`)
- [x] calcolare la valutazione della Fase A e riportare i risultati (sezione "Valutazione: risultati", figure `analytics/phase_a/01`–`08`)
- [ ] conferma finale sul test set (una sola volta, a fine progetto): modelli finali, soglia e fasce di `cutpoints.csv`
- [x] procurare i PDF di Bang et al. 2007 (SCORED, solo abstract) ed Echouffo-Tcheugui & Kengne 2012 (testo completo)

---

## Bibliografia

Citazioni nel testo in formato autore-anno. ✅ = PDF in `papers/`, metadati letti dal file. Tutte le voci con DOI sono state verificate su Crossref il 18/09/2026, quelle della sezione "Fase A" il 19/09/2026 (titolo, primo autore, rivista, volume, fascicolo, pagine, anno). Senza DOI e quindi non verificabili su Crossref: linee guida WHO, bozze KDIGO 2026, Platt 1999 (capitolo di libro), Hastie et al. 2009 (libro), Xu et al. 2019 e Lundberg & Lee 2017 (atti NeurIPS, indicato l'identificativo arXiv), Bergstra et al. 2011 (atti NeurIPS), Cawley & Talbot 2010 (JMLR), Elkan 2001 (atti IJCAI).

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

### Valutazione della Fase A: soglia, concordanza, intervalli, confronti
Verificate su Crossref il 19/09/2026. Contenuto letto dal testo completo: Boyd et al. 2013, Forman & Scholz 2010; dall'abstract: Wynants et al. 2019, Perkins & Schisterman 2006, Bates, Hastie & Tibshirani 2024, Brown, Cai & DasGupta 2001, Sim & Wright 2005, Feinstein & Cicchetti 1990, Byrt et al. 1993, Carpenter & Bithell 2000, Van Calster et al. 2019. Holm 1979 non ha DOI.
- **Wynants et al. 2019** — Wynants L., van Smeden M., McLernon D.J., Timmerman D., Steyerberg E.W., Van Calster B. *Three myths about risk thresholds for prediction models.* BMC Med 17:192 (2019). doi:10.1186/s12916-019-1425-3
- **Youden 1950** — Youden W.J. *Index for rating diagnostic tests.* Cancer 3(1):32–35 (1950). doi:10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
- **Perkins & Schisterman 2006** — Perkins N.J., Schisterman E.F. *The inconsistency of "optimal" cutpoints obtained using two criteria based on the receiver operating characteristic curve.* Am J Epidemiol 163(7):670–675 (2006). doi:10.1093/aje/kwj063
- **Van Calster et al. 2019** — Van Calster B., McLernon D.J., van Smeden M., Wynants L., Steyerberg E.W. *Calibration: the Achilles heel of predictive analytics.* BMC Med 17:230 (2019). doi:10.1186/s12916-019-1466-7
- **Vickers & Elkin 2006** — Vickers A.J., Elkin E.B. *Decision curve analysis: a novel method for evaluating prediction models.* Med Decis Making 26(6):565–574 (2006). doi:10.1177/0272989X06295361
- **Feinstein & Cicchetti 1990** — Feinstein A.R., Cicchetti D.V. *High agreement but low kappa: I. The problems of two paradoxes.* J Clin Epidemiol 43(6):543–549 (1990). doi:10.1016/0895-4356(90)90158-L
- **Byrt et al. 1993** — Byrt T., Bishop J., Carlin J.B. *Bias, prevalence and kappa.* J Clin Epidemiol 46(5):423–429 (1993). doi:10.1016/0895-4356(93)90018-V
- **Sim & Wright 2005** — Sim J., Wright C.C. *The kappa statistic in reliability studies: use, interpretation, and sample size requirements.* Phys Ther 85(3):257–268 (2005). doi:10.1093/ptj/85.3.257
- **Fleiss & Cohen 1973** — Fleiss J.L., Cohen J. *The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability.* Educ Psychol Meas 33(3):613–619 (1973). doi:10.1177/001316447303300309
- **Hollander, Wolfe & Chicken 2014** — Hollander M., Wolfe D.A., Chicken E. *Nonparametric Statistical Methods*, 3ª ed. Wiley (2014; su Crossref 2015). doi:10.1002/9781119196037. Test di Jonckheere-Terpstra con pareggi; la formula implementata è verificata numericamente contro il test di Kendall
- **Forman & Scholz 2010** — Forman G., Scholz M. *Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement.* SIGKDD Explor 12(1):49–57 (2010). doi:10.1145/1882471.1882479
- **DeLong et al. 1988** — DeLong E.R., DeLong D.M., Clarke-Pearson D.L. *Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.* Biometrics 44(3):837–845 (1988). doi:10.2307/2531595
- **Boyd et al. 2013** — Boyd K., Eng K.H., Page C.D. *Area under the precision-recall curve: point estimates and confidence intervals.* ECML PKDD 2013, LNCS 8190:451–466 (2013). doi:10.1007/978-3-642-40994-3_29
- **Wilson 1927** — Wilson E.B. *Probable inference, the law of succession, and statistical inference.* J Am Stat Assoc 22(158):209–212 (1927). doi:10.1080/01621459.1927.10502953
- **Brown, Cai & DasGupta 2001** — Brown L.D., Cai T.T., DasGupta A. *Interval estimation for a binomial proportion.* Stat Sci 16(2):101–133 (2001). doi:10.1214/ss/1009213286
- **Carpenter & Bithell 2000** — Carpenter J., Bithell J. *Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians.* Stat Med 19(9):1141–1164 (2000). doi:10.1002/(SICI)1097-0258(20000515)19:9<1141::AID-SIM479>3.0.CO;2-F
- **Bates, Hastie & Tibshirani 2024** — Bates S., Hastie T., Tibshirani R. *Cross-validation: what does it estimate and how well does it do it?* J Am Stat Assoc 119(546):1434–1445 (2024). doi:10.1080/01621459.2023.2197686
- **Holm 1979** — Holm S. *A simple sequentially rejective multiple test procedure.* Scand J Stat 6(2):65–70 (1979)
- **McNemar 1947** — McNemar Q. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika 12(2):153–157 (1947). doi:10.1007/BF02295996

### Fase B: bilanciamento, generazione sintetica, calibrazione
Verificate su Crossref il 19/09/2026 (titolo, primo autore, rivista, volume, pagine, anno); contenuto letto dall'abstract o dal testo. Senza DOI: Lemaître et al. 2017, Kotelnikov et al. 2023, Camino et al. 2020, Seedat et al. 2024 (atti PMLR/JMLR), Elor & Averbuch-Elor 2022 e Manousakas & Aydöre 2023 (arXiv).
- **Carriero et al. 2025** — Carriero A., Luijken K., de Hond A., Moons K.G.M., Van Calster B., van Smeden M. *The harms of class imbalance corrections for machine learning based prediction models: a simulation study.* Stat Med 44(3-4):e10320 (2025). doi:10.1002/sim.10320
- **Elor & Averbuch-Elor 2022** — Elor Y., Averbuch-Elor H. *To SMOTE, or not to SMOTE?* arXiv:2201.08528 (2022) — preprint
- **Roesler et al. 2026** — Roesler M. et al. *Class imbalance correction in artificial intelligence models leads to miscalibrated clinical predictions: a real-world evaluation.* medRxiv/openRxiv (2026). doi:10.64898/2026.03.04.26347634 — preprint
- **Zadrozny, Langford & Abe 2003** — Zadrozny B., Langford J., Abe N. *Cost-sensitive learning by cost-proportionate example weighting.* Proc. 3rd IEEE ICDM, 435–442 (2003). doi:10.1109/ICDM.2003.1250950
- **Correa Bahnsen et al. 2015** — Correa Bahnsen A., Aouada D., Ottersten B. *Example-dependent cost-sensitive decision trees.* Expert Syst Appl 42(19):6609–6619 (2015). doi:10.1016/j.eswa.2015.04.042
- **King & Zeng 2001** — King G., Zeng L. *Logistic regression in rare events data.* Political Analysis 9(2):137–163 (2001). doi:10.1093/oxfordjournals.pan.a004868
- **Batista et al. 2004** — Batista G.E.A.P.A., Prati R.C., Monard M.C. *A study of the behavior of several methods for balancing machine learning training data.* SIGKDD Explor 6(1):20–29 (2004). doi:10.1145/1007730.1007735
- **Japkowicz & Stephen 2002** — Japkowicz N., Stephen S. *The class imbalance problem: a systematic study.* Intell Data Anal 6(5):429–449 (2002). doi:10.3233/IDA-2002-6504
- **Fernández et al. 2018** — Fernández A., García S., Herrera F., Chawla N.V. *SMOTE for learning from imbalanced data: progress and challenges, marking the 15-year anniversary.* J Artif Intell Res 61:863–905 (2018). doi:10.1613/jair.1.11192
- **Han et al. 2005** — Han H., Wang W.-Y., Mao B.-H. *Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning.* ICIC 2005, LNCS 3644:878–887 (2005). doi:10.1007/11538059_91
- **He et al. 2008** — He H., Bai Y., Garcia E.A., Li S. *ADASYN: adaptive synthetic sampling approach for imbalanced learning.* IJCNN 2008, 1322–1328. doi:10.1109/IJCNN.2008.4633969
- **Jo & Japkowicz 2004** — Jo T., Japkowicz N. *Class imbalances versus small disjuncts.* SIGKDD Explor 6(1):40–49 (2004). doi:10.1145/1007730.1007737
- **Pérez-Ortiz et al. 2015** — Pérez-Ortiz M., Gutiérrez P.A., Hervás-Martínez C., Yao X. *Graph-based approaches for over-sampling in the context of ordinal regression.* IEEE Trans Knowl Data Eng 27(5):1233–1245 (2015). doi:10.1109/TKDE.2014.2365780
- **Chakraborty et al. 2021** — Chakraborty J., Majumder S., Menzies T. *Bias in machine learning software: why? how? what to do?* ESEC/FSE 2021, 429–440. doi:10.1145/3468264.3468537
- **Demircioğlu 2024** — Demircioğlu A. *Applying oversampling before cross-validation will lead to high bias in radiomics.* Sci Rep 14:11563 (2024). doi:10.1038/s41598-024-62585-z
- **Lemaître et al. 2017** — Lemaître G., Nogueira F., Aridas C.K. *Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced datasets in machine learning.* J Mach Learn Res 18(17):1–5 (2017)
- **Zhao et al. 2024** — Zhao Z. et al. *CTAB-GAN+: enhancing tabular data synthesis.* Front Big Data 6:1296508 (2024). doi:10.3389/fdata.2023.1296508
- **Kotelnikov et al. 2023** — Kotelnikov A., Baranchuk D., Rubachev I., Babenko A. *TabDDPM: modelling tabular data with diffusion models.* ICML 2023, PMLR 202:17564–17579. arXiv:2209.15421
- **Camino et al. 2020** — Camino R.D., State R., Hammerschmidt C.A. *Oversampling tabular data with deep generative models: is it worth the effort?* ICBINB@NeurIPS 2020, PMLR 137:148–157 — workshop
- **Manousakas & Aydöre 2023** — Manousakas D., Aydöre S. *On the usefulness of synthetic tabular data generation.* arXiv:2306.15636 (2023) — workshop
- **Seedat et al. 2024** — Seedat N., Huynh N., van Breugel B., van der Schaar M. *Curated LLM: synergy of LLMs and data curation for tabular augmentation in low-data regimes.* ICML 2024. arXiv:2312.12112
- **Hameed & Alamgir 2022** — Hameed M.A.B., Alamgir Z. *Improving mortality prediction in acute pancreatitis by machine learning and data augmentation.* Comput Biol Med 150:106077 (2022). doi:10.1016/j.compbiomed.2022.106077
- **Saerens et al. 2002** — Saerens M., Latinne P., Decaestecker C. *Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure.* Neural Comput 14(1):21–41 (2002). doi:10.1162/089976602753284446
- **Dal Pozzolo et al. 2015** — Dal Pozzolo A., Caelen O., Johnson R.A., Bontempi G. *Calibrating probability with undersampling for unbalanced classification.* IEEE SSCI 2015, 159–166. doi:10.1109/SSCI.2015.33
- **Niculescu-Mizil & Caruana 2005** — Niculescu-Mizil A., Caruana R. *Predicting good probabilities with supervised learning.* ICML 2005, 625–632. doi:10.1145/1102351.1102430
- **Van Calster et al. 2016** — Van Calster B., Nieboer D., Vergouwe Y., De Cock B., Pencina M.J., Steyerberg E.W. *A calibration hierarchy for risk models was defined: from utopia to empirical data.* J Clin Epidemiol 74:167–176 (2016). doi:10.1016/j.jclinepi.2015.12.005
- **Newcombe 1998** — Newcombe R.G. *Improved confidence intervals for the difference between binomial proportions based on paired data.* Stat Med 17(22):2635–2650 (1998). doi:10.1002/(SICI)1097-0258(19981130)17:22<2635::AID-SIM954>3.0.CO;2-C

### Fase C: sottogruppi, equità e misure di performance
Verificate su Crossref il 20/09/2026 (DOI, titolo, autori, rivista, anno). Testo completo letto: Matos et al. 2026 e Van Calster et al. 2025 (dal preprint arXiv; le versioni pubblicate su Lancet Digital Health non sono ad accesso libero), Riley et al. 2024, TRIPOD+AI (checklist estesa ufficiale). Letto dall'abstract: McDermott et al. 2024, la cui tesi centrale è però confermata dal testo di Matos et al. 2026 e Van Calster et al. 2025, che lo citano. Solo metadati: Vickers & Holland 2021.
- **Matos et al. 2026** — Matos J., Van Calster B., Celi L.A., Dhiman P., Gichoya J.W., Riley R.D., Russell C., Khalid S., Collins G.S. *Critical appraisal of fairness metrics for artificial intelligence-based clinical prediction models: a scoping review.* Lancet Digit Health 8(7):101001 (2026). doi:10.1016/j.landig.2026.101001 — preprint arXiv:2506.17035. **Fonte della scelta di confrontare i sottogruppi solo con l'AUROC**: raccomanda AUROC Parity e sconsiglia AUPRC Parity perché favorisce i sottogruppi ad alta prevalenza
- **Van Calster et al. 2025** — Van Calster B., Collins G.S., Vickers A.J., Wynants L., Kerr K.F., Barrenada L., Varoquaux G., Singh K., Moons K.G.M., Hernandez-Boussard T., Timmerman D., McLernon D.J., van Smeden M., Steyerberg E.W. *Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance.* Lancet Digit Health (2025). doi:10.1016/j.landig.2025.100916 — preprint arXiv:2412.10288. **Fonte della scelta di non ristimare la soglia sul sottogruppo** (Box 1: la soglia è un argomento medico) e della dipendenza della PR-AUC dalla prevalenza
- **McDermott et al. 2024** — McDermott M.B.A., Hansen L.H., Zhang H., Angelotti G., Gallifant J. *A Closer Look at AUROC and AUPRC under Class Imbalance.* NeurIPS 37 (2024). doi:10.52202/079017-1400 — arXiv:2401.06091. Dimostrazione formale che la PR-AUC favorisce le sottopopolazioni con più positivi
- **Collins et al. 2024 (TRIPOD+AI)** — Collins G.S., Moons K.G.M., Dhiman P., Riley R.D., Beam A.L., Van Calster B., et al. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods.* BMJ 385:e078378 (2024). doi:10.1136/bmj-2023-078378. Item 23a: performance nei sottogruppi **con intervalli di confidenza**
- **Riley et al. 2024 (parte 3)** — Riley R.D., Snell K.I.E., Archer L., Ensor J., Debray T.P.A., Van Calster B., van Smeden M., Collins G.S. *Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study.* BMJ 384:e074821 (2024). doi:10.1136/bmj-2023-074821. **Fonte della scelta di non fare test formali**: servono almeno 100 eventi e 100 non-eventi per stimare la c-statistic con precisione accettabile; il sottogruppo ne ha 68
- **Vickers & Holland 2021** — Vickers A.J., Holland F. *Decision curve analysis to evaluate the clinical benefit of prediction models.* Spine J 21(10):1643–1648 (2021). doi:10.1016/j.spinee.2021.02.024. A supporto: la soglia riflette il compromesso clinico fra danni e benefici, non una quantità stimata dai dati

### Fase D: tetto di prestazione e qualità dell'etichetta
Verificate il 22/09/2026 (Crossref, PubMed, sito dell'editore; dettaglio in `docs/verifica_stato_arte.md`). Abstract letto per Muntner, Tanner, Saydah, Rasaratnam, Hollmann, Ambroise; testo completo per Khitan; pagina degli atti per Menon.
- **Muntner et al. 2011** — Muntner P., Woodward M., Carson A.P., et al. *Am J Kidney Dis* 58(2):196–205 (2011). doi:10.1053/j.ajkd.2011.01.027 — albuminuria senza esami, 8 domande, C 0,709–0,714
- **Tanner et al. 2015** — Tanner R.M., Woodward M., Peralta C., et al. *Ethn Dis* 25(4):427–434 (2015). doi:10.18865/ed.25.4.427 — stesso strumento in MESA, C 0,728–0,761
- **Khitan et al. 2021** — Khitan Z., Nath T., Santhanam P. *J Clin Hypertens* 23(12):2137–2145 (2021). doi:10.1111/jch.14397 — albuminuria nel diabete tipo 2, con esami del sangue compresa la creatinina, AUC 0,61–0,67
- **Saydah et al. 2013** — Saydah S.H., Pavkov M.E., Zhang C., et al. *Clin Chem* 59(4):675–683 (2013). doi:10.1373/clinchem.2012.195644 — il 43,5% degli ACR ≥ 30 su urina casuale confermato sulla prima urina del mattino
- **Rasaratnam et al. 2024** — Rasaratnam N., Salim A., Blackberry I., et al. *Am J Kidney Dis* 84(1):8–17.e1 (2024). doi:10.1053/j.ajkd.2023.12.018 — variabilità intra-individuale dell'ACR 48,8% nel diabete tipo 2
- **Menon et al. 2015** — Menon A.K., van Rooyen B., Ong C.S., Williamson R.C. *Learning from corrupted binary labels via class-probability estimation.* ICML 2015, PMLR 37:125–134 — etichette rumorose e AUROC
- **Hollmann et al. 2025** — Hollmann N., Müller S., Purucker L., et al. *Accurate predictions on small data with a tabular foundation model.* Nature 637(8045):319–326 (2025). doi:10.1038/s41586-024-08328-6 — TabPFN
- **Ambroise & McLachlan 2002** — Ambroise C., McLachlan G.J. *Selection bias in gene extraction on the basis of microarray gene-expression data.* PNAS 99(10):6562–6566 (2002). doi:10.1073/pnas.102102699 — distorsione da selezione delle variabili fuori dalla validazione

### Utilità clinica e misure di qualità (post-hoc)
Verificate su Crossref il 22/09/2026 (DOI, titolo, autori, rivista, anno). Testo completo letto: Bragg-Gresham et al. 2024 (PMC), Benitez-Aurioles et al. 2024 (arXiv), Vickers, van Calster & Steyerberg 2019 e Vickers et al. 2023 (PMC, accesso aperto); Cusick et al. 2023 letto nel testo completo il 20/09/2026. Solo metadati: Harrell 2015 (libro). Servono anche Vickers & Elkin 2006, Van Calster et al. 2016 e 2019 (sezioni Fase A e Fase B), Van Calster et al. 2025, Matos et al. 2026 e TRIPOD+AI (sezione Fase C); di Van Calster et al. 2019 il testo completo è stato letto il 22/09/2026 (PMC).
- **Bragg-Gresham et al. 2024** — Bragg-Gresham J.L., Annadanam S., Gillespie B., Li Y., Powe N.R., Saran R. *Using Risk Assessment to Improve Screening for Albuminuria among US Adults without Diabetes.* J Gen Intern Med 40(13):3159–3169 (online 2024, fascicolo 2025). doi:10.1007/s11606-024-09185-9 — **fonte delle soglie 5% e 7%** (esempi di punto operativo per mandare all'esame dell'albuminuria un adulto senza diabete) e riferimento esterno del confronto. Il loro modello include eGFR < 60 fra i predittori; c-statistic 0,752 in validazione; concludono che serve una valutazione di costo-efficacia
- **Cusick et al. 2023** — Cusick M.M., Tisdale R.L., Chertow G.M., Owens D.K., Goldhaber-Fiebert J.D. *Population-Wide Screening for Chronic Kidney Disease: A Cost-Effectiveness Analysis.* Ann Intern Med 176(6):788–797 (2023). doi:10.7326/M22-3228 — **fonte del costo del test ($49, intervallo $36–$64) e del limite inferiore dell'intervallo di soglie**: lo screening di popolazione è costo-efficace ($86.300 per QALY, una tantum a 55 anni)
- **Benitez-Aurioles et al. 2024** — Benitez-Aurioles J., Joules A., Brusini I., Peek N., Sperrin M. *Understanding algorithmic fairness for clinical prediction in terms of subgroup net benefit and health equity.* arXiv:2412.07879 (2024) — preprint, riferimento 88 di Matos et al. 2026. **Definizione formale del subgroup net benefit** (equazione 5): `1 − π_g + λ × NB_g(t)`
- **Vickers, van Calster & Steyerberg 2019** — Vickers A.J., van Calster B., Steyerberg E.W. *A simple, step-by-step guide to interpreting decision curve analysis.* Diagn Progn Res 3:18 (2019). doi:10.1186/s41512-019-0064-7 — **intervallo di soglie fissato prima** e net benefit espresso come esami evitati quando il riferimento è "testare tutti"
- **Vickers et al. 2023** — Vickers A.J., Van Calster B., Wynants L., Steyerberg E.W. *Decision curve analysis: confidence intervals and hypothesis testing for net benefit.* Diagn Progn Res 7:11 (2023). doi:10.1186/s41512-023-00148-y — **fonte della scelta di non fare test né intervalli sul net benefit** (su Crossref il secondo autore compare come "Van Claster", refuso dei metadati)
- **Harrell 2015** — Harrell F.E. Jr. *Regression Modeling Strategies*, 2ª ed. Springer (2015). doi:10.1007/978-3-319-19425-7 — quantili dei nodi della spline cubica ristretta usata per la curva di calibrazione flessibile

### Interpretazione dei modelli
Verificate su Crossref il 19/09/2026; Lundberg et al. 2020 anche nell'abstract (algoritmo esatto in tempo polinomiale per gli alberi, con un'applicazione alla malattia renale cronica).
- **Lundberg et al. 2020** — Lundberg S.M., Erion G., Chen H., DeGrave A., Prutkin J.M., Nair B., Katz R., Himmelfarb J., Bansal N., Lee S.-I. *From local explanations to global understanding with explainable AI for trees.* Nat Mach Intell 2(1):56–67 (2020). doi:10.1038/s42256-019-0138-9
- **Hosmer, Lemeshow & Sturdivant 2013** — Hosmer D.W., Lemeshow S., Sturdivant R.X. *Applied Logistic Regression*, 3ª ed. Wiley (2013). doi:10.1002/9781118548387

### Fase A: modelli, dimensione del campione e ottimizzazione
- **Bang et al. 2007** — Bang H. et al. *SCreening for Occult REnal Disease (SCORED): a simple prediction model for chronic kidney disease.* Arch Intern Med 167(4):374 (2007). doi:10.1001/archinte.167.4.374 — abstract letto (`papers/kdigo/`), testo completo non ad accesso libero
- **Echouffo-Tcheugui & Kengne 2012** — Echouffo-Tcheugui J.B., Kengne A.P. *Risk models to predict chronic kidney disease and its progression: a systematic review.* PLoS Med 9(11):e1001344 (2012). doi:10.1371/journal.pmed.1001344 ✅ (`papers/kdigo/file.pdf`)
- **Christodoulou et al. 2019** — Christodoulou E. et al. *A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models.* J Clin Epidemiol 110:12–22 (2019). doi:10.1016/j.jclinepi.2019.02.004
- **Pavlou et al. 2015** — Pavlou M. et al. *How to develop a more accurate risk prediction model when there are few events.* BMJ 351:h3868 (2015). doi:10.1136/bmj.h3868
- **Peduzzi et al. 1996** — Peduzzi P. et al. *A simulation study of the number of events per variable in logistic regression analysis.* J Clin Epidemiol 49(12):1373–1379 (1996). doi:10.1016/S0895-4356(96)00236-3
- **Riley et al. 2020** — Riley R.D. et al. *Calculating the sample size required for developing a clinical prediction model.* BMJ 368:m441 (2020). doi:10.1136/bmj.m441
- **Tibshirani 1996** — Tibshirani R. *Regression shrinkage and selection via the lasso.* J R Stat Soc B 58(1):267–288 (1996). doi:10.1111/j.2517-6161.1996.tb02080.x
- **Hoerl & Kennard 1970** — Hoerl A.E., Kennard R.W. *Ridge regression: biased estimation for nonorthogonal problems.* Technometrics 12(1):55–67 (1970). doi:10.1080/00401706.1970.10488634
- **Zou & Hastie 2005** — Zou H., Hastie T. *Regularization and variable selection via the elastic net.* J R Stat Soc B 67(2):301–320 (2005). doi:10.1111/j.1467-9868.2005.00503.x
- **Friedman et al. 2010** — Friedman J. et al. *Regularization paths for generalized linear models via coordinate descent.* J Stat Softw 33(1) (2010). doi:10.18637/jss.v033.i01
- **Chen & Guestrin 2016** — Chen T., Guestrin C. *XGBoost: a scalable tree boosting system.* Proc. KDD 2016, 785–794. doi:10.1145/2939672.2939785
- **Varma & Simon 2006** — Varma S., Simon R. *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics 7:91 (2006). doi:10.1186/1471-2105-7-91
- **Cawley & Talbot 2010** — Cawley G.C., Talbot N.L.C. *On over-fitting in model selection and subsequent selection bias in performance evaluation.* J Mach Learn Res 11:2079–2107 (2010)
- **Akiba et al. 2019** — Akiba T. et al. *Optuna: a next-generation hyperparameter optimization framework.* Proc. KDD 2019, 2623–2631. doi:10.1145/3292500.3330701
- **Bergstra et al. 2011** — Bergstra J., Bardenet R., Bengio Y., Kégl B. *Algorithms for hyper-parameter optimization.* NeurIPS 24 (2011)
- **Elkan 2001** — Elkan C. *The foundations of cost-sensitive learning.* Proc. IJCAI 2001, 973–978
