# La pipeline passo per passo

Schema ragionato di tutta la pipeline, in ordine di esecuzione. Per ogni passo: **cosa** si fa, **come** e **perché**. Le sezioni "Per capire" spiegano i concetti più difficili: cross-validation annidata, soglia, accuratezza, AUROC, confronto con KDIGO. In fondo ci sono i risultati in sintesi.

I numeri vengono dalle tabelle in `analytics/` e dal `Notepad.md`; le altre domande probabili sono in [`incontro_relatori.md`](incontro_relatori.md).

## Il quadro in una figura

```
dataset grezzo (5.922 × 190)
 └─ 1–3. controlli, soggetti eleggibili (5.801), bersaglio e livello KDIGO
     └─ 4. split 75/25 ─────────────── test (1.451): chiuso fino al passo 17
         └─ training (4.350)
             └─ 5–7. colonne (190 → 74), codifiche, scelta dell'imputazione
                 └─ 8. fold: 5 esterni × 5 interni
                     └─ 9. per ogni fold: standardizzazione → moda → MissForest
                         ├─ 10. Fase A: Optuna → modello → previsioni out-of-fold
                         └─ 11–12. Fase B: bilanciamento → Optuna → modello → Platt
             └─ 13–16. valutazione, interpretazione, Fasi C e D, utilità clinica
 └─ 17. test: preprocessore del training → modelli finali → conferma (una volta)
```

---

## Dati e bersaglio (prima di tutto)

### 1. Analisi del dataset
- **Cosa**: il file Excel di Li et al. 2026 (5.922 soggetti × 190 variabili), con impronta SHA-256, e il dizionario delle variabili letto dal PDF.
- **Verifiche**, fatte prima di usare qualsiasi colonna:
  - la colonna `GFR` non corrisponde a nessuna equazione standard (provate CKD-EPI 2009 e 2021, MDRD, MDRD cinese e Cockcroft-Gault) e arriva a 562, mentre le equazioni non superano circa 150–160;
  - `Gender` vale 1/2, non 0/1 come dice il dizionario. Dalla fisiologia, 1 = maschio e 2 = femmina: creatinina mediana 76,7 contro 54,9 µmol/L, emoglobina 155 contro 135 g/L;
  - `DM` coincide con i criteri OMS 1999: anamnesi di diabete, glicemia a digiuno ≥ 7,0 mmol/L oppure glicemia a 2 ore ≥ 11,1;
  - il 9 significa "sconosciuto" in 26 colonne. Non va mai ricodificato su tutto il dataset: ALT, TBIL e MPV hanno valori reali pari a 9.
- **Perché**: il bersaglio dipende da eGFR e sesso. Con la colonna `GFR` o con il sesso letto male sarebbe sbagliato.

### 2. Soggetti eleggibili
- 5.801 soggetti con creatinina, ACR, età e sesso; 121 esclusi.
- **Perché**: senza creatinina o ACR il bersaglio non si può calcolare.

### 3. Bersaglio e livelli
- **eGFR**: ricalcolato con CKD-EPI 2021, senza coefficiente etnico, come raccomanda KDIGO 2024:
  - eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^−1,200 × 0,9938^età × 1,012 (se donna);
  - la creatinina Scr è in mg/dL (µmol/L ÷ 88,4); κ vale 0,7 per le donne e 0,9 per gli uomini, α vale −0,241 e −0,302.
- **Bersaglio**: y = 1 se ACR ≥ 30 mg/g oppure eGFR < 60, i due marcatori KDIGO di malattia renale cronica.
  - 567 positivi, il 9,8%;
  - nel training i positivi sono 362 per la sola albuminuria, 25 per entrambi i criteri e 38 per il solo eGFR < 60: il bersaglio è al 91% albuminuria.
- **Livello KDIGO**: incrocio delle categorie G (eGFR) e A (ACR) della heatmap: basso 5.234, moderato 473, alto 67, molto alto 27.
- **Perché binario**: i "molto alto" sono 27 in tutto, e un modello a 4 classi sarebbe instabile. Il modello impara il binario, i livelli servono a valutarlo.
- **Perché non una regressione sull'ACR**: l'ACR è molto asimmetrica, e la decisione clinica (fare o no l'esame) è una soglia.
- **Perché non il primo bersaglio, `DN`**:
  - era una soglia sulla concentrazione di albumina, non sull'ACR: coincide con `UmALB` ≥ 30 in 455 casi su 456;
  - ignorava il filtrato;
  - l'82,5% dei positivi non era diabetico.
- **Bersaglio e livello non entrano mai fra le variabili**: altrimenti il modello ricopierebbe la regola KDIGO (leakage).

### 4. Split
- **Come**: `train_test_split` 75/25, seed 42, stratificato su livello KDIGO × diabete. Sono 8 strati; il più piccolo, i diabetici "molto alto", ha 6 soggetti.
- **Training**: 4.350 soggetti, 425 positivi (50 alto, 21 molto alto), 263 diabetici (68 positivi).
- **Test**: 1.451 soggetti, 142 positivi (17 alto, 6 molto alto), 88 diabetici (23 positivi).
- **Perché 75/25**: servono almeno 100 eventi nel test (Collins et al. 2016; Riley et al. 2021). Con 80/20 nel test restavano 5 "molto alto"; con 70/30 il training perdeva 28 positivi.
- **Perché stratificare sul livello**: su 1.000 split simulati, i "molto alto" nel test andavano da 0 a 15 senza stratificazione, e da 1 a 14 stratificando sul solo bersaglio. Stratificando sul livello sono sempre 6.
- **Perché anche sul diabete**: la prevalenza è del 25,9% nei diabetici contro l'8,7% negli altri, e il sottogruppo serve alla domanda 6.
- **Controllo**: differenza media standardizzata (SMD) sotto 0,1 su 17 variabili, con un massimo di 0,059.
- Da qui il test resta chiuso fino al passo 17.

*Codice*: `src/data/load.py`, `src/data/kidney.py`, `src/data/split.py`.

---

## Preprocessing "senza stato" (una volta, prima della CV)

Queste trasformazioni lavorano riga per riga e non imparano nulla dai dati: si fanno una volta, prima di dividere in fold.

### 5. Colonne, da 190 a 74
- **Cosa**: restano 74 colonne in ingresso, 60 numeriche e 14 categoriche. Nella mappa di `configs/config.yaml` ogni colonna ha un gruppo e un motivo; un test fallisce se una colonna non è assegnata.
- **Escluse**, in 12 gruppi:
  - le colonne da cui si calcola il bersaglio (`SCRE`, `UMAUCR`, `UmALB`, `UCRE`, `GFR`, `DN` e 10 indicatori derivati) e gli esami renali (`BUN`, `RF`). Con queste l'AUROC arriverebbe a 0,96–0,998;
  - colonne amministrative, non documentate, vuote (oltre il 50% di mancanti) e il questionario sul diabete;
  - derivati a soglia, doppioni, combinazioni lineari esatte;
  - comorbidità con il codice 9, `Homaβ` (formula instabile), colonne con oltre il 15% di mancanti.
- **Controlli**:
  - un'asserzione ferma il codice se entra una colonna vietata;
  - nessuna variabile da sola supera AUROC 0,75; la più alta è l'età, con 0,653.
- **Perché non è leakage farla prima della CV**: le regole decisive non usano l'esito (tipo di variabile, esami renali, mancanti). L'unica selezione basata sull'esito, quella del tri-ensemble, è stata fatta dentro ogni fold (passo 16).
- **Set di sensibilità `no_consequence`** (67 colonne): toglie le 7 variabili che la malattia renale altera, cioè emoglobina, globuli rossi, ematocrito, acido urico, albumina, proteine totali e albumina glicata. Serve a verificare che il modello non riconosca la malattia dalle sue conseguenze: l'AUROC cambia fra −0,012 e +0,001.

### 6. Codifiche, nessun one-hot
- **Come**:
  - `Gender` 1/2 → 0/1;
  - fumo, alcol e tè 1/2/3 → 0/1/2, ordinali;
  - `HypertenHis` vuoto → 0;
  - le altre 9 categoriche sono già 0/1 nel file grezzo.
- **Perché niente one-hot**: le categoriche sono tutte binarie o ordinali. Con il one-hot SMOTE e CTGAN potrebbero generare combinazioni impossibili, come "non fuma" e "fuma regolarmente" entrambe a 1. Gli alberi tagliano bene le ordinali.

### 7. Scelte decise una volta, con una CV a 5 fold
**Imputazione con MissForest.** Il 38% dei soggetti del training ha almeno un valore mancante, quindi eliminarli è escluso. Quattro metodi sono stati confrontati sugli stessi 5 fold, con due criteri:

| metodo | errore di ricostruzione (RMSE) ↓ | PR-AUC a valle |
|---|---|---|
| mediana | 1,038 | 0,252 |
| KNN (k = 5) | 0,793 | 0,259 |
| MICE | 1,252 | 0,257 |
| **MissForest** | **0,734** | 0,257 |

- a valle i quattro metodi sono equivalenti;
- nel confronto appaiato MissForest ricostruisce meglio in tutti i fold;
- serve alla Fase B, dove SMOTE e CTGAN lavorano sui valori imputati;
- da dire con onestà: la regola di scelta in due fasi è stata introdotta dopo i primi risultati, e sul set principale darebbe KNN per un margine piccolo.

**Niente logaritmo**:
- sulle 25 variabili molto asimmetriche vale +0,008 ± 0,005 di PR-AUC, non significativo;
- agli alberi non cambia nulla;
- la scala clinica rende leggibili odds ratio e SHAP.

**Valori estremi tenuti**: sono clinicamente possibili, e toglierli solo dal training renderebbe il modello impreparato sul test.

*Codice*: `src/data/preprocess.py` (`select_features`, `encode`), `src/data/imputation.py`.

---

## Dentro la cross-validation annidata

### 8. Fold
- **Come**: `StratifiedKFold` a 5, stratificato sul livello KDIGO, seed 42. Ci sono 5 fold esterni e, dentro il training di ognuno, 5 fold interni; con il training intero sono 31 "fold".
- **Stessi fold per tutto**: confronto delle imputazioni, Fase A, Fase B e Fase D. Così le differenze dipendono solo da modello e tecnica.

| parte | soggetti | positivi |
|---|---|---|
| fold esterno: training | 3.480 | 340 |
| fold esterno: validazione | 870 | 85 |
| fold interno: training | 2.784 | 272 |
| fold interno: validazione | 696 | 68 |

#### Per capire: fold interni ed esterni
Con i dati del training si fanno due lavori diversi: **scegliere** gli iperparametri e **misurare** le prestazioni. Se li fai sugli stessi dati la misura esce gonfiata, perché hai scelto la combinazione che funzionava meglio proprio lì, fortuna compresa. Per questo una CV sta dentro l'altra.

```
giro esterno 1, per un modello con iperparametri (per esempio XGBoost):
1. metti da parte 870 soggetti            → non li tocchi fino al punto 6
2. restano 3.480: li dividi in 5 parti da 696 (i fold interni)
3. Optuna prova una combinazione: 5 addestramenti su 2.784 soggetti, PR-AUC sui 696 → media;
   ripete, fino a 100 combinazioni
4. vince la combinazione migliore; i modelli di prova si buttano
5. RIADDESTRAMENTO: un modello nuovo, con la combinazione vincente, su tutti i 3.480
6. solo ora lo usi sugli 870 messi da parte → 870 previsioni

giri 2–5: uguali, ogni volta con altri 870 → 4.350 previsioni "out-of-fold"
```

- **Chi fa cosa**: i fold interni **scelgono**, i fold esterni **misurano**, il test **conferma**. Il fold esterno non serve mai a scegliere.
- **Modelli senza iperparametri** (maggioranza e SCORED): saltano i punti 2–4.
- **Il modello finale** è un passo a parte, dopo i 5 giri. La procedura si ripete un livello più su:
  - il training intero (4.350) sceglie gli iperparametri con le sue 5 parti;
  - si addestra un solo modello su tutti i 4.350;
  - il ruolo degli 870 messi da parte lo fa il test, usato una volta sola.
- **Perché così**:
  - le stime non escono gonfiate (Varma & Simon 2006; Cawley & Talbot 2010);
  - il test resta per la fine;
  - c'è una previsione per ogni soggetto del training, quindi l'analisi per livello usa 50 "alto" e 21 "molto alto" invece di 17 e 6.

### 9. Preprocessing "con stato", per ogni fold
Stimato solo sulla parte di training del fold, senza vedere y, e poi applicato alla validazione con `transform`:
1. **standardizzazione** delle 60 numeriche con `StandardScaler`: z = (x − media) / deviazione standard, con media e deviazione standard della parte di training; i mancanti sono ignorati nel calcolo;
2. **moda** per le 14 categoriche (`SimpleImputer`), che restano intere e non vengono standardizzate;
3. **MissForest** sulle 74 colonne (`IterativeImputer` con `ExtraTreesRegressor`, 50 alberi, 5 iterazioni): ogni numerica con buchi è predetta dalle altre, categoriche comprese, per esempio `DM` per stimare `HbA1c`.

- **Perché dentro il fold**: queste trasformazioni imparano dai dati (medie, deviazioni standard, foreste). Stimarle su tutto il training farebbe entrare nella stima i soggetti di validazione.
- **Perché standardizzare**: serve a due cose; agli alberi invece è indifferente.
  - alla logistica penalizzata, perché la penalità tratta tutti i coefficienti allo stesso modo e gli odds ratio si leggono "per 1 deviazione standard";
  - alle distanze di KNN e SMOTE, che altrimenti sarebbero dominate dalle variabili con valori grandi (piastrine circa 250 contro HbA1c circa 6).
- **Perché la moda per le categoriche**: KNN o MICE su una variabile binaria danno valori senza senso, come `DM` = 0,37.
- **Cache**: le matrici si calcolano una volta sola (31 fold × 2 set, circa 1 ora) e si riusano per tutti i modelli e le tecniche. È lecito perché l'imputer non vede mai y.
- **Com'è una riga dentro un fold**:
  - 60 numeriche in deviazioni standard dalla media del fold, senza mancanti;
  - 14 categoriche intere, 0/1 o 0/1/2;
  - nessuna colonna renale, nessun ID;
  - esito e livello KDIGO stanno fuori dalla matrice.

### 10. Fase A, dati originali
- **I 5 modelli**, uno per ruolo:
  - **classificatore di maggioranza**: prevede a tutti la prevalenza; è il controllo di coerenza, deve dare AUROC 0,5;
  - **logistica SCORED**: i 5 predittori disponibili del punteggio clinico SCORED (età, sesso, emoglobina, pressione sistolica, diabete), senza penalizzazione;
  - **logistica penalizzata** (elastic net) su 74 variabili: con 5,7 eventi per variabile la penalizzazione è necessaria;
  - **Random Forest** (bagging) e **XGBoost** (boosting): gli ensemble di alberi, uno per famiglia.
- **Optuna**: campionatore TPE (seed 42), 100 tentativi per modello e fold. Il `MedianPruner` interrompe un tentativo che, dopo un fold interno, è sotto la mediana dei precedenti.
- **Metrica PR-AUC**: con il 9,8% di positivi guarda dove sta il problema; il suo valore di base è la prevalenza, non 0,5.
- **Iperparametri finali**:
  - logistica penalizzata: C = 1,53, `l1_ratio` = 0,85;
  - Random Forest: 885 alberi, profondità 24;
  - XGBoost: 392 alberi, learning rate 0,025, profondità 3;
  - fra i fold cambiano molto: è il segno di un ottimo piatto.
- **Analisi di sensibilità**: XGBoost con profondità 1–12 sceglie profondità 1, cioè una somma di effetti delle singole variabili. Il segnale è additivo, senza interazioni.
- **Nessun peso di classe**: è una tecnica di bilanciamento, appartiene alla Fase B.

### 11. Fase B, bilanciamento sugli stessi fold
Avviene dopo il passo 9 e solo sulla parte di training del fold. La validazione resta sempre reale.

| tecnica | righe di training in un fold esterno (3.480 reali) |
|---|---|
| pesi di classe; pesi per livello 1:2:3 | 3.480: nessuna riga nuova, cambiano i pesi (`sample_weight`) |
| undersampling 1:1 | 680 (340 positivi + 340 negativi estratti a caso) |
| oversampling 1:1 | 6.280 (positivi duplicati fino a 3.140) |
| SMOTE-NC, anche dentro ogni livello | 6.280 (340 positivi reali + 2.800 sintetici) |
| CTGAN, su y o sul livello (esplorative) | 6.280 (340 reali + 2.800 generati) |

- **Perché solo sulla parte di training**: bilanciare prima di dividere metterebbe copie o interpolazioni dello stesso paziente in training e in validazione (Santos et al. 2018).
- **Optuna**: 30 tentativi, partendo dall'ottimo della Fase A. Nei log della Fase A, a 30 tentativi la PR-AUC interna era già in mediana entro 0,005 da quella a 100.
- **CTGAN**: solo fold esterni, senza ottimizzazione né ricalibrazione. Con circa 340 positivi per fold una rete generativa ha troppo pochi dati.
- **Scartate, con motivo scritto**:
  - Borderline-SMOTE e ADASYN: con 17–40 casi gravi per fold moltiplicano il rumore;
  - SMOTE-ENN e SMOTE-Tomek: cancellano anche negativi reali;
  - TVAE, TabDDPM, CTAB-GAN+: hanno gli stessi limiti di CTGAN;
  - EasyEnsemble, RUSBoost, Balanced Random Forest: cambiano il modello.
- **Prevista e non eseguita**: ripetere una tecnica con KNN al posto di MissForest (Notepad, Passo 10). Va dichiarata.

### 12. Ricalibrazione di Platt (Fase B)
- **Come**: una logistica su logit(p), stimata sulle previsioni che il modello fa sulle righe **reali** dei fold interni, e poi applicata al fold esterno.
- **Perché**: il bilanciamento gonfia le probabilità, e l'intercetta di calibrazione scende da 0 a valori fra −1,4 e −2,2. Platt la riporta a −0,01.
- **Dove si usa**: per le probabilità medie e la calibrazione. Soglia, fasce ed esito primario usano le probabilità grezze, perché dipendono solo dall'ordinamento.

*Codice*: `src/data/folds.py`, `src/data/imputed.py`, `src/models/zoo.py`, `src/models/phase_a.py`, `src/data/augmented.py`, `src/models/phase_b.py`.

---

## Valutazione (sulle previsioni out-of-fold)

### 13. Domande 1–4
- **Soglia**: la più alta con sensibilità ≥ 0,90 sulle previsioni out-of-fold, fra 0,042 e 0,058 secondo il modello. Si stima sul training e si applica invariata al test.
- **Fasce**: 4, con le stesse proporzioni dei livelli KDIGO nel training (90,2 / 8,1 / 1,1 / 0,5%).
- **Metriche**:
  - AUROC e PR-AUC;
  - precision, recall, specificità, quota da esaminare;
  - probabilità media e sensibilità per livello;
  - concordanza di Jonckheere-Terpstra e kappa pesato.
- **Intervalli**: DeLong per l'AUROC, logit di Boyd per la PR-AUC, Wilson per le proporzioni, bootstrap stratificato (2.000 campioni) per kappa e medie.
- **Confronti**: t corretto di Nadeau-Bengio con correzione di Holm.
- **Fase B**:
  - l'esito primario sono i casi gravi (alto + molto alto, 71) riconosciuti a sensibilità 0,90, con McNemar esatto e IC di Newcombe;
  - in più, la valutazione "ingenua" alla soglia 0,5.

#### Per capire: la soglia è un prezzo
Il modello dà una probabilità, e per dire sì o no serve una soglia. Abbassarla trova più malati, ma manda agli esami molti più sani. Logistica penalizzata, su 100 persone (9,8 con marcatori):

| soglia | esami | malati trovati | malati persi | esami per ogni malato in più |
|---|---|---|---|---|
| 0,5 (predefinita) | 0,7 | 0,3 | 9,4 | — |
| 0,070 | 53,5 | 7,3 | 2,5 | 8 |
| 0,063 (sensibilità 0,80) | 60,7 | 7,8 | 2,0 | 14 |
| 0,057 (sensibilità 0,85) | 69,2 | 8,3 | 1,4 | 17 |
| **0,048 (sensibilità 0,90)** | **78,9** | **8,8** | **1,0** | **20** |
| 0,039 (sensibilità 0,95) | 88,5 | 9,3 | 0,5 | 20 |
| 0 (testare tutti) | 100 | 9,8 | 0 | 24 |

- **Abbassare la soglia non migliora il modello**: ci si sposta lungo la stessa curva, scambiando malati persi con esami inutili. Migliorare il modello vuol dire trovare più malati **con gli stessi esami**.
- **Non esiste una soglia "giusta" in senso statistico**. Con probabilità calibrate, una soglia t vuol dire circa "accetto 1/t esami per trovare un malato in più": a 0,048, circa 20. È una scelta clinica ed economica (Van Calster et al. 2025).
- **La tesi la gestisce in due modi**:
  - una regola fissata prima dei risultati: trovare almeno il 90% dei malati, come il punto operativo di SCORED. Sul test tiene, con recall 0,887–0,937;
  - la decision curve, che invece di scegliere una soglia le prova tutte dal 2% al 20%.

#### Per capire: perché non l'accuratezza
Alla soglia 0,5 il modello non segnala quasi nessuno, perché le probabilità sono calibrate sulla prevalenza: in media valgono 0,098, e solo l'1–3% delle persone supera 0,30.

| alla soglia 0,5 | segnalati su 4.350 | malati trovati su 425 | accuratezza |
|---|---|---|---|
| regola "sono tutti sani" | 0 | 0 | 90,2% |
| logistica SCORED | 7 | 5 | 90,3% |
| logistica penalizzata | 30 | 14 | 90,2% |
| Random Forest | 0 | 0 | 90,2% |
| XGBoost | 27 | 15 | 90,3% |

- **Il 90% non dice nulla**: con il 90,2% di sani, chi dice "no" a tutti ha già il 90,2% di risposte giuste. È aritmetica, quindi certo al 100%.
- **Pesa gli errori allo stesso modo**: l'accuratezza conta un malato perso quanto un esame inutile, mentre clinicamente non valgono lo stesso. F1 e MCC hanno lo stesso limite.
- **In Fase B lo stesso effetto si vede al contrario**: il bilanciamento gonfia le probabilità, e alla soglia 0,5 il recall sale dal 2% al 60% mentre l'accuratezza scende al 65–69%. L'ordine dei pazienti non cambia: a parità di sensibilità nessuna tecnica trova più casi gravi.

#### Per capire: AUROC e le altre misure
- **AUROC**: non dipende dalla soglia e misura quanto il modello ordina bene malati e sani. Serve a **confrontare** modelli e tecniche.
- **Per dire se il modello serve non basta**. La tesi riporta l'insieme minimo raccomandato da Van Calster et al. 2025:
  - calibrazione;
  - misure alla soglia scelta: sensibilità, specificità, valori predittivi, esami ogni 100 persone;
  - decision curve;
  - distribuzione delle probabilità fra malati e sani.

#### Per capire: il confronto con KDIGO
- **Cosa si confronta**: per ogni soggetto, quello che dice il modello (probabilità, classe, fascia) accanto al livello vero, calcolato dagli esami che il modello non vede.
- **Tre misure, e solo una usa le fasce**:
  - domanda 2: probabilità contro livello, senza fasce (Jonckheere-Terpstra);
  - domanda 3: sensibilità per livello alla soglia 0/1, senza fasce;
  - domanda 4: fasce contro livelli, con tabella 4 × 4 e kappa pesato.
- **Le fasce non sono i livelli**: sono la proposta di gravità del modello, tagliata sui quantili della probabilità. Per XGBoost la fascia 2 parte da 0,193, la 3 da 0,352, la 4 da 0,456. Con soglie fisse come 30% e 70% le fasce alte resterebbero vuote. Che le fasce coincidano poco con i livelli è il risultato, non un difetto del metodo.
- **Il confronto non è indipendente**: il bersaglio è "livello moderato o superiore", quindi il 98,5% delle coppie della concordanza sono "basso contro positivo", le stesse dell'AUROC. La gravità vera si misura fra i soli positivi.
- **Risultato**:
  - la probabilità media cresce con il livello (XGBoost: 0,086 → 0,142 → 0,194 → 0,284);
  - fra i soli positivi la concordanza è 0,61–0,64 (0,5 = nessun ordine) e il kappa è circa 0,2;
  - XGBoost mette in fascia 4 solo 5 "molto alto" su 21, e 6 restano in fascia 1.

  Il modello ordina la gravità debolmente: **decide chi testare, il livello lo decidono gli esami**.

### 14. Interpretazione
- **Come**: sui modelli finali, odds ratio per le logistiche e SHAP esatto (TreeSHAP) per Random Forest e XGBoost.
- **SCORED**:
  - diabete: OR 2,58 (1,89–3,52);
  - età: 1,47 ogni 13,8 anni;
  - pressione sistolica: 1,34 ogni 16,5 mmHg;
  - emoglobina e sesso non significativi.
- **SHAP**:
  - l'età è prima in entrambi gli alberi; seguono ALP, FIB-4, peptide C, albumina glicata, glicemie, lipidi, acido urico;
  - l'importanza è diffusa: le prime 5 variabili fanno il 23–26% del totale.
- **Cautela**: descrivono il modello, non la causalità.

### 15. Fase C, diabetici
- **Come**: le stesse previsioni filtrate sui 263 diabetici del training (68 positivi, 16 casi gravi), con soglia e fasce globali. Il confronto usa solo la differenza di AUROC.
- **Risultati**:
  - discrimina come sugli altri: differenze di AUROC da −0,038 a +0,042;
  - la PR-AUC sembra doppia (0,44 contro 0,19), ma solo per la prevalenza;
  - alla soglia globale segnala il 97–100% dei diabetici, cioè equivale a "testare tutti".

### 16. Analisi post-hoc
**Fase D, il tetto di prestazione.** 11 candidati sugli stessi fold, con una regola scritta prima: ensemble, CatBoost, LightGBM, EBM, TabPFN v2, varianti di XGBoost, bersaglio scomposto, tri-ensemble. Nessuno supera la regola: AUROC fra 0,692 e 0,710.
- **diagnostiche**:
  - con l'albumina urinaria fra le variabili l'AUROC sale a 0,933: la pipeline impara quando l'informazione c'è;
  - dal 60% al 100% del training si guadagna solo 0,006–0,008;
  - l'eGFR < 60 si riconosce con 0,80–0,86, la sola albuminuria con 0,67–0,69: il limite è l'albuminuria.
- **tri-ensemble**: 0,703 scegliendo le 21 variabili dentro ogni fold, 0,716 scegliendole su tutto il training. È distorsione da selezione.

**Utilità clinica.**
- **decision curve** contro "testare tutti":
  - al 7% il modello evita 9–13 esami inutili ogni 100 persone, al 10% 25–28;
  - fra i diabetici non evita niente fino al 10%.
- **calibrazione**: buona in media. Random Forest schiaccia le probabilità (pendenza 1,31), XGBoost le esaspera (0,86).

*Codice*: `src/models/evaluation.py`, `evaluation_b.py`, `interpretation.py`, `phase_c.py`, `phase_d.py`, `clinical_utility.py`.

---

## Test (una volta sola)

### 17. Conferma finale
- **Come**:
  - il preprocessore è ristimato su tutto il training, e risulta identico bit a bit alla cache;
  - si applica un solo `transform` al test, poi si usano i modelli finali già salvati;
  - soglie e fasce restano quelle fissate out-of-fold.
- **Sicurezza**:
  - il codice legge il test solo con `authorized: true` in config;
  - `RUN.json` registra data, versioni delle librerie e impronta del file;
  - eseguito il 22/09 dalle 18:20 alle 18:37, una volta sola.

**Le 5 affermazioni fissate prima**:

| affermazione | esito |
|---|---|
| (A) discriminazione come out-of-fold | non contraddetta: AUROC 0,714–0,743 |
| (B) il bilanciamento non trova più casi gravi | non contraddetta: da −2 a +2 su 23 |
| (C) sui diabetici il modello segnala quasi tutti | confermata: 97,7–100% |
| (D) utilità clinica al 7% e al 10% | confermata: 13–19 e 29–33 esami evitati ogni 100 |
| (E) alla soglia 0,5 il bilanciamento sembra aiutare | confermata: recall 4,4% contro 40–65% |

### 18. Dopo il test, sul solo training
- **Audit del tetto**:
  - una variabile semi-sintetica che da sola vale 0,75 o 0,80 porta l'AUROC a 0,795 e 0,831: la pipeline trova i segnali quando ci sono;
  - il rumore dell'etichetta vicino alla soglia vale al massimo +0,009;
  - l'unità dell'ACR (fattore 176,8, non standard) vale al massimo +0,016.
- **Bersaglio continuo dell'albuminuria**: addestrare sul logaritmo dell'ACR invece che sulla soglia dà −0,004 (IC da −0,038 a +0,030). Non aiuta.

*Codice*: `src/models/final_test.py`, `src/models/phase_d.py`.

---

## I risultati in sintesi

### Quanto distingue

| modello | AUROC training (IC 95%) | AUROC test (IC 95%) | PR-AUC training |
|---|---|---|---|
| logistica SCORED (5 variabili) | 0,675 (0,646–0,704) | 0,723 (0,677–0,769) | 0,224 |
| logistica penalizzata | 0,697 (0,669–0,725) | 0,714 (0,668–0,760) | 0,242 |
| Random Forest | 0,703 (0,676–0,731) | 0,743 (0,699–0,786) | 0,246 |
| XGBoost | 0,699 (0,671–0,726) | 0,725 (0,680–0,771) | 0,251 |

La colonna "training" si riferisce alle previsioni out-of-fold della cross-validation.

- **AUROC circa 0,70**: prendendo a caso un malato e un sano, 7 volte su 10 il modello dà al malato la probabilità più alta. Nella scala usata spesso in medicina (Hosmer, Lemeshow & Sturdivant 2013) è "accettabile", al limite inferiore.
- **PR-AUC**: 2,3–2,6 volte il valore del caso (0,098).
- **Nessun modello è migliore degli altri in modo dimostrabile**: la logistica con 5 variabili resta vicina ai modelli con 74.

### Quanto ci prende, su 100 persone (9,8 con marcatori)

| su 100 persone | soglia di lavoro (90% dei malati) | soglia 7% |
|---|---|---|
| mandate agli esami | 77–80 | 48–64 |
| malati trovati, su 9,8 | 8,8 | 7,0–8,1 |
| malati persi | 1,0 | 1,6–2,8 |
| sani esclusi correttamente, su 90,2 | 19–22 | 35–49 |
| fra i segnalati, quanti sono malati | 11% | 13–15% |
| fra i non segnalati, quanti sono sani | 95–96% | 94–96% |

### Sono buoni?
- **In assoluto, modesti**: AUROC 0,70 non basta per diagnosticare, e la specificità a sensibilità 0,90 è solo 0,21–0,25.
- **Rispetto al possibile, in linea**: i modelli pubblicati per un bersaglio simile, senza esami delle urine, stanno fra 0,58 e 0,76, e quello sviluppato nello stesso ospedale arriva a 0,70–0,72. Dodici strategie in più restano fra 0,69 e 0,71.
- **Per lo scopo, utili**: al 7% evitano 9–13 esami inutili ogni 100 persone, e chi non viene segnalato è sano nel 95–96% dei casi.
- **Solidi**: il test su 1.451 soggetti mai visti conferma tutto, e le probabilità sono calibrate in media.
- **Sulla gravità, deboli**: il modello ordina i livelli in media ma non li distingue sul singolo paziente.
