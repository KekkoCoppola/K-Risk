# Valorizzazione della tesi e posizionamento scientifico di K-Risk

Documento di sintesi metodologica, clinica e bibliografica ad uso della tesi. Definisce il posizionamento di K-Risk rispetto allo stato dell'arte e raccoglie tutti i risultati che ne misurano il valore.

**Regole di questo documento.** Ogni numero del progetto viene da una tabella in `analytics/`: previsioni out-of-fold sul training (4.350 soggetti) e, nella sezione 5bis, la conferma unica sul test set (1.451 soggetti, 22/09/2026). Alcuni numeri non compaiono in nessuna tabella: sono ricalcolati dalle stesse tabelle, dai record della Fase D o dal training con gli script dell'audit del 22–23/09/2026, e l'appendice lo segnala caso per caso. Ogni fonte esterna è stata verificata il 22–23/09/2026 su Crossref (DataCite per i dataset), PubMed, Europe PMC o sul sito dell'editore: **TC** = contenuto letto nel testo completo, **AB** = letto nell'abstract. I codici fra parentesi quadre, come [T1], rimandano all'appendice di tracciabilità. Le analisi della Fase D, del blocco "qualità e utilità clinica" e dell'audit del tetto dei dati sono post-hoc e vanno presentate come esplorative. [T1]

---

## 0. Il valore del progetto in sei punti

1. **Serve a qualcosa, e lo dimostra con la misura giusta.** Con la decision curve analysis, alla soglia del 7% il modello evita **9–13 esami inutili ogni 100 persone** rispetto a "testare tutti", a parità di casi trovati; al 10% ne evita 25–28 (sezione 5) [T2]. Sul test set ne evita 13–19 al 7% e 29–33 al 10% (sezione 5bis) [T3]. Nella revisione sistematica di Haris et al. 2024 nessuno dei 12 modelli di CKD incidente sviluppati o validati su cartelle cliniche di comunità riporta un'analisi di utilità clinica [T4].
2. **Il confronto con SCORED è onesto: il vantaggio è piccolo e sul test regge solo per la Random Forest.** Out-of-fold i tre modelli con gli esami del sangue di routine superano SCORED in stima puntuale in AUROC, PR-AUC ed esami necessari a parità di sensibilità; nel net benefit la logistica penalizzata e la Random Forest lo superano a 31 soglie su 31, XGBoost a 26. Nessuna differenza è statisticamente significativa [T5]. Sul test la Random Forest resta sopra SCORED in AUROC (0,743 contro 0,723) e in net benefit (28 soglie su 31), ma non in PR-AUC (0,249 contro 0,266); XGBoost ha un'AUROC simile a SCORED e la logistica penalizzata scende sotto (sezione 4) [T6]. Il tri-ensemble su 21 variabili, con selezione verificata dentro ogni fold, ottiene le stesse prestazioni della Random Forest con meno di un terzo delle variabili e, out-of-fold, è il migliore sul piano operativo: 7,8 esami in meno ogni 100 persone rispetto a SCORED per trovare l'85% dei casi (sezione 4) [T7].
3. **Il tetto di prestazione è spiegato con i dati, non ipotizzato, e un audit dedicato lo conferma.** L'eGFR < 60 si riconosce bene (AUROC 0,80–0,86), l'albuminuria no (0,67–0,69). La letteratura mostra lo stesso schema (sezione 6) [T8]. Tre diagnostiche, registrate in un commit prima di eseguirle, mostrano che la pipeline trova un segnale moderato quando c'è (0,795 e 0,831 con una variabile semi-sintetica che da sola vale 0,75 e 0,80), che il rumore dell'etichetta vicino alla soglia costa al massimo +0,009 di AUROC e che l'incertezza sull'unità dell'ACR la sposta al massimo di +0,016 (sezione 6bis) [T9].
4. **Tutti i controlli di leakage previsti sono superati.** Esami renali esclusi per costruzione e bloccati da un controllo automatico; nessuna delle 74 variabili, da sola, supera AUROC 0,75 (test automatico); controllo positivo con l'albumina urinaria a AUROC 0,933: la pipeline impara quando l'informazione c'è (sezione 7) [T10].
5. **Rigore metodologico da articolo, e conferma su dati mai visti.** Validazione incrociata annidata, regole di decisione scritte nella configurazione, 177 test automatici [T11]. Il test set (1.451 soggetti) è stato aperto una sola volta, dopo aver fissato protocollo e codice, e la sequenza è certificata da git e da `analytics/test/RUN.json`: **nessuna delle cinque affermazioni pre-registrate è contraddetta**, AUROC 0,71–0,74 (sezione 5bis) [T12].
6. **Quattro "risultati apparenti" smascherati.** La soglia 0,5 che gonfia il recall dal 2% al 60%, la PR-AUC dei diabetici che sembra doppia, la pendenza di calibrazione "media 1,02" che nasconde due errori opposti, e la selezione delle variabili fatta su tutti i dati che porta l'AUROC da 0,703 a 0,716 e rende "significativo" un vantaggio che non lo è (sezione 7) [T13].

---

## 1. Il problema clinico e il ruolo di K-Risk

### Il problema
- La CKD si definisce con due esami: **eGFR < 60 ml/min/1,73 m²** (dalla creatinina sierica) oppure **albuminuria, ACR ≥ 30 mg/g**, presenti per almeno 3 mesi (KDIGO 2024, Tabella 1; TC) [T14].
- **La maggior parte delle persone con CKD non sa di averla**: negli USA fino a 9 adulti su 10 (CDC 2023; TC), l'87% degli adulti dai 20 anni in su nell'aggiornamento CDC 2026 (TC) [T15]. In Cina, nella sorveglianza nazionale 2018–2019 su 176.874 adulti, prevalenza 8,2% (albuminuria 6,7%, eGFR ridotto 2,2%) e **consapevolezza del 10,0%** (Wang L et al. 2023; AB) [T16].
- KDIGO 2024 indica di testare le persone a rischio con **entrambi** gli esami, albumina urinaria ed eGFR (Practice Point 1.1.1.1; TC) [T17]. Nel diabete l'ADA raccomanda ACR ed eGFR almeno una volta l'anno, nel diabete tipo 1 da almeno 5 anni e in tutto il diabete tipo 2 (Standards of Care 2024 e 2026, raccomandazione 11.1a; TC); la CKD attribuita al diabete riguarda il 20–40% delle persone con diabete (ADA, sezione 11; TC) [T18].
- Le linee guida raccomandano lo screening annuale dell'albuminuria solo nelle persone con diabete (Bragg-Gresham et al. 2025; AB): fra chi non ha il diabete serve decidere **chi** mandare all'esame [T19].

### Il ruolo di K-Risk
- **A monte** del percorso diagnostico: un triage di primo livello che non sostituisce gli esami renali, ma decide a chi prescriverli.
- **Input**: dati anagrafici, antropometrici, pressori ed esami del sangue di routine (glicemia, lipidi, enzimi epatici, emocromo, peptide C…) [T20].
- **Vincolo**: tutti gli esami renali sono esclusi dall'input (`SCRE`, `UMAUCR`, `GFR`, `BUN` e derivati), con un controllo automatico che blocca il codice se una colonna vietata entra fra le feature [T21].
- **Uscita**: una probabilità di avere marcatori di CKD. In media è calibrata (O:E 0,95–1,01 out-of-fold e sul test); la pendenza varia fra modelli e fra campioni, e la ricalibrazione di Platt la riporta vicino a 1 (1,01–1,15 sul test; sezioni 5 e 5bis) [T22].

### I dati
Li J et al. 2026, *Sci Data* (TC): **reparto di Diabetologia ed Endocrinologia dello Shanghai Sixth People's Hospital, febbraio-aprile 2012**, 5.922 record e 190 variabili [T23]. Dopo aver escluso chi non ha creatinina, ACR, età o sesso restano 5.801 soggetti (training 4.350, test 1.451) [T24]. Nel training solo il 6% ha la variabile `DM` = 1 e la prevalenza dei marcatori di CKD è 9,8% [T25]. L'eGFR è ricalcolato con l'equazione CKD-EPI 2021 senza coefficiente etnico (Inker et al. 2021), perché il dataset non documenta l'equazione della sua colonna `GFR` [T26]. L'etichetta ACR ≥ 30 coincide, su tutte le righe del training, con la colonna `HighACR` fornita dagli autori [T27].

**Attenzione per la tesi**: gli autori descrivono dati ospedalieri di **pazienti con diabete**, **non uno screening di popolazione**, mentre nel training `DM` = 1 riguarda solo il 6% dei soggetti: la discrepanza va dichiarata [T28]. Il tipo di campione urinario e le unità di `UCRE`, `UmALB` e `UMAUCR` **non sono documentati** (nel dizionario dei dati degli autori l'unità è "None"), e nel training `UMAUCR` vale 176,8 × `UmALB`/`UCRE` su ogni riga, un fattore che non corrisponde a nessuna conversione standard (sezione 6bis) [T29]. Ci sono piccole incongruenze interne (dati trasversali o di follow-up, durata della raccolta) [T30]. K-Risk va quindi presentato come modello sviluppato su una coorte ospedaliera cinese, da validare in popolazioni di screening.

---

## 2. Stato dell'arte

I modelli si distinguono per **bersaglio**, e il bersaglio decide l'AUROC raggiungibile. [T31]

| modello | popolazione | bersaglio | esami usati | AUROC / C | fonte |
|---|---|---|---|---|---|
| **Bersaglio: eGFR ridotto, stato attuale** | | | | | |
| SCORED | NHANES 1999–2002, n = 8.530 | eGFR < 60 | 9 variabili, **incluse proteinuria e anemia** | 0,88 interna; **0,71 esterna** (ARIC) | Bang et al. 2007 (AB) |
| Kwon | KNHANES, Corea | eGFR < 60 | 7 variabili, incluse proteinuria e anemia | 0,83; 0,87 e 0,78 in validazione | Kwon et al. 2012 (AB) |
| Sabanayagam, soli fattori di rischio | Singapore; test a Singapore e Pechino | eGFR < 60 | nessuno (età, sesso, etnia, diabete, ipertensione) | 0,916 interna; 0,829 e 0,887 esterne | Sabanayagam et al. 2020 (AB) |
| **Bersaglio: composito con albuminuria, stato attuale** | | | | | |
| Thakkinstian | Thailandia, comunità, n = 3.459 | CKD stadi 1–5 (albuminuria o ematuria negli stadi 1–2) | nessuno (età, diabete, ipertensione, calcoli) | 0,77; 0,74 con bootstrap | Thakkinstian et al. 2011 (TC/AB) |
| MERWACS | NHANES, **≥ 50 anni**; esterna KNHANES | ACR ≥ 30 o eGFR sotto il 2,5° percentile per età e sesso | nessuno (12 parametri) | 0,68–0,70 interna; 0,71–0,73 esterna | Yoo et al. 2026 (TC) |
| **K-Risk** | coorte ospedaliera, Shanghai | **ACR ≥ 30 o eGFR < 60 (KDIGO)** | esami del sangue di routine, **nessun esame renale** | **0,697–0,703** out-of-fold (Fase A); 0,708–0,710 (migliori candidati Fase D); **0,714–0,743** sul test | questa tesi |
| **Bersaglio: sola albuminuria** | | | | | |
| Muntner | REGARDS, ≥ 45 anni; esterna NHANES | ACR ≥ 30 | nessuno (8 domande) | 0,709; 0,714 esterna | Muntner et al. 2011 (AB) |
| Tanner | MESA, 45–84 anni | ACR ≥ 30 | nessuno (stesso strumento) | 0,728–0,761 (0,761 nei cino-americani) | Tanner et al. 2015 (AB) |
| Bragg-Gresham | NHANES, adulti senza diabete, n = 44.322 | ACR ≥ 30 | modello 1: nessun esame renale (età, sesso, etnia, fumo, prediabete dall'HbA1c, ipertensione, malattia cardiovascolare); modello 3: aggiunge **eGFR < 60**, BMI, HDL, acido urico, farmaci e interazioni | 0,734 (modello 1) e 0,752 (modello 3) in validazione | Bragg-Gresham et al. 2025 (TC) |
| Khitan | diabete tipo 2 (Look AHEAD) | ACR ≥ 30 | **creatinina sierica**, HbA1c, lipidi | 0,61–0,67 | Khitan et al. 2021 (TC) |
| **Bersaglio: CKD futura (incidente)** | | | | | |
| QKidney | cure primarie UK, > 1,5 milioni | CKD moderata-grave a 5 anni | nessuno (dati clinici) | 0,875–0,876 | Hippisley-Cox & Coupland 2010 (AB) |
| Chien | Taiwan, coorte | eGFR < 60 a 4 anni | modello clinico | 0,768; 0,667 esterna | Chien et al. 2010 (AB) |
| Wen | Cina rurale, n = 3.266 | eGFR < 60 o ACR ≥ 30 in circa 6 anni (95,8% dei casi per albuminuria) | nessuno (Simple Score) | 0,717 | Wen J et al. 2020 (TC) |
| **A valle: progressione della CKD accertata** | | | | | |
| KFRE | 721.357 pazienti con CKD G3–G5, 31 coorti, oltre 30 paesi | dialisi o trapianto a 2 e 5 anni | **età, sesso, eGFR e ACR** (versione a 4 variabili) | 0,90 (2 anni); 0,88 (5 anni) | Tangri et al. 2016 (TC) |
| **Confronto locale** | | | | | |
| Wu M | stesso ospedale del dataset, diabete tipo 2 ricoverati | malattia renale diabetica (criteri NKF-KDOQI 2007, proteinuria sulle urine delle 24 h) | nessuno (sesso, BMI, PAS, durata del diabete) | 0,713 sviluppo; 0,720 validazione trasversale; 0,696 prospettica | Wu M et al. 2017 (TC) |

### Come leggere la tabella
- **Quando il bersaglio è l'eGFR ridotto, i modelli arrivano a 0,83–0,92 nelle stime interne e a 0,71–0,89 in validazione.** L'eGFR dipende soprattutto dall'età, che tutti i modelli conoscono. Due di questi (SCORED e Kwon) usano anche la proteinuria, cioè un esame delle urine. [T32]
- **Quando il bersaglio contiene l'albuminuria, i modelli stanno fra 0,61 e 0,77**, anche con decine di migliaia di soggetti; il valore più basso (Khitan, 0,61–0,67) viene da diabetici tipo 2, pur con la creatinina fra i predittori. Bragg-Gresham arriva a 0,734 già senza eGFR (con l'HbA1c per il prediabete) e a 0,752 aggiungendo eGFR, altri esami e interazioni. [T33]
- **K-Risk sta dove deve stare**: 0,70–0,71 su un bersaglio composito al 91% di albuminuria, senza nessun esame renale, in linea con MERWACS (0,68–0,73), Muntner (0,71) e il punteggio senza esami di laboratorio sviluppato nello stesso ospedale (Wu 2017, 0,70–0,72); sul test K-Risk arriva a 0,71–0,74. [T34]
- **Originalità, formulata con prudenza**: non abbiamo trovato modelli per il composito KDIGO attuale (eGFR < 60 o ACR ≥ 30) stimato senza esami renali in una popolazione cinese con e senza diabete (ricerca su PubMed ed Europe PMC del 23/09/2026) [T35]. Esistono però punteggi senza esami di laboratorio molto vicini: Wen et al. 2020 (composito KDIGO incidente, Cina rurale, 0,717) e soprattutto Wu et al. 2017, sviluppato nello **stesso ospedale** del dataset (malattia renale diabetica nei diabetici tipo 2 ricoverati, 0,70–0,72). Wu 2017 toglie parte dell'originalità, ma conferma che 0,70 è il livello realistico per questo centro e per predittori di questo tipo. Poiché il periodo di validazione di Wu (gennaio 2011–aprile 2015) comprende la raccolta del dataset (febbraio-aprile 2012), non si può escludere che alcuni pazienti siano in comune [T36].
- Le revisioni sistematiche confermano il quadro: molti punteggi, AUROC di sviluppo spesso > 0,70, poche validazioni esterne (Echouffo-Tcheugui & Kengne 2012; González-Rocha et al. 2023, AUROC 0,63–0,91 nelle popolazioni sane); rischio di bias alto e **nessuna analisi di utilità clinica** nei modelli di CKD incidente su cartelle cliniche di comunità (Haris et al. 2024); machine learning raramente validato fuori dal contesto di sviluppo (Sanmarchi et al. 2023). [T37]

---

## 3. K-Risk e KFRE: due strumenti per due momenti diversi

| | **K-Risk (questa tesi)** | **KFRE (Tangri et al. 2016)** |
|---|---|---|
| popolazione | coorte ospedaliera, soggetti in maggioranza senza diabete noto (n = 5.801) | pazienti con CKD già diagnosticata, G3–G5 (n = 721.357) |
| input | esami del sangue di routine, **zero esami renali** | età, sesso, **eGFR e ACR** |
| bersaglio | marcatori di CKD presenti ora (eGFR < 60 o ACR ≥ 30) | insufficienza renale terminale a 2 o 5 anni |
| AUROC / C | 0,70–0,71 out-of-fold; 0,71–0,74 sul test | 0,90 (2 anni), 0,88 (5 anni) |
| ruolo | **a monte**: chi deve fare i primi esami renali | **a valle**: prognosi e pianificazione della dialisi |

[T38] Il confronto non è una gara: il KFRE parte dagli stessi due esami che K-Risk serve a decidere se prescrivere. Se K-Risk li usasse come input, ricopierebbe la definizione del bersaglio e perderebbe ogni utilità.

---

## 4. K-Risk e SCORED

### Il confronto
SCORED è il riferimento storico per lo screening della CKD in cure primarie (Bang et al. 2007) ed è il modello di rischio per la CKD più validato esternamente secondo Echouffo-Tcheugui & Kengne 2012 (TC) [T39]. Nel dataset sono disponibili 5 dei suoi 9 predittori (età, sesso, emoglobina al posto dell'anemia, pressione sistolica al posto dell'ipertensione, diabete); mancano proteinuria, storia cardiovascolare, scompenso e vasculopatia periferica [T40]. La logistica SCORED della tesi è quindi **ristimata sui nostri dati**: questo la avvantaggia, perché i coefficienti sono adattati alla popolazione, ma le mancano le variabili che non abbiamo.

### Risultato out-of-fold: superiori in stima puntuale, di poco

| misura | SCORED ristimato | logistica penalizzata | Random Forest | XGBoost |
|---|---|---|---|---|
| AUROC (IC 95%) | 0,675 (0,646–0,704) | 0,697 (0,669–0,725) | **0,703** (0,676–0,731) | 0,699 (0,671–0,726) |
| differenza di AUROC contro SCORED (IC 95%, 5 fold) | — | +0,021 (−0,003; +0,046) | +0,029 (−0,008; +0,066) | +0,025 (−0,004; +0,054) |
| PR-AUC | 0,224 | 0,242 | 0,246 | 0,251 |
| persone da esaminare per trovare l'85% dei casi | 73,8 ogni 100 | 69,2 | 68,5 | **66,2** |
| persone da esaminare per trovare il 90% dei casi | 80,2 ogni 100 | 78,9 | 78,1 | 76,9 |
| soglie 5–20% in cui il net benefit supera quello di SCORED | — | **31 su 31** | **31 su 31** | 26 su 31 |
| esami inutili evitati in più rispetto a SCORED, soglia 7% | — | +1,8 ogni 100 | +4,0 | +3,0 |

[T41] Fonti: `analytics/phase_a/evaluation/` (AUROC, PR-AUC, punti operativi, confronti appaiati) e `analytics/quality/evaluation/decision_curve.csv`. Nella decision curve XGBoost è la versione con profondità 1–12, come in tutte le analisi dalla Fase B in poi (AUROC 0,696): la colonna XGBoost unisce quindi due varianti dello stesso modello. Gli IC delle differenze vengono dal t corretto di Nadeau-Bengio, che tiene conto della sovrapposizione dei fold [T42].

### Risultato sul test: il vantaggio regge solo per la Random Forest

| misura (test set, 1.451 soggetti) | SCORED ristimato | logistica penalizzata | Random Forest | XGBoost |
|---|---|---|---|---|
| AUROC (IC 95%) | 0,723 (0,677–0,769) | 0,714 (0,668–0,760) | **0,743** (0,699–0,786) | 0,725 (0,680–0,771) |
| PR-AUC | **0,266** | 0,239 | 0,249 | 0,254 |
| AUROC fra i non diabetici | 0,705 | 0,690 | **0,726** | 0,705 |
| soglie 5–20% in cui il net benefit supera quello di SCORED | — | 10 su 31 | **28 su 31** | 18 su 31 |
| esami inutili evitati in più rispetto a SCORED, soglia 7% | — | −5,7 ogni 100 | +0,8 | +0,2 |

[T43] Fonti: `analytics/test/run_1/q1_discrimination.csv` (modelli della Fase A) e `analytics/test/run_1/decision_curve.csv` (come out-of-fold, XGBoost con profondità 1–12). Le tabelle del test non contengono un confronto appaiato fra modelli: le differenze di AUROC non hanno un IC, e gli intervalli dei singoli modelli si sovrappongono.

**Come scriverlo nella tesi**: out-of-fold i modelli con gli esami del sangue di routine sono **superiori a SCORED in stima puntuale**, ma **nessuna differenza è statisticamente significativa** (AUROC: p di Holm 0,44; PR-AUC: p = 1,00) [T44]. Sul test il vantaggio si conferma solo per la Random Forest, in AUROC e net benefit ma non in PR-AUC; la logistica penalizzata finisce sotto SCORED [T43]. In pratica, out-of-fold, per trovare l'85% dei casi servono **4,6–7,6 esami in meno ogni 100 persone**, cioè 460–760 esami ACR in meno ogni 10.000 persone valutate (22.500–37.200 dollari ai 49 dollari per test di Cusick et al. 2023; stima illustrativa) [T45].

**La lettura corretta del "di poco"**: il risultato è coerente con Christodoulou et al. 2019, che su 145 confronti a basso rischio di bias non trovano differenze fra machine learning e regressione logistica (differenza di logit(AUROC) 0,00; IC −0,18; 0,18). Nei confronti ad alto rischio di bias lo stesso lavoro trova un vantaggio apparente del machine learning: l'assenza di un grande vantaggio è quindi compatibile con una valutazione corretta, anche se da sola non la dimostra [T46].

**Il candidato con il miglior equilibrio, out-of-fold**: la **logistica penalizzata** è l'unico modello della Fase A superiore a SCORED su tutte e quattro le misure e anche ben calibrato (pendenza 0,94, IC 0,80–1,07) [T47]. **Sul test set però la sua pendenza è 0,79 (IC 0,62–0,98)**: la buona calibrazione non si conferma, e va ricalibrata come gli altri (dopo Platt 1,01); sul test perde anche il vantaggio su SCORED (tabella sopra) [T48]. Random Forest comprime le probabilità in entrambi i campioni (1,31 e 1,27). È un'osservazione descrittiva, non una nuova scelta del modello [T49].

### Il tri-ensemble su 21 variabili, verificato
Secondo una nota della configurazione, un'analisi preliminare non registrata, e non riproducibile dal repository, riportava per un "tri-ensemble sulle 21 variabili più importanti" AUROC 0,717 [T50]. Il 22/09/2026 il tri-ensemble è stato registrato come candidato della Fase D (`configs/config.yaml`, `tri_ensemble_top21`), con la stessa regola degli altri candidati: media delle probabilità di logistica penalizzata, Random Forest e XGBoost, addestrati sulle prime 21 variabili con gli iperparametri della Fase A di ogni fold. Le 21 variabili si scelgono **dentro ogni fold, sul solo training**, per rango medio fra i tre modelli (|coefficiente| per la logistica, SHAP medio per gli alberi; per la Random Forest su 500 righe del training). Come diagnostica si è calcolata anche la versione con le variabili scelte su tutto il training [T51]. La configurazione dichiara la registrazione precedente al calcolo, ma git non lo può certificare: registrazione, codice e risultati sono in commit dello stesso minuto [T52].

| versione | AUROC (IC 95%) | PR-AUC | contro RF (IC 95%) | contro SCORED (IC 95%, 5 fold) |
|---|---|---|---|---|
| **selezione dentro ogni fold (candidato)** | **0,703** (0,676–0,731) | 0,252 | −0,002 (−0,025; +0,020), p di Holm 1,00 | +0,027 (−0,004; +0,057), p = 0,07 |
| selezione su tutto il training (diagnostica) | 0,716 (0,689–0,742) | 0,264 | — | +0,039 (+0,006; +0,071), p = 0,03 |

[T53]

**Cosa si ricava:**
- **la selezione fatta fuori dalla validazione gonfia l'AUROC**: con le variabili scelte su tutto il training si arriva a 0,716, vicino allo 0,717 dell'analisi preliminare; con la selezione corretta si scende a 0,703. La distorsione vale +0,012 (Ambroise & McLachlan 2002). Lo 0,717 non va quindi citato come risultato [T54];
- **la distorsione fabbrica anche la significatività**: contro SCORED la versione sbagliata sembrerebbe superiore in modo significativo (p = 0,03), quella corretta no (p = 0,07). È il quarto "risultato apparente" della tesi (sezione 7) [T53];
- **il valore reale è la parsimonia**: con 21 variabili invece di 74 il modello discrimina come la Random Forest (0,703) ed è il migliore sul piano operativo contro SCORED. Per trovare il 90% dei casi fa esaminare 75,0 persone ogni 100 contro 80,2; per l'85%, 66,0 contro 73,8, cioè **7,8 esami in meno ogni 100**. Il net benefit supera quello di SCORED a 31 soglie su 31, e al 7% evita 5,6 esami inutili in più ogni 100. Un modello con meno esami in ingresso è più facile da usare in un ambulatorio [T55];
- **nucleo stabile di 9 variabili**, scelte in tutti e 5 i fold: età, pressione sistolica, glicemia a digiuno e a 2 ore, peptide C a digiuno, acido urico, fosfatasi alcalina (ALP), indice FIB-4 e colesterolo LDL. Altre 6 compaiono in 4 fold su 5 (colesterolo totale, trigliceridi, globuli bianchi, peptide C a 2 ore, emoglobina, GGT). Sono le variabili su cui il modello si appoggia davvero, qualunque sia il campione di addestramento: descrivono ciò che il modello ha imparato, non relazioni causali [T56].

I p del confronto con SCORED sono post-hoc e non corretti per confronti multipli.

### Le stime migliori della Fase D
| candidato | AUROC (IC 95%) | AUROC media sui fold | contro SCORED, AUROC complessiva | contro SCORED, appaiato sui fold (IC 95%) |
|---|---|---|---|---|
| TabPFN v2 (Hollmann et al. 2025) | 0,710 (0,682–0,737) | 0,717 | +0,034 | +0,038 (+0,014; +0,061) |
| ensemble, media di logistica penalizzata, RF e XGBoost (74 variabili) | 0,708 (0,680–0,735) | 0,710 | +0,032 | +0,030 (+0,001; +0,060) |
| tri-ensemble, 21 variabili scelte nel fold | 0,703 (0,676–0,731) | 0,706 | +0,028 | +0,027 (−0,004; +0,057) |

[T57] Nessuno supera la regola fissata per la Fase D contro il miglior modello della Fase A (differenza ≥ 0,01, IC sopra zero, Holm < 0,05): contro Random Forest TabPFN guadagna +0,009 (IC −0,014; +0,032) [T58]. Contro SCORED gli IC appaiati di TabPFN e dell'ensemble stanno sopra zero, ma sono confronti post-hoc: con la correzione di Holm sugli 11 candidati nessuno resta significativo (TabPFN: p = 0,011, p di Holm 0,12) [T59]. Sono le stime più alte del progetto, da citare come esplorative; che cosa vuol dire "nessun candidato migliora" è spiegato nella sezione 6bis.

---

## 5. Utilità clinica: il modello serve?

Blocco post-hoc, dichiarato tale nella configurazione; il protocollo è nel Notepad ("Qualità e utilità clinica"), ma git non può certificare che precedesse i calcoli, perché protocollo e risultati sono in commit dello stesso minuto [T60]. La misura è il net benefit (Vickers & Elkin 2006), raccomandato fra le misure essenziali insieme ad AUROC e curva di calibrazione (Van Calster et al. 2025; AB) [T61]. Il codice calcola il net benefit con due formule indipendenti e si ferma se divergono [T62]. Figure `analytics/quality/01`–`06`.

### Decision curve (popolazione del dataset, prevalenza 9,8%)
| soglia | esami inutili evitati ogni 100 persone rispetto a "testare tutti", out-of-fold | sul test (run 1) |
|---|---|---|
| 2–4,5% | da −4,4 a +3,1: vicino a "testare tutti" (la Random Forest batte entrambe le strategie fra il 2% e il 4%) | da −2,8 a +8,9 |
| 5% | da +0,1 a +2,3 | da +4,6 a +11,3 |
| **7%** | **da +9,2 a +13,1** | **da +12,8 a +19,2** |
| 10% | da +24,9 a +27,8 | da +29,5 a +33,1 |
| 20% | da +54,9 a +56,8 | da +54,3 a +58,0 |

[T63]
- dal 5–6% in su tutti e quattro i modelli battono sia "testare tutti" sia "non testare nessuno" [T64];
- le soglie 5% e 7% sono quelle usate come esempi da Bragg-Gresham et al. 2025 per la stessa decisione [T65];
- MERWACS (Yoo et al. 2026) riporta anch'esso un'analisi delle curve decisionali: la tesi si allinea al lavoro più recente sul tema [T66].

### Sui diabetici il modello non serve, e questo conferma le linee guida
Fra i diabetici (prevalenza 25,9%) il guadagno rispetto a "testare tutti" resta trascurabile fino al 10%: al massimo 3,4 esami inutili evitati ogni 100 (Random Forest alla soglia del 10%) [T67]. Alla soglia globale del progetto il modello segnala dal 97% al 100% dei diabetici (dal 97,7% al 100% sul test) [T68]. È coerente con l'ADA, che prescrive l'esame ogni anno a tutti i diabetici tipo 2 e ai tipo 1 da almeno 5 anni: **K-Risk serve a decidere fra i soggetti senza diabete**.

### Costi
"Testare tutti" costa **502 dollari per caso trovato** (10,2 esami da 49 dollari). Alla soglia del 7% il modello abbassa il costo per caso trovato a 336–383 dollari e fa spendere **1.800–2.600 dollari in meno ogni 100 persone**, al prezzo di 1,6–2,8 casi non trovati ogni 100: recuperarli testando tutti costerebbe 880–1.100 dollari ciascuno [T69]. Il costo di 49 dollari (intervallo 36–64) è quello del test ACR in Cusick et al. 2023 (TC) [T70]. Non è un'analisi di costo-efficacia. Il contesto economico però sostiene l'idea: lo screening della CKD è più costo-efficace nei gruppi con punteggi di rischio alti (Yeo et al. 2024, revisione di 21 valutazioni economiche) e in Cina è costo-efficace anche nella popolazione generale (Wen F et al. 2025). La stessa revisione di Yeo lo trova invece non costo-efficace nelle popolazioni senza diabete né ipertensione, e già Boulware et al. 2003 lo trovavano conveniente solo se rivolto ai gruppi a rischio più alto: è un argomento in più per un triage che scelga chi testare [T71].

### Calibrazione
- **In media è buona**: intercetta fra −0,01 e +0,01 e rapporto O:E fra 0,99 e 1,01 per tutti i modelli [T72].
- **out-of-fold le due logistiche sono calibrate** (pendenze 0,97 e 0,94; sul test la penalizzata scende a 0,79, sezione 5bis); **Random Forest schiaccia le probabilità** (pendenza 1,31, IC 1,13–1,48), **XGBoost le esaspera** (0,86, IC 0,75–0,97). Gli IC sono quelli del bootstrap semplice; con il bootstrap stratificato per livello KDIGO le conclusioni sulle pendenze non cambiano [T73].
- **Controllo nei sottogruppi** (TRIPOD+AI, item 23a: prestazioni con IC anche nei sottogruppi chiave): fra i non diabetici nessun problema; fra i diabetici Random Forest sottostima il rischio di circa un quinto (O:E 1,26, IC 1,01–1,51). Non cambia nessuna decisione, perché i diabetici vanno testati comunque, ma va dichiarato. Con il bootstrap stratificato l'O:E dei non diabetici esclude di poco 1 per la logistica penalizzata (0,97) e la Random Forest (0,96): una lieve sovrastima del rischio [T74].

### Confronto con Bragg-Gresham (riferimento esterno)
Per trovare il 73% dei casi di albuminuria fra i non diabetici servono 12–22 persone esaminate in più ogni 100 rispetto al loro modello; per l'85%, almeno 19–27 in più [T75]. Il loro modello però usa l'HbA1c, la pressione e, nel modello finale, **eGFR, HDL e acido urico** su 44.322 adulti: il divario è in parte dovuto alla rinuncia alla creatinina, in parte alla diversa popolazione (NHANES contro coorte ospedaliera cinese) e agli altri predittori [T76].

---

## 5bis. Conferma sul test set (22/09/2026)

Esecuzione unica su 1.451 soggetti mai visti, con protocollo, criteri e codice fissati prima (commit di autorizzazione `b7d5f61`, test richiuso subito dopo). `analytics/test/RUN.json` registra una sola esecuzione (22/09/2026, 18:20–18:37) con l'impronta sha256 del file letto. Tabelle in `analytics/test/run_1/`. [T77]

| affermazione fissata prima | esito sul test |
|---|---|
| (A) discriminazione come out-of-fold | **non contraddetta**: AUROC 0,714–0,743, dentro l'IC per 3 modelli su 4 (per SCORED il test è più alto) |
| (B) nessuna tecnica fa riconoscere più casi gravi | **non contraddetta**: differenze da −2 a +2 su 23, p di Holm 1,00 |
| (C) sui diabetici il modello segnala quasi tutti | **confermata**: dal 97,7% al 100% |
| (D) utilità clinica al 7% e al 10% | **confermata** per tutti e 4 i modelli: 13–19 esami inutili evitati ogni 100 persone al 7%, 29–33 al 10% |
| (E) alla soglia 0,5 il bilanciamento sembra migliorare il recall | **confermata**: 4,4% contro valori dal 40% al 65% |

[T78]
- l'AUROC più alta del test non è un miglioramento: gli intervalli (±0,045) comprendono la stima out-of-fold per 3 modelli su 4 [T79]
- la calibrazione in media si conferma (O:E 0,95–0,97); la pendenza della Random Forest (1,27) anche; quella della logistica penalizzata no (0,79) [T80]
- limite: stessa coorte ospedaliera e stessa finestra temporale, 23 casi gravi e 88 diabetici: è una conferma interna, non una validazione esterna [T81]

---

## 6. Il tetto di prestazione: spiegato con i dati

Le prove dirette sono cinque; la sezione 6bis le mette alla prova con l'audit del 22–23/09/2026.

1. **Il limite è l'albuminuria.** Con le stesse previsioni, i positivi per eGFR < 60 si distinguono dai negativi con AUROC 0,80–0,86; i positivi per sola albuminuria con 0,67–0,69. Il bersaglio è per il 91% albuminuria (362 + 25 casi su 425 nel training). Lo stesso schema compare in letteratura (sezione 2) [T82].
2. **Non è la tecnica.** Undici strategie diverse nella Fase D (ensemble, NaN gestiti dall'algoritmo, spazio degli iperparametri allargato, bersaglio scomposto, CatBoost, LightGBM, EBM, TabPFN, tri-ensemble…) restano fra 0,692 e 0,710 [T83]. Con una sola partizione, però, la regola della Fase D riconosce solo differenze di almeno 0,02–0,04 (sezione 6bis).
3. **Non sono i dati che mancano.** Dal 60% al 100% del training l'AUROC sale di 0,006–0,008: più soggetti aiuterebbero poco [T84].
4. **La pipeline funziona.** Aggiungendo l'albumina urinaria l'AUROC sale a 0,933: quando l'informazione c'è, il modello la trova [T85]. Questo controllo dimostra poco, perché l'albumina è il numeratore dell'ACR; l'audit aggiunge un controllo a intensità nota (sezione 6bis).
5. **L'etichetta è rumorosa.** Un ACR su un solo campione è instabile: solo il 43,5% degli ACR ≥ 30 su urina casuale viene confermato sulla prima urina del mattino (Saydah et al. 2013); la variabilità dell'ACR nella stessa persona è del 48,8% nel diabete tipo 2 (Rasaratnam et al. 2024); KDIGO chiede infatti di ripetere un ACR anomalo (Practice Point 1.1.1.2) e di confermare un ACR ≥ 30 su urina casuale con la prima urina del mattino (Practice Point 1.3.1.2) [T86]. Nel dataset il tipo di campione non è documentato, e nelle giornate con creatinina urinaria mediana bassa la quota di ACR ≥ 30 sale (rho di Spearman −0,70) senza che salga l'albumina: è un'anomalia da dichiarare [T87].

In sintesi: con esami del sangue di routine e un'albuminuria misurata su un solo campione, 0,70–0,71 è il livello raggiunto da tutte le strategie provate, in linea con i modelli pubblicati più vicini per bersaglio e predittori (MERWACS 0,68–0,73; Wen 0,717; Wu 0,70–0,72) [T88].

---

## 6bis. Audit del tetto dei dati (22–23/09/2026)

L'audit ha controllato se il tetto dipende dalla pipeline, dall'etichetta o dall'unità dell'ACR. Le tre diagnostiche sono state registrate nella configurazione con il commit `9fd7939` (23/09/2026, 09:28), già su GitHub, ed eseguite dopo con il codice del progetto (`src/models/phase_d.py`: `semi_synthetic_control`, `label_noise_auroc`, `acr_label_sensitivity`); le tabelle sono in `analytics/phase_d/` [T89]. Sono analisi post-hoc, sul solo training: il test set non è stato riaperto. La configurazione dichiara che i numeri della sensibilità all'unità dell'ACR erano già stati esplorati durante l'audit, prima della registrazione [T90].

### 1. La pipeline trova un segnale moderato quando c'è
Il controllo positivo con l'albumina urinaria (0,933) dimostra poco, perché l'albumina è il numeratore dell'ACR. Per questo ai 74 predittori si è aggiunta una variabile semi-sintetica, il logaritmo dell'ACR più un rumore gaussiano, con il rumore tarato perché la variabile da sola abbia AUROC 0,75 oppure 0,80. Con gli stessi fold e gli iperparametri della Fase A, XGBoost arriva a **0,795 (IC 0,771–0,818)** e a **0,831 (0,809–0,853)**, contro 0,696 senza la variabile. Il criterio registrato (almeno l'AUROC della variabile meno 0,01) è superato in entrambi i casi: la pipeline trova il segnale e lo somma a quello degli altri predittori. [T91]

### 2. Il rumore dell'etichetta vicino alla soglia pesa poco
Sulle previsioni out-of-fold di cinque modelli (logistica penalizzata, Random Forest, XGBoost, ensemble, TabPFN):
- l'AUROC cresce con l'ACR dei positivi: 0,645–0,682 per ACR 30–45, 0,675–0,684 per 45–100, 0,698–0,718 per 100–300, 0,758–0,790 per ACR ≥ 300. I casi appena sopra la soglia sono i più difficili da riconoscere [T92];
- escludendo la zona grigia di 17,7–35,4 (2,0–4,0 mg/mmol, la zona di incertezza diagnostica dopo un solo ACR secondo Rasaratnam et al. 2024) l'AUROC passa da 0,696–0,710 a 0,701–0,717, cioè da +0,005 a +0,009 secondo il modello. Togliere i casi vicini alla soglia alza l'AUROC anche con un'etichetta perfetta (effetto spettro): questi valori sono un **limite superiore** del costo del rumore vicino alla soglia, non una stima [T93];
- sulle sole coppie positivo-negativo della stessa giornata l'AUROC è 0,696–0,711, non più bassa di quella complessiva (differenze da −0,002 a +0,009): l'anomalia delle giornate (sezione 6, punto 5) non gonfia la discriminazione [T94].

Il rumore lontano dalla soglia, per esempio un campione raccolto male, non si può misurare senza un secondo ACR.

### 3. L'unità dell'ACR: un'ipotesi da dichiarare, che non cambia le conclusioni
- Nel training `UMAUCR` = 176,8 × `UmALB`/`UCRE` su tutte le 4.350 righe (da 173,7 a 178,9). Con l'albumina in mg/L, i fattori standard per un ACR in mg/g sarebbero 100 con la creatinina urinaria in mg/dL, 8.840 in µmol/L e 8,84 in mmol/L: 176,8 non corrisponde a nessuno. Il dataset non documenta le unità (nel dizionario dei dati degli autori l'unità è "None") [T95].
- La mediana di `UCRE` nel training, 185, è compatibile solo con i mg/dL: nella popolazione USA la mediana della creatinina urinaria su campione singolo è 118,6 mg/dL (Barr et al. 2005; TC); in µmol/L, 185 corrisponderebbe a circa 2 mg/dL, un'urina più diluita del limite di validità di 30 mg/dL riportato dallo stesso lavoro [T96].
- **Ipotesi, non dimostrata**: se `UCRE` è in mg/dL, il fattore corretto è 100 e la soglia `UMAUCR` ≥ 30 corrisponde a un ACR vero di circa 17 mg/g. I positivi per albuminuria scenderebbero da 387 a 248 e quelli del bersaglio da 425 a 293 [T97].
- **Sensibilità** (previsioni out-of-fold esistenti, modelli addestrati sulla soglia attuale: misura quanto l'ordinamento si trasferisce, non un modello riaddestrato): con la soglia equivalente al fattore 100 (`UMAUCR` ≥ 53,04) l'AUROC passa da 0,696–0,710 a 0,705–0,720, cioè da +0,003 a +0,016 secondo il modello. Anche escludendo le zone grigie resta fra 0,722 e 0,755, un valore che contiene l'effetto spettro [T98].
- L'etichetta resta quella degli autori: `UMAUCR` ≥ 30 coincide con la loro colonna `HighACR` su tutte le righe del training [T27].

### 4. Che cosa significa "nessun candidato migliora"
La regola della Fase D chiede una differenza di almeno 0,01, un IC sopra zero e un p di Holm sotto 0,05 contro la Random Forest. Con 5 fold e una sola partizione, la deviazione standard delle differenze per fold fra candidato e Random Forest ha mediana 0,0105; con il t corretto di Nadeau-Bengio usato dal progetto, la differenza media minima che può risultare significativa è circa **0,02** per un solo confronto (0,0195) e circa **0,04** con Holm su 11 candidati (0,0403, primo passo) [T99]. La CV ripetuta prevista dalla configurazione (`repeat_seeds`) non è stata eseguita: nessun codice la usava, e la configurazione lo annota insieme alla decisione di non farla (circa 16 ore di calcolo con esito atteso invariato) [T100]. Quindi "nessun candidato migliora" va letto così: **nessun candidato migliora di 0,04 o più; differenze di 0,01–0,02 non si possono né escludere né dimostrare**. Il risultato ha però anche un lato forte: l'estremo superiore più alto degli IC contro la Random Forest è +0,039 (+0,032 per TabPFN), quindi un guadagno superiore a 0,04 è escluso per tutti gli undici candidati [T101].

### 5. Che cosa dice la letteratura sul tetto
- Con uno standard di riferimento imperfetto, anche un marcatore perfetto ha un'AUROC apparente bassa: con sensibilità 80%, specificità 90% e prevalenza 10% dello standard, l'AUROC apparente è 0,72 (Waikar et al. 2012; TC) [T102].
- L'ACR su un solo campione è uno standard imperfetto: sulla prima urina del mattino si conferma il 43,5% degli ACR ≥ 30 su urina casuale, il 56,3% nei diabetici (Saydah et al. 2013; AB); la variabilità nella stessa persona è del 48,8% (Rasaratnam et al. 2024; AB); KDIGO chiede di ripetere e confermare (Practice Point 1.1.1.2 e 1.3.1.2; TC); il CDC avverte che le stime basate su un solo ACR possono sovrastimare la CKD (CDC 2026; TC) [T103].
- I modelli pubblicati per l'albuminuria senza esami delle urine stanno fra 0,58 e 0,76: 0,58 per la nefropatia diabetica con soli predittori clinici (Blech, nella revisione di Echouffo-Tcheugui & Kengne 2012; TC), 0,61–0,67 nel diabete tipo 2 con la creatinina (Khitan), 0,709–0,714 (Muntner), 0,717 (Wen), 0,70–0,72 per la malattia renale diabetica nello stesso ospedale (Wu), 0,728–0,761 (Tanner), 0,734 senza eGFR e 0,752 con eGFR su 44.322 adulti (Bragg-Gresham). Anche con l'ACR di base, un modello per l'albuminuria di nuova insorgenza nel diabete arriva a 0,65 (ADVANCE, nella stessa revisione; TC) [T104].

**Verdetto dell'audit**: il tetto è un limite dei dati. La pipeline sa estrarre un segnale moderato; il rumore vicino alla soglia e l'incertezza sull'unità spostano l'AUROC al massimo di 0,009 e 0,016; il limite resta l'informazione sull'albuminuria contenuta negli esami di routine (0,67–0,69, contro 0,80–0,86 dell'eGFR) [T105].

---

## 7. Perché il risultato è credibile

### Controlli di leakage superati
- esami renali esclusi dall'input e bloccati da un'asserzione nel codice, verificata da un test automatico;
- nessuna delle 74 variabili, da sola, supera AUROC 0,75 (test automatico sul training);
- controllo positivo con l'albumina urinaria (0,933) e controllo semi-sintetico (0,795 e 0,831): se ci fosse leakage nascosto, le prestazioni senza esami renali non sarebbero ferme a 0,70;
- test set (1.451 soggetti) letto una sola volta, dopo aver congelato protocollo e codice (run 1 del 22/09/2026, `analytics/test/RUN.json`). [T106]

### Come si riconoscono le prestazioni gonfiate in letteratura
- **Esami renali fra le feature**: il dataset UCI "Chronic Kidney Disease" (Rubini et al. 2015, 400 pazienti, 24 variabili) contiene creatinina, urea, albumina e peso specifico urinari; sopra questi dati gli articoli arrivano al 98,9–99,8% di accuratezza (Qin et al. 2020; Chittora et al. 2021). È la regola diagnostica ricopiata, non una previsione. [T107]
- **Solo validazione interna**: il deep learning sulle retinografie riconosce l'eGFR < 60 con AUROC 0,911 in validazione interna, ma 0,733 e 0,835 sui dati esterni; un modello con soli età, sesso, etnia, diabete e ipertensione fa altrettanto bene (Sabanayagam et al. 2020). [T108]
- **Rischio di bias**: nei confronti ad alto rischio di bias il machine learning sembra migliore della logistica, in quelli a basso rischio no (Christodoulou et al. 2019). [T46]

### Metodo
Validazione incrociata annidata 5 × 5 stratificata; iperparametri scelti solo sui fold interni; regole di decisione scritte nella configurazione (per la Fase D: differenza ≥ 0,01, IC sopra zero, Holm < 0,05); confronti fra modelli con il t corretto di Nadeau & Bengio 2003, che tiene conto della sovrapposizione dei fold, e con la correzione di Holm; intervalli di confidenza per le stime principali; 177 test automatici. La precedenza delle regole rispetto ai risultati è certificabile da git solo per il test set (sezione 5bis) e per le diagnostiche dell'audit (sezione 6bis). [T109]

**Robustezza alla causalità inversa.** Togliendo le 7 variabili che la malattia renale può alterare (emoglobina, globuli rossi, ematocrito, acido urico, albumina, proteine totali, albumina glicata) l'AUROC cambia fra −0,012 e +0,001, mai in modo significativo (p ≥ 0,23) [T110].

### Quattro risultati apparenti, misurati
1. **Fase B**: alla soglia 0,5 le tecniche di bilanciamento portano il recall dal 2% al 60% (media dei quattro modelli, senza correzione e con l'undersampling), ma a parità di sensibilità non riconoscono più casi gravi: il saldo va da −3 a +3 casi su 71, sempre con p di Holm 1,00. È coerente con van den Goorbergh et al. 2022, per cui le correzioni dello sbilanciamento non aumentano l'AUROC (AB) [T111].
2. **Fase C**: fra i diabetici la PR-AUC sembra doppia (0,44 contro 0,19), ma rapportata alla prevalenza del gruppo è più bassa [T112].
3. **Blocco qualità**: la pendenza di calibrazione "media 1,02" nasconde Random Forest a 1,31 e XGBoost a 0,86 [T113].
4. **Fase D, tri-ensemble**: scegliendo le 21 variabili su tutti i dati l'AUROC sale da 0,703 a 0,716 e il vantaggio su SCORED diventa "significativo" (p = 0,03 contro 0,07). È la distorsione da selezione di Ambroise & McLachlan 2002, misurata sui nostri dati [T53].

Quattro esempi indipendenti di come un numero, letto senza controllare come è stato ottenuto, può ingannare: è un contributo metodologico della tesi, non un effetto collaterale.

---

## 8. Come presentare i punti deboli alla commissione

Per ogni punto: la formulazione onesta, l'argomento di valore e le prove. Nessuna frase va detta senza il numero o la fonte che la sostiene.

**a. Il tetto a 0,70** [T114]
- *Formulazione onesta*: "K-Risk discrimina in modo moderato: AUROC 0,70 out-of-fold e 0,71–0,74 sul test."
- *Argomento di valore*: il tetto è misurato, non ipotizzato. La componente eGFR si riconosce (0,80–0,86), l'albuminuria no (0,67–0,69); undici strategie restano fra 0,692 e 0,710; la pipeline trova un segnale moderato quando c'è (0,795 e 0,831); rumore e unità dell'ACR spostano l'AUROC al massimo di 0,009 e 0,016. È lo stesso livello dei modelli pubblicati senza esami delle urine (0,58–0,76) e del punteggio sviluppato nello stesso ospedale (Wu 2017: 0,70–0,72).
- *Prove*: sezioni 6 e 6bis.

**b. Il test set usato una volta** [T115]
- *Formulazione onesta*: "Il test è una conferma interna nella stessa coorte e nella stessa finestra temporale (142 positivi, 23 casi gravi, 88 diabetici), non una validazione esterna. Non verrà riaperto: ogni scelta fatta dopo averlo visto lo trasformerebbe in dati di sviluppo."
- *Argomento di valore*: il test è stato letto una sola volta, dopo aver congelato protocollo, codice e cinque affermazioni da verificare, e nessuna è contraddetta. È la separazione chiesta da TRIPOD+AI: i dati di valutazione devono essere distinti da quelli usati per addestrare, ottimizzare o selezionare il modello (Box 1; TC).
- *Prove*: `analytics/test/RUN.json` (una esecuzione, dalle 18:20:04 alle 18:37:20, con impronta sha256); commit `62bf5d5` e `a7a0a6b` (protocollo e codice, 18:17:47), `b7d5f61` (autorizzazione, 18:18:18), `4075bfa` e `5ffadf6` (risultati e chiusura, 18:38); `authorized: false` nella configurazione; test automatici che impediscono di leggere il test set fuori dall'esecuzione e di rieseguirla senza un motivo dichiarato; `analytics/test/run_1/claims.csv`.

**c. Il confronto con SCORED** [T116]
- *Formulazione onesta*: "Out-of-fold i modelli con gli esami del sangue superano SCORED di poco e senza significatività statistica. Sul test il vantaggio regge per la Random Forest (AUROC 0,743 contro 0,723; net benefit sopra SCORED a 28 soglie su 31), non in PR-AUC; la logistica penalizzata finisce sotto. SCORED è ristimato sui nostri dati con 5 dei suoi 9 predittori."
- *Argomento di valore*: SCORED è il modello più validato esternamente (Echouffo-Tcheugui & Kengne 2012), e ristimarlo sui nostri dati lo avvantaggia. Nei confronti a basso rischio di bias la letteratura non trova vantaggi del machine learning sulla logistica (Christodoulou et al. 2019): un vantaggio piccolo e dichiarato è ciò che ci si aspetta da una valutazione corretta. Il valore di K-Risk sta anche nell'uso: utilità clinica misurata con la decision curve, parsimonia (21 variabili) e controlli di leakage.
- *Prove*: sezione 4.

**d. La natura post-hoc della Fase D, del blocco qualità e dell'audit** [T117]
- *Formulazione onesta*: "La Fase D e il blocco qualità sono stati decisi dopo aver visto i risultati delle fasi precedenti, e la configurazione lo dichiara: sono esplorativi. La precedenza del protocollo rispetto ai calcoli è certificabile da git solo per il test set e per le tre diagnostiche dell'audit; per la sensibilità all'unità dell'ACR i numeri erano già stati esplorati prima della registrazione."
- *Argomento di valore*: le analisi post-hoc servono a spiegare il tetto, non a scegliere il modello. La regola della Fase D è scritta nella configurazione, nessun candidato l'ha superata e il test non è stato riaperto: nessuna analisi post-hoc cambia una conclusione della tesi.
- *Prove*: `configs/config.yaml` (Fase D e blocco qualità dichiarati post-hoc; regola di decisione); `analytics/phase_d/comparison.csv` (nessun candidato migliora); commit `9fd7939`; `analytics/test/RUN.json` (una sola esecuzione).

**e. L'unità dell'ACR** [T118]
- *Formulazione onesta*: "Il dataset non documenta le unità di albumina e creatinina urinarie, e l'ACR fornito vale 176,8 × UmALB/UCRE, un fattore che non corrisponde a conversioni standard. Se la creatinina urinaria fosse in mg/dL, come suggerisce la sua mediana, la soglia usata corrisponderebbe a circa 17 mg/g invece di 30. È un'ipotesi, non dimostrata."
- *Argomento di valore*: il problema è stato trovato e quantificato dal progetto. L'etichetta è quella degli autori (`HighACR`), e con la soglia alternativa l'AUROC cambia da +0,003 a +0,016: la conclusione sul tetto non dipende dall'unità.
- *Prove*: sezione 6bis, punto 3.

**f. La popolazione del dataset** [T119]
- *Formulazione onesta*: "I dati vengono da un reparto ospedaliero di diabetologia e gli autori li descrivono come dati di pazienti con diabete, ma nel training `DM` = 1 riguarda solo il 6%. Non è un campione di screening."
- *Argomento di valore*: la discrepanza è dichiarata e il modello è presentato per quello che è: sviluppato su una coorte ospedaliera cinese, da validare in popolazioni di screening. L'analisi per sottogruppi mostra dove serve (fra i non diabetici) e dove no (fra i diabetici, che le linee guida mandano comunque all'esame).
- *Prove*: sezioni 1 e 5.

---

## 9. Limiti da dichiarare

- dataset **ospedaliero** (Shanghai, 2012), descritto dagli autori come dati di pazienti con diabete ma con `DM` = 1 nel 6% del training, non di screening: la validità esterna va dimostrata;
- ACR da un solo campione, di tipo non documentato; unità di `UCRE`, `UmALB` e `UMAUCR` non documentate e fattore 176,8 non spiegato (sezione 6bis);
- nessuna validazione esterna: il test set interno (eseguito una volta, nessuna conclusione contraddetta) è una conferma nella stessa coorte;
- SCORED ristimato con 5 dei 9 predittori originali;
- Fase D, blocco qualità e audit sono post-hoc ed esplorativi; la regola della Fase D usa una sola partizione (differenza minima rilevabile 0,02–0,04) e la CV ripetuta prevista non è stata eseguita;
- le soglie operative a sensibilità fissata sono stimate e valutate sulle stesse previsioni out-of-fold, quindi lievemente ottimistiche; sul test si usano soglie congelate;
- il tri-ensemble usa gli iperparametri scelti in Fase A su 74 variabili: con un'ottimizzazione dedicata alle 21 variabili potrebbe cambiare di poco; il numero 21 viene dall'analisi preliminare e non è stato ottimizzato.

[T120]

---

## 10. Bibliografia verificata

Verifica del 22–23/09/2026 (TC = testo completo, AB = abstract). DOI controllati su Crossref; per i dataset su DataCite.

**Dati, linee guida ed epidemiologia**
- Li J, Zheng H, Zhou Y, Jiang F. A bimodal dataset for diabetes research. *Sci Data* 2026;13:652. doi:10.1038/s41597-026-06923-y. PMID 41813689. Dati: Zenodo doi:10.5281/zenodo.18270337 (DataCite); dizionario dei dati nel repository degli autori (github.com/Zhoushanshen/Diabetes-dataset) (TC)
- KDIGO 2024 CKD Work Group. KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD. *Kidney Int* 2024;105(4S):S117-S314. doi:10.1016/j.kint.2023.10.018. PMID 38490803 (TC: Tabella 1; Practice Point 1.1.1.1, 1.1.1.2 e 1.3.1.2)
- ADA Professional Practice Committee. 11. Chronic Kidney Disease and Risk Management: Standards of Care in Diabetes—2024. *Diabetes Care* 2024;47(Suppl 1):S219-S230. doi:10.2337/dc24-S011. PMID 38078574 (TC). Edizione 2026: *Diabetes Care* 2026;49(Suppl 1):S246-S260. doi:10.2337/dc26-S011 (TC)
- CDC. *Chronic Kidney Disease in the United States, 2023*. Atlanta: US DHHS, CDC; 2023 (CS 338890; TC). Aggiornamento: *Chronic Kidney Disease in the United States*, CS 363495-A, 30/03/2026 (TC)
- Wang L, Xu X, Zhang M, et al. Prevalence of Chronic Kidney Disease in China. *JAMA Intern Med* 2023;183(4):298-310. doi:10.1001/jamainternmed.2022.6817. PMID 36804760 (AB)
- Inker LA, Eneanya ND, Coresh J, et al. New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race. *N Engl J Med* 2021;385(19):1737-49. doi:10.1056/NEJMoa2102953. PMID 34554658 (AB)
- Barr DB, Wilder LC, Caudill SP, et al. Urinary creatinine concentrations in the U.S. population: implications for urinary biologic monitoring measurements. *Environ Health Perspect* 2005;113(2):192-200. doi:10.1289/ehp.7337 (TC)

**Modelli di confronto**
- Bang H, Vupputuri S, Shoham DA, et al. SCreening for Occult REnal Disease (SCORED). *Arch Intern Med* 2007;167(4):374-81. doi:10.1001/archinte.167.4.374. PMID 17325299 (AB)
- Kwon KS, Bang H, Bomback AS, et al. A simple prediction score for kidney disease in the Korean population. *Nephrology* 2012;17(3):278-84. doi:10.1111/j.1440-1797.2011.01552.x. PMID 22171932 (AB)
- Hippisley-Cox J, Coupland C. Predicting the risk of chronic kidney disease in men and women in England and Wales: prospective derivation and external validation of the QKidney Scores. *BMC Fam Pract* 2010;11:49. doi:10.1186/1471-2296-11-49. PMID 20565929 (AB)
- Yoo D, Nguyen VK, Maggiore U, Jolliet O. MERWACS: development and external validation of a non-invasive machine learning tool for identifying subjects to be screened for CKD. *PLOS Digit Health* 2026;5(7):e0001486. doi:10.1371/journal.pdig.0001486. PMID 42424235 (TC)
- Tangri N, Grams ME, Levey AS, et al. Multinational Assessment of Accuracy of Equations for Predicting Risk of Kidney Failure: A Meta-analysis. *JAMA* 2016;315(2):164-74. doi:10.1001/jama.2015.18202. PMID 26757465 (TC)
- Bragg-Gresham JL, Annadanam S, Gillespie B, Li Y, Powe NR, Saran R. Using Risk Assessment to Improve Screening for Albuminuria among US Adults without Diabetes. *J Gen Intern Med* 2025;40(13):3159-69. doi:10.1007/s11606-024-09185-9. PMID 39557751 (TC)
- Muntner P, Woodward M, Carson AP, et al. *Am J Kidney Dis* 2011;58(2):196-205. doi:10.1053/j.ajkd.2011.01.027. PMID 21620547 (AB)
- Tanner RM, Woodward M, Peralta C, et al. *Ethn Dis* 2015;25(4):427-34. doi:10.18865/ed.25.4.427. PMID 26676090 (AB)
- Khitan Z, Nath T, Santhanam P. *J Clin Hypertens* 2021;23(12):2137-45. doi:10.1111/jch.14397. PMID 34847294 (TC)
- Thakkinstian A, Ingsathit A, Chaiprasert A, et al. *BMC Nephrol* 2011;12:45. doi:10.1186/1471-2369-12-45. PMID 21943205 (TC/AB)
- Chien KL, Lin HJ, Lee BC, et al. *Am J Med* 2010;123(9):836-846.e2. doi:10.1016/j.amjmed.2010.05.010. PMID 20800153 (AB)
- Wen J, Hao J, Zhang Y, et al. *BMC Nephrol* 2020;21:120. doi:10.1186/s12882-020-01787-9. PMID 32252667 (TC)
- Wu M, Lu J, Zhang L, et al. A non-laboratory-based risk score for predicting diabetic kidney disease in Chinese patients with type 2 diabetes. *Oncotarget* 2017;8(60):102550-8. doi:10.18632/oncotarget.21684. PMID 29254270 (TC)
- Sabanayagam C, Xu D, Ting DSW, et al. *Lancet Digit Health* 2020;2(6):e295-e302. doi:10.1016/S2589-7500(20)30063-7. PMID 33328123 (AB)

**Revisioni e metodo**
- Echouffo-Tcheugui JB, Kengne AP. Risk models to predict chronic kidney disease and its progression: a systematic review. *PLoS Med* 2012;9(11):e1001344. doi:10.1371/journal.pmed.1001344. PMID 23185136 (TC)
- González-Rocha A, Colli VA, Denova-Gutiérrez E. *Prev Chronic Dis* 2023;20:E30. doi:10.5888/pcd20.220380. PMID 37079751 (AB)
- Haris M, et al. *Clin Kidney J* 2024;17(5):sfae098. doi:10.1093/ckj/sfae098. PMID 38737345 (AB)
- Sanmarchi F, et al. *J Nephrol* 2023;36(4):1101-17. doi:10.1007/s40620-023-01573-4. PMID 36786976 (AB)
- Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol* 2019;110:12-22. doi:10.1016/j.jclinepi.2019.02.004. PMID 30763612 (AB)
- Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 2002;99(10):6562-6. doi:10.1073/pnas.102102699. PMID 11983868 (AB)
- Nadeau C, Bengio Y. Inference for the generalization error. *Mach Learn* 2003;52(3):239-81. doi:10.1023/A:1024068626366 (AB)
- van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *J Am Med Inform Assoc* 2022;29(9):1525-34. doi:10.1093/jamia/ocac093 (AB)
- Waikar SS, Betensky RA, Emerson SC, Bonventre JV. Imperfect gold standards for kidney injury biomarker evaluation. *J Am Soc Nephrol* 2012;23(1):13-21. doi:10.1681/ASN.2010111124 (TC)
- Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model. *Nature* 2025;637(8045):319-26. doi:10.1038/s41586-024-08328-6. PMID 39780007 (AB)
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006;26(6):565-74. doi:10.1177/0272989X06295361 (metadati)
- Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. *Lancet Digit Health* 2025;7(12):100916. doi:10.1016/j.landig.2025.100916 (AB)
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378 (TC: Box 1; Tabella 2, item 23a)

**Qualità dell'etichetta ACR**
- Saydah SH, Pavkov ME, Zhang C, et al. *Clin Chem* 2013;59(4):675-83. doi:10.1373/clinchem.2012.195644. PMID 23315482 (AB)
- Rasaratnam N, Salim A, Blackberry I, et al. *Am J Kidney Dis* 2024;84(1):8-17.e1. doi:10.1053/j.ajkd.2023.12.018. PMID 38551531 (AB)

**Economia dello screening**
- Cusick MM, Tisdale RL, Chertow GM, Owens DK, Goldhaber-Fiebert JD. Population-Wide Screening for Chronic Kidney Disease: A Cost-Effectiveness Analysis. *Ann Intern Med* 2023;176(6):788-97. doi:10.7326/M22-3228. PMID 37216661 (TC: Tabella 1)
- Yeo SC, Wang H, Ang YG, Lim CK, Ooi XY. *Clin Kidney J* 2024;17(1):sfad137. doi:10.1093/ckj/sfad137. PMID 38186904 (AB)
- Wen F, Wang J, Yang C, et al. *Lancet Reg Health West Pac* 2025;56:101493. doi:10.1016/j.lanwpc.2025.101493. PMID 40226778 (AB)
- Boulware LE, Jaar BG, Tarver-Carr ME, Brancati FL, Powe NR. *JAMA* 2003;290(23):3101-14. doi:10.1001/jama.290.23.3101. PMID 14679273 (AB)

**Studi con prestazioni gonfiate (esempi critici)**
- Rubini L, Soundarapandian P, Eswaran P. *Chronic Kidney Disease* [dataset]. UCI Machine Learning Repository; 2015. doi:10.24432/C5G020 (DataCite elenca i primi due autori; la pagina UCI tutti e tre)
- Qin J, Chen L, Liu Y, Liu C, Feng C, Chen B. *IEEE Access* 2020;8:20991-21002. doi:10.1109/ACCESS.2019.2963053 (AB)
- Chittora P, et al. *IEEE Access* 2021;9:17312-34. doi:10.1109/ACCESS.2021.3053763 (AB)

---

## Appendice — tracciabilità

Ogni codice [T] del testo rimanda a una riga di questa tabella. Per i numeri del progetto la verifica è di due tipi:
- **05**: numero che si ottiene con un filtro su una sola tabella; file, filtro pandas, colonna, aggregazione e arrotondamento sono in `05_numeri.csv` (righe T*n*.*xx*);
- **S18**: numero che richiede un calcolo (conteggi, differenze fra file diversi, valori × 100, segni invertiti, t corretto, dati del training), ricalcolato e confrontato con il testo da `18_numeri_da_script.py`.

Entrambi i file sono in `docs/audit/`, e dalla radice del repository si ricontrolla tutto con due comandi: `python docs/audit/check_numbers.py . docs/audit/05_numeri.csv valorizzazione_tesi.md` per le righe 05 e `python docs/audit/18_numeri_da_script.py` per le righe S18. Leggono solo `analytics/` e `data/processed/train.csv`.

Abbreviazioni: `pa/` = `analytics/phase_a/evaluation/`; `pb/` = `analytics/phase_b/evaluation/`; `pc/` = `analytics/phase_c/evaluation/`; `pd/` = `analytics/phase_d/`; `q/` = `analytics/quality/evaluation/`; `t/` = `analytics/test/run_1/`; `cfg` = `configs/config.yaml`. Per le fonti esterne: DOI e posizione; TC = testo completo, AB = abstract.

| id | affermazione | fonte primaria (posizione) e verifica |
|---|---|---|
| T1 | regole; 4.350 e 1.451 soggetti; analisi post-hoc | `q/calibration.csv` (tutti, `n`) e `t/q1_discrimination.csv` (`n`): 05; `cfg`:248-251 (Fase D post-hoc), 365-368 (blocco qualità post-hoc), 289 (diagnostiche dell'audit) |
| T2 | 9–13 esami evitati al 7%, 25–28 al 10% | `q/decision_curve.csv`, tutti, modelli ≠ dummy, soglie 0,07 e 0,1, `avoided_per_100` min–max: 05 |
| T3 | sul test 13–19 al 7%, 29–33 al 10% | `t/decision_curve.csv`, stessi filtri: 05 |
| T4 | Haris 2024: nessuna analisi di utilità clinica nei 12 modelli | doi:10.1093/ckj/sfae098, abstract (Results) |
| T5 | out-of-fold sopra SCORED in stima puntuale; net benefit 31, 31, 26 soglie su 31; nessuna significatività | `pa/q1_discrimination.csv` (`auc`, `pr_auc`): 05; `pa/q1_operating_points.csv` (`alert_rate` a 0,85 e 0,90): S18; `q/decision_curve.csv` (conteggio delle soglie 0,05–0,20 con `net_benefit` sopra `lr_scored`): S18; `pa/comparison_models.csv` (`p_holm`): 05 |
| T6 | sul test RF 0,743 contro 0,723, PR-AUC 0,249 contro 0,266, net benefit 28 su 31 | `t/q1_discrimination.csv`, `technique == 'fase_a'`, tutti: 05; conteggio da `t/decision_curve.csv`: S18 |
| T7 | tri-ensemble: 21 variabili su 74, come la RF, 7,8 esami in meno all'85% | `pd/discrimination.csv` (0,7031 contro 0,7032); `cfg`:348 (21 variabili); `tests/test_preprocess.py`:49 (74 variabili); 7,8: S18 da `pd/tri_ensemble_top21/outer0–4.json`, `analytics/phase_a/oof_predictions.csv` e bersaglio del training |
| T8 | eGFR < 60: 0,80–0,86; sola albuminuria: 0,67–0,69 | `pd/label_quality.csv`, `analysis` = `componente: eGFR < 60` e `componente: solo albuminuria`, `auc` min–max: 05 |
| T9 | tre diagnostiche registrate prima di eseguirle; 0,795 e 0,831; +0,009; +0,016 | commit `9fd7939` (`cfg`:289-313); `pd/semi_synthetic_control.csv`, `pd/label_noise_auroc.csv`, `pd/acr_label_sensitivity.csv`: 05 |
| T10 | esclusione degli esami renali; nessuna variabile sopra 0,75; controllo positivo 0,933 | `src/data/preprocess.py`:22-26 e 49-54; `tests/test_preprocess.py`:40-43 e 67-72; `pd/discrimination.csv` (`positive_control`): 05 |
| T11 | CV annidata; regole nella configurazione; 177 test | `src/data/folds.py`:17-30; `cfg`:107 e 119 (5 fold esterni e interni), 258-263 (regola della Fase D); `pytest --collect-only` del 23/09/2026: 177 test, di cui 7 dell'audit ancora non in un commit; 169 eseguiti e superati (esclusi gli 8 di `tests/test_split.py`, che leggono i dati grezzi) |
| T12 | test aperto una volta, sequenza certificata; cinque affermazioni; AUROC 0,71–0,74 | `analytics/test/RUN.json` (una esecuzione, 2026-09-22T18:20:04–18:37:20, sha256); commit `62bf5d5` e `a7a0a6b` (18:17:47), `b7d5f61` (18:18:18), `4075bfa` e `5ffadf6` (18:38); `t/claims.csv`; AUROC: 05 |
| T13 | recall dal 2% al 60%; PR-AUC 0,44 contro 0,19; pendenza media 1,02; 0,703 → 0,716 | `pb/naive.csv` (modelli ≠ dummy, soglia 0,5, media di `recall` per tecnica: none 0,020, undersampling 0,605); `pc/q1_discrimination.csv`; `q/calibration.csv` (media di `slope`, tutti); `pd/discrimination.csv`: 05 |
| T14 | definizione KDIGO, per almeno 3 mesi | doi:10.1016/j.kint.2023.10.018, p. S137 (Tabella 1: "present for a minimum of 3 months"), TC |
| T15 | fino a 9 su 10; 87% degli adulti ≥ 20 anni | CDC, *Chronic Kidney Disease in the United States, 2023* (CS 338890), p. 1; CDC 2026 (CS 363495-A), p. 1; TC |
| T16 | Cina: 176.874 adulti, 8,2%, 6,7%, 2,2%, consapevolezza 10,0% | doi:10.1001/jamainternmed.2022.6817, abstract (Results) |
| T17 | KDIGO PP 1.1.1.1 | doi:10.1016/j.kint.2023.10.018, p. S149, TC |
| T18 | ADA, raccomandazione 11.1a; CKD attribuita al diabete nel 20–40% | doi:10.2337/dc24-S011 (PMC10725805) e doi:10.2337/dc26-S011 (PMC12690176): raccomandazione 11.1a e paragrafo introduttivo della sezione, TC |
| T19 | screening annuale raccomandato solo nel diabete | doi:10.1007/s11606-024-09185-9, abstract (Background) |
| T20 | input di K-Risk | `cfg`:84-91 (feature numeriche e categoriche) |
| T21 | esami renali esclusi, controllo automatico | `cfg`:57-60 (leakage, renal_exams); `src/data/preprocess.py`:22-26 e 49-54 |
| T22 | O:E 0,95–1,01; Platt 1,01–1,15 sul test | `q/calibration.csv` (tutti, `oe_ratio` max); `t/calibration.csv` (none, tutti; grezza: `oe_ratio` min; ricalibrata: `slope` min–max): 05 |
| T23 | reparto, febbraio-aprile 2012, 5.922 record e 190 variabili | doi:10.1038/s41597-026-06923-y: abstract (5.922 × 190) e Methods "Participants" (PMC13111635), TC |
| T24 | esclusioni; 5.801 = 4.350 + 1.451 | `src/data/load.py`:8 e 17 (righe senza SCRE, UMAUCR, Age o Gender escluse); `tests/test_split.py`:14 (5801); n del training e del test: 05 |
| T25 | DM = 1 nel 6%; prevalenza 9,8% | `q/calibration.csv`: `n` diabetici 263 su 4.350 (S18); `prevalence` tutti 0,0977 (05) |
| T26 | eGFR ricalcolato con CKD-EPI 2021 | `src/data/kidney.py`:41-52; doi:10.1056/NEJMoa2102953; equazione di `GFR` non indicata nel dataset: doi:10.1038/s41597-026-06923-y, Tab. 5 ("Calculated GFR"), TC |
| T27 | etichetta uguale a `HighACR` degli autori | `data/processed/train.csv`: `HighACR` == (`UMAUCR` ≥ 30) su 4.350 righe su 4.350: S18 |
| T28 | "pazienti con diabete" contro DM = 1 nel 6% | doi:10.1038/s41597-026-06923-y, Methods ("cross-sectional dataset of diabetes patients"), TC; quota: come T25 |
| T29 | unità non documentate ("None"); fattore 176,8 | dizionario dei dati degli autori (github.com/Zhoushanshen/Diabetes-dataset, Supplementary Table 1); fattore: S18 su `data/processed/train.csv` |
| T30 | incongruenze interne | doi:10.1038/s41597-026-06923-y ("recorded in two months" e febbraio-aprile 2012); record Zenodo doi:10.5281/zenodo.18270337 ("follow-up records") |
| T31 | tabella dello stato dell'arte | riga K-Risk: `pa/q1_discrimination.csv` (main; logistica penalizzata, RF, XGBoost: 0,697–0,703), `pd/discrimination.csv` (0,708 e 0,710), `t/q1_discrimination.csv` (0,714–0,743): 05. Altre righe: SCORED doi:10.1001/archinte.167.4.374 (abstract); Kwon doi:10.1111/j.1440-1797.2011.01552.x (abstract); Sabanayagam doi:10.1016/S2589-7500(20)30063-7 (abstract, Findings); Thakkinstian doi:10.1186/1471-2369-12-45 (abstract; definizione degli stadi nei Methods, TC); MERWACS doi:10.1371/journal.pdig.0001486 (abstract; Methods: esclusi i minori di 50 anni, TC); Muntner doi:10.1053/j.ajkd.2011.01.027 (abstract); Tanner doi:10.18865/ed.25.4.427 (abstract); Bragg-Gresham doi:10.1007/s11606-024-09185-9 (Results, modelli 1–3, TC); Khitan doi:10.1111/jch.14397 (abstract); QKidney doi:10.1186/1471-2296-11-49 (abstract); Chien doi:10.1016/j.amjmed.2010.05.010 (abstract); Wen doi:10.1186/s12882-020-01787-9 (abstract; 95,8% nei Results, TC); KFRE doi:10.1001/jama.2015.18202 (abstract; versione a 4 variabili nei Methods, TC); Wu doi:10.18632/oncotarget.21684 (Methods, Results, Fig. 2B, TC) |
| T32 | eGFR: 0,83–0,92 interne, 0,71–0,89 in validazione | righe SCORED, Kwon e Sabanayagam della tabella (T31) |
| T33 | albuminuria: 0,61–0,77; Bragg-Gresham 0,734 senza eGFR, 0,752 con | righe Thakkinstian, MERWACS, Muntner, Tanner, Bragg-Gresham e Khitan (T31) |
| T34 | K-Risk 0,70–0,71 out-of-fold, 0,71–0,74 sul test; 91% albuminuria | 05 (T34.01–04); 91%: T82; confronti: T31 |
| T35 | nessun modello trovato per il composito KDIGO attuale in questa popolazione | ricerca bibliografica del 23/09/2026: PubMed (E-utilities, 35 risultati vagliati) ed Europe PMC (primi 100 risultati per titolo); stringa conservata nel registro delle fonti dell'audit |
| T36 | Wen 0,717; Wu nello stesso ospedale, 0,70–0,72; periodi | doi:10.1186/s12882-020-01787-9 (abstract); doi:10.18632/oncotarget.21684 (Methods: ricoverati dello Shanghai Clinical Center for Diabetes, Sixth People's Hospital, sviluppo 2005–2010, validazione gennaio 2011–aprile 2015), TC |
| T37 | revisioni sistematiche | doi:10.1371/journal.pmed.1001344 (abstract); doi:10.5888/pcd20.220380 (abstract: 0,63–0,91); doi:10.1093/ckj/sfae098 (abstract: rischio di bias alto nel 64% dei modelli); doi:10.1007/s40620-023-01573-4 (abstract) |
| T38 | K-Risk e KFRE | KFRE: doi:10.1001/jama.2015.18202 (abstract; Methods, TC); K-Risk: T24, T34 |
| T39 | SCORED il più validato esternamente | doi:10.1371/journal.pmed.1001344, Discussion ("the most externally validated model"), TC (PMC3502517) |
| T40 | 5 dei 9 predittori | `cfg`:135 (`scored_predictors`); 9 predittori: doi:10.1001/archinte.167.4.374, abstract |
| T41 | tabella out-of-fold | AUROC, IC, PR-AUC e differenza della logistica penalizzata: `pa/q1_discrimination.csv`, `pa/comparison_models.csv` (05); differenze di RF e XGBoost come differenza di `auc_folds_mean` (05); loro IC = colonne `high` e `low` con segno invertito delle righe `lr_scored`–`random_forest` e `lr_scored`–`xgboost` (S18); persone da esaminare = `alert_rate` × 100 di `pa/q1_operating_points.csv` (S18); soglie vinte (S18); esami evitati in più = differenze di `avoided_per_100` in `q/decision_curve.csv` a 0,07 (05) |
| T42 | XGBoost profondità 1–12 (0,696); t corretto | `cfg`:256; `pd/discrimination.csv` (riferimento `xgboost`): 05; `src/models/evaluation.py`:167-177 |
| T43 | tabella del test | `t/q1_discrimination.csv` (`technique == 'fase_a'`, gruppi tutti e non diabetici): 05; differenze di `avoided_per_100` a 0,07 in `t/decision_curve.csv`: 05; soglie vinte: S18; decision curve sul braccio "none" (XGBoost 1–12): `src/models/final_test.py`:63-65 e 284-289 |
| T44 | p di Holm 0,44 e 1,00 | `pa/comparison_models.csv` (main; `auc` logistica penalizzata–SCORED; `pr_auc` SCORED–RF): 05 |
| T45 | 4,6–7,6 esami in meno; 460–760; 22.500–37.200 dollari | differenze di `alert_rate` × 100 a sensibilità 0,85 in `pa/q1_operating_points.csv`: S18; 49 dollari: `cfg`:379 e Cusick 2023, Tabella 1 (doi:10.7326/M22-3228, TC) |
| T46 | Christodoulou 2019 | doi:10.1016/j.jclinepi.2019.02.004, abstract (Results) |
| T47 | logistica penalizzata: pendenza 0,94 (0,80–1,07) | `q/calibration.csv` (tutti; `slope`, `slope_low_simple`, `slope_high_simple`): 05 |
| T48 | sul test 0,79 (0,62–0,98); dopo Platt 1,01 | `t/calibration.csv` (none, tutti; grezza e ricalibrata): 05 |
| T49 | RF 1,31 e 1,27 | `q/calibration.csv` e `t/calibration.csv`: 05 |
| T50 | analisi preliminare: AUROC 0,717, non riproducibile | `cfg`:340-341 (nota della configurazione; nessuna altra fonte) |
| T51 | costruzione del tri-ensemble | `cfg`:340-348 e 288; `src/models/phase_d.py`:143-173 (classifiche e rango medio), 191-212 (`run_tri_ensemble`) |
| T52 | precedenza non certificabile da git | `git log`: `68f349a`, `2224c59` e `2030e01` del 22/09/2026 alle 12:48 |
| T53 | tabella del tri-ensemble; p = 0,07 e 0,03 | `pd/discrimination.csv`, `pd/comparison.csv`: 05; confronti con SCORED: S18 (`pd/folds_auc.csv` e `pa/folds.csv` con il t di `src/models/evaluation.py`:167-177) |
| T54 | distorsione +0,012; 0,717 da non citare | `pd/discrimination.csv` (leaky − top21): 05; `cfg`:340-341; doi:10.1073/pnas.102102699 (abstract) |
| T55 | parsimonia: 75,0 contro 80,2; 66,0 contro 73,8; 31 su 31; +5,6 | S18 (record del tri-ensemble, previsioni out-of-fold di SCORED, bersaglio del training); RF 0,703: 05 |
| T56 | 9 variabili in 5 fold, 6 in 4 | `pd/tri_ensemble_top21/outer0–4.json` (campo `features`): S18 |
| T57 | stime migliori della Fase D | `pd/discrimination.csv` (`auc`, IC, `auc_folds_mean`): 05; differenze con SCORED complessive (`pd/discrimination.csv` − `pa/q1_discrimination.csv`) e appaiate (`pd/folds_auc.csv` − `pa/folds.csv`): S18 |
| T58 | nessuno supera la regola; TabPFN +0,009 (−0,014; +0,032) | `pd/comparison.csv` (`improves` falso per tutti; riga `tabpfn`): 05; regola: `cfg`:258-263 |
| T59 | TabPFN p = 0,011; Holm 0,12 | S18 (t corretto e Holm sugli 11 candidati contro SCORED) |
| T60 | blocco qualità post-hoc; precedenza non certificabile | `cfg`:365-368; `git log`: `66266ee`, `b126c88`, `c9c2829`, `6a6179f` del 22/09/2026 alle 12:48 |
| T61 | net benefit fra le misure essenziali | doi:10.1177/0272989X06295361 (metadati); doi:10.1016/j.landig.2025.100916, abstract |
| T62 | due formule del net benefit | `src/models/clinical_utility.py`:76-96 e 147-180 |
| T63 | tabella della decision curve | `q/decision_curve.csv` e `t/decision_curve.csv` (tutti, modelli ≠ dummy, soglie ≤ 0,045, 0,05, 0,07, 0,1, 0,2; `avoided_per_100` min–max): 05; RF fra 2% e 4%: `q/decision_summary.csv` (`ranges_beating_both` "0,020-0,040; 0,050-0,200") |
| T64 | dal 5–6% tutti battono entrambe le strategie | `q/decision_curve.csv` (`beats_both`): S18 (logistica penalizzata e RF dal 5%, XGBoost dal 4,5%, SCORED dal 6%) |
| T65 | soglie 5% e 7% di Bragg-Gresham | doi:10.1007/s11606-024-09185-9, Results, TC; `cfg`:376-377 |
| T66 | decision curve in MERWACS | doi:10.1371/journal.pdig.0001486, Methods e Results ("Decision curve analysis"), TC |
| T67 | diabetici: 25,9%; al massimo 3,4 | `q/calibration.csv` (`prevalence`); `q/decision_curve.csv` (diabetici, modelli ≠ dummy, soglie ≤ 0,1, `avoided_per_100` max): 05 |
| T68 | dal 97% al 100% dei diabetici; dal 97,7% al 100% sul test | `pc/threshold_descriptive.csv` (none, diabetici, `alert_rate_global`); `t/q1_operating_points.csv` (none, diabetici, `alert_rate`): 05 |
| T69 | costi | `q/costs.csv` (tutti, soglia 0,07: `cost_per_case_all`, `tests_per_case_all`, `cost_per_case`, `saving_per_100`, `missed_per_100`, `incremental_cost_per_case`): 05 |
| T70 | 49 dollari (36–64) | `cfg`:379; doi:10.7326/M22-3228, Tabella 1, TC |
| T71 | contesto economico | doi:10.1093/ckj/sfad137 (abstract: 21 studi; punteggi di rischio; non costo-efficace senza diabete né ipertensione); doi:10.1016/j.lanwpc.2025.101493 (abstract); doi:10.1001/jama.290.23.3101 (abstract, Conclusions) |
| T72 | intercetta −0,01…+0,01; O:E 0,99–1,01 | `q/calibration.csv` (tutti, `intercept` e `oe_ratio` min–max): 05 |
| T73 | pendenze e IC; bootstrap semplice e stratificato | `q/calibration.csv` (`slope`, `slope_low_simple`, `slope_high_simple`; `slope_low` e `slope_high` per lo stratificato): 05; `src/models/clinical_utility.py`:300-336 |
| T74 | sottogruppi: O:E 1,26 (1,01–1,51); non diabetici 0,97 e 0,96 | `q/calibration.csv` (diabetici e non diabetici; colonne `_simple` e stratificate): 05; TRIPOD+AI doi:10.1136/bmj-2023-078378, Tabella 2, item 23a, TC |
| T75 | 12–22 e almeno 19–27 persone in più | `q/screening_curves.csv` e `q/external_comparison.csv`: S18; 73,2% con il 37,7% esaminati e "just under half" al 5%: doi:10.1007/s11606-024-09185-9, Results, TC; `cfg`:383-387 |
| T76 | predittori del modello di Bragg-Gresham | doi:10.1007/s11606-024-09185-9, Results (modelli 1–3), TC |
| T77 | esecuzione unica, RUN.json, sha256 | `analytics/test/RUN.json`; commit `b7d5f61`, `4075bfa`, `5ffadf6`; `cfg`:398 (`authorized: false`) |
| T78 | tabella delle affermazioni (A)–(E) | `t/claims.csv` (verdetti); `t/q1_discrimination.csv`, `t/primary_comparison.csv` (`severe_n`, `p_holm`), `t/q1_operating_points.csv`, `t/decision_curve.csv`, `t/naive.csv` (modelli ≠ dummy, soglia 0,5, media di `recall`): 05; differenze da −2 a +2 (`gained` − `lost`): S18 |
| T79 | intervalli circa ±0,045 | `t/q1_discrimination.csv`: semiampiezze 0,043–0,046 (S18) |
| T80 | O:E 0,95–0,97; pendenze 1,27 e 0,79 | `t/calibration.csv` (none, grezza, tutti): 05 |
| T81 | 23 casi gravi e 88 diabetici | `analytics/test/RUN.json` (counts: alto 17, molto alto 6, diabetici 88); `t/primary_comparison.csv` e `t/q1_discrimination.csv`: 05 |
| T82 | 0,80–0,86 e 0,67–0,69; 91% (362 + 25 su 425) | `pd/label_quality.csv` (componenti; `positives` 362 e 425): 05; 25 e 91% (387 = somma dei positivi delle 4 fasce di ACR in `pd/label_noise_auroc.csv`): S18 |
| T83 | undici strategie fra 0,692 e 0,710 | `pd/discrimination.csv` (`role == 'candidato'`, 11 righe, `auc` min–max): 05 |
| T84 | dal 60% al 100% del training +0,006–0,008 | `pd/learning_curve.csv` (media di `auc` per modello e quota): S18 |
| T85 | controllo positivo 0,933 | `pd/discrimination.csv`: 05; `cfg`:277 |
| T86 | Saydah, Rasaratnam, KDIGO PP 1.1.1.2 e 1.3.1.2 | doi:10.1373/clinchem.2012.195644 (abstract); doi:10.1053/j.ajkd.2023.12.018 (abstract); doi:10.1016/j.kint.2023.10.018, p. S149 (PP 1.1.1.2) e p. S154 (PP 1.3.1.2), TC |
| T87 | rho di Spearman −0,70 senza aumento dell'albumina | `data/processed/train.csv`: 39 giornate, rho −0,702 (quota di ACR ≥ 30) e +0,16 (albumina urinaria mediana): S18; stesso calcolo in `src/analytics/phase_d_report.py`:97-113 |
| T88 | sintesi del tetto | T31, T34, T83 |
| T89 | registrazione e codice dell'audit | `git show 9fd7939` (23/09/2026 09:28:44, su `origin/experimental/tetto-dati`; `cfg`:289-313); `src/models/phase_d.py`:342-493 (codice delle diagnostiche, al 23/09 non ancora in un commit); tabelle in `pd/` (file del 23/09 alle 09:45, non ancora in un commit) |
| T90 | numeri della sensibilità all'unità già esplorati | `cfg`:305-309 |
| T91 | controllo semi-sintetico | `pd/semi_synthetic_control.csv` (`z_auc`, `oof_auc` con IC, `reference_auc`, `passes`): 05; criterio: `cfg`:289-296; codice: `src/models/phase_d.py`:347-398 |
| T92 | AUROC per fascia di ACR | `pd/label_noise_auroc.csv` (`analysis` = `fascia ACR 30-45` … `fascia ACR 300-inf`, `auc` min–max sui 5 modelli): 05 |
| T93 | casi netti: da 0,696–0,710 a 0,701–0,717, +0,005/+0,009 | `pd/label_noise_auroc.csv` (`tutti` e `casi netti (zona grigia 17.7-35.4 esclusa)`; differenze per modello): 05; `cfg`:297-304; zona grigia: doi:10.1053/j.ajkd.2023.12.018, abstract (2,0–4,0 mg/mmol; × 8,84 = 17,7–35,4 mg/g) |
| T94 | coppie della stessa giornata | `pd/label_noise_auroc.csv` (`coppie della stessa giornata`; differenze per modello): 05; `src/models/phase_d.py`:413-425 |
| T95 | fattore 176,8 (173,7–178,9) su 4.350 righe; fattori standard | `data/processed/train.csv`: S18; dizionario dei dati: T29; fattori: 1 mg/dL = 88,4 µmol/L |
| T96 | mediana di UCRE 185; Barr 118,6 mg/dL; circa 2 mg/dL | `data/processed/train.csv`: S18; doi:10.1289/ehp.7337, Tab. 2 (mediana) e Introduction (limiti di validità 30–300 mg/dL), TC |
| T97 | circa 17 mg/g; 387 → 248; 425 → 293 | S18 (`train.csv`: 30 × 100/176,8; conteggi); `pd/acr_label_sensitivity.csv` (`positives`): 05 |
| T98 | sensibilità all'unità: 0,705–0,720, +0,003/+0,016; 0,722–0,755 | `pd/acr_label_sensitivity.csv` (scala osservata, soglie 30 e 53,04, `grey_zone` vuota; scala alternativa con zone grigie): 05; codice: `src/models/phase_d.py`:463-493 |
| T99 | differenza minima rilevabile circa 0,02 (0,0195) e 0,04 (0,0403) | S18: `pd/folds_auc.csv`, DS mediana delle differenze contro RF 0,0105; t 2,776 e 5,747; fattore √(1/5 + 870/3480) = 0,671; formula di `src/models/evaluation.py`:167-177 |
| T100 | CV ripetuta non eseguita (16 ore) | `cfg`:264-268; `repeat_seeds` compare solo nella configurazione, non nel codice |
| T101 | estremo superiore massimo +0,039; TabPFN +0,032 | `pd/comparison.csv` (`high` max; riga `tabpfn`): 05 |
| T102 | Waikar: AUROC apparente 0,72 | doi:10.1681/ASN.2010111124, Tab. 2, TC (PMC3695762) |
| T103 | l'ACR su un solo campione è uno standard imperfetto | doi:10.1373/clinchem.2012.195644 (abstract: 43,5%, 56,3%); doi:10.1053/j.ajkd.2023.12.018 (abstract: 48,8%); KDIGO p. S149 e S154, TC; CDC 2026, p. 4 ("How estimates were calculated"), TC |
| T104 | letteratura sull'albuminuria: 0,58–0,76; ADVANCE 0,65 | doi:10.1371/journal.pmed.1001344, Tabella 1 (Blech 0,58; ADVANCE, modello finale, 0,65), TC; Khitan, Muntner, Wen, Wu, Tanner e Bragg-Gresham come in T31 |
| T105 | verdetto dell'audit | T8, T91–T98 |
| T106 | controlli di leakage | T10, T91, T12 |
| T107 | UCI: 400 pazienti, 24 variabili; 98,9–99,8% | pagina UCI del dataset 336 (doi:10.24432/C5G020): "Dataset Information" e "Variables"; doi:10.1109/ACCESS.2019.2963053 (abstract: 99,75% e 99,83%); doi:10.1109/ACCESS.2021.3053763 (abstract: 98,86%) |
| T108 | retinografie: 0,911 interna, 0,733 e 0,835 esterne | doi:10.1016/S2589-7500(20)30063-7, abstract (Findings) |
| T109 | metodo | `src/data/folds.py`:17-30; `cfg`:107, 119, 258-263; `src/models/evaluation.py`:167-187; doi:10.1023/A:1024068626366 (abstract); 177 test: T11; precedenza: T12, T89 |
| T110 | causalità inversa: da −0,012 a +0,001, p ≥ 0,23 | `pa/comparison_sets.csv` (`metric == 'auc'`, modelli ≠ dummy): 05; le 7 variabili: `cfg`:99 |
| T111 | Fase B: 2% e 60%; saldo da −3 a +3 su 71; p di Holm 1,00 | `pb/naive.csv` e `pb/primary_comparison.csv` (confermative, `p_holm`): 05; `q/operating_points.csv` (`severe_n` 71): 05; saldo (`gained` − `lost`): S18; doi:10.1093/jamia/ocac093 (abstract) |
| T112 | PR-AUC 0,44 contro 0,19 | `pc/q1_discrimination.csv` (none, RF, diabetici e non diabetici; `pr_auc` e `prevalence`): 05 |
| T113 | pendenza media 1,02; RF 1,31; XGBoost 0,86 | `q/calibration.csv` (tutti): 05 |
| T114 | presentazione: tetto | T8, T83, T91–T98, T104, T12 |
| T115 | presentazione: test usato una volta | T12, T77, T81; `tests/test_final_test.py`:90-117; `cfg`:398; TRIPOD+AI doi:10.1136/bmj-2023-078378, Box 1, TC |
| T116 | presentazione: SCORED | T39, T40, T43, T46 |
| T117 | presentazione: analisi post-hoc | `cfg`:248-251, 289, 305-309, 365-368; `pd/comparison.csv` (`improves`); T12, T89 |
| T118 | presentazione: unità dell'ACR | T27, T95–T98 |
| T119 | presentazione: popolazione | T23, T25, T28, T67, T68 |
| T120 | limiti | T28–T29, T40, T99–T100; soglie stimate e valutate sulle stesse previsioni: `src/models/evaluation.py`:260-263; soglie lette e non stimate sul test: `src/models/final_test.py`:111-122; tri-ensemble: `cfg`:340-348 |
