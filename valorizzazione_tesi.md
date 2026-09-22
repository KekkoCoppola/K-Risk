# Valorizzazione della tesi e posizionamento scientifico di K-Risk

Documento di sintesi metodologica, clinica e bibliografica ad uso della tesi. Definisce il posizionamento di K-Risk rispetto allo stato dell'arte e raccoglie tutti i risultati che ne misurano il valore.

**Regole di questo documento.** Ogni numero del progetto viene da una tabella in `analytics/` (previsioni out-of-fold sul training, 4.350 soggetti; il test set non è mai stato letto). Ogni fonte esterna è stata verificata il 22/09/2026 su Crossref, PubMed, Europe PMC o sul sito dell'editore: **TC** = contenuto letto nel testo completo, **AB** = letto nell'abstract. Rapporto completo della verifica in `docs/verifica_stato_arte.md`. Le analisi della Fase D e del blocco "qualità e utilità clinica" sono post-hoc e vanno presentate come esplorative.

---

## 0. Il valore del progetto in sei punti

1. **Serve a qualcosa, e lo dimostra con la misura giusta.** Con la decision curve analysis, alla soglia del 7% il modello evita **9–13 esami inutili ogni 100 persone** rispetto a "testare tutti", a parità di casi trovati; al 10% ne evita 25–28 (sezione 5). Le revisioni recenti segnalano che i modelli di CKD per la comunità quasi mai riportano un'analisi di utilità clinica (Haris et al. 2024).
2. **Batte SCORED, di poco ma in modo coerente.** È superiore su tutte e quattro le misure (AUROC, PR-AUC, esami necessari a parità di sensibilità, net benefit), senza differenze statisticamente significative. Il tri-ensemble su 21 variabili, con selezione verificata dentro ogni fold, ottiene le stesse prestazioni della Random Forest con meno di un terzo delle variabili ed è il migliore sul piano operativo: 7,8 esami in meno ogni 100 persone rispetto a SCORED per trovare l'85% dei casi (sezione 4).
3. **Il tetto di prestazione è spiegato con i dati, non ipotizzato.** L'eGFR < 60 si riconosce bene (AUROC 0,81–0,86), l'albuminuria no (0,67–0,69). La letteratura mostra lo stesso schema (sezione 6).
4. **Non c'è leakage, ed è verificato.** Esami renali esclusi per costruzione e bloccati da un controllo automatico; controllo positivo con l'albumina urinaria a AUROC 0,933: la pipeline impara quando l'informazione c'è (sezione 7).
5. **Rigore metodologico da articolo.** Protocolli scritti prima dei risultati, validazione incrociata annidata, test set mai toccato, 143 test automatici, revisioni indipendenti del codice.
6. **Quattro "risultati apparenti" smascherati.** La soglia 0,5 che gonfia il recall dal 2% al 61%, la PR-AUC dei diabetici che sembra doppia, la pendenza di calibrazione "media 1,02" che nasconde due errori opposti, e la selezione delle variabili fatta su tutti i dati che porta l'AUROC da 0,703 a 0,716 e rende "significativo" un vantaggio che non lo è (sezione 7).

---

## 1. Il problema clinico e il ruolo di K-Risk

### Il problema
- La CKD si definisce con due esami: **eGFR < 60 ml/min/1,73 m²** (dalla creatinina sierica) oppure **albuminuria, ACR ≥ 30 mg/g**, presenti per almeno 3 mesi (KDIGO 2024, Tabella 1; TC).
- **La maggior parte delle persone con CKD non sa di averla**: negli USA fino a 9 adulti su 10 (CDC 2023; TC), l'87% nell'aggiornamento CDC 2026 (TC). In Cina, nella sorveglianza nazionale 2018–2019 su 176.874 adulti, prevalenza 8,2% (albuminuria 6,7%, eGFR ridotto 2,2%) e **consapevolezza del 10,0%** (Wang L et al. 2023; TC).
- KDIGO 2024 indica di testare le persone a rischio con **entrambi** gli esami, albumina urinaria ed eGFR (Practice Point 1.1.1.1; TC). Nel diabete l'ADA raccomanda ACR ed eGFR almeno una volta l'anno (Standards of Care 2024 e 2026, raccomandazione 11.1a; TC); fra i diabetici la CKD riguarda il 20–40% (ADA, sezione 11; TC).
- Nelle persone senza diabete l'ACR si prescrive raramente: è lì che serve decidere **chi** mandare all'esame.

### Il ruolo di K-Risk
- **A monte** del percorso diagnostico: un triage di primo livello che non sostituisce gli esami renali, ma decide a chi prescriverli.
- **Input**: dati anagrafici, antropometrici, pressori ed esami del sangue di routine (glicemia, lipidi, enzimi epatici, emocromo, peptide C…).
- **Vincolo**: tutti gli esami renali sono esclusi dall'input (`SCRE`, `UMAUCR`, `GFR`, `BUN` e derivati), con un controllo automatico che blocca il codice se una colonna vietata entra fra le feature.
- **Uscita**: una probabilità di avere marcatori di CKD. È calibrata per le due logistiche; Random Forest e XGBoost lo sono solo in media (sezione 5).

### I dati
Li J et al. 2026, *Sci Data* (TC): **reparto di Diabetologia ed Endocrinologia dello Shanghai Sixth People's Hospital, febbraio-aprile 2012**, 5.922 record e 190 variabili. Dopo aver escluso chi non ha creatinina, ACR, età o sesso restano 5.801 soggetti (training 4.350, test 1.451). Nel training solo il 6% ha la variabile `DM` = 1 e la prevalenza dei marcatori di CKD è 9,8%.

**Attenzione per la tesi**: gli autori descrivono dati ospedalieri, **non uno screening di popolazione**; il tipo di campione urinario per l'ACR e le unità di `UCRE` e `UmALB` **non sono documentati**; ci sono piccole incongruenze interne (dati trasversali o di follow-up, durata della raccolta). K-Risk va quindi presentato come modello sviluppato su una coorte ospedaliera cinese, da validare in popolazioni di screening.

---

## 2. Stato dell'arte

I modelli si distinguono per **bersaglio**, e il bersaglio decide l'AUROC raggiungibile.

| modello | popolazione | bersaglio | esami usati | AUROC / C | fonte |
|---|---|---|---|---|---|
| **Bersaglio: eGFR ridotto, stato attuale** | | | | | |
| SCORED | NHANES 1999–2002, n = 8.530 | eGFR < 60 | 9 variabili, **incluse proteinuria e anemia** | 0,88 interna; **0,71 esterna** (ARIC) | Bang et al. 2007 (AB) |
| Kwon | KNHANES, Corea | eGFR < 60 | 7 variabili, incluse proteinuria e anemia | 0,83; 0,87 e 0,78 in validazione | Kwon et al. 2012 (AB) |
| Sabanayagam, soli fattori di rischio | Singapore; test a Singapore e Pechino | eGFR < 60 | nessuno (età, sesso, etnia, diabete, ipertensione) | 0,916 interna; 0,829 e 0,887 esterne | Sabanayagam et al. 2020 (AB) |
| **Bersaglio: composito con albuminuria, stato attuale** | | | | | |
| Thakkinstian | Thailandia, comunità, n = 3.459 | CKD stadi 1–5 (albuminuria o ematuria negli stadi 1–2) | nessuno (età, diabete, ipertensione, calcoli) | 0,77; 0,74 con bootstrap | Thakkinstian et al. 2011 (TC/AB) |
| MERWACS | NHANES, **≥ 50 anni**; esterna KNHANES | ACR ≥ 30 o eGFR sotto il 2,5° percentile per età e sesso | nessuno (12 parametri) | 0,68–0,70 interna; 0,71–0,73 esterna | Yoo et al. 2026 (TC) |
| **K-Risk** | coorte ospedaliera, Shanghai | **ACR ≥ 30 o eGFR < 60 (KDIGO)** | esami del sangue di routine, **nessun esame renale** | **0,675–0,703** (Fase A); 0,708–0,710 (miglior candidato Fase D) | questa tesi |
| **Bersaglio: sola albuminuria** | | | | | |
| Muntner | REGARDS, ≥ 45 anni; esterna NHANES | ACR ≥ 30 | nessuno (8 domande) | 0,709; 0,714 esterna | Muntner et al. 2011 (AB) |
| Tanner | MESA, 45–84 anni | ACR ≥ 30 | nessuno (stesso strumento) | 0,728–0,761 (0,761 nei cino-americani) | Tanner et al. 2015 (AB) |
| Bragg-Gresham | NHANES, adulti senza diabete, n = 44.322 | ACR ≥ 30 | **eGFR < 60**, HbA1c, HDL, acido urico | 0,752 in validazione | Bragg-Gresham et al. 2025 (TC) |
| Khitan | diabete tipo 2 (Look AHEAD) | ACR ≥ 30 | **creatinina sierica**, HbA1c, lipidi | 0,61–0,67 | Khitan et al. 2021 (TC) |
| **Bersaglio: CKD futura (incidente)** | | | | | |
| QKidney | cure primarie UK, > 1,5 milioni | CKD moderata-grave a 5 anni | nessuno (dati clinici) | 0,875–0,876 | Hippisley-Cox & Coupland 2010 (AB) |
| Chien | Taiwan, coorte | eGFR < 60 a 4 anni | modello clinico | 0,768; 0,667 esterna | Chien et al. 2010 (AB) |
| Wen | Cina rurale, n = 3.266 | eGFR < 60 o ACR ≥ 30 in circa 6 anni | nessuno (Simple Score) | 0,717 | Wen J et al. 2020 (TC) |
| **A valle: progressione della CKD accertata** | | | | | |
| KFRE | 721.357 pazienti con CKD G3–G5, 31 coorti, oltre 30 paesi | dialisi o trapianto a 2 e 5 anni | **eGFR e ACR** | 0,90 (2 anni); 0,88 (5 anni) | Tangri et al. 2016 (AB) |
| **Confronto locale** | | | | | |
| Wu M | stesso ospedale del dataset, diabete tipo 2 ricoverati | malattia renale diabetica (proteinuria 24 h) | nessuno (sesso, BMI, PAS, durata del diabete) | 0,70–0,72; 0,70 prospettica | Wu M et al. 2017 (TC) |

### Come leggere la tabella
- **Quando il bersaglio è l'eGFR ridotto, i modelli arrivano a 0,78–0,92.** L'eGFR dipende soprattutto dall'età, che tutti i modelli conoscono. Due di questi (SCORED e Kwon) usano anche la proteinuria, cioè un esame delle urine.
- **Quando il bersaglio contiene l'albuminuria, i modelli stanno fra 0,68 e 0,77**, anche con decine di migliaia di soggetti; Bragg-Gresham arriva a 0,75 solo aggiungendo l'eGFR, cioè la creatinina, fra i predittori.
- **K-Risk sta dove deve stare**: 0,70–0,71 su un bersaglio composito al 91% di albuminuria, senza nessun esame renale, in linea con MERWACS (0,68–0,73) e Muntner (0,71).
- **Originalità, formulata con prudenza**: non abbiamo trovato modelli che stimino il composito KDIGO attuale (eGFR < 60 o ACR ≥ 30) senza esami renali in una popolazione cinese con e senza diabete.
- Le revisioni sistematiche confermano il quadro: molti punteggi, AUROC di sviluppo spesso > 0,70, poche validazioni esterne (Echouffo-Tcheugui & Kengne 2012; González-Rocha et al. 2023, AUROC 0,63–0,91 nelle popolazioni sane); rischio di bias alto e **nessuna analisi di utilità clinica** nei modelli di comunità (Haris et al. 2024); machine learning raramente validato fuori dal contesto di sviluppo (Sanmarchi et al. 2023).

---

## 3. K-Risk e KFRE: due strumenti per due momenti diversi

| | **K-Risk (questa tesi)** | **KFRE (Tangri et al. 2016)** |
|---|---|---|
| popolazione | coorte ospedaliera, soggetti in maggioranza senza diabete noto (n = 5.801) | pazienti con CKD già diagnosticata, G3–G5 (n = 721.357) |
| input | esami del sangue di routine, **zero esami renali** | età, sesso, **eGFR e ACR** |
| bersaglio | marcatori di CKD presenti ora (eGFR < 60 o ACR ≥ 30) | insufficienza renale terminale a 2 o 5 anni |
| AUROC / C | 0,70–0,71 | 0,90 (2 anni), 0,88 (5 anni) |
| ruolo | **a monte**: chi deve fare i primi esami renali | **a valle**: prognosi e pianificazione della dialisi |

Il confronto non è una gara: il KFRE parte dagli stessi due esami che K-Risk serve a decidere se prescrivere. Se K-Risk li usasse come input, ricopierebbe la definizione del bersaglio e perderebbe ogni utilità.

---

## 4. K-Risk e SCORED

### Il confronto
SCORED è il riferimento storico per lo screening della CKD in cure primarie (Bang et al. 2007) ed è il modello di rischio per la CKD più validato esternamente secondo Echouffo-Tcheugui & Kengne 2012. Nel dataset sono disponibili 5 dei suoi 9 predittori (età, sesso, emoglobina al posto dell'anemia, pressione sistolica al posto dell'ipertensione, diabete); mancano proteinuria, storia cardiovascolare, scompenso e vasculopatia periferica. La logistica SCORED della tesi è quindi **ristimata sui nostri dati**: questo la avvantaggia, perché i coefficienti sono adattati alla popolazione, ma le mancano le variabili che non abbiamo.

### Risultato: superiori su tutte le misure, di poco

| misura | SCORED ristimato | logistica penalizzata | Random Forest | XGBoost |
|---|---|---|---|---|
| AUROC (IC 95%) | 0,675 (0,646–0,704) | 0,697 (0,669–0,725) | **0,703** (0,676–0,731) | 0,699 (0,671–0,726) |
| differenza di AUROC contro SCORED (IC 95%, 5 fold) | — | +0,021 (−0,003; +0,046) | +0,029 (−0,008; +0,066) | +0,025 (−0,004; +0,054) |
| PR-AUC | 0,224 | 0,242 | 0,246 | 0,251 |
| persone da esaminare per trovare l'85% dei casi | 73,8 ogni 100 | 69,2 | 68,5 | **66,2** |
| persone da esaminare per trovare il 90% dei casi | 80,2 ogni 100 | 78,9 | 78,1 | 76,9 |
| soglie 5–20% in cui il net benefit supera quello di SCORED | — | **31 su 31** | **31 su 31** | 26 su 31 |
| esami inutili evitati in più rispetto a SCORED, soglia 7% | — | +1,8 ogni 100 | +4,0 | +3,0 |

Fonti: `analytics/phase_a/evaluation/` (AUROC, PR-AUC, punti operativi, confronti appaiati) e `analytics/quality/evaluation/decision_curve.csv`. Nella decision curve XGBoost è la versione con profondità 1–12, come in tutte le analisi dalla Fase B in poi.

**Come scriverlo nella tesi**: i modelli con gli esami del sangue di routine sono **superiori a SCORED in stima puntuale su tutte e quattro le misure**, ma **nessuna differenza è statisticamente significativa** (AUROC: p di Holm 0,44; PR-AUC: p = 1,00). In pratica, per trovare l'85% dei casi servono **4,6–7,6 esami in meno ogni 100 persone**, cioè 460–760 esami ACR in meno ogni 10.000 persone valutate (22.500–37.200 dollari ai 49 dollari per test di Cusick et al. 2023; stima illustrativa).

**La lettura corretta del "di poco"**: è la replica, su un dataset nuovo, di Christodoulou et al. 2019, che su 145 confronti a basso rischio di bias non trovano differenze fra machine learning e regressione logistica (differenza di logit(AUROC) 0,00; IC −0,18; 0,18). Nei confronti ad alto rischio di bias lo stesso lavoro trova un vantaggio apparente del machine learning: l'assenza di un grande vantaggio è quindi un segnale di correttezza.

**Il candidato con il miglior equilibrio**: la **logistica penalizzata** è l'unico modello della Fase A superiore a SCORED su tutte e quattro le misure e anche ben calibrato (pendenza 0,94, IC 0,80–1,07). Random Forest discrimina un po' meglio ma è mal calibrato (sezione 5). È un'osservazione descrittiva, non una nuova scelta del modello.

### Il tri-ensemble su 21 variabili, verificato
Un'analisi preliminare non registrata riportava per un "tri-ensemble sulle 21 variabili più importanti" AUROC 0,717 e PR-AUC 0,273. Il 22/09/2026 è stato registrato come candidato della Fase D **prima** di calcolarlo (`configs/config.yaml`, `tri_ensemble_top21`): media delle probabilità di logistica penalizzata, Random Forest e XGBoost, addestrati sulle prime 21 variabili con gli iperparametri della Fase A di ogni fold. Le 21 variabili si scelgono **dentro ogni fold, sul solo training**, per rango medio fra i tre modelli (|coefficiente| per la logistica, SHAP medio per gli alberi). Come diagnostica si è calcolata anche la versione con le variabili scelte su tutto il training.

| versione | AUROC (IC 95%) | PR-AUC | contro RF (IC 95%) | contro SCORED (IC 95%, 5 fold) |
|---|---|---|---|---|
| **selezione dentro ogni fold (candidato)** | **0,703** (0,676–0,731) | 0,252 | −0,002 (−0,025; +0,020), p di Holm 1,00 | +0,027 (−0,004; +0,057), p = 0,07 |
| selezione su tutto il training (diagnostica) | 0,716 (0,689–0,742) | 0,264 | — | +0,039 (+0,007; +0,071), p = 0,03 |

**Cosa si ricava:**
- **lo 0,717 originale nasceva dalla distorsione da selezione**: la versione con le variabili scelte su tutti i dati lo riproduce quasi esattamente (0,716 e 0,264), quella corretta scende a 0,703. La selezione fuori dalla validazione gonfia l'AUROC di +0,013 (Ambroise & McLachlan 2002). Non va quindi citato come risultato;
- **la distorsione fabbrica anche la significatività**: contro SCORED la versione sbagliata sembrerebbe superiore in modo significativo (p = 0,03), quella corretta no (p = 0,07). È il quarto "risultato apparente" della tesi (sezione 7);
- **il valore reale è la parsimonia**: con 21 variabili invece di 74 il modello discrimina come la Random Forest (0,703) ed è il migliore sul piano operativo contro SCORED. Per trovare il 90% dei casi fa esaminare 75,0 persone ogni 100 contro 80,2; per l'85%, 66,0 contro 73,8, cioè **7,8 esami in meno ogni 100**. Il net benefit supera quello di SCORED a 31 soglie su 31, e al 7% evita 5,6 esami inutili in più ogni 100. Un modello con meno esami in ingresso è più facile da usare in un ambulatorio;
- **nucleo stabile di 9 variabili**, scelte in tutti e 5 i fold: età, pressione sistolica, glicemia a digiuno e a 2 ore, peptide C a digiuno, acido urico, fosfatasi alcalina (ALP), indice FIB-4 e colesterolo LDL. Altre 6 compaiono in 4 fold su 5 (colesterolo totale, trigliceridi, globuli bianchi, peptide C a 2 ore, emoglobina, GGT). Sono le variabili su cui il modello si appoggia davvero, qualunque sia il campione di addestramento: descrivono ciò che il modello ha imparato, non relazioni causali.

I p del confronto con SCORED sono post-hoc e non corretti per confronti multipli.

### Le stime migliori della Fase D
| candidato | AUROC (IC 95%) | AUROC media sui fold | contro SCORED |
|---|---|---|---|
| TabPFN v2 (Hollmann et al. 2025) | 0,710 (0,682–0,737) | 0,717 | +0,035 |
| ensemble, media di logistica penalizzata, RF e XGBoost (74 variabili) | 0,708 (0,680–0,735) | 0,710 | +0,033 |
| tri-ensemble, 21 variabili scelte nel fold | 0,703 (0,676–0,731) | 0,706 | +0,028 |

Nessuno supera la regola fissata per la Fase D contro il miglior modello della Fase A (differenza ≥ 0,01, IC sopra zero, Holm < 0,05): contro Random Forest TabPFN guadagna +0,009 (IC −0,014; +0,032). Sono le stime più alte del progetto, da citare come esplorative.

---

## 5. Utilità clinica: il modello serve?

Blocco post-hoc, protocollo scritto prima dei calcoli (Notepad, "Qualità e utilità clinica"). La misura è il net benefit (Vickers & Elkin 2006), raccomandato fra le misure essenziali insieme ad AUROC e curva di calibrazione (Van Calster et al. 2025). Figure `analytics/quality/01`–`06`.

### Decision curve (popolazione del dataset, prevalenza 9,8%)
| soglia | esami inutili evitati ogni 100 persone, rispetto a "testare tutti" |
|---|---|
| 2–4,5% | da −4 a +2: equivalente a testare tutti |
| 5% | da +0,1 a +2,3 |
| **7%** | **da +9,2 a +13,1** |
| 10% | da +24,9 a +27,8 |
| 20% | da +54,9 a +56,8 |

- dal 5–6% in su tutti e quattro i modelli battono sia "testare tutti" sia "non testare nessuno";
- le soglie 5% e 7% sono quelle usate come esempi da Bragg-Gresham et al. 2025 per la stessa decisione;
- MERWACS (Yoo et al. 2026) riporta anch'esso un'analisi delle curve decisionali: la tesi si allinea al lavoro più recente sul tema.

### Sui diabetici il modello non serve, e questo conferma le linee guida
Fra i diabetici (prevalenza 25,9%) nessun modello fa meglio di "testare tutti" fino al 10%. Alla soglia globale del progetto il modello segnala il 97–100% dei diabetici. È coerente con l'ADA, che prescrive l'esame ogni anno a tutti i diabetici: **K-Risk serve a decidere fra i soggetti senza diabete**.

### Costi
"Testare tutti" costa **502 dollari per caso trovato** (10,2 esami da 49 dollari). Alla soglia del 7% il modello abbassa il costo per caso trovato a 336–383 dollari e fa spendere **1.800–2.600 dollari in meno ogni 100 persone**, al prezzo di 1,6–2,8 casi non trovati ogni 100: recuperarli testando tutti costerebbe 880–1.100 dollari ciascuno. Non è un'analisi di costo-efficacia. Il contesto economico però sostiene l'idea: lo screening della CKD è più costo-efficace nei gruppi con punteggi di rischio alti (Yeo et al. 2024, revisione di 21 valutazioni economiche) e in Cina è costo-efficace anche nella popolazione generale (Wen F et al. 2025).

### Calibrazione
- **In media è buona**: intercetta 0,00 e rapporto O:E 1,00 per tutti i modelli.
- **Le due logistiche sono calibrate** (pendenze 0,97 e 0,94); **Random Forest schiaccia le probabilità** (pendenza 1,31, IC 1,13–1,48), **XGBoost le esaspera** (0,86, IC 0,75–0,97).
- **Controllo di equità** (TRIPOD+AI, item 23a): fra i non diabetici nessun problema; fra i diabetici Random Forest sottostima il rischio di circa un quinto (O:E 1,26, IC 1,01–1,52). Non cambia nessuna decisione, perché i diabetici vanno testati comunque, ma va dichiarato.

### Confronto con Bragg-Gresham (riferimento esterno)
Per trovare il 73% dei casi di albuminuria fra i non diabetici servono 12–22 persone esaminate in più ogni 100 rispetto al loro modello; per l'85%, almeno 19–27 in più. Il loro modello però usa **eGFR, HbA1c, HDL e acido urico** su 44.322 adulti: il divario misura quanto costa rinunciare alla creatinina.

---

## 6. Il tetto di prestazione: spiegato con i dati

La vecchia versione di questo documento attribuiva il tetto di AUROC 0,70–0,75 a ragioni biologiche non documentate. Il progetto ora ha prove dirette.

1. **Il limite è l'albuminuria.** Con le stesse previsioni, i positivi per eGFR < 60 si distinguono dai negativi con AUROC 0,81–0,86; i positivi per sola albuminuria con 0,67–0,69. Il bersaglio è per il 91% albuminuria (362 + 25 casi su 425 nel training). Lo stesso schema compare in letteratura (sezione 2).
2. **Non è la tecnica.** Dieci strategie diverse nella Fase D (ensemble, NaN gestiti dall'algoritmo, spazio degli iperparametri allargato, bersaglio scomposto, CatBoost, LightGBM, EBM, TabPFN…) restano fra 0,692 e 0,710.
3. **Non sono i dati che mancano.** Dal 60% al 100% del training l'AUROC sale di 0,006–0,008: più soggetti aiuterebbero poco.
4. **La pipeline funziona.** Aggiungendo l'albumina urinaria l'AUROC sale a 0,933: quando l'informazione c'è, il modello la trova.
5. **L'etichetta è rumorosa.** Un ACR su un solo campione è instabile: solo il 43,5% degli ACR ≥ 30 su urina casuale viene confermato sulla prima urina del mattino (Saydah et al. 2013); la variabilità dell'ACR nella stessa persona è del 48,8% nel diabete tipo 2 (Rasaratnam et al. 2024); KDIGO chiede infatti di ripetere un ACR anomalo (Practice Point 1.1.1.2). Nel dataset il tipo di campione non è documentato, e nelle giornate con creatinina urinaria mediana bassa la quota di ACR ≥ 30 sale (rho di Spearman −0,70) senza che salga l'albumina: è un'anomalia da dichiarare.

In sintesi: con esami del sangue di routine e un'albuminuria misurata su un solo campione, 0,70–0,71 è il livello raggiunto da tutte le strategie provate e da tutti i modelli pubblicati con lo stesso bersaglio.

---

## 7. Perché il risultato è credibile

### Assenza di leakage, verificata
- esami renali esclusi dall'input e bloccati da un'asserzione nel codice;
- audit riga per riga della pipeline, controllato da un revisore indipendente;
- controllo positivo con l'albumina urinaria (0,933): se ci fosse leakage nascosto, le prestazioni senza esami renali non sarebbero ferme a 0,70;
- test set (1.451 soggetti) mai letto.

### Come si riconoscono le prestazioni gonfiate in letteratura
- **Esami renali fra le feature**: il dataset UCI "Chronic Kidney Disease" (Rubini et al. 2015, 400 pazienti ospedalieri) contiene creatinina, urea, albumina e peso specifico urinari; sopra questi dati gli articoli arrivano al 98,9–99,8% di accuratezza (Qin et al. 2020; Chittora et al. 2021). È la regola diagnostica ricopiata, non una previsione.
- **Solo validazione interna**: il deep learning sulle retinografie riconosce l'eGFR < 60 con AUROC 0,911 in validazione interna, ma 0,733 e 0,835 sui dati esterni; un modello con soli età, sesso, etnia, diabete e ipertensione fa altrettanto bene (Sabanayagam et al. 2020).
- **Rischio di bias**: nei confronti ad alto rischio di bias il machine learning sembra migliore della logistica, in quelli a basso rischio no (Christodoulou et al. 2019).

### Metodo
Validazione incrociata annidata 5 × 5 stratificata; iperparametri scelti solo sui fold interni; soglie, fasce e regole di decisione scritte nel Notepad prima di calcolare; intervalli di confidenza per ogni stima; 143 test automatici; revisioni indipendenti del codice per ogni fase.

### Quattro risultati apparenti, misurati
1. **Fase B**: alla soglia 0,5 le tecniche di bilanciamento portano il recall dal 2% al 61%, ma a parità di sensibilità non riconoscono un solo caso grave in più.
2. **Fase C**: fra i diabetici la PR-AUC sembra doppia (0,44 contro 0,19), ma rapportata alla prevalenza del gruppo è più bassa.
3. **Blocco qualità**: la pendenza di calibrazione "media 1,02" nasconde Random Forest a 1,31 e XGBoost a 0,86.
4. **Fase D, tri-ensemble**: scegliendo le 21 variabili su tutti i dati l'AUROC sale da 0,703 a 0,716 e il vantaggio su SCORED diventa "significativo" (p = 0,03 contro 0,07). È la distorsione da selezione di Ambroise & McLachlan 2002, misurata sui nostri dati.

Quattro esempi indipendenti di come un numero, letto senza controllare come è stato ottenuto, può ingannare: è un contributo metodologico della tesi, non un effetto collaterale.

---

## 8. Limiti da dichiarare

- dataset **ospedaliero** (Shanghai, 2012), non di screening: la validità esterna va dimostrata;
- ACR da un solo campione, di tipo non documentato, unità di `UCRE` e `UmALB` non documentate;
- nessuna validazione esterna; il test set interno darà solo una conferma;
- SCORED ristimato con 5 dei 9 predittori originali;
- Fase D e blocco qualità sono post-hoc ed esplorativi;
- il tri-ensemble usa gli iperparametri scelti in Fase A su 74 variabili: con un'ottimizzazione dedicata alle 21 variabili potrebbe cambiare di poco; il numero 21 viene dall'analisi preliminare e non è stato ottimizzato.

---

## 9. Bibliografia verificata

Verifica del 22/09/2026 (TC = testo completo, AB = abstract).

**Dati, linee guida ed epidemiologia**
- Li J, Zheng H, Zhou Y, Jiang F. A bimodal dataset for diabetes research. *Sci Data* 2026;13:652. doi:10.1038/s41597-026-06923-y. PMID 41813689. Dati: Zenodo doi:10.5281/zenodo.18270337 (TC)
- KDIGO 2024 CKD Work Group. KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD. *Kidney Int* 2024;105(4S):S117-S314. doi:10.1016/j.kint.2023.10.018. PMID 38490803 (TC)
- ADA Professional Practice Committee. 11. Chronic Kidney Disease and Risk Management: Standards of Care in Diabetes—2024. *Diabetes Care* 2024;47(Suppl 1):S219-S230. doi:10.2337/dc24-S011. PMID 38078574 (TC). Edizione 2026: doi:10.2337/dc26-S011 (TC)
- CDC. *Chronic Kidney Disease in the United States, 2023*. Atlanta: US DHHS, CDC; 2023 (TC). Aggiornamento 2026 (TC)
- Wang L, Xu X, Zhang M, et al. Prevalence of Chronic Kidney Disease in China. *JAMA Intern Med* 2023;183(4):298-310. doi:10.1001/jamainternmed.2022.6817. PMID 36804760 (TC)
- Inker LA, Eneanya ND, Coresh J, et al. New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race. *N Engl J Med* 2021;385(19):1737-49. doi:10.1056/NEJMoa2102953. PMID 34554658

**Modelli di confronto**
- Bang H, Vupputuri S, Shoham DA, et al. SCreening for Occult REnal Disease (SCORED). *Arch Intern Med* 2007;167(4):374-81. doi:10.1001/archinte.167.4.374. PMID 17325299 (AB)
- Kwon KS, Bang H, Bomback AS, et al. A simple prediction score for kidney disease in the Korean population. *Nephrology* 2012;17(3):278-84. doi:10.1111/j.1440-1797.2011.01552.x. PMID 22171932 (AB)
- Hippisley-Cox J, Coupland C. Predicting the risk of chronic kidney disease in men and women in England and Wales: prospective derivation and external validation of the QKidney Scores. *BMC Fam Pract* 2010;11:49. doi:10.1186/1471-2296-11-49. PMID 20565929 (AB)
- Yoo D, Nguyen VK, Maggiore U, Jolliet O. MERWACS: development and external validation of a non-invasive machine learning tool for identifying subjects to be screened for CKD. *PLOS Digit Health* 2026;5(7):e0001486. doi:10.1371/journal.pdig.0001486. PMID 42424235 (TC)
- Tangri N, Grams ME, Levey AS, et al. Multinational Assessment of Accuracy of Equations for Predicting Risk of Kidney Failure: A Meta-analysis. *JAMA* 2016;315(2):164-74. doi:10.1001/jama.2015.18202. PMID 26757465 (AB)
- Bragg-Gresham JL, Annadanam S, Gillespie B, Li Y, Powe NR, Saran R. Using Risk Assessment to Improve Screening for Albuminuria among US Adults without Diabetes. *J Gen Intern Med* 2025;40(13):3159-69. doi:10.1007/s11606-024-09185-9. PMID 39557751 (TC)
- Muntner P, Woodward M, Carson AP, et al. *Am J Kidney Dis* 2011;58(2):196-205. doi:10.1053/j.ajkd.2011.01.027. PMID 21620547 (AB)
- Tanner RM, Woodward M, Peralta C, et al. *Ethn Dis* 2015;25(4):427-34. doi:10.18865/ed.25.4.427. PMID 26676090 (AB)
- Khitan Z, Nath T, Santhanam P. *J Clin Hypertens* 2021;23(12):2137-45. doi:10.1111/jch.14397. PMID 34847294 (TC)
- Thakkinstian A, Ingsathit A, Chaiprasert A, et al. *BMC Nephrol* 2011;12:45. doi:10.1186/1471-2369-12-45. PMID 21943205 (TC/AB)
- Chien KL, Lin HJ, Lee BC, et al. *Am J Med* 2010;123(9):836-846.e2. doi:10.1016/j.amjmed.2010.05.010. PMID 20800153 (AB)
- Wen J, Hao J, Zhang Y, et al. *BMC Nephrol* 2020;21:120. doi:10.1186/s12882-020-01787-9. PMID 32252667 (TC)
- Wu M, Lu J, Zhang L, et al. *Oncotarget* 2017;8(60):102550-8. doi:10.18632/oncotarget.21684. PMID 29254270 (TC)
- Sabanayagam C, Xu D, Ting DSW, et al. *Lancet Digit Health* 2020;2(6):e295-e302. doi:10.1016/S2589-7500(20)30063-7. PMID 33328123 (AB)

**Revisioni e metodo**
- Echouffo-Tcheugui JB, Kengne AP. Risk models to predict chronic kidney disease and its progression: a systematic review. *PLoS Med* 2012;9(11):e1001344. doi:10.1371/journal.pmed.1001344. PMID 23185136 (AB)
- González-Rocha A, Colli VA, Denova-Gutiérrez E. *Prev Chronic Dis* 2023;20:E30. doi:10.5888/pcd20.220380. PMID 37079751 (AB)
- Haris M, et al. *Clin Kidney J* 2024;17(5):sfae098. doi:10.1093/ckj/sfae098. PMID 38737345 (AB)
- Sanmarchi F, et al. *J Nephrol* 2023;36(4):1101-17. doi:10.1007/s40620-023-01573-4. PMID 36786976 (AB)
- Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol* 2019;110:12-22. doi:10.1016/j.jclinepi.2019.02.004. PMID 30763612 (AB)
- Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 2002;99(10):6562-6. doi:10.1073/pnas.102102699. PMID 11983868 (AB)
- Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model. *Nature* 2025;637(8045):319-26. doi:10.1038/s41586-024-08328-6. PMID 39780007 (AB)
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making* 2006;26(6):565-74. doi:10.1177/0272989X06295361
- Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. *Lancet Digit Health* 2025. doi:10.1016/j.landig.2025.100916 (TC, preprint arXiv:2412.10288)
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378

**Qualità dell'etichetta ACR**
- Saydah SH, Pavkov ME, Zhang C, et al. *Clin Chem* 2013;59(4):675-83. doi:10.1373/clinchem.2012.195644. PMID 23315482 (AB)
- Rasaratnam N, Salim A, Blackberry I, et al. *Am J Kidney Dis* 2024;84(1):8-17.e1. doi:10.1053/j.ajkd.2023.12.018. PMID 38551531 (AB)

**Economia dello screening**
- Cusick MM, Tisdale RL, Chertow GM, Owens DK, Goldhaber-Fiebert JD. Population-Wide Screening for Chronic Kidney Disease: A Cost-Effectiveness Analysis. *Ann Intern Med* 2023;176(6):788-97. doi:10.7326/M22-3228. PMID 37216661 (TC)
- Yeo SC, Wang H, Ang YG, Lim CK, Ooi XY. *Clin Kidney J* 2024;17(1):sfad137. doi:10.1093/ckj/sfad137. PMID 38186904 (AB)
- Wen F, Wang J, Yang C, et al. *Lancet Reg Health West Pac* 2025;56:101493. doi:10.1016/j.lanwpc.2025.101493. PMID 40226778 (AB)
- Boulware LE, Jaar BG, Tarver-Carr ME, Brancati FL, Powe NR. *JAMA* 2003;290(23):3101-14. doi:10.1001/jama.290.23.3101. PMID 14679273 (AB)

**Studi con prestazioni gonfiate (esempi critici)**
- Rubini L, Soundarapandian P, Eswaran P. *Chronic Kidney Disease* [dataset]. UCI Machine Learning Repository; 2015. doi:10.24432/C5G020
- Qin J, Chen L, Liu Y, Liu C, Feng C, Chen B. *IEEE Access* 2020;8:20991-21002. doi:10.1109/ACCESS.2019.2963053 (AB)
- Chittora P, et al. *IEEE Access* 2021;9:17312-34. doi:10.1109/ACCESS.2021.3053763 (AB)
