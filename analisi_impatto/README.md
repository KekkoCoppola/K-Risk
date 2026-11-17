# analisi_impatto — analisi post-hoc

Codice e risultati aggregati delle analisi successive alle Fasi A–E: apprendimento non
supervisionato confrontato con KDIGO, confine dell'AI rispetto alla soglia di ACR 30,
replica esterna e mortalità su NHANES, separazione dei due criteri KDIGO (filtrazione e
albuminuria). Sono tutte analisi **post-hoc**, decise dopo aver visto i risultati delle fasi
registrate in `configs/config.yaml`.

Ogni script dichiara nella docstring il tipo di esperimento:

- **PREDITTIVO**: nessun esame renale fra gli ingressi (stesse 74 colonne del progetto);
  misura quanto si riconosce la CKD.
- **ESPLICATIVO**: gli ingressi contengono le misure che definiscono il target; descrive come
  ragiona il modello rispetto a KDIGO e non stima una capacità predittiva.

Tutte le analisi sugli Jinuo usano solo il training (4.350 soggetti) e i 5 fold esterni del
progetto. L'insieme di test non viene letto. In uscita ci sono solo statistiche aggregate.

## Il filone non supervisionato

### Domanda

I relatori hanno chiesto che il modello impari da solo, senza etichetta, senza formula KDIGO e
senza soglie, cosa distingue i malati, e che il risultato venga poi confrontato con KDIGO. Il
filone risponde con tre esperimenti da non mescolare:

| Esperimento | Tipo | Ingressi | Etichetta | Domanda |
|---|---|---|---|---|
| Passo 0 e parte A | PREDITTIVO | le 74 colonne del progetto, senza esami renali | sì | scegliendo chi controllare, il modello trova più malati della regola clinica "diabete o ipertensione"? |
| Split G | ESPLICATIVO | le 74 colonne + 4 esami renali grezzi (`SCRE`, `UmALB`, `UCRE`, `BUN`) = 78 | no | senza formula, l'AI ritrova KDIGO? Come ragiona? |
| Split T | ESPLICATIVO | tutte le colonne del file (188 nei modelli) | no | con tutto il file, cosa trova? |

L'etichetta KDIGO (`target()` in `src/data/kidney.py`) si usa solo per valutare, mai per stimare
i modelli non supervisionati.

### Disegno

- **Dati**: solo training (4.350 soggetti, 425 positivi), stessi 5 fold esterni delle Fasi A–E.
  Il modello si stima sul fold di addestramento e il punteggio si calcola sul fold escluso.
- **Preprocessing** (`prep.py`), stimato dentro il fold: moda per le categoriche, MissForest
  come nel progetto, `log1p` sulle continue asimmetriche, standardizzazione.
- **Modelli** (`03_parteB.py`), ognuno trasformato in un punteggio di "atipicità" continuo:
  - k-prototypes (k-means per dati misti, Huang 1998) con k = 2, k = 4 e k scelto dalla gap
    statistic; punteggio = distanza dal centro del gruppo più numeroso;
  - misture gaussiane separate per sesso (k = 2, k = 4, k scelto con il BIC); punteggio =
    1 − probabilità di appartenere alla componente più numerosa;
  - distanza di Mahalanobis robusta (MinCovDet) sulle sole variabili continue;
  - autoencoder (MLP 32-8-32); punteggio = errore quadratico medio di ricostruzione.
- **Confronto con KDIGO**: AUROC del punteggio contro l'etichetta (IC di DeLong), prevalenza dei
  malati nei gruppi trovati, punteggio medio per fascia di ACR.
- **Riferimento con etichetta** (`04_supervisionato_G.py`): elastic net, Random Forest e XGBoost
  stimati sulle stesse 78 colonne di G, per misurare quanto vale l'etichetta.
- **XAI** (`05_xai.py`): profili dei gruppi (d di Cohen), importanza per permutazione, surrogato
  XGBoost + TreeSHAP con interazioni, importanza per famiglie di variabili, verso con cui il
  punteggio usa gli esami renali.

### Risultati

Numeri letti dai CSV in `04_esperimenti/non_supervisionato/` (percorso, poi colonna o riga).

**Parte A [PREDITTIVO].** A parità di persone testate (378, quante ne testa la regola "diabete
o ipertensione"), il Random Forest della Fase A trova 120 malati contro 98 della regola: +22
(IC bootstrap 8–36, McNemar p = 0,008). La regola più larga, che aggiunge la pressione misurata
≥ 140/90, testa 929 persone: 203 contro 162, cioè +41 (20–60). Fonti: `figure/fig1_parteA_malati_trovati.csv`;
`../impatto/A_parita_di_testati.csv`, colonna `p_mcnemar`. Gran parte del vantaggio è età:
contro la regola "diabete, ipertensione o età > 55" (864 testati) il modello trova 188 contro
181, +7 (p = 0,48) (`../impatto/A_parita_di_testati.csv`).

**Split G [ESPLICATIVO]: senza etichetta l'AI arriva al livello del supervisionato senza reni,
non a KDIGO.**

| Modello | AUROC (IC 95%) |
|---|---|
| autoencoder su G, senza etichetta | 0,713 (0,688–0,738) |
| k-means k = 4 su G, senza etichetta | 0,706 (0,680–0,732) |
| Random Forest Fase A, 74 colonne senza esami renali, con etichetta | 0,703 (0,676–0,731) |
| Mahalanobis su G, senza etichetta | 0,644 (0,621–0,667) |
| mistura gaussiana k = 2 su G, senza etichetta | 0,551 (0,521–0,582) |
| XGBoost su G, con etichetta | 0,998 (0,997–0,999) |

Fonte: `figure/fig2_G_auroc_con_e_senza_etichetta.csv`. Con l'etichetta, sulle stesse 78
colonne, KDIGO si ricostruisce quasi perfettamente: l'informazione c'è, ma senza etichetta
l'AI non sa quali colonne contano (`parteB/G/B7_valore_etichetta.csv`).

Come ragiona l'AI rispetto al clinico (split G):

- usa l'albumina urinaria nel verso clinico (più albumina, punteggio più alto) e la creatinina
  urinaria nel verso del denominatore dell'ACR, ma non le combina in un rapporto e non corregge
  la creatinina per età e sesso (`figure/fig3_G_verso_degli_esami_renali.csv`,
  `xai/G/C2_criteri_sintesi.csv`);
- gli esami renali pesano quanto una colonna qualunque; contano anche diabete, sesso, fegato ed
  emocromo (`xai/G/C1_famiglie.csv`);
- **il punteggio sale già sotto la soglia KDIGO di 30 mg/g**: fra ACR 13–26 e 7–13 la
  differenza media è +0,175 per k-means k = 4 (p = 0,002) e +0,110 per l'autoencoder
  (p = 0,003); fra 7–13 e 4–7 nessuna differenza (p = 0,31 per k-means). Il segnale non compare
  con Mahalanobis e con le misture (`parteB/G/posthoc_fasce_acr_confronti.csv`,
  `figure/fig4_punteggio_per_fascia_di_acr.csv`).

**Split T [ESPLICATIVO]: con tutto il file l'AI "trova" la CKD perché la diagnosi è già
scritta in una colonna.** La colonna `EUGFR90UACR01` vale 1 per tutti i 425 malati del
training e per 12 sani (`figure/fig5_T_diagnosi_gia_scritta.csv`, pannello a). Nei fold 0–3 il
gruppo di k-means k = 4 con più malati coincide quasi del tutto con chi ha quella colonna a 1
(pannello b). Ablazione, stessi modelli, AUROC media dei 5 fold
(`figure/fig6_T_da_dove_viene_lo_0_84.csv`):

| Colonne guardate | k-means k = 4 | autoencoder |
|---|---|---|
| tutte le 188 | 0,838 | 0,840 |
| senza le 14 diagnosi già scritte | 0,674 | 0,670 |
| senza nessuna delle 18 colonne renali | 0,649 | 0,645 |
| solo le 18 colonne renali | 0,993 | 0,992 |
| split G, per confronto | 0,707 | 0,713 |

Nella classifica SHAP la colonna non è in testa: la scorciatoia si vede dal profilo del gruppo
e dall'ablazione, non dall'importanza (`xai/T/C1_importanze.csv`).

### Prosecuzione su NHANES (`codice/confine_ai/`, punti 6–7) [PREDITTIVO]

- Sugli Jinuo, k-means k = 4 sulle 74 colonne senza esami renali né etichetta non batte la
  regola clinica: 98 malati contro 98 a parità di testati, AUROC 0,648; il Random Forest
  supervisionato trova 120 (`04_esperimenti/confine_ai/punto6/P6A_nonsupervisionato_vs_regola.csv`).
- Random Forest stimato sugli Jinuo (14 variabili comuni) e applicato a NHANES senza
  riaddestramento: 4.710 malati contro 4.300 della regola su 13.788 testati, +410 (IC 351–469),
  AUROC 0,790 (`punto6/P6B_validazione_esterna.csv`).
- k-means stimato sugli Jinuo senza etichetta, applicato ai 28.433 adulti NHANES "sani" per
  KDIGO (ACR < 30, eGFR ≥ 60): il punteggio sale con l'ACR anche sotto 30 (pendenza 0,092 DS per
  raddoppio, IC 0,080–0,104) e predice la mortalità, HR 1,14 per DS (1,07–1,21) aggiustato per
  età e sesso, 1,09 (1,02–1,16) aggiungendo log ACR; cardiovascolare 1,38 (1,20–1,58)
  (`punto7/P7a_gradiente_sotto_30.csv`, `punto7/P7b_mortalita_sani_kdigo.csv`). Cause di morte
  in `punto7/P7c_cause_di_morte.csv`.

### Limiti dichiarati

- Tutto il filone è post-hoc rispetto alle fasi registrate; il passo 0 è esplorativo e la parte A
  è una verifica di robustezza sugli stessi dati, non una conferma indipendente.
- Negli split G e T gli ingressi contengono le misure che definiscono il target: le AUROC non
  stimano una capacità predittiva.
- Il segnale sotto 30 mg/g vale per 4 configurazioni su 7 ed è un'analisi post-hoc.

## Struttura

```
analisi_impatto/
├── codice/
│   ├── giro2/                 elenco dei modelli con previsioni OOF (ingresso del passo 0)
│   ├── non_supervisionato/    passo 0, parte A, parte B (split G e T), XAI, post-hoc, figure
│   ├── impatto/               screening a parità di testati contro la regola diabete/ipertensione
│   ├── confine_ai/            punti 2–7: corpo lungo l'ACR, NHANES, mortalità, trasporto
│   └── filtrazione/           eGFR < 60 e ACR ≥ 30 separati, replica e trasporto su NHANES
└── 04_esperimenti/            uscite (CSV, figure, log) con la stessa suddivisione
```

## Ordine di esecuzione

Dalla radice del repository, con l'ambiente di `requirements.txt`. Prima servono
`data/processed/train.csv` (generato dagli script di `src/data/`) e le previsioni OOF delle
Fasi A–E in `analytics/`.

1. `codice/giro2/01_inventario_previsioni.py` → `codice/giro2/out/01_membri_rashomon.csv`
2. Non supervisionato (`codice/non_supervisionato/`):
   1. `00_passo0_linea_guida.py`: modelli supervisionati contro la linea guida, a parità di testati
   2. `01_parteA.py`: robustezza del passo 0
   3. `03_parteB.py G` e `03_parteB.py T`: modelli non supervisionati sui due split
   4. `04_supervisionato_G.py`: riferimento supervisionato sullo split G
   5. `05_xai.py G` e `05_xai.py T`: spiegazione dei punteggi e dei cluster
   6. `06_posthoc_G.py`, `07_posthoc_T.py`, `08_mancanti_T.py`: confronti e ablazioni post-hoc
   7. `09_figure.py`: figure, ognuna con il CSV dei numeri disegnati
   8. `10_fasce_acr_G.py`: punteggi per fascia di ACR

   `prep.py` è il preprocessing dentro il fold, importato dagli altri script.
3. `codice/impatto/01_abcd.py` (usa le funzioni di `01_parteA.py`)
4. Confine AI (`codice/confine_ai/`): `punto2_corpo_lungo_acr.py`, `punto2b_pendenze.py`,
   `punto3_reni_o_muscoli.py`, `punto4_nhanes.py`, `punto5_eta_nice.py`,
   `punto6_nonsup_e_validazione.py`, `punto7_modello_sotto_30.py`, `punto7c_cause_di_morte.py`.
   I punti 4–7 richiedono i dati NHANES (sotto).
5. Filtrazione (`codice/filtrazione/`): `01_addestra.py --bersaglio {G,A} --modelli ...`
   (CV annidata 5 × 5), poi `02_valuta.py`, `03_nhanes.py`, `04_nhanes_trasporto.py`.

Le figure dei punti del confine AI sono copiate anche in `analytics/soglia_ai/`.

## Dati NHANES

Non sono nel repository. Servono in `data/external/nhanes/` i file pubblici CDC dei cicli
2005–2006 … 2017–2018 (suffissi D–J): `DEMO`, `ALB_CR`, `BIOPRO`, `BMX`, `BPX`, `BPQ`, `CBC`,
`DIQ`, `GHB`, `GLU`, `HDL`, `TRIGLY` in formato `.xpt`, più i file di mortalità collegata
`NHANES_<anni>_MORT_2019_PUBLIC.dat`.

Due modi di usarli, dichiarati nelle docstring:

- **trasporto** (`punto6_nonsup_e_validazione.py` parte 6B, `punto7_modello_sotto_30.py`,
  `filtrazione/04_nhanes_trasporto.py`): modelli stimati una sola volta sugli Jinuo e applicati
  a NHANES senza riaddestramento;
- **replica in CV** (`punto4_nhanes.py` parte 4c, `punto5_eta_nice.py`, `filtrazione/03_nhanes.py`):
  modelli stimati dentro NHANES con una CV a 5 fold.
