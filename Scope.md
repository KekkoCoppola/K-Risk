# Scope del progetto

## In una frase
Allenare modelli di classificazione che, **senza esami renali**, stimino chi presenta marcatori di malattia renale cronica, confrontare le stime con la stratificazione clinica **KDIGO** e misurare quanto le tecniche di **data augmentation** migliorano il riconoscimento dei casi più gravi.

## Dati
- dati clinici del reparto di Diabetologia ed Endocrinologia dello Shanghai Sixth People's Hospital, febbraio-aprile 2012: 5.922 soggetti, 190 variabili (dataset pubblico, CC BY 4.0). Gli autori non li descrivono come screening di popolazione (verifica del 22/09/2026 sul testo completo di Li J. et al. 2026; dettaglio in `docs/verifica_stato_arte.md`)
- soggetti utilizzabili: **5.801**, quelli con creatinina sierica e rapporto albumina/creatinina urinaria (ACR)
- dati **trasversali**: una sola misurazione per soggetto, nessun follow-up

## Target
Classificazione **binaria**:

```
y = 1  se  ACR >= 30 mg/g  oppure  eGFR < 60 ml/min/1,73 m²
y = 0  altrimenti
```

- sono i criteri KDIGO per i marcatori di malattia renale cronica
- **567 positivi su 5.801 (9,77%)**: classi sbilanciate, da cui la data augmentation
- l'eGFR è ricalcolato con **CKD-EPI 2021**: la colonna `GFR` del dataset non corrisponde a nessuna equazione standard
- nel sottogruppo dei **diabetici** (351 soggetti, 91 positivi) il target coincide con la definizione di malattia renale cronica nel diabete (DKD)

### Livelli di rischio KDIGO
Incrociando eGFR (asse G) e ACR (asse A) si ottengono 4 livelli. Servono per **valutare**, non per allenare:

| livello | soggetti | corrisponde a |
|---------|----------|---------------|
| basso | 5.234 | y = 0 |
| moderato | 473 | y = 1 |
| alto | 67 | y = 1 |
| molto alto | 27 | y = 1 |

## Input e output del modello
- **input**: solo dati disponibili senza esami renali — anagrafica, antropometria, pressione, glicemia, lipidi, altri esami del sangue, anamnesi, stili di vita
- **escluse** tutte le variabili renali (`GFR`, `UMAUCR`, `UmALB`, `SCRE`, `BUN`, `DN`, indicatori derivati): con quelle il modello copierebbe la regola KDIGO, senza alcun valore
- **output**: probabilità, classe 0/1 e fascia di rischio

Perché non una regressione: l'ACR è fortemente asimmetrica, le tecniche di augmentation da confrontare sono pensate per la classificazione e KDIGO è già una classificazione.

## Impianto sperimentale
1. **Split 75/25**, stratificato su livello KDIGO × diabete, seed fisso. Test congelato: nessuna tecnica lo tocca.
2. **Preprocessing**: selezione di 74 feature (nessuna variabile renale, nessuna colonna con codice 9, NA ≤ 15%), codifiche senza one-hot, imputazione e scaling stimati solo sul training, dentro ogni fold. Metodo di imputazione scelto con un confronto in cross-validation: **MissForest** (motivazione in Notepad, Passi 9–10).
3. **Fase A — cinque modelli sui dati originali**, senza augmentation, scelti per famiglia e ruolo (motivazione e bibliografia nel Notepad, sezione "Fase A — modelli e protocollo"):
   - classificatore di maggioranza: soglia minima e controllo di coerenza
   - regressione logistica con i predittori del punteggio clinico SCORED (Bang et al. 2007) disponibili nel dataset, ristimata sui nostri dati
   - regressione logistica penalizzata (L1, L2 o elastic net): modello lineare di riferimento
   - Random Forest: ensemble ad albero, bagging
   - XGBoost: ensemble ad albero, gradient boosting

   Protocollo: cross-validation annidata (5 fold esterni per le previsioni, 5 interni per gli iperparametri), stesso budget di ottimizzazione per tutti i modelli (Optuna), metrica di ottimizzazione PR-AUC.
4. **Fase B — stesse domande, stessi cinque modelli, per ogni tecnica di bilanciamento**: nessuna correzione, pesi di classe, pesi per livello KDIGO, undersampling, oversampling, SMOTE-NC, SMOTE-NC per livello KDIGO; CTGAN (anche condizionato al livello KDIGO) come analisi esplorativa. Esito principale: sensibilità sui casi gravi a parità di sensibilità complessiva (protocollo nel Notepad, sezione "Fase B").
5. **Fase C — sottogruppo diabetico** (domanda 6): le stesse previsioni filtrate sui diabetici, con soglia e fasce globali fisse (protocollo nel Notepad, sezione "Fase C").
6. **Analisi post-hoc, dichiarate tali**: la Fase D (ricerca del tetto di prestazione: dieci strategie più il tri-ensemble, e tre diagnostiche) e il blocco "qualità e utilità clinica" (decision curve, calibrazione anche nei sottogruppi, costi, confronto con la letteratura). Regole scritte prima dei calcoli, risultati esplorativi.
7. **Conclusioni** delle domande 1–6 sulle previsioni out-of-fold, poi **conferma finale sul test set**, una sola volta, con il protocollo fissato prima (Notepad, sezione "Conferma finale sul test set — protocollo").

## Le domande della tesi
Valgono per la Fase A e, identiche, per ogni tecnica della Fase B:

1. il modello distingue chi ha marcatori di malattia renale? → AUC, PR-AUC, precision, recall
2. il rischio stimato cresce con la gravità KDIGO? → probabilità per livello, test di tendenza
3. quanti casi "alto" e "molto alto" riconosce? → sensibilità per livello. **Mancare un "molto alto" è l'errore più grave**
4. le fasce di rischio del modello corrispondono alla stratificazione clinica? → concordanza con i livelli KDIGO

E infine:

5. l'augmentation migliora il riconoscimento dei casi gravi o solo la metrica media?
6. come si comporta lo stesso modello sui diabetici (91 casi)? → risultati descrittivi

## Regole metodologiche
- augmentation **solo sul training** e **dentro ogni fold**: i dati sintetici non devono mai finire in validazione o nel test
- **ricalibrare** le probabilità dopo l'augmentation (il bilanciamento le distorce); riportare anche metriche basate sull'ordinamento, che la distorsione non tocca
- confronto con KDIGO **solo su soggetti reali**: i sintetici non hanno un livello KDIGO
- analisi principale per livello sulle previsioni **out-of-fold** del training (21 casi "molto alto", 50 "alto"), conferma sul test (6 e 17)
- test set usato **una volta sola**, alla fine: scelte di modello, tecnica e soglia si fanno in cross-validation

## Cosa il progetto non è
- non propone un nuovo algoritmo: il contributo è una pipeline completa, riproducibile e metodologicamente corretta
- non sostituisce gli esami: nelle persone con diabete le linee guida prescrivono ACR ed eGFR ogni anno. L'utilità del modello è **stabilire la priorità degli esami nei soggetti senza diabete**; va dimostrata in popolazioni di screening, perché il dataset è ospedaliero
- non è uno strumento diagnostico: il prototipo è dimostrativo

## Limiti dichiarati
- una sola misurazione: KDIGO richiede alterazioni persistenti oltre 3 mesi, quindi si parla di **marcatori**, non di diagnosi. Con un solo esame un danno acuto non si distingue da uno cronico
- confronto con KDIGO **non indipendente**: target e livelli usano gli stessi marcatori. La domanda è "senza esami renali il modello ordina i soggetti come KDIGO?", non "il modello batte KDIGO"
- livelli gravi poco numerosi (67 "alto", 27 "molto alto") → stime per livello instabili
- sottogruppo diabetico piccolo (91 positivi) → solo descrittivo
- dati trasversali: il modello riconosce lo stato presente, non prevede la progressione
- nessuna validazione esterna: una sola popolazione, una sola finestra temporale
- dataset ospedaliero, non di screening: tipo di campione urinario per l'ACR e unità di `UCRE` e `UmALB` non documentati dagli autori
- soggetti raggruppati per giornata di raccolta (prevalenza dal 4,5% al 24,5%): possibile struttura per comunità, non valutata

## Riferimenti
- dataset: Li J. et al., *A bimodal dataset for diabetes research*, Scientific Data (2026) — vedi `data/README.md`
- KDIGO 2026, linea guida su diabete e malattia renale cronica (bozza per revisione pubblica)
- KDIGO 2026, linee guida su AKI/AKD e su anemia nella malattia renale cronica
- KDIGO 2024, valutazione e gestione della malattia renale cronica (da procurare)

Il dettaglio operativo, le verifiche fatte sui dati e le motivazioni di ogni scelta sono in [`Notepad.md`](Notepad.md).
