# La pipeline passo per passo

Schema breve di tutta la pipeline, in ordine di esecuzione: per ogni passo cosa si fa, come e su quali dati. Serve a ricostruire il progetto a voce; i numeri e le motivazioni sono in [`incontro_relatori.md`](incontro_relatori.md) e nel `Notepad.md`.

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

## Dati e bersaglio (prima di tutto)

1. **Analisi del dataset.** File Excel di Li et al. 2026 (5.922 × 190), con impronta SHA-256; dizionario delle variabili letto dal PDF. Verifiche:
   - la colonna `GFR` non corrisponde a nessuna equazione standard;
   - `Gender` vale 1 = maschio e 2 = femmina, non 0/1 come dice il dizionario;
   - `DM` segue i criteri OMS 1999;
   - il 9 significa "sconosciuto" in 26 colonne.
2. **Soggetti eleggibili.** Si tengono i 5.801 con creatinina, ACR, età e sesso; 121 esclusi.
3. **Bersaglio e livelli.**
   - eGFR ricalcolato con CKD-EPI 2021 (creatinina da µmol/L a mg/dL dividendo per 88,4);
   - `y = 1` se ACR ≥ 30 oppure eGFR < 60: 567 positivi (9,8%);
   - livello KDIGO dall'incrocio delle categorie G (eGFR) e A (ACR).

   Bersaglio e livello non entrano mai fra le variabili del modello.
4. **Split.** `train_test_split` 75/25, seed 42, stratificato su livello KDIGO × diabete (8 strati):
   - training 4.350 soggetti (425 positivi), test 1.451 (142);
   - controllo con la differenza media standardizzata (SMD): tutte sotto 0,1;
   - da qui il test resta chiuso fino al passo 17.

*Codice*: `src/data/load.py`, `src/data/kidney.py`, `src/data/split.py`.

## Preprocessing "senza stato" (sul training, prima della CV)

Queste trasformazioni lavorano riga per riga e non imparano nulla dai dati: si fanno una volta, prima di dividere in fold.

5. **Selezione delle colonne, da 190 a 74** (60 numeriche + 14 categoriche). La mappa completa è in `configs/config.yaml`: ogni colonna ha un gruppo e un motivo. Escluse:
   - le colonne da cui si calcola il bersaglio e gli esami renali;
   - amministrative, non documentate, vuote, questionario sul diabete;
   - derivati a soglia, doppioni, combinazioni lineari esatte;
   - comorbidità con il codice 9;
   - `Homaβ`;
   - colonne con oltre il 15% di mancanti.

   Un'asserzione ferma il codice se entra una colonna vietata. Il set di sensibilità `no_consequence` ha 67 colonne.
6. **Codifiche, nessun one-hot.**
   - `Gender` 1/2 → 0/1;
   - fumo, alcol e tè 1/2/3 → 0/1/2, ordinali;
   - `HypertenHis` vuoto → 0;
   - le altre 9 categoriche sono già binarie 0/1 e restano così.
7. **Scelte decise una volta sul training**, con una CV a 5 fold:
   - imputazione con MissForest, dopo il confronto con mediana, KNN e MICE;
   - niente logaritmo;
   - valori estremi tenuti.

*Codice*: `src/data/preprocess.py` (`select_features`, `encode`), `src/data/imputation.py`.

## Cross-validation annidata (dentro ogni fold)

8. **Fold.** `StratifiedKFold` a 5, stratificato sul livello KDIGO, seed 42:
   - 5 fold esterni; dentro il training di ognuno, 5 fold interni;
   - con il training intero (per il modello finale) sono 31 "fold";
   - gli stessi fold servono al confronto delle imputazioni, alla Fase A, alla Fase B e alla Fase D.
9. **Preprocessing "con stato", per ogni fold.** È stimato solo sulla parte di training del fold, senza vedere y, e poi applicato alla validazione:
   1. **standardizzazione** delle 60 numeriche con `StandardScaler`: z = (x − media) / deviazione standard, con media e deviazione standard della parte di training; i mancanti sono ignorati nel calcolo;
   2. **moda** per le 14 categoriche (`SimpleImputer`), che non vengono standardizzate;
   3. **MissForest** sulle 74 colonne (`IterativeImputer` con `ExtraTreesRegressor`, 50 alberi, 5 iterazioni): ogni numerica con buchi viene predetta dalle altre, categoriche comprese.

   La standardizzazione serve alla logistica penalizzata e alle distanze di KNN e SMOTE; agli alberi è indifferente. Le matrici, senza mancanti, sono salvate una volta sola (31 fold × 2 set, circa 1 ora) in `data/processed/imputed/` e riusate da tutti i modelli e da tutte le tecniche.
10. **Fase A, dati originali.** Per ogni modello e fold esterno:
    - Optuna (campionatore TPE, seed 42, 100 tentativi, `MedianPruner`) sceglie gli iperparametri massimizzando la PR-AUC media sui 5 fold interni;
    - il modello si riaddestra sul training del fold esterno e predice la sua validazione: sono le previsioni out-of-fold, una per ciascuno dei 4.350 soggetti;
    - modello finale: iperparametri scelti sui 5 fold esterni, poi riaddestramento su tutto il training;
    - hanno iperparametri solo logistica penalizzata, Random Forest e XGBoost; maggioranza e SCORED no;
    - analisi di sensibilità: XGBoost con profondità 1–12 sceglie profondità 1.
11. **Fase B, bilanciamento sugli stessi fold.** Avviene dopo il passo 9 e solo sulla parte di training del fold:
    - pesi di classe e pesi per livello 1:2:3, passati al modello come `sample_weight`, senza righe nuove;
    - undersampling e oversampling casuali 1:1 (`imbalanced-learn`);
    - SMOTE-NC (k = 5) e SMOTE-NC dentro ogni livello;
    - CTGAN condizionato su y o sul livello, solo esplorativo.

    I training bilanciati sono salvati una volta in `data/augmented/`. Optuna usa 30 tentativi, partendo dall'ottimo della Fase A. La validazione resta sempre reale.
12. **Ricalibrazione di Platt (Fase B).** Il bilanciamento gonfia le probabilità. Le corregge una logistica su logit(p), stimata sulle previsioni delle righe interne reali.

*Codice*: `src/data/folds.py`, `src/data/imputed.py`, `src/models/zoo.py`, `src/models/phase_a.py`, `src/data/augmented.py`, `src/models/phase_b.py`.

## Valutazione (sulle previsioni out-of-fold)

13. **Domande 1–4.**
    - soglia: la più alta con sensibilità ≥ 0,90; 4 fasce con le proporzioni dei livelli KDIGO;
    - metriche: AUROC, PR-AUC, precision, recall, specificità, sensibilità per livello, concordanza di Jonckheere-Terpstra, kappa pesato;
    - intervalli: DeLong, Wilson, Boyd, bootstrap;
    - confronti fra modelli: t corretto di Nadeau-Bengio con correzione di Holm;
    - in Fase B l'esito primario sono i casi gravi (test di McNemar), più la valutazione alla soglia 0,5.
14. **Interpretazione.** Odds ratio per le logistiche, SHAP per Random Forest e XGBoost, calcolati sui modelli finali.
15. **Fase C.** Le stesse previsioni filtrate sui 263 diabetici del training, con la soglia e le fasce globali.
16. **Analisi post-hoc.**
    - Fase D: 11 candidati sugli stessi fold, con una regola scritta prima, più le diagnostiche;
    - utilità clinica: decision curve, calibrazione, costi.

*Codice*: `src/models/evaluation.py`, `evaluation_b.py`, `interpretation.py`, `phase_c.py`, `phase_d.py`, `clinical_utility.py`.

## Test (una volta sola)

17. **Conferma finale.**
    - il preprocessore è ristimato su tutto il training, e risulta identico bit a bit alla cache;
    - si applica un solo `transform` al test, poi si calcolano le previsioni dei modelli finali già salvati;
    - soglie e fasce restano quelle fissate out-of-fold;
    - si verificano 5 affermazioni fissate prima: nessuna è contraddetta.
18. **Dopo il test**, sul solo training: audit del tetto dei dati e bersaglio continuo dell'albuminuria.

*Codice*: `src/models/final_test.py`, `src/models/phase_d.py`.
