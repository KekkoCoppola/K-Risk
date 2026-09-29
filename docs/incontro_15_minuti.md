# L'incontro in 15 minuti

Versione breve di [`incontro_relatori.md`](incontro_relatori.md), da dire a voce. Obiettivo: in 15 minuti far capire che cosa hai fatto, che cosa hai ottenuto e perché il lavoro è solido. I dettagli restano nel documento lungo, per le domande.

Aggiornato al 29/09/2026: comprende il bersaglio continuo dell'albuminuria (25/09) e il primo capitolo della tesi (26/09).

## La scaletta

| minuti | blocco | cosa deve restare |
|---|---|---|
| 0–1 | 1. Il problema | una domanda clinica concreta |
| 1–3 | 2. Dati e bersaglio KDIGO | il bersaglio è quello delle linee guida, costruito con cura |
| 3–6 | 3. La pipeline | corposa dove conta: nei controlli |
| 6–11 | 4. I risultati | cinque risultati, confermati su dati mai visti |
| 11–13 | 5. Perché è un buon lavoro | rigore, utilità misurata, un risultato metodologico |
| 13–15 | 6. Limiti e prossimi passi | i limiti li ho trovati io |

Tre regole:
- non leggere: questa pagina è una traccia, le frasi vanno dette con parole tue;
- se sei in ritardo, riduci 4.2 e 4.4 a una frase ciascuno; il blocco 5 non si taglia;
- presenta i limiti come cose che hai trovato e misurato tu: sono una prova di rigore, non una debolezza.

---

## 1. Il problema (1 minuto)

> In una frase: ho costruito un modello che, senza esami renali, indica a chi fare gli esami per la malattia renale cronica, e ho misurato con rigore quanto funziona e dove serve.
>
> La malattia renale cronica è silenziosa: in Cina solo il 10% di chi ce l'ha sa di averla. Per riconoscerla bastano due esami, la creatinina nel sangue e l'albumina nelle urine, ma le linee guida li prescrivono ogni anno a chi ha il diabete. Per gli altri resta una domanda: a chi farli?
>
> Ho provato a rispondere con i dati che un ambulatorio ha già: età, pressione, esami del sangue di routine. Con un vincolo preciso: nessun esame renale fra i dati in ingresso. Il modello non sostituisce gli esami, decide a chi prescriverli.

## 2. Dati e bersaglio KDIGO (2 minuti)

> Ho usato un dataset pubblico uscito quest'anno su *Scientific Data*: 5.922 soggetti di un reparto ospedaliero di Shanghai, con 190 variabili.
>
> All'inizio il bersaglio era la colonna `DN`, "nefropatia diabetica". Verificandola ho visto che era solo una soglia sulla concentrazione di albumina, e che l'82% dei positivi non era diabetico. L'ho abbandonata e sono passato alla definizione KDIGO: eGFR sotto 60 oppure rapporto albumina/creatinina sopra 30. L'eGFR l'ho ricalcolato con l'equazione CKD-EPI 2021, perché la colonna del dataset non corrispondeva a nessuna equazione standard.
>
> Restano 5.801 soggetti, con il 9,8% di positivi. Dalla heatmap KDIGO ho ricavato anche i 4 livelli di rischio, da basso a molto alto. Non li uso per allenare il modello, ma per dividere i dati in modo equilibrato e per chiedermi se il modello riconosce i casi gravi.

## 3. La pipeline (3 minuti)

> La pipeline ha dieci passaggi, ognuno con il suo codice e i suoi test.
> - **Variabili**: da 190 a 74. Ogni colonna ha un motivo scritto per entrare o uscire, e un test automatico blocca il codice se entra un esame renale.
> - **Valori mancanti**: il 38% dei soggetti ne ha almeno uno. Ho confrontato quattro metodi di imputazione e ho scelto MissForest.
> - **Modelli**: cinque, uno per ruolo. Un classificatore banale come controllo, una logistica con i predittori del punteggio clinico SCORED, una logistica penalizzata, Random Forest e XGBoost. Cross-validation annidata 5 × 5, iperparametri scelti con Optuna.
> - **Bilanciamento**: otto tecniche, dai pesi di classe a SMOTE, fino a CTGAN, una rete generativa.
> - **Ricerca del tetto**: altre dodici strategie, fra cui TabPFN, CatBoost e LightGBM.
>
> Alla domanda che mi avete fatto, corposa o minimal: **corposa nei controlli, volutamente semplice nei modelli**. Ogni protocollo è scritto prima di calcolare i risultati, ci sono 182 test automatici, e il test set l'ho aperto una sola volta, alla fine, con le affermazioni da verificare già fissate.

## 4. I risultati (5 minuti)

### 4.1 Discriminazione modesta, ma onesta (1 minuto)
> Il modello distingue chi ha i marcatori con un'AUROC di circa 0,70 in cross-validation, e fra 0,71 e 0,74 sul test. Nessun modello è migliore degli altri in modo dimostrabile: la logistica con i 5 predittori di SCORED resta vicina ai modelli con 74 variabili. È quello che trova la letteratura quando i confronti sono fatti bene.

### 4.2 Ordina la gravità, ma solo debolmente (30 secondi)
> La probabilità stimata cresce in media a ogni livello KDIGO. Fra i soli positivi, però, l'ordine è debole: la concordanza con i livelli è fra 0,61 e 0,64, dove 0,5 vuol dire nessun ordine. E le fasce del modello non riproducono i livelli: il modello riconosce la presenza del danno, non il grado.

### 4.3 Il risultato centrale: il bilanciamento non aiuta, ma sembra aiutare (1 minuto e mezzo)
> La domanda principale era: le tecniche di bilanciamento fanno riconoscere più casi gravi? No. A parità di sensibilità nessuna tecnica ne trova di più, e SMOTE e CTGAN peggiorano.
>
> Però alla soglia di default, 0,5, sembrano trasformare il modello: il recall passa dal 2% al 60%. È un effetto della soglia, non del modello, e si ripete identico sul test.
>
> Ho misurato quattro casi come questo, numeri che sembrano buoni e non lo sono: la soglia 0,5, la PR-AUC confrontata fra gruppi con prevalenza diversa, una media di calibrazione che nasconde due errori opposti e la selezione delle variabili fatta fuori dalla validazione. Li considero un contributo della tesi.

### 4.4 Il tetto è nei dati, non nel metodo (1 minuto)
> Perché 0,70? Ho scomposto il bersaglio: l'eGFR basso si riconosce bene, con AUROC 0,80–0,86; l'albuminuria no, 0,67–0,69. E nel training il 91% dei positivi è albuminuria.
>
> Dodici strategie in più restano fra 0,69 e 0,71, anche addestrando sul valore continuo dell'ACR invece che sulla soglia. Aggiungendo l'albumina urinaria si sale a 0,93: la pipeline impara quando l'informazione c'è. In letteratura i modelli per l'albuminuria senza esami delle urine stanno fra 0,58 e 0,76, e un punteggio sviluppato nello stesso ospedale arriva a 0,70–0,72.

### 4.5 Serve, fra i non diabetici (1 minuto)
> Il modello serve? L'ho misurato con la decision curve, che lo confronta con "testare tutti". Alla soglia del 7% evita 9–13 esami inutili ogni 100 persone; al 10%, 25–28. Fra i diabetici invece no: li segnala quasi tutti, e lì hanno ragione le linee guida, che li testano comunque. Il modello serve proprio dove manca una regola.
>
> Tutto questo l'ho confermato sul test set, 1.451 soggetti mai visti: nessuna delle cinque affermazioni fissate prima è stata contraddetta.

## 5. Perché è un buon lavoro (2 minuti)

> Riassumo il valore in quattro punti.
> 1. **Rigore**: cross-validation annidata, protocolli scritti prima dei calcoli, test set usato una volta sola e documentato da git. Il risultato è credibile perché non potevo aggiustarlo a posteriori.
> 2. **Utilità misurata**: non mi fermo all'AUROC, dico quanti esami si risparmiano. In una revisione del 2024, nessuno dei 12 modelli per la malattia renale sviluppati o validati su cartelle cliniche di comunità riportava un'analisi di questo tipo.
> 3. **Un risultato metodologico**: il bilanciamento, che molti usano di default, qui non aiuta; e ho misurato quattro modi in cui un numero può ingannare.
> 4. **Onestà sul tetto**: 0,70 non è un difetto nascosto. È un limite dei dati, spiegato e in linea con la letteratura.
>
> E tutto è riproducibile: un solo file di configurazione, 182 test, ogni decisione registrata.

## 6. Limiti e prossimi passi (2 minuti)

> I limiti li ho trovati e dichiarati io:
> - è una coorte ospedaliera, non uno screening di popolazione;
> - c'è una sola misura per soggetto, quindi parlo di marcatori, non di diagnosi;
> - manca una validazione esterna: il test è una conferma nella stessa coorte.
>
> Ora sto scrivendo la tesi: il primo capitolo è pronto, poi vengono stato dell'arte, metodologia, risultati e conclusioni. Resta anche il prototipo dimostrativo.
>
> Vi chiederei due cose: cosa mettere nel testo e cosa in appendice, perché il materiale è tanto; e che cosa vi aspettate dal prototipo.

---

## Se ti interrompono

| domanda | risposta in una frase |
|---|---|
| 0,70 non è poco? | È il tetto di questo bersaglio: l'albuminuria dagli esami del sangue si riconosce male, e la letteratura sta fra 0,58 e 0,76. |
| Come sai che non c'è leakage? | Gli esami renali sono vietati dal codice, nessuna variabile da sola supera 0,75, e senza esami renali le prestazioni restano ferme a 0,70. |
| Perché niente deep learning? | L'ho provato dove aveva senso: CTGAN peggiora, TabPFN arriva a 0,71 senza differenze significative. |
| Quale modello sceglieresti? | Nessuno è dimostrabilmente migliore: per il prototipo è una scelta pratica, da concordare con voi. |
| Il test è una validazione esterna? | No, è una conferma interna nella stessa coorte; la validazione esterna è il limite principale. |

Per tutto il resto: [`incontro_relatori.md`](incontro_relatori.md), sezione 3.

## I tre numeri da far ricordare
- **0,70**: il tetto, misurato e spiegato;
- **dal 2% al 60%**: il miglioramento che sembra vero e non lo è;
- **9–13 esami inutili evitati ogni 100 persone**: l'utilità clinica.
