# Dama Italiana AI & Computer Vision

Project work ITS dedicato allo sviluppo di un sistema completo per giocare a Dama Italiana combinando sviluppo software, intelligenza artificiale, reinforcement learning e computer vision.

Il progetto include una web app giocabile da browser, un motore di gioco con regole ufficiali della dama italiana, un agente AI basato su ricerca/minimax, una sperimentazione con modello PPO e una modalità fisica che permette di riconoscere una scacchiera reale tramite webcam.

## Obiettivo del progetto

L’obiettivo principale del progetto è realizzare un sistema capace di collegare il mondo digitale e quello fisico attraverso una pipeline completa:

input dell’utente, validazione della mossa, aggiornamento dello stato della partita, risposta della macchina e riconoscimento della scacchiera tramite computer vision.

Il progetto nasce come project work ITS in ambito Artificial Intelligence Developer and Data Analyst.

## Funzionalità principali

* Gioco di Dama Italiana con regole ufficiali.
* Interfaccia web giocabile da browser.
* Motore di gioco con gestione di mosse, catture, promozioni e turni.
* Cattura obbligatoria e catture multiple.
* Regola del soffio.
* Motore AI basato su ricerca/minimax.
* Notebook per sperimentazione e training tramite reinforcement learning.
* Modello PPO addestrato per test e sperimentazione.
* Modalità Computer Vision con webcam.
* Calibrazione della scacchiera fisica tramite selezione dei quattro angoli.
* Correzione prospettica della scacchiera.
* Riconoscimento dei pezzi tramite analisi dei colori.
* Identificazione della mossa confrontando lo stato prima e dopo lo spostamento.

## Struttura del progetto

```text
dama-ai-computer-vision/
├── web/
│   ├── index.html
│   ├── styles.css
│   └── game.js
│
├── src/
│   ├── dama_core.py
│   └── physical_camera_game.py
│
├── models/
│   └── dama_ppo_model.zip
│
├── notebooks/
│   └── dama.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Tecnologie utilizzate

* Python
* JavaScript
* HTML
* CSS
* NumPy
* OpenCV
* Gymnasium
* Stable-Baselines3
* PPO
* Minimax
* Computer Vision
* Reinforcement Learning

## Web App

La web app permette di giocare a Dama Italiana direttamente dal browser, senza installazioni aggiuntive.

La logica del gioco è implementata in JavaScript e include:

* generazione della scacchiera;
* gestione dei turni;
* validazione delle mosse;
* catture obbligatorie;
* promozione delle pedine;
* cronologia delle mosse;
* risposta automatica della macchina tramite logica AI/minimax.

Per avviare la web app è possibile aprire il file:

```text
web/index.html
```

oppure avviare un server locale:

```bash
cd web
python -m http.server 8000
```

e aprire nel browser:

```text
http://localhost:8000
```

## Modalità Computer Vision

La modalità Computer Vision permette di giocare utilizzando una scacchiera fisica reale e una webcam.

Il sistema esegue questi passaggi:

1. Acquisizione del frame dalla webcam.
2. Selezione dei quattro angoli della scacchiera.
3. Correzione prospettica per ottenere una vista dall’alto.
4. Apprendimento dei colori iniziali di pezzi e caselle vuote.
5. Lettura dello stato della scacchiera.
6. Confronto tra stato precedente e stato successivo.
7. Riconoscimento della mossa effettuata dall’utente.
8. Validazione della mossa tramite motore di gioco.
9. Calcolo della risposta della macchina.

Per avviare la modalità webcam:

```bash
python src/physical_camera_game.py --camera 0 --engine master --depth 5
```

Se si vuole usare il modello PPO:

```bash
python src/physical_camera_game.py --camera 0 --engine ppo --ppo-path models/dama_ppo_model.zip
```

## Notebook di training

Il notebook contiene la parte sperimentale del progetto legata al reinforcement learning.

Include:

* definizione dell’ambiente Gymnasium;
* configurazione del modello PPO;
* training dell’agente;
* valutazione del modello;
* confronto con avversari automatici;
* sperimentazione della modalità fisica con webcam.

Il notebook si trova in:

```text
notebooks/dama.ipynb
```

## Motore di gioco

Il file `dama_core.py` contiene il cuore logico del progetto.

Gestisce:

* rappresentazione della scacchiera;
* regole della Dama Italiana;
* mosse legali;
* catture;
* promozioni;
* cambio turno;
* verifica del vincitore;
* euristiche;
* algoritmo minimax;
* inferenza della mossa a partire dallo stato della scacchiera.

## Competenze sviluppate

Questo progetto mi ha permesso di sviluppare competenze in:

* sviluppo software;
* logica di gioco;
* intelligenza artificiale;
* reinforcement learning;
* computer vision;
* image processing;
* progettazione di pipeline;
* integrazione tra software e input fisico;
* problem solving;
* lavoro in team.

## Autori

Project work realizzato da:

* Nicolò Romare
* Arion Halili
* Diana Popovici
* Elyass Rochdi

## Contesto

Progetto sviluppato durante il percorso ITS in Artificial Intelligence Developer and Data Analyst.

## Possibili sviluppi futuri

* Miglioramento del modello di reinforcement learning.
* Aumento del numero di episodi di training.
* Integrazione di self-play.
* Miglioramento della robustezza della Computer Vision in condizioni di luce diverse.
* Riconoscimento automatico delle dame tramite marker fisici o classificazione avanzata.
* Creazione di una dashboard per analizzare le partite.
* Integrazione tra web app e backend Python.
