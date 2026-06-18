(function () {
  "use strict";

  // Valori usati dentro la matrice della scacchiera:
  // 0 = casella vuota, numeri positivi = bianco, numeri negativi = nero.
  const EMPTY = 0;
  const WHITE_MAN = 1;
  const WHITE_KING = 2;
  const BLACK_MAN = -1;
  const BLACK_KING = -2;

  // Giocatori: bianco vale 1, nero vale -1.
  // Questo rende facile cambiare turno facendo currentPlayer *= -1.
  const WHITE = 1;
  const BLACK = -1;

  // La dama si gioca su una scacchiera 8x8.
  const SIZE = 8;

  // Limite massimo di mosse per evitare partite infinite.
  const MAX_MOVES = 180;

  // Stato centrale del gioco.
  // Tutto quello che cambia durante la partita viene salvato qui.
  const state = {
    // Matrice 8x8 con pedine e caselle vuote.
    board: null,

    // Giocatore che deve muovere adesso.
    currentPlayer: WHITE,

    // Colore scelto dall'utente.
    humanPlayer: WHITE,

    // Casella selezionata dall'utente, oppure null se non ha selezionato nulla.
    selected: null,

    // Ultima mossa fatta, usata per evidenziarla sulla scacchiera.
    lastMove: null,

    // Testi delle mosse giocate, mostrati nella cronologia.
    history: [],

    // Diventa true quando qualcuno vince o la partita finisce.
    gameOver: false,

    // Diventa true mentre la macchina sta calcolando.
    thinking: false,
  };

  // Controlla se riga e colonna sono dentro la scacchiera.
  function inside(row, col) {
    return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
  }

  // Nella dama si usano solo le caselle scure.
  // Con questa formula alterniamo caselle chiare e scure.
  function playable(row, col) {
    return inside(row, col) && (row + col) % 2 === 1;
  }

  // Restituisce il proprietario di una pedina:
  // bianco, nero oppure 0 se la casella e' vuota.
  function owner(piece) {
    if (piece === EMPTY) return 0;
    return piece > 0 ? WHITE : BLACK;
  }

  // Una dama vale 2 o -2, quindi controllo il valore assoluto.
  function isKing(piece) {
    return Math.abs(piece) === 2;
  }

  // Converte riga/colonna in nome leggibile, per esempio a3 o f6.
  function squareName(row, col) {
    return String.fromCharCode(97 + col) + (SIZE - row);
  }

  // Crea una copia della scacchiera.
  // Serve per simulare mosse senza modificare subito la partita vera.
  function cloneBoard(board) {
    return board.map((row) => row.slice());
  }

  // Crea la posizione iniziale della partita.
  function initialBoard() {
    // Parte da una scacchiera vuota.
    const board = Array.from({ length: SIZE }, () => Array(SIZE).fill(EMPTY));

    // Riempie le prime 3 righe scure con il nero e le ultime 3 con il bianco.
    for (let row = 0; row < SIZE; row += 1) {
      for (let col = 0; col < SIZE; col += 1) {
        if (!playable(row, col)) continue;
        if (row < 3) board[row][col] = BLACK_MAN;
        if (row > 4) board[row][col] = WHITE_MAN;
      }
    }
    return board;
  }

  // Trasforma una pedina semplice in dama quando arriva in fondo.
  function promote(piece, row) {
    if (piece === WHITE_MAN && row === 0) return WHITE_KING;
    if (piece === BLACK_MAN && row === SIZE - 1) return BLACK_KING;
    return piece;
  }

  // Direzioni in cui una pedina puo' muoversi.
  // Le dame vanno avanti e indietro; le pedine semplici solo in avanti.
  function dirs(piece) {
    if (isKing(piece)) return [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    return piece > 0 ? [[-1, -1], [-1, 1]] : [[1, -1], [1, 1]];
  }

  // Regola di gerarchia della Dama Italiana:
  // una pedina semplice non puo' mai catturare una dama.
  function canCapturePiece(attacker, target) {
    if (target === EMPTY) return false;
    if (owner(attacker) === owner(target)) return false;
    if (!isKing(attacker) && isKing(target)) return false;
    return true;
  }

  // Crea un oggetto mossa.
  // path contiene le caselle attraversate, captures contiene le pedine mangiate.
  function makeMove(path, captures) {
    return { path, captures: captures || [] };
  }

  // Prima casella della mossa.
  function moveStart(move) {
    return move.path[0];
  }

  // Ultima casella della mossa.
  function moveEnd(move) {
    return move.path[move.path.length - 1];
  }

  // Controlla se due caselle sono uguali.
  function sameSquare(a, b) {
    return a && b && a[0] === b[0] && a[1] === b[1];
  }

  // Trasforma una mossa in testo leggibile.
  // Usa "-" per una mossa normale e "x" per una cattura.
  function moveText(move) {
    const sep = move.captures.length ? "x" : "-";
    return move.path.map(([row, col]) => squareName(row, col)).join(sep);
  }

  // Cerca tutte le pedine di un giocatore sulla scacchiera.
  function pieces(board, player) {
    const out = [];
    for (let row = 0; row < SIZE; row += 1) {
      for (let col = 0; col < SIZE; col += 1) {
        const piece = board[row][col];
        if (owner(piece) === player) out.push([row, col, piece]);
      }
    }
    return out;
  }

  // Trova tutte le sequenze di cattura possibili partendo da una pedina.
  // E' ricorsiva: dopo una cattura controlla se la stessa pedina puo' catturare ancora.
  function captureSequences(board, row, col, piece, path, captures) {
    const found = [];

    // Prova ogni direzione diagonale permessa alla pedina.
    for (const [dr, dc] of dirs(piece)) {
      // Casella della pedina avversaria da mangiare.
      const midRow = row + dr;
      const midCol = col + dc;

      // Casella dove atterrare dopo aver mangiato.
      const landRow = row + dr * 2;
      const landCol = col + dc * 2;

      // Se una delle caselle non e' valida, questa direzione non si puo' usare.
      if (!playable(midRow, midCol) || !playable(landRow, landCol)) continue;

      const middle = board[midRow][midCol];

      // Per catturare serve una pedina avversaria catturabile in mezzo e una casella libera dopo.
      if (!canCapturePiece(piece, middle) || board[landRow][landCol] !== EMPTY) continue;

      // Crea una scacchiera temporanea con questa cattura applicata.
      const next = cloneBoard(board);
      next[row][col] = EMPTY;
      next[midRow][midCol] = EMPTY;
      const nextPiece = promote(piece, landRow);
      next[landRow][landCol] = nextPiece;

      // Dopo la cattura, cerca altre catture dalla nuova posizione.
      const tails = captureSequences(
        next,
        landRow,
        landCol,
        nextPiece,
        path.concat([[landRow, landCol]]),
        captures.concat([[midRow, midCol]])
      );

      // Se ci sono catture successive, salva le sequenze complete.
      if (tails.length) found.push(...tails);

      // Se non ci sono catture successive, questa sequenza finisce qui.
      else found.push(makeMove(path.concat([[landRow, landCol]]), captures.concat([[midRow, midCol]])));
    }
    return found;
  }

  // Calcola solo le catture disponibili.
  function captureMoves(board, player) {
    const captures = [];

    for (const [row, col, piece] of pieces(board, player)) {
      captures.push(...captureSequences(board, row, col, piece, [[row, col]], []));
    }

    if (!captures.length) return [];

    // Cattura multipla obbligatoria: se una cattura puo' continuare,
    // mantiene le sequenze complete con il massimo numero di salti.
    const maxCaptures = Math.max(...captures.map((move) => move.captures.length));
    return captures.filter((move) => move.captures.length === maxCaptures);
  }

  // Calcola solo le mosse semplici, cioe' senza cattura.
  function quietMoves(board, player) {
    const quiet = [];
    for (const [row, col, piece] of pieces(board, player)) {
      for (const [dr, dc] of dirs(piece)) {
        const nextRow = row + dr;
        const nextCol = col + dc;
        if (playable(nextRow, nextCol) && board[nextRow][nextCol] === EMPTY) {
          quiet.push(makeMove([[row, col], [nextRow, nextCol]], []));
        }
      }
    }
    return quiet;
  }

  // Calcola le mosse disponibili.
  // allowSoffio=false: se esiste una cattura, restituisce solo catture.
  // allowSoffio=true: restituisce anche mosse semplici; se il giocatore le usa
  // mentre poteva catturare, il soffio verra' applicato dopo la mossa.
  function legalMoves(board, player, allowSoffio = false) {
    const captures = captureMoves(board, player);
    const quiet = quietMoves(board, player);
    if (captures.length) return allowSoffio ? captures.concat(quiet) : captures;
    return quiet;
  }

  // Applica una mossa e restituisce una nuova scacchiera aggiornata.
  // Non modifica direttamente la scacchiera ricevuta.
  function applyMoveToBoard(board, move) {
    const next = cloneBoard(board);
    const [startRow, startCol] = moveStart(move);
    const [endRow, endCol] = moveEnd(move);
    const piece = next[startRow][startCol];

    // Svuota la casella di partenza.
    next[startRow][startCol] = EMPTY;

    // Toglie dalla scacchiera tutte le pedine catturate.
    for (const [row, col] of move.captures) {
      next[row][col] = EMPTY;
    }

    // Mette la pedina nella casella finale, promuovendola se serve.
    next[endRow][endCol] = promote(piece, endRow);
    return next;
  }

  // Conta quante pedine ha un giocatore.
  function countPieces(board, player) {
    return pieces(board, player).length;
  }

  // Applica il soffio: se un giocatore poteva catturare ma fa una mossa
  // normale, viene rimossa una pedina obbligata alla cattura.
  function applySoffio(board, playedMove, forcedCaptures) {
    if (!forcedCaptures.length || playedMove.captures.length) return null;

    const playedStart = moveStart(playedMove);
    const playedEnd = moveEnd(playedMove);
    const forcedStarts = forcedCaptures.map(moveStart);
    const movedPieceWasForced = forcedStarts.some((square) => sameSquare(square, playedStart));
    const blownSquare = movedPieceWasForced ? playedEnd : forcedStarts[0];
    const [row, col] = blownSquare;

    if (board[row] && board[row][col] !== EMPTY) {
      board[row][col] = EMPTY;
      return blownSquare;
    }
    return null;
  }

  // Controlla se qualcuno ha vinto.
  function winner(board, currentPlayer) {
    // Se il bianco non ha pezzi, vince il nero.
    if (countPieces(board, WHITE) === 0) return BLACK;

    // Se il nero non ha pezzi, vince il bianco.
    if (countPieces(board, BLACK) === 0) return WHITE;

    // Se il giocatore di turno non puo' muovere, perde.
    if (legalMoves(board, currentPlayer).length === 0) return -currentPlayer;

    // Altrimenti la partita continua.
    return null;
  }

  // Valuta la scacchiera contando il materiale.
  // Le dame valgono piu' delle pedine semplici.
  function materialScore(board, player) {
    let raw = 0;
    for (const row of board) {
      for (const piece of row) {
        if (piece === WHITE_MAN) raw += 1;
        if (piece === WHITE_KING) raw += 2.2;
        if (piece === BLACK_MAN) raw -= 1;
        if (piece === BLACK_KING) raw -= 2.2;
      }
    }
    return raw * player;
  }

  // Valutazione piu' completa usata dalla macchina.
  // Considera materiale, posizione centrale e avanzamento verso la promozione.
  function positionalScore(board, player) {
    let score = materialScore(board, player) * 10;
    for (let row = 0; row < SIZE; row += 1) {
      for (let col = 0; col < SIZE; col += 1) {
        const piece = board[row][col];
        if (owner(piece) === player) {
          score += 0.18 * (3.5 - Math.abs(col - 3.5));
          if (!isKing(piece)) score += 0.12 * (player === WHITE ? 7 - row : row);
        } else if (owner(piece) === -player) {
          score -= 0.18 * (3.5 - Math.abs(col - 3.5));
        }
      }
    }
    return score;
  }

  // Sceglie la mossa della macchina usando minimax con potatura alpha-beta.
  // In parole semplici: prova le mosse future e sceglie quella che sembra migliore.
  function minimaxMove(board, player, depth) {
    const moves = legalMoves(board, player);
    if (!moves.length) return null;

    // Funzione interna che simula il futuro della partita.
    function search(nodeBoard, currentPlayer, remainingDepth, alpha, beta) {
      const win = winner(nodeBoard, currentPlayer);

      // Vittoria della macchina: punteggio molto alto.
      if (win === player) return 10000 + remainingDepth;

      // Sconfitta della macchina: punteggio molto basso.
      if (win === -player) return -10000 - remainingDepth;

      // Se siamo arrivati alla profondita' scelta, valuta la posizione senza continuare.
      if (remainingDepth === 0) return positionalScore(nodeBoard, player);

      // Ordina prima le catture, perche' spesso sono mosse importanti.
      const nodeMoves = legalMoves(nodeBoard, currentPlayer)
        .slice()
        .sort((a, b) => b.captures.length - a.captures.length);

      // Turno della macchina: cerca il punteggio massimo.
      if (currentPlayer === player) {
        let value = -Infinity;
        for (const move of nodeMoves) {
          value = Math.max(value, search(applyMoveToBoard(nodeBoard, move), -currentPlayer, remainingDepth - 1, alpha, beta));
          alpha = Math.max(alpha, value);

          // Potatura: se questa strada non puo' migliorare, smette di provarla.
          if (alpha >= beta) break;
        }
        return value;
      }

      // Turno dell'avversario: assume che l'avversario scelga la mossa peggiore per la macchina.
      let value = Infinity;
      for (const move of nodeMoves) {
        value = Math.min(value, search(applyMoveToBoard(nodeBoard, move), -currentPlayer, remainingDepth - 1, alpha, beta));
        beta = Math.min(beta, value);

        // Anche qui taglia i rami inutili della ricerca.
        if (alpha >= beta) break;
      }
      return value;
    }

    // Prova ogni mossa possibile e conserva quella con il punteggio migliore.
    let bestScore = -Infinity;
    let bestMoves = [];
    const ordered = moves.slice().sort((a, b) => b.captures.length - a.captures.length);
    for (const move of ordered) {
      const score = search(applyMoveToBoard(board, move), -player, depth - 1, -Infinity, Infinity);
      if (score > bestScore) {
        bestScore = score;
        bestMoves = [move];
      } else if (score === bestScore) {
        bestMoves.push(move);
      }
    }

    // Se piu' mosse hanno lo stesso punteggio, ne sceglie una a caso.
    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  // Recupera gli elementi HTML che il JavaScript deve leggere o aggiornare.
  function elements() {
    return {
      board: document.getElementById("board"),
      status: document.getElementById("status"),
      whiteCount: document.getElementById("white-count"),
      blackCount: document.getElementById("black-count"),
      side: document.getElementById("human-side"),
      difficulty: document.getElementById("difficulty"),
      newGame: document.getElementById("new-game"),
      help: document.getElementById("move-help"),
      log: document.getElementById("move-log"),
    };
  }

  // Restituisce le classi CSS giuste per disegnare una pedina.
  function pieceClass(piece) {
    const color = piece > 0 ? "white" : "black";
    const crown = isKing(piece) ? " king" : "";
    return `piece ${color}${crown}`;
  }

  // Ridisegna tutta la pagina in base allo stato attuale.
  // Viene chiamata dopo ogni click, ogni mossa e ogni nuova partita.
  function render() {
    const el = elements();

    // Svuota la scacchiera HTML prima di ricrearla aggiornata.
    el.board.innerHTML = "";

    // Se l'utente ha selezionato una pedina, trova le sue mosse legali.
    const movesFromSelected = state.selected
      ? legalMoves(state.board, state.humanPlayer, true).filter((move) => sameSquare(moveStart(move), state.selected))
      : [];

    // Mappa delle caselle di arrivo, usata per colorare le destinazioni cliccabili.
    const targets = new Map(movesFromSelected.map((move) => [moveEnd(move).join(","), move]));

    // Caselle dell'ultima mossa, usate per evidenziarla in giallo.
    const lastSquares = new Set(state.lastMove ? state.lastMove.path.map((square) => square.join(",")) : []);

    // Crea i 64 bottoni della scacchiera.
    for (let row = 0; row < SIZE; row += 1) {
      for (let col = 0; col < SIZE; col += 1) {
        const button = document.createElement("button");
        const key = `${row},${col}`;
        const piece = state.board[row][col];
        button.type = "button";
        button.className = `square ${playable(row, col) ? "dark" : "light"}`;
        button.dataset.row = String(row);
        button.dataset.col = String(col);
        button.setAttribute("role", "gridcell");
        button.setAttribute("aria-label", squareName(row, col));

        // Colora la casella selezionata.
        if (state.selected && sameSquare(state.selected, [row, col])) button.classList.add("selected");

        // Colora le destinazioni disponibili: verde per mossa normale, rosso per cattura.
        if (targets.has(key)) button.classList.add(targets.get(key).captures.length ? "capture-target" : "target");

        // Evidenzia l'ultima mossa.
        if (lastSquares.has(key)) button.classList.add("last");

        // Disattiva i click quando non e' il turno dell'utente o la partita e' finita.
        button.disabled = state.gameOver || state.thinking || state.currentPlayer !== state.humanPlayer;

        // Se nella casella c'e' una pedina, crea il disco grafico.
        if (piece !== EMPTY) {
          const disk = document.createElement("span");
          disk.className = pieceClass(piece);
          button.appendChild(disk);
        }

        // Collega il click della casella alla funzione che gestisce la mossa.
        button.addEventListener("click", () => onSquareClick(row, col));
        el.board.appendChild(button);
      }
    }

    // Aggiorna conteggio pezzi, messaggi e cronologia.
    el.whiteCount.textContent = String(countPieces(state.board, WHITE));
    el.blackCount.textContent = String(countPieces(state.board, BLACK));
    el.status.textContent = statusText();
    el.help.textContent = helpText(movesFromSelected);
    renderLog();
  }

  // Decide il testo principale da mostrare sopra la scacchiera.
  function statusText() {
    const win = winner(state.board, state.currentPlayer);

    // Mostra messaggi diversi in base al risultato o al turno.
    if (win === state.humanPlayer) return "Hai vinto.";
    if (win === -state.humanPlayer) return "Ha vinto la macchina.";
    if (win) return "Partita terminata.";
    if (state.thinking) return "La macchina sta pensando...";
    return state.currentPlayer === state.humanPlayer ? "Tocca a te." : "Tocca alla macchina.";
  }

  // Decide il testo di aiuto nel pannello "Mossa".
  // Serve a dire all'utente cosa deve cliccare.
  function helpText(movesFromSelected) {
    if (state.gameOver) return "Premi Nuova partita per ricominciare.";
    if (state.thinking) return "Attendi la mossa della macchina.";
    if (state.currentPlayer !== state.humanPlayer) return "La macchina giochera' automaticamente.";

    // Se non e' selezionata nessuna pedina, spiega se c'e' una cattura obbligatoria.
    if (!state.selected) {
      const captures = captureMoves(state.board, state.humanPlayer).some((move) => move.captures.length);
      return captures
        ? "Cattura obbligatoria: se giochi una mossa normale, la pedina obbligata viene soffiata."
        : "Clicca una tua pedina, poi una casella evidenziata.";
    }

    // Se una pedina e' selezionata, mostra le sue mosse disponibili.
    return movesFromSelected.length
      ? `Mosse disponibili: ${movesFromSelected.map(moveText).join(", ")}`
      : "Questa pedina non ha mosse legali.";
  }

  // Ridisegna la cronologia delle mosse nel pannello laterale.
  function renderLog() {
    const el = elements();

    // Svuota la lista e la ricrea partendo da state.history.
    el.log.innerHTML = "";
    for (const item of state.history) {
      const li = document.createElement("li");
      li.textContent = item;
      el.log.appendChild(li);
    }
  }

  // Gestisce il click su una casella della scacchiera.
  function onSquareClick(row, col) {
    // Ignora i click se la partita e' finita, se la macchina pensa o se non e' il turno umano.
    if (state.gameOver || state.thinking || state.currentPlayer !== state.humanPlayer) return;

    const clicked = [row, col];

    // Calcola tutte le mosse dell'utente, includendo quelle che causano soffio.
    const legal = legalMoves(state.board, state.humanPlayer, true);

    // Elenco delle caselle da cui parte almeno una mossa legale.
    const starts = legal.map(moveStart);

    // Primo click: seleziona una pedina solo se quella pedina puo' muovere.
    if (!state.selected) {
      if (starts.some((square) => sameSquare(square, clicked))) state.selected = clicked;
      render();
      return;
    }

    // Se clicchi di nuovo la stessa pedina, la deselezioni.
    if (sameSquare(state.selected, clicked)) {
      state.selected = null;
      render();
      return;
    }

    // Controlla se il secondo click e' una destinazione valida per la pedina selezionata.
    const candidates = legal.filter((move) => sameSquare(moveStart(move), state.selected) && sameSquare(moveEnd(move), clicked));

    // Se la destinazione e' valida, gioca la mossa umana e poi avvia la macchina.
    if (candidates.length) {
      const move = candidates.sort((a, b) => b.captures.length - a.captures.length)[0];
      playMove(move, "Tu");
      render();
      queueMachineMove();
      return;
    }

    // Se invece clicchi un'altra tua pedina valida, cambia selezione.
    if (starts.some((square) => sameSquare(square, clicked))) state.selected = clicked;
    render();
  }

  // Applica una mossa alla partita vera.
  function playMove(move, label) {
    const forcedCaptures = captureMoves(state.board, state.currentPlayer);

    // Aggiorna la scacchiera.
    state.board = applyMoveToBoard(state.board, move);

    const blownSquare = applySoffio(state.board, move, forcedCaptures);

    // Cambia turno.
    state.currentPlayer *= -1;

    // Pulisce la selezione e ricorda l'ultima mossa.
    state.selected = null;
    state.lastMove = move;

    // Aggiunge la mossa alla cronologia.
    const soffioText = blownSquare ? ` (soffio in ${squareName(...blownSquare)})` : "";
    state.history.push(`${label}: ${moveText(move)}${soffioText}`);

    // Controlla se la partita e' finita.
    const win = winner(state.board, state.currentPlayer);
    state.gameOver = Boolean(win) || state.history.length >= MAX_MOVES;
  }

  // Fa giocare la macchina dopo la mossa dell'utente.
  function queueMachineMove() {
    // Se non serve la mossa della macchina, non fa nulla.
    if (state.gameOver || state.currentPlayer === state.humanPlayer) return;

    // Mostra lo stato "sta pensando".
    state.thinking = true;
    render();

    // Piccola pausa per far vedere il cambio turno e non bloccare subito l'interfaccia.
    window.setTimeout(() => {
      // La difficolta' e' la profondita' di ricerca minimax.
      const depth = Number(elements().difficulty.value);

      // Calcola la miglior mossa trovata dalla macchina.
      const move = minimaxMove(state.board, state.currentPlayer, depth);

      // Se trova una mossa, la gioca. Se non la trova, la partita finisce.
      if (move) playMove(move, "Macchina");
      else state.gameOver = true;

      // Finito il calcolo, riabilita la scacchiera.
      state.thinking = false;
      render();
    }, 180);
  }

  // Inizia una nuova partita.
  function newGame() {
    // Crea la scacchiera iniziale.
    state.board = initialBoard();

    // Il bianco muove sempre per primo.
    state.currentPlayer = WHITE;

    // Legge dal menu se l'utente vuole bianco o nero.
    state.humanPlayer = elements().side.value === "white" ? WHITE : BLACK;

    // Azzera selezione, ultima mossa, cronologia e stato finale.
    state.selected = null;
    state.lastMove = null;
    state.history = [];
    state.gameOver = false;
    state.thinking = false;

    // Disegna la partita nuova.
    render();

    // Se l'utente ha scelto nero, la macchina gioca subito il bianco.
    queueMachineMove();
  }

  // Avvia il gioco quando la pagina e' pronta.
  function boot() {
    const el = elements();

    // Se manca la scacchiera nell'HTML, interrompe per evitare errori.
    if (!el.board) return;

    // Collega i controlli HTML alle funzioni JavaScript.
    el.newGame.addEventListener("click", newGame);
    el.side.addEventListener("change", newGame);

    // Crea subito la prima partita.
    newGame();
  }

  // Nel browser aspetta che l'HTML sia caricato, poi avvia boot().
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", boot);
  }

  // Questa parte serve solo per i test con Node.
  // Nel browser non viene usata, ma permette di verificare le funzioni da terminale.
  if (typeof module !== "undefined") {
    module.exports = {
      EMPTY,
      WHITE,
      BLACK,
      initialBoard,
      captureMoves,
      legalMoves,
      applyMoveToBoard,
      applySoffio,
      winner,
      minimaxMove,
      moveText,
      countPieces,
    };
  }
}());
