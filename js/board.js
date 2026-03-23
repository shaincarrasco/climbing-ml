// ── Board state & rendering ────────────────────────────────────────────────────

var boardHolds   = [];
var boardLoaded  = false;
var currentAngle = 40;
var creatorAngle = 40;
var hiddenTypes  = new Set();
var currentBoard = 'original';
var currentSet   = 'all';
var boardMeta    = {};

// ── Board Meta ─────────────────────────────────────────────────────────────────

async function loadBoardMeta() {
  try {
    var res = await fetch(API + '/api/boards');
    boardMeta = await res.json();
    renderBoardSwitcher();
  } catch(e) { /* API offline */ }
}

function renderBoardSwitcher() {
  var container = document.getElementById('board-switcher');
  if (!container) return;
  container.innerHTML = '';
  Object.entries(boardMeta).forEach(function([key, info]) {
    var btn = document.createElement('button');
    btn.className = 'angle-btn' + (key === currentBoard ? ' active' : '');
    btn.textContent = key === 'original' ? 'Original' : 'Homewall';
    btn.title = info.description || '';
    btn.onclick = function() { switchBoard(key); };
    container.appendChild(btn);
  });
  renderSetSwitcher();
}

function renderSetSwitcher() {
  var container = document.getElementById('set-switcher');
  if (!container) return;
  container.innerHTML = '';
  var meta = boardMeta[currentBoard];
  if (!meta) return;

  var LABELS = {
    'all': 'All', 'bolt-ons': 'Bolt Ons', 'screw-ons': 'Screw Ons',
    'mainline': 'Mainline', 'auxiliary': 'Auxiliary', 'full-ride': 'Full Ride',
  };
  meta.sets.forEach(function(setKey) {
    var btn = document.createElement('button');
    btn.className = 'filter-chip' + (setKey === currentSet ? ' active' : '');
    btn.textContent = LABELS[setKey] || setKey;
    btn.onclick = function() { switchSet(setKey); };
    container.appendChild(btn);
  });

  var desc = document.getElementById('board-char-desc');
  if (desc) desc.textContent = meta.hold_character || '';
}

async function switchBoard(boardType) {
  currentBoard = boardType;
  currentSet   = (boardMeta[boardType] && boardMeta[boardType].default_set) || 'all';
  renderBoardSwitcher();
  await loadBoardHolds();
  if (currentRoute) { currentRoute = null; clearRouteDetail(); }
}

async function switchSet(setKey) {
  currentSet = setKey;
  document.querySelectorAll('#set-switcher .filter-chip').forEach(function(b) {
    var LABELS = { 'all':'All','bolt-ons':'Bolt Ons','screw-ons':'Screw Ons',
                   'mainline':'Mainline','auxiliary':'Auxiliary','full-ride':'Full Ride' };
    b.classList.toggle('active', b.textContent === (LABELS[setKey] || setKey));
  });
  await loadBoardHolds();
}

// ── Board Hold Loading ─────────────────────────────────────────────────────────

async function loadBoardHolds() {
  var loadingEl = document.getElementById('board-loading');
  if (loadingEl) loadingEl.style.display = 'flex';

  try {
    var res  = await fetch(API + '/api/board?board_type=' + currentBoard + '&set_filter=' + currentSet);
    var data = await res.json();
    boardHolds = data.holds || [];

    var charEl = document.getElementById('board-char-desc');
    if (charEl) charEl.textContent = data.hold_character || '';

    var tcEl = document.getElementById('type-counts');
    if (tcEl && data.type_counts) {
      tcEl.innerHTML = Object.entries(data.type_counts)
        .filter(function([, n]) { return n > 0; })
        .sort(function(a, b) { return b[1] - a[1]; })
        .map(function([t, n]) {
          return '<span class="hc-chip"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--hold-' + t + ');margin-right:4px;"></span>' + t + ' <b>' + n + '</b></span>';
        }).join('');
    }

    renderBoard('board', false);
    if (loadingEl) loadingEl.style.display = 'none';
    boardLoaded = true;
  } catch(e) {
    if (loadingEl) loadingEl.innerHTML =
      '<div style="color:var(--coral);font-size:12px;text-align:center;padding:20px;">' +
      'API offline — run: <code style="font-family:\'Space Mono\',monospace;">python3 api/app.py</code></div>';
  }
}

// ── Board Rendering ────────────────────────────────────────────────────────────

function renderBoard(boardId, clickable) {
  var board = document.getElementById(boardId);
  if (!board) return;

  var wrapper = board.parentElement;
  var wh  = wrapper.offsetHeight - 36;
  var ww  = wrapper.offsetWidth  - 36;
  // Leave a few % margin so perspective corners stay within bounds
  var size = Math.floor(Math.min(wh, ww, 680) * 0.97);
  if (size <= 0) return; // board not visible yet
  board.style.width  = size + 'px';
  board.style.height = size + 'px';

  board.querySelectorAll('.hold').forEach(function(h) { h.remove(); });

  var frag = document.createDocumentFragment();
  boardHolds.forEach(function(hold) {
    var el = document.createElement('div');
    el.className  = 'hold hold-' + hold.hold_type;
    el.dataset.id  = hold.id;
    el.dataset.pid = hold.pid;
    el.dataset.x   = hold.x;
    el.dataset.y   = hold.y;
    el.dataset.type = hold.hold_type;
    el.style.cssText = 'left:' + hold.x_pct + '%;top:' + hold.y_pct + '%';
    el.title = hold.hold_type + '  (' + hold.x + ', ' + hold.y + ')';

    if (clickable) {
      el.addEventListener('click', function() { onCreatorHoldClick(el, hold); });
      el.addEventListener('mouseenter', function(e) { showHoldPreview(el, hold, e); });
      el.addEventListener('mouseleave', hideHoldPreview);
    }

    if (hiddenTypes.has(hold.hold_type)) el.style.display = 'none';
    frag.appendChild(el);
  });
  board.appendChild(frag);
}

// ── Angle Controls ─────────────────────────────────────────────────────────────

function setAngle(a) {
  currentAngle = a;
  document.getElementById('board').setAttribute('data-angle', a);
  document.querySelectorAll('#angle-btns .angle-btn').forEach(function(b) {
    b.classList.toggle('active', parseInt(b.textContent) === a);
  });
  if (currentRoute) renderRouteOnBoard(currentRoute);
  reloadRoutes();
}

function setCreatorAngle(a) {
  creatorAngle = a;
  var cb = document.getElementById('creator-board');
  if (cb) cb.setAttribute('data-angle', a);
  document.querySelectorAll('#creator-angle-btns .angle-btn').forEach(function(b) {
    b.classList.toggle('active', parseInt(b.textContent) === a);
  });
  triggerPredict();
}

function toggleType(type) {
  if (hiddenTypes.has(type)) hiddenTypes.delete(type);
  else hiddenTypes.add(type);
  document.querySelectorAll('.hold-' + type).forEach(function(h) {
    h.style.display = hiddenTypes.has(type) ? 'none' : '';
  });
}

// ── Arrow Drawing ──────────────────────────────────────────────────────────────

function drawArrow(svg, x1, y1, x2, y2) {
  var ns = 'http://www.w3.org/2000/svg';
  var path = document.createElementNS(ns, 'path');
  var mx = (x1 + x2) / 2;
  var my = (y1 + y2) / 2 - 12;
  path.setAttribute('d', 'M ' + x1 + ' ' + y1 + ' Q ' + mx + ' ' + my + ' ' + x2 + ' ' + y2);
  path.setAttribute('class', 'move-arrow');

  var angle = Math.atan2(y2 - my, x2 - mx);
  var ah = 7;
  var ar = document.createElementNS(ns, 'polygon');
  ar.setAttribute('points',
    x2 + ',' + y2 + ' ' +
    (x2 - ah * Math.cos(angle - 0.4)) + ',' + (y2 - ah * Math.sin(angle - 0.4)) + ' ' +
    (x2 - ah * Math.cos(angle + 0.4)) + ',' + (y2 - ah * Math.sin(angle + 0.4))
  );
  ar.setAttribute('fill', 'var(--teal)');
  ar.setAttribute('opacity', '0.7');
  svg.appendChild(path);
  svg.appendChild(ar);
}
