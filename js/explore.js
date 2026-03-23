// ── Explore tab — route list, selection, board display ─────────────────────────

var routeOffset      = 0;
var routeGradeFilter = '';
var routeSearchQ     = '';
var routeBetaOnly    = false;
var currentRoute     = null;
var searchTimer      = null;

// ── Route List ─────────────────────────────────────────────────────────────────

async function loadRoutes(reset) {
  if (reset) {
    routeOffset = 0;
    document.getElementById('route-list').innerHTML =
      '<div class="loading-overlay" style="position:relative;height:200px;">' +
      '<div class="spinner"></div><span class="loading-text">Loading…</span></div>';
  }

  var params = new URLSearchParams({
    limit: 40,
    offset: routeOffset,
    angle: currentAngle,
    min_sends: 5,
  });
  if (routeGradeFilter) params.set('grade', routeGradeFilter);
  if (routeSearchQ)     params.set('q', routeSearchQ);
  if (routeBetaOnly)    params.set('has_pose', '1');

  try {
    var res  = await fetch(API + '/api/routes?' + params);
    var data = await res.json();
    document.getElementById('route-count').textContent = data.total.toLocaleString();

    if (reset) document.getElementById('route-list').innerHTML = '';

    var list = document.getElementById('route-list');
    data.routes.forEach(function(r) {
      var card = document.createElement('div');
      card.className = 'route-card';
      card.dataset.id = r.id;
      var gc = gradeColor(r.community_grade);
      card.style.setProperty('--grade-color', gc);
      card.innerHTML =
        '<div class="rc-top">' +
          '<span class="rc-name">' + (r.name || 'Unnamed') + '</span>' +
          '<span style="display:flex;align-items:center;gap:5px;flex-shrink:0;">' +
            (r.has_pose ? '<span title="Has beta video data" style="font-size:9px;background:rgba(139,92,246,0.15);border:1px solid var(--purple);color:var(--purple);border-radius:3px;padding:1px 5px;font-family:\'Space Mono\',monospace;letter-spacing:0.05em;">BETA</span>' : '') +
            '<span class="rc-grade" style="background:' + gc + ';">' + (r.community_grade || '?') + '</span>' +
          '</span>' +
        '</div>' +
        '<div class="rc-meta">' +
          '<span>' + r.board_angle_deg + '°</span>' +
          '<span><b>' + (r.send_count || 0).toLocaleString() + '</b> sends</span>' +
          (r.setter_name ? '<span>by ' + r.setter_name + '</span>' : '') +
        '</div>';
      card.addEventListener('click', function() { selectRoute(r.id, card); });
      list.appendChild(card);
    });

    if (data.routes.length === 40) {
      var btn = document.createElement('button');
      btn.className = 'load-more-btn';
      btn.textContent = 'Load more…';
      btn.onclick = function() { routeOffset += 40; btn.remove(); loadRoutes(false); };
      list.appendChild(btn);
    }
  } catch(e) {
    document.getElementById('route-list').innerHTML =
      '<div class="empty-state">Could not load routes.<br>Is the API running?</div>';
  }
}

function reloadRoutes() { loadRoutes(true); }

function filterGrade(el, grade) {
  document.querySelectorAll('#grade-filters .filter-chip').forEach(function(c) {
    c.classList.remove('active');
  });
  el.classList.add('active');
  routeGradeFilter = grade;
  reloadRoutes();
}

function toggleBetaFilter(el) {
  routeBetaOnly = !routeBetaOnly;
  el.classList.toggle('beta-active', routeBetaOnly);
  reloadRoutes();
}

function debounceSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(function() {
    routeSearchQ = document.getElementById('route-search').value.trim();
    reloadRoutes();
  }, 350);
}

// ── Route Selection ────────────────────────────────────────────────────────────

async function selectRoute(routeId, cardEl) {
  document.querySelectorAll('.route-card').forEach(function(c) { c.classList.remove('selected'); });
  cardEl.classList.add('selected');

  try {
    var res   = await fetch(API + '/api/route/' + routeId);
    currentRoute = await res.json();
    renderRouteOnBoard(currentRoute);
    renderRouteDetail(currentRoute);
  } catch(e) {
    console.error('Failed to load route:', e);
  }
}

// ── Board Display ──────────────────────────────────────────────────────────────

function renderRouteOnBoard(route) {
  var board = document.getElementById('board');
  if (!board) return;

  board.querySelectorAll('.hold').forEach(function(h) {
    h.className = 'hold hold-' + h.dataset.type;
  });

  var svg = document.getElementById('arrows-svg');
  svg.innerHTML = '';
  if (!route || !route.holds) return;

  var size    = board.offsetWidth;
  var handSeq = [];

  route.holds.forEach(function(rh) {
    if (rh.position_x_cm == null) return;
    var el = board.querySelector('.hold[data-x="' + Math.round(rh.position_x_cm) + '"][data-y="' + Math.round(rh.position_y_cm) + '"]');
    if (!el) return;

    var roleClass = { start:'role-start', hand:'role-hand', foot:'role-foot-sel', finish:'role-finish' }[rh.role] || 'role-hand';
    el.classList.add(roleClass);

    if (rh.role !== 'foot' && rh.x_pct != null) {
      handSeq.push({ x_pct: rh.x_pct, y_pct: rh.y_pct, seq: rh.hand_sequence || 999, role: rh.role });
    }
  });

  handSeq.sort(function(a, b) { return a.seq - b.seq; });
  for (var i = 1; i < handSeq.length; i++) {
    var h1 = handSeq[i - 1], h2 = handSeq[i];
    drawArrow(svg,
      h1.x_pct / 100 * size, h1.y_pct / 100 * size,
      h2.x_pct / 100 * size, h2.y_pct / 100 * size
    );
  }
}

// ── Route Detail ──────────────────────────────────────────────────────────────

function renderRouteDetail(route) {
  var el = document.getElementById('route-detail');
  var holds = route.holds || [];
  var handHolds = holds
    .filter(function(h) { return h.role !== 'foot' && h.position_x_cm != null; })
    .sort(function(a, b) { return (a.hand_sequence || 99) - (b.hand_sequence || 99); });

  el.innerHTML =
    '<div class="route-detail">' +
      '<div class="rd-title">' + (route.name || 'Unnamed Route') + '</div>' +
      '<div class="rd-meta">' + (route.setter_name ? 'by ' + route.setter_name : '') + '</div>' +
      '<div class="rd-stat"><span class="rd-stat-key">Grade</span><span class="rd-stat-val">' + (route.community_grade || '—') + '</span></div>' +
      '<div class="rd-stat"><span class="rd-stat-key">Angle</span><span class="rd-stat-val">' + route.board_angle_deg + '°</span></div>' +
      '<div class="rd-stat"><span class="rd-stat-key">Sends</span><span class="rd-stat-val">' + (route.send_count || 0).toLocaleString() + '</span></div>' +
      '<div class="rd-stat"><span class="rd-stat-key">Quality</span><span class="rd-stat-val">' + (route.avg_quality_rating ? route.avg_quality_rating.toFixed(1) : '—') + '</span></div>' +
      '<div class="rd-stat"><span class="rd-stat-key">Holds</span><span class="rd-stat-val">' + holds.length + '</span></div>' +
      '<div style="margin-top:14px;">' +
        '<div class="creator-label">Hold Sequence</div>' +
        '<div class="move-chips" style="margin-top:6px;">' +
          handHolds.map(function(h, i) {
            var label = h.role === 'start' ? 'S' : h.role === 'finish' ? 'F' : i;
            var xPct  = h.x_pct  != null ? h.x_pct  : '';
            var yPct  = h.y_pct  != null ? h.y_pct  : '';
            return '<span class="move-chip ' + h.role + '" style="cursor:pointer;" ' +
              'data-xpct="' + xPct + '" data-ypct="' + yPct + '" ' +
              'onclick="_onMoveChipClick(this)">' + label + '</span>';
          }).join('') +
        '</div>' +
      '</div>' +
      '<div style="margin-top:14px;">' +
        '<div class="creator-label">Type Breakdown</div>' +
        '<div style="margin-top:6px;font-size:11px;line-height:2;">' + buildTypeBreakdown(holds) + '</div>' +
      '</div>' +
      '<div id="route-dna-section" style="margin-top:14px;">' +
        '<div class="creator-label">Route DNA <span style="font-size:9px;color:var(--mist);">ML</span></div>' +
        '<div id="route-dna-bars" style="margin-top:6px;">' +
          '<div style="font-size:10px;color:var(--mist);">Loading…</div>' +
        '</div>' +
      '</div>' +
      '<div class="pose-phases" id="pose-phases-section" style="display:none;">' +
        '<div class="creator-label">Move Breakdown <span style="color:var(--purple);font-size:9px;margin-left:4px;">BETA VIDEO</span></div>' +
        '<div class="pose-phase-grid" id="pose-phase-grid"></div>' +
      '</div>' +
      '<div style="margin-top:16px;">' +
        '<button id="animate-route-btn" onclick="toggleRouteAnimation(currentRoute)" ' +
          'style="width:100%;padding:8px;background:var(--panel);border:1px solid var(--border);' +
          'border-radius:6px;color:var(--chalk);font-size:11px;cursor:pointer;' +
          'display:flex;align-items:center;justify-content:center;gap:6px;">' +
          '<span style="font-size:13px;">🧗</span> Animate Route' +
        '</button>' +
      '</div>' +
    '</div>';

  // Stick figure from start hold positions, then try loading pose animation
  renderRouteFigure(route);
  if (route.external_id) {
    loadPoseAnimation(route.external_id);
  } else {
    stopPosePlay();
    poseFrames = [];
    var btn = document.getElementById('pose-play-btn');
    var ctr = document.getElementById('pose-frame-counter');
    if (btn) { btn.disabled = true; btn.textContent = '▶ Play Beta'; }
    if (ctr) ctr.textContent = '';
  }

  // Load ML-predicted route DNA
  loadRouteDNA(route);
}

async function loadRouteDNA(route) {
  var barsEl = document.getElementById('route-dna-bars');
  if (!barsEl) return;

  var holds = (route.holds || [])
    .filter(function(h) { return h.position_x_cm != null; })
    .map(function(h) {
      return { x_cm: h.position_x_cm, y_cm: h.position_y_cm, role: h.role, hold_type: h.hold_type };
    });

  if (holds.length < 2) {
    barsEl.innerHTML = '<div style="font-size:10px;color:var(--mist);">Not enough hold data.</div>';
    return;
  }

  try {
    var res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ holds: holds, angle: route.board_angle_deg }),
    });
    var data = await res.json();
    if (data.error) { barsEl.innerHTML = ''; return; }

    var dna = data.dna || {};
    var confPct = Math.round((data.confidence || 0) * 100);
    var gc = gradeColor(data.grade || '');

    var dnaKeys = [
      ['reach',      'Reach',      'dna-reach'],
      ['sustained',  'Sustained',  'dna-sustained'],
      ['dynamic',    'Dynamic',    'dna-dynamic'],
      ['lateral',    'Lateral',    'dna-lateral'],
      ['complexity', 'Complexity', 'dna-complexity'],
      ['power',      'Power',      'dna-power'],
    ];

    var predRow = '<div class="route-dna-pred">' +
      '<span style="color:var(--mist);font-size:10px;">ML Prediction:</span>' +
      '<span class="ml-grade" style="color:' + gc + ';">' + (data.grade || '—') + '</span>' +
      '<span style="color:var(--mist);font-size:10px;">(' + confPct + '%)</span>' +
    '</div>';

    var barRows = dnaKeys.map(function(k) {
      var val = dna[k[0]] || 0;
      return '<div class="dna-row">' +
        '<span class="dna-label">' + k[1] + '</span>' +
        '<div class="dna-track"><div class="dna-fill ' + k[2] + '" style="width:' + val + '%"></div></div>' +
        '<span class="dna-val">' + val + '</span>' +
      '</div>';
    }).join('');

    barsEl.innerHTML = predRow + barRows;
  } catch(e) {
    // API down — fail silently
    if (barsEl) barsEl.innerHTML = '';
  }
}

function clearRouteDetail() {
  stopRouteAnimation();
  document.getElementById('route-detail').innerHTML =
    '<div class="empty-state"><div class="big">→</div>Select a route to see its holds and stats</div>';
  document.getElementById('route-figure-panel').style.display = 'none';
}

function _onMoveChipClick(chipEl) {
  // Remove selected from all chips in the same .move-chips container
  var container = chipEl.closest('.move-chips');
  if (container) {
    container.querySelectorAll('.move-chip').forEach(function(c) { c.classList.remove('selected'); });
  }
  chipEl.classList.add('selected');

  var xPct = parseFloat(chipEl.dataset.xpct);
  var yPct = parseFloat(chipEl.dataset.ypct);
  if (isNaN(xPct) || isNaN(yPct)) return;
  _showMoveOnFigure(xPct, yPct);
}

function _showMoveOnFigure(xPct, yPct) {
  stopPosePlay();
  var svg = document.getElementById('stick-figure-svg');
  if (!svg) return;
  var angle = (currentRoute && currentRoute.board_angle_deg) || 40;
  var joints = computeHoldReachPose(xPct, yPct, angle);
  renderStickFigure(svg, joints, +svg.getAttribute('width'), +svg.getAttribute('height'));
}

function buildTypeBreakdown(holds) {
  var counts = {};
  holds.forEach(function(h) {
    if (h.hold_type) counts[h.hold_type] = (counts[h.hold_type] || 0) + 1;
  });
  return Object.entries(counts).sort(function(a, b) { return b[1] - a[1]; })
    .map(function([t, n]) {
      return '<span style="color:var(--chalk)">' + t + '</span> <span style="color:var(--mist)">' + n + ' holds</span><br>';
    }).join('') || '<span style="color:var(--mist)">No hold type data</span>';
}
