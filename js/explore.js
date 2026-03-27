// ── Explore tab — route list, selection, board display ─────────────────────────

var routeOffset      = 0;
var routeGradeFilter = '';
var routeSearchQ     = '';
var routeBetaOnly    = false;
var routeHoldFilter  = '';   // 'crimp' | 'jug' | 'sloper' | 'pinch' | ''
var routeSortSends   = false;
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
  if (routeHoldFilter)  params.set('hold_type', routeHoldFilter);
  params.set('sort', routeSortSends ? 'sends' : 'quality');

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

function toggleHoldTypeFilter(type) {
  routeHoldFilter = (routeHoldFilter === type) ? '' : type;
  ['crimp','jug','sloper','pinch'].forEach(function(t) {
    var btn = document.getElementById('ht-' + t);
    if (btn) btn.classList.toggle('active', routeHoldFilter === t);
  });
  reloadRoutes();
}

function toggleSortSends(btn) {
  routeSortSends = !routeSortSends;
  btn.classList.toggle('active', routeSortSends);
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
    // Store difficulty score for PD calibration when user logs this route
    window._lastRouteDifficultyScore = currentRoute.difficulty_score || null;
    renderRouteOnBoard(currentRoute);
    renderRouteDetail(currentRoute);
  } catch(e) {
    console.error('Failed to load route:', e);
  }
}

async function selectRouteById(routeId) {
  try {
    var res = await fetch(API + '/api/route/' + routeId);
    currentRoute = await res.json();
    window._lastRouteDifficultyScore = currentRoute.difficulty_score || null;
    renderRouteOnBoard(currentRoute);
    renderRouteDetail(currentRoute);
    // Highlight the matching card if it's visible in the list
    document.querySelectorAll('.route-card').forEach(function(c) {
      c.classList.toggle('selected', c.dataset.id == routeId);
    });
  } catch(e) {
    console.error('Failed to load route by id:', e);
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

  var routeId   = route.external_id || route.id || '';
  var routeName = route.name || 'Unnamed Route';
  var routeGrade = route.community_grade || '';
  var isSaved   = typeof isRouteSaved === 'function' && isRouteSaved(routeId);

  el.innerHTML =
    '<div class="route-detail">' +
      '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">' +
        '<div>' +
          '<div class="rd-title" id="route-detail-name">' + (routeName) + '</div>' +
          '<div class="rd-meta">' + (route.setter_name ? 'by ' + route.setter_name : '') + '</div>' +
        '</div>' +
        '<button id="save-route-btn" onclick="typeof toggleSaveRoute===\'function\' && toggleSaveRoute(\'' + routeId + '\',\'' + routeName.replace(/'/g,"\\'") + '\',\'' + routeGrade + '\')" ' +
          'style="flex-shrink:0;background:none;border:1px solid var(--border);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:13px;color:' + (isSaved?'var(--amber)':'var(--mist)') + ';white-space:nowrap;transition:color 0.15s;">' +
          (isSaved ? '★ Saved' : '☆ Save') +
        '</button>' +
      '</div>' +
      '<div class="rd-stat"><span class="rd-stat-key">Grade</span><span class="rd-stat-val" id="route-detail-grade">' + (route.community_grade || '—') + '</span></div>' +
      (route.style_tags && route.style_tags.length ? '<div style="margin-top:4px;margin-bottom:6px;display:flex;flex-wrap:wrap;gap:4px;">' + route.style_tags.map(function(t) { var c = {dynamic:'#f97316',technical:'#3b82f6',endurance:'#10b981',compression:'#8b5cf6',slab:'#eab308',span:'#06b6d4'}[t]||'#888'; return '<span style="font-size:9px;text-transform:uppercase;letter-spacing:0.06em;padding:2px 6px;border-radius:10px;border:1px solid '+c+';color:'+c+';font-weight:600;">'+t+'</span>'; }).join('') + '</div>' : '') +
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
      '<div style="margin-top:14px;display:flex;gap:6px;">' +
        '<button onclick="typeof openLogRouteModal===\'function\' && openLogRouteModal()" ' +
          'style="flex:1;padding:7px 0;background:rgba(0,229,200,0.1);border:1px solid rgba(0,229,200,0.25);border-radius:6px;color:var(--teal);font-size:11px;font-weight:600;cursor:pointer;letter-spacing:0.03em;">+ Log to Session</button>' +
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
      '<div id="similar-routes-section" style="margin-top:14px;">' +
        '<div class="creator-label">Similar Routes <span style="font-size:9px;color:var(--mist);margin-left:4px;">AI</span></div>' +
        '<div id="similar-routes-list" style="margin-top:6px;font-size:10px;color:var(--mist);">Loading…</div>' +
      '</div>' +
    '</div>';

  // Physics beta always available — enable Play Beta immediately
  renderRouteFigure(route);
  var btn = document.getElementById('pose-play-btn');
  var ctr = document.getElementById('pose-frame-counter');
  stopRouteAnimation();

  // Reset physics mode UI if it was active for a previous route
  if (typeof _physicsMode !== 'undefined' && _physicsMode) {
    _physicsMode = false;
    var modeBtn   = document.getElementById('physics-mode-btn');
    var modeLabel = document.getElementById('physics-mode-label');
    if (modeBtn) {
      modeBtn.textContent          = '⚛ Physics Mode';
      modeBtn.style.background     = 'rgba(139,92,246,0.18)';
      modeBtn.style.borderColor    = 'rgba(139,92,246,0.45)';
      modeBtn.style.color          = '#a78bfa';
    }
    if (modeLabel) modeLabel.style.display = 'none';
    var svgOverlay = document.getElementById('board-figure-overlay');
    if (svgOverlay) svgOverlay.style.display = '';
    if (typeof _physicsAnimator !== 'undefined' && _physicsAnimator) _physicsAnimator.hide();
  }

  if (btn) {
    btn.disabled = false;
    btn.textContent = '▶ Play Beta';
    btn.onclick = function() { togglePosePlay(); };
  }
  if (ctr) ctr.textContent = route.has_pose ? 'real beta available' : 'AI-generated beta';

  // If scraped frames exist, load them in background for sparkline + phase breakdown
  if (route.external_id && route.has_pose) {
    loadPoseAnimation(route.external_id);
  }

  // Load similar routes in background
  if (route.external_id) {
    loadSimilarRoutes(route.external_id);
  } else {
    var simEl = document.getElementById('similar-routes-list');
    if (simEl) simEl.style.display = 'none';
    var simSec = document.getElementById('similar-routes-section');
    if (simSec) simSec.style.display = 'none';
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


// ── Setter leaderboard ─────────────────────────────────────────────────────

var _settersVisible = false;

function toggleSettersView() {
  _settersVisible = !_settersVisible;
  var routeList   = document.getElementById('route-list');
  var settersList = document.getElementById('setters-list');
  var filterRows  = document.querySelectorAll('#grade-filters, #hold-type-filters');
  var searchBox   = document.querySelector('.search-box');
  var btn         = document.getElementById('setters-toggle-btn');

  if (_settersVisible) {
    if (routeList)  routeList.style.display  = 'none';
    if (settersList) settersList.style.display = '';
    filterRows.forEach(function(r) { r.style.display = 'none'; });
    if (searchBox)  searchBox.style.display   = 'none';
    if (btn) { btn.style.color = 'var(--teal)'; btn.style.borderColor = 'var(--teal)'; }
    _loadSetters();
  } else {
    if (routeList)  routeList.style.display  = '';
    if (settersList) settersList.style.display = 'none';
    filterRows.forEach(function(r) { r.style.display = ''; });
    if (searchBox)  searchBox.style.display   = '';
    if (btn) { btn.style.color = 'var(--mist)'; btn.style.borderColor = 'var(--border)'; }
  }
}

function _loadSetters(sort) {
  var el = document.getElementById('setters-list');
  if (!el) return;
  el.innerHTML = '<div style="padding:20px;text-align:center;color:var(--mist);font-size:12px;">Loading setters…</div>';

  var sortParam = sort || 'count';
  fetch(API + '/api/leaderboard/setters?sort=' + sortParam)
    .then(function(r) { return r.json(); })
    .then(function(d) {
      var setters = d.setters || [];
      if (!setters.length) {
        el.innerHTML = '<div style="padding:20px;text-align:center;color:var(--mist);">No setter data</div>';
        return;
      }

      var gradeLabels = {'easy':'V0-V3','mid':'V4-V6','hard':'V7+'};

      var html = '<div style="padding:6px 0;">';
      // Sort tabs
      html += '<div style="display:flex;gap:6px;padding:6px 10px;border-bottom:1px solid var(--border);">';
      ['count','hardest','creative'].forEach(function(s) {
        var label = {count:'Most Routes', hardest:'Hardest', creative:'Most Diverse'}[s];
        var active = sortParam === s;
        html += '<button onclick="_loadSetters(\''+s+'\')" style="font-size:9px;text-transform:uppercase;letter-spacing:0.06em;padding:3px 8px;border-radius:10px;border:1px solid '+(active?'var(--teal)':'var(--border)')+';background:none;color:'+(active?'var(--teal)':'var(--mist)')+';cursor:pointer;">'+label+'</button>';
      });
      html += '</div>';

      setters.forEach(function(s, i) {
        var rank = i + 1;
        var rankStr = rank <= 3 ? ['🥇','🥈','🥉'][rank-1] : '#' + rank;
        html += '<div style="padding:10px 12px;border-bottom:1px solid var(--border);cursor:pointer;" onclick="filterBySetter(\''+s.setter_name.replace(/'/g,"\\'")+'\')">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;">' +
            '<span style="font-size:13px;font-weight:600;">' + rankStr + ' ' + s.setter_name + '</span>' +
            '<span style="font-size:11px;color:var(--teal);font-family:var(--mono);">' + (s.hardest_grade || '?') + '</span>' +
          '</div>' +
          '<div style="font-size:10px;color:var(--mist);margin-top:3px;">' +
            s.route_count + ' routes · ' + (s.total_sends || 0).toLocaleString() + ' total sends' +
          '</div>' +
        '</div>';
      });
      html += '</div>';
      el.innerHTML = html;
    })
    .catch(function() {
      el.innerHTML = '<div style="padding:20px;text-align:center;color:var(--mist);">Could not load setters</div>';
    });
}

function filterBySetter(setterName) {
  _settersVisible = false;
  var routeList   = document.getElementById('route-list');
  var settersList = document.getElementById('setters-list');
  var filterRows  = document.querySelectorAll('#grade-filters, #hold-type-filters');
  var searchBox   = document.querySelector('.search-box');
  var btn         = document.getElementById('setters-toggle-btn');
  var searchInput = document.getElementById('route-search');

  if (routeList)  routeList.style.display  = '';
  if (settersList) settersList.style.display = 'none';
  filterRows.forEach(function(r) { r.style.display = ''; });
  if (searchBox)  searchBox.style.display   = '';
  if (btn) { btn.style.color = 'var(--mist)'; btn.style.borderColor = 'var(--border)'; }

  // Populate search with setter name to filter
  if (searchInput) {
    searchInput.value = setterName;
    if (typeof debounceSearch === 'function') debounceSearch();
  }
}


// ── Daily Challenge ───────────────────────────────────────────────────────────

var _dailyChallengeRoute = null;

async function initDailyChallenge() {
  var banner = document.getElementById('daily-challenge-banner');
  var text   = document.getElementById('daily-challenge-text');
  if (!banner || !text) return;
  try {
    var res  = await fetch(API + '/api/daily_challenge?grade=V4');
    if (!res.ok) return;  // No challenge today — keep banner hidden
    var data = await res.json();
    _dailyChallengeRoute = data;
    text.textContent = (data.grade || 'V4') + ' · ' + (data.name || 'Challenge Route') +
      (data.board_angle_deg ? ' @ ' + data.board_angle_deg + '°' : '');
    banner.style.display = '';
  } catch(e) {}
}

async function loadDailyChallenge() {
  if (!_dailyChallengeRoute) return;
  var id = _dailyChallengeRoute.id;
  if (!id) return;
  try {
    var res = await fetch(API + '/api/route/' + id);
    var route = await res.json();
    currentRoute = route;
    window._lastRouteDifficultyScore = route.difficulty_score || null;
    renderRouteOnBoard(route);
    renderRouteDetail(route);
    if (typeof showToast === 'function') showToast("Today's V4 challenge loaded!", 'success');
  } catch(e) {
    console.error('Failed to load daily challenge:', e);
  }
}

// ── Similar Routes ────────────────────────────────────────────────────────────

async function loadSimilarRoutes(externalId) {
  var listEl = document.getElementById('similar-routes-list');
  var secEl  = document.getElementById('similar-routes-section');
  if (!listEl || !secEl) return;

  try {
    var url = API + '/api/routes/similar?id=' + encodeURIComponent(externalId) + '&limit=5';
    var res  = await fetch(url);
    var data = await res.json();
    var routes = (data.similar || []).filter(function(r) { return r.similarity > 0.7; });

    if (!routes.length) {
      secEl.style.display = 'none';
      return;
    }

    secEl.style.display = '';
    listEl.innerHTML = routes.map(function(r) {
      var simPct = Math.round(r.similarity * 100);
      var simColor = simPct >= 90 ? 'var(--teal)' : simPct >= 80 ? 'var(--amber)' : 'var(--mist)';
      return '<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);cursor:pointer;" ' +
        'onclick="selectRouteById(' + JSON.stringify(r.id) + ')" ' +
        'title="Similarity: ' + simPct + '%">' +
        '<span style="min-width:28px;text-align:center;font-size:9px;font-weight:700;padding:2px 5px;border-radius:4px;background:rgba(0,229,200,0.12);color:var(--teal);">' + (r.grade || '?') + '</span>' +
        '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;">' + (r.name || 'Unnamed') + '</span>' +
        '<span style="font-size:9px;color:' + simColor + ';white-space:nowrap;">' + simPct + '% match</span>' +
        '</div>';
    }).join('');
  } catch(e) {
    if (secEl) secEl.style.display = 'none';
  }
}

