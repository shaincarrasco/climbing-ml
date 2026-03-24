// ── Creator tab — route building with live grade prediction ───────────────────

var creatorHolds = [];
var activeRole   = 'start';
var suggestMode  = false;
var predictTimer = null;
var _creatorAnimator = null;

// ── Role Selector ─────────────────────────────────────────────────────────────

function setRole(role) {
  activeRole = role;
  document.querySelectorAll('.role-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.role === role);
  });
}

// ── Hold Click ────────────────────────────────────────────────────────────────

function onCreatorHoldClick(el, hold) {
  var alreadyIdx = creatorHolds.findIndex(function(h) { return h.holdId === hold.id; });

  if (alreadyIdx >= 0) {
    creatorHolds.splice(alreadyIdx, 1);
    el.className = 'hold hold-' + hold.hold_type;
  } else {
    if (activeRole === 'finish' && creatorHolds.filter(function(h) { return h.role === 'finish'; }).length >= 1) {
      document.getElementById('pf-status').textContent = 'Routes can only have 1 finish hold.';
      document.getElementById('pf-status').className = 'pf-status warn';
      return;
    }
    creatorHolds.push({
      holdId:    hold.id,
      x_cm:      hold.x,
      y_cm:      hold.y,
      role:      activeRole,
      hold_type: hold.hold_type,
      hand_sequence: null,
    });
    var roleClass = { start:'role-start', hand:'role-hand', foot:'role-foot-sel', finish:'role-finish' }[activeRole];
    el.className = 'hold hold-' + hold.hold_type + ' ' + roleClass;

    // After 2 starts, auto-advance to hand
    var starts = creatorHolds.filter(function(h) { return h.role === 'start'; }).length;
    if (starts === 2 && activeRole === 'start') setRole('hand');
  }

  refreshCreatorBoard();
  updatePathfinderStatus();
  triggerPredict();
}

// ── Board State Sync ──────────────────────────────────────────────────────────

function refreshCreatorBoard() {
  var board = document.getElementById('creator-board');
  if (!board) return;
  var svg  = document.getElementById('creator-arrows-svg');
  var size = board.offsetWidth;
  if (svg) svg.innerHTML = '';

  // Assign hand sequences by Y position (bottom of board first = lower hold = earlier in sequence)
  var seq = 1;
  creatorHolds
    .filter(function(h) { return ['start','hand','finish'].includes(h.role); })
    .sort(function(a, b) { return b.y_cm - a.y_cm; }) // higher y_cm = lower on board = earlier
    .forEach(function(h) { h.hand_sequence = seq++; });

  // Update counters
  document.getElementById('ct-start').textContent  = creatorHolds.filter(function(h) { return h.role === 'start'; }).length;
  document.getElementById('ct-hand').textContent   = creatorHolds.filter(function(h) { return h.role === 'hand'; }).length;
  document.getElementById('ct-foot').textContent   = creatorHolds.filter(function(h) { return h.role === 'foot'; }).length;
  document.getElementById('ct-finish').textContent = creatorHolds.filter(function(h) { return h.role === 'finish'; }).length;

  // Sync hold visuals
  boardHolds.forEach(function(bh) {
    var el = board.querySelector('.hold[data-id="' + bh.id + '"]');
    if (!el) return;
    var placed = creatorHolds.find(function(c) { return c.holdId === bh.id; });
    if (placed) {
      var rc = { start:'role-start', hand:'role-hand', foot:'role-foot-sel', finish:'role-finish' }[placed.role];
      el.className = 'hold hold-' + bh.hold_type + ' ' + rc;
    } else {
      el.className = 'hold hold-' + bh.hold_type;
    }
  });

  // Draw sequence arrows
  var handSeq = creatorHolds
    .filter(function(h) { return ['start','hand','finish'].includes(h.role); })
    .sort(function(a, b) { return a.hand_sequence - b.hand_sequence; });

  if (svg) {
    for (var i = 1; i < handSeq.length; i++) {
      var h1 = boardHolds.find(function(b) { return b.id === handSeq[i-1].holdId; });
      var h2 = boardHolds.find(function(b) { return b.id === handSeq[i].holdId; });
      if (!h1 || !h2) continue;
      drawArrow(svg, h1.x_pct/100*size, h1.y_pct/100*size, h2.x_pct/100*size, h2.y_pct/100*size);
    }
  }

  // Move chips
  var chips = document.getElementById('creator-move-chips');
  if (creatorHolds.length === 0) {
    chips.innerHTML = '<div style="font-size:11px;color:var(--mist);">No holds placed yet.</div>';
  } else {
    chips.innerHTML = handSeq.map(function(h, i) {
      return '<span class="move-chip ' + h.role + '">' + (h.role === 'start' ? 'S' : h.role === 'finish' ? 'F' : i) + '</span>';
    }).join('');
  }

  // Type breakdown
  document.getElementById('creator-type-breakdown').innerHTML =
    buildTypeBreakdown(creatorHolds.map(function(h) { return { hold_type: h.hold_type }; }));

  // Update stick figure with current start hold positions
  _updateCreatorFigure();
}

function _updateCreatorFigure() {
  var svg = document.getElementById('creator-figure-svg');
  if (!svg) return;
  var W = +svg.getAttribute('width');
  var H = +svg.getAttribute('height');

  // Collect start holds with board positions
  var startBoardHolds = creatorHolds
    .filter(function(h) { return h.role === 'start'; })
    .map(function(h) { return boardHolds.find(function(b) { return b.id === h.holdId; }); })
    .filter(Boolean)
    .sort(function(a, b) { return a.x_pct - b.x_pct; });

  var lHold = startBoardHolds[0] || null;
  var rHold = startBoardHolds[startBoardHolds.length - 1] || null;

  var holdSpread   = lHold && rHold ? Math.abs(rHold.x_pct - lHold.x_pct) / 100 : 0;
  var lateral_spread = Math.max(0.30, Math.min(0.65, 0.18 + holdSpread * 0.75));
  var avgY         = lHold ? ((lHold.y_pct + (rHold ? rHold.y_pct : lHold.y_pct)) / 2) : 70;
  var com_height   = Math.max(0.05, Math.min(0.50, 1 - (avgY / 100) - 0.10));
  var armReach     = Math.max(0.70, Math.min(1.10, 1.05 - avgY / 160));

  var joints = computeClimbingPose({
    angle: creatorAngle, avg_arm_reach: armReach,
    com_height: com_height, tension: 0.50, lateral_spread: lateral_spread,
  });

  var shoulderY = (joints.left_shoulder.y + joints.right_shoulder.y) / 2;
  if (lHold) {
    var lWY = shoulderY - 0.22 + (lHold.y_pct / 100) * 0.40;
    joints.left_wrist = { x: lHold.x_pct / 100, y: lWY };
    joints.left_index = { x: lHold.x_pct / 100 - 0.025, y: lWY - 0.025 };
    joints.left_elbow = {
      x: (joints.left_wrist.x + joints.left_shoulder.x) / 2 + 0.01,
      y: (joints.left_wrist.y + joints.left_shoulder.y) / 2 - 0.02,
    };
  }
  if (rHold && rHold !== lHold) {
    var rWY = shoulderY - 0.22 + (rHold.y_pct / 100) * 0.40;
    joints.right_wrist = { x: rHold.x_pct / 100, y: rWY };
    joints.right_index = { x: rHold.x_pct / 100 + 0.025, y: rWY - 0.025 };
    joints.right_elbow = {
      x: (joints.right_wrist.x + joints.right_shoulder.x) / 2 - 0.01,
      y: (joints.right_wrist.y + joints.right_shoulder.y) / 2 - 0.02,
    };
  }

  // Pin ankles to actual foot hold positions.
  var footBoardHolds = creatorHolds
    .filter(function(h) { return h.role === 'foot'; })
    .map(function(h) { return boardHolds.find(function(b) { return b.id === h.holdId; }); })
    .filter(function(b) { return b && b.x_pct != null; })
    .sort(function(a, b) { return a.x_pct - b.x_pct; });

  var lFoot = null, rFoot = null;
  if (footBoardHolds.length === 1) {
    if (footBoardHolds[0].x_pct >= 50) { rFoot = footBoardHolds[0]; }
    else                                { lFoot = footBoardHolds[0]; }
  } else if (footBoardHolds.length >= 2) {
    lFoot = footBoardHolds[0];
    rFoot = footBoardHolds[footBoardHolds.length - 1];
  }

  var hipY = (joints.left_hip.y + joints.right_hip.y) / 2;

  if (lFoot) {
    var lAnkleY = hipY + (lFoot.y_pct / 100) * 0.36;
    var lAnkleX = lFoot.x_pct / 100;
    joints.left_ankle      = { x: lAnkleX, y: lAnkleY };
    joints.left_knee       = {
      x: (joints.left_hip.x + lAnkleX) / 2 - 0.015,
      y: (joints.left_hip.y + lAnkleY) / 2,
    };
    joints.left_foot_index = { x: lAnkleX - 0.03, y: lAnkleY + 0.025 };
  }
  if (rFoot) {
    var rAnkleY = hipY + (rFoot.y_pct / 100) * 0.36;
    var rAnkleX = rFoot.x_pct / 100;
    joints.right_ankle      = { x: rAnkleX, y: rAnkleY };
    joints.right_knee       = {
      x: (joints.right_hip.x + rAnkleX) / 2 + 0.015,
      y: (joints.right_hip.y + rAnkleY) / 2,
    };
    joints.right_foot_index = { x: rAnkleX + 0.03, y: rAnkleY + 0.025 };
  }

  renderStickFigure(svg, joints, W, H);
  _drawHoldMarkers(svg, joints, lHold, rHold, lFoot, rFoot, W, H);
}

function playCreatorBeta() {
  if (!_creatorAnimator) {
    var board = document.getElementById('creator-board');
    _creatorAnimator = new PoseAnimator(board, 'creator-board-figure-overlay');
  }
  var handHolds = creatorHolds
    .filter(function(h) { return h.role !== 'foot' && h.x_cm != null; })
    .map(function(h) {
      // Convert cm to pct (board is 140cm wide/tall)
      return Object.assign({}, h, {
        x_pct: (h.x_cm / 140) * 100,
        y_pct: 100 - (h.y_cm / 140) * 100,  // kilter y_cm=0 is bottom, y_pct=0 is top
      });
    });
  var footHolds = creatorHolds
    .filter(function(h) { return h.role === 'foot' && h.x_cm != null; })
    .map(function(h) {
      return Object.assign({}, h, {
        x_pct: (h.x_cm / 140) * 100,
        y_pct: 100 - (h.y_cm / 140) * 100,
      });
    });
  if (handHolds.length < 2) { showToast('Need at least 2 hand holds to animate', 'warn'); return; }
  var fakeRoute = {
    holds: handHolds.concat(footHolds),
    board_angle_deg: creatorAngle || 40,
  };
  var btn = document.getElementById('creator-play-btn');
  if (_creatorAnimator.isPlaying()) {
    _creatorAnimator.reset();
    if (btn) btn.textContent = '▶ Play Beta';
  } else {
    _creatorAnimator.reset();
    _creatorAnimator.loadRoute(fakeRoute);
    _creatorAnimator.play();
    if (btn) btn.textContent = '⏹ Stop';
  }
}

function updatePathfinderStatus() {
  var status   = document.getElementById('pf-status');
  var starts   = creatorHolds.filter(function(h) { return h.role === 'start'; }).length;
  var finishes = creatorHolds.filter(function(h) { return h.role === 'finish'; }).length;
  var total    = creatorHolds.length;

  if (starts < 1) {
    status.className = 'pf-status warn';
    status.innerHTML = 'Place at least <b>1 start hold</b> to begin.';
  } else if (finishes < 1 && total >= 4) {
    status.className = 'pf-status warn';
    status.innerHTML = total + ' holds placed. Don\'t forget a <b>finish hold</b>.';
  } else {
    status.className = 'pf-status ok';
    status.innerHTML = '✓ ' + total + ' hold' + (total !== 1 ? 's' : '') + ' · ' + starts + ' start' + (starts !== 1 ? 's' : '') + ' · ' + finishes + ' finish';
  }
}

// ── Hold Hover Preview ────────────────────────────────────────────────────────

function showHoldPreview(el, hold, e) {
  var preview = document.getElementById('hold-hover-preview');
  if (!preview) return;

  var holdXPct = parseFloat(el.style.left);
  var holdYPct = parseFloat(el.style.top);

  var infoEl = preview.querySelector('.preview-label');
  if (infoEl) infoEl.textContent = hold.hold_type + '  (' + Math.round(hold.x || 0) + ', ' + Math.round(hold.y || 0) + ')';

  // Position: relative to creator-board-area
  var areaEl = document.getElementById('creator-board-area');
  var board  = document.getElementById('creator-board');
  if (!areaEl || !board) { preview.style.display = 'flex'; return; }

  var boardRect = board.getBoundingClientRect();
  var areaRect  = areaEl.getBoundingClientRect();
  var holdPx    = boardRect.left - areaRect.left + holdXPct / 100 * board.offsetWidth;
  var holdPy    = boardRect.top  - areaRect.top  + holdYPct / 100 * board.offsetHeight;

  var left = holdPx + 22;
  if (left + 110 > areaRect.width) left = holdPx - 115;
  var top = Math.max(4, holdPy - 65);

  preview.style.left    = left + 'px';
  preview.style.top     = top  + 'px';
  preview.style.display = 'flex';
}

function hideHoldPreview() {
  var preview = document.getElementById('hold-hover-preview');
  if (preview) preview.style.display = 'none';
}

// ── Suggestions ───────────────────────────────────────────────────────────────

async function toggleSuggestions() {
  if (creatorHolds.filter(function(h) { return h.role !== 'foot'; }).length < 1) {
    showToast('Place at least one start hold first.', 'warn');
    return;
  }
  suggestMode = !suggestMode;
  _clearSuggestions();
  if (!suggestMode) return;
  await _fetchAndShowSuggestions();
}

function _clearSuggestions() {
  document.querySelectorAll('.hold.suggested').forEach(function(h) {
    h.classList.remove('suggested');
    h.removeAttribute('data-impact');
  });
  var list = document.getElementById('suggest-list');
  if (list) list.innerHTML = '';
}

async function _fetchAndShowSuggestions() {
  try {
    var res = await fetch(API + '/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        holds: creatorHolds.map(function(h) { return { x_cm: h.x_cm, y_cm: h.y_cm, role: h.role }; }),
        angle: creatorAngle,
        count: 8,
      }),
    });
    var data = await res.json();
    var board = document.getElementById('creator-board');
    var listEl = document.getElementById('suggest-list');
    var items = [];

    data.suggestions.forEach(function(s) {
      var elS = board.querySelector('.hold[data-x="' + s.x + '"][data-y="' + s.y + '"]');
      if (elS && !creatorHolds.find(function(c) { return c.holdId === s.id; })) {
        var impact = (s.impact || []).join(' · ');
        elS.classList.add('suggested');
        elS.setAttribute('data-impact', impact);
        elS.title = s.hold_type + ' — ' + impact + ' (' + s.dist + 'cm)';
        items.push('<div class="suggest-item" onclick="_pickSuggestion(' + JSON.stringify(s) + ')">' +
          '<span class="suggest-type hold-chip ' + s.hold_type + '">' + s.hold_type + '</span>' +
          '<span class="suggest-dist">' + s.dist + 'cm</span>' +
          '<span class="suggest-impact">' + impact + '</span>' +
        '</div>');
      }
    });

    if (listEl) listEl.innerHTML = items.length
      ? items.join('')
      : '<div style="font-size:11px;color:var(--mist);padding:6px 0;">No suggestions — try adding another hold first.</div>';

  } catch(e) { console.error('Suggest failed:', e); }
}

function _pickSuggestion(s) {
  // Add the suggested hold as a hand hold
  creatorHolds.push({
    holdId: s.id, x_cm: s.x, y_cm: s.y,
    role: activeRole === 'foot' ? 'hand' : activeRole,
    hold_type: s.hold_type, hand_sequence: null,
  });
  _clearSuggestions();
  refreshCreatorBoard();
  updatePathfinderStatus();
  triggerPredict();
  if (suggestMode) _fetchAndShowSuggestions();
}

// ── Auto-Generate ─────────────────────────────────────────────────────────────

var _selectedAutoGrade = '';  // '' = freeform, 'V6' etc = specific grade

function selectAutoGrade(el, grade) {
  _selectedAutoGrade = grade;
  document.querySelectorAll('.grade-chip').forEach(function(c) { c.classList.remove('active'); });
  el.classList.add('active');
  // Show/hide freeform slider
  var row = document.getElementById('auto-difficulty-row');
  if (row) row.style.display = grade ? 'none' : 'flex';
}

async function autoGenerate() {
  var diffSlider  = document.getElementById('auto-difficulty');
  var countSlider = document.getElementById('auto-hold-count');
  var difficulty  = diffSlider  ? parseFloat(diffSlider.value) : 0.4;
  var holdCount   = countSlider ? parseInt(countSlider.value)   : 8;

  var btn = document.getElementById('btn-auto-generate');
  if (btn) { btn.textContent = 'Generating…'; btn.disabled = true; }

  var reqBody = { angle: creatorAngle, hold_count: holdCount };
  if (_selectedAutoGrade) {
    reqBody.grade = _selectedAutoGrade;
  } else {
    reqBody.difficulty = difficulty;
  }

  try {
    var res = await fetch(API + '/api/auto_generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody),
    });
    var data = await res.json();
    if (data.error) { showToast('Auto-generate failed: ' + data.error, 'error'); return; }

    creatorHolds = data.holds.map(function(h) {
      return { holdId: h.id, x_cm: h.x, y_cm: h.y, role: h.role, hold_type: h.hold_type, hand_sequence: null };
    });

    refreshCreatorBoard();
    updatePathfinderStatus();
    triggerPredict();
    var gradeLabel = _selectedAutoGrade ? ' · targeting ' + _selectedAutoGrade : '';
    showToast('Generated ' + creatorHolds.length + ' hold route' + gradeLabel + '!', 'success');
  } catch(e) {
    showToast('API offline — cannot auto-generate.', 'error');
  } finally {
    if (btn) { btn.textContent = 'Auto-Generate'; btn.disabled = false; }
  }
}

// ── Live Prediction ───────────────────────────────────────────────────────────

function triggerPredict() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(runPredict, 400);
}

async function runPredict() {
  var handHolds = creatorHolds.filter(function(h) { return ['start','hand','finish'].includes(h.role); });
  if (handHolds.length < 2) {
    document.getElementById('pred-grade').textContent = '—';
    document.getElementById('pred-conf').textContent  = '';
    document.getElementById('grade-bar-fill').style.width = '0%';
    return;
  }

  try {
    // Include climber profile if set — enables personal difficulty score
    var reqBody = {
      holds: creatorHolds.map(function(h) {
        return { x_cm: h.x_cm, y_cm: h.y_cm, role: h.role, hold_type: h.hold_type, hand_sequence: h.hand_sequence };
      }),
      angle: creatorAngle,
    };
    var savedClimberId = localStorage.getItem('climberId');
    if (savedClimberId) reqBody.climber_id = savedClimberId;
    var savedProfile = _getProfileForPDScore();
    if (savedProfile) reqBody.profile = savedProfile;

    var res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody),
    });
    var data = await res.json();
    if (data.error) return;

    var gradeEl = document.getElementById('pred-grade');
    var confPct = Math.round(data.confidence * 100);
    // Below 15% confidence the model is uncertain — show grade with a tilde
    gradeEl.textContent = confPct < 15 ? '~' + data.grade : data.grade;
    gradeEl.style.color = confPct < 15 ? 'var(--mist)' : gradeColor(data.grade);
    document.getElementById('pred-conf').textContent = confPct < 15
      ? confPct + '% — uncertain'
      : confPct + '% conf';
    document.getElementById('grade-bar-fill').style.width = (data.score * 100).toFixed(1) + '%';

    // Personal difficulty score (1.0–10.0 scale) — only shown when profile has body dims
    var pdEl = document.getElementById('pred-pd-grade');
    if (pdEl) {
      if (data.pd_personal != null) {
        pdEl.style.display = '';
        var pdInner = document.getElementById('pred-pd-grade-val');
        if (pdInner) {
          pdInner.textContent = data.pd_personal.toFixed(1);
          // Colour: green <4, yellow 4–7, coral >7
          pdInner.style.color = data.pd_personal < 4 ? 'var(--teal)' :
                                 data.pd_personal < 7 ? '#f59e0b' : 'var(--coral)';
        }
      } else {
        pdEl.style.display = 'none';
      }
    }

    var dna = data.dna || {};
    ['reach','sustained','dynamic','lateral','complexity','power'].forEach(function(k) {
      document.getElementById('dna-' + k).style.width = (dna[k] || 0) + '%';
      document.getElementById('dna-' + k + '-v').textContent = dna[k] || 0;
    });
  } catch(e) { /* API down — fail silently */ }
}

// Read profile from localStorage and convert to metric for PDScore.
// Returns null if height/wingspan not set (PDScore won't be computed).
function _getProfileForPDScore() {
  try {
    var p = JSON.parse(localStorage.getItem('climberProfile') || '{}');
    if (!p.height || !p.wingspan) return null;
    var h  = p.height,   ws = p.wingspan, wt = p.weight;
    if (p.units === 'imperial') {
      h  = h  ? h  * 2.54      : null;
      ws = ws ? ws * 2.54      : null;
      wt = wt ? wt * 0.453592  : null;
    }
    return { height_cm: h, wingspan_cm: ws, weight_kg: wt || null };
  } catch(e) { return null; }
}

// ── Actions ───────────────────────────────────────────────────────────────────

function clearCreator() {
  if (creatorHolds.length > 0) {
    if (!confirm('Clear all holds?')) return;
  }
  creatorHolds = [];
  refreshCreatorBoard();
  updatePathfinderStatus();
  document.getElementById('pred-grade').textContent = '—';
  document.getElementById('pred-conf').textContent  = '';
  document.getElementById('grade-bar-fill').style.width = '0%';
  var pdEl = document.getElementById('pred-pd-grade');
  if (pdEl) pdEl.style.display = 'none';
  ['reach','sustained','dynamic','lateral','complexity','power'].forEach(function(k) {
    document.getElementById('dna-' + k).style.width = '0%';
    document.getElementById('dna-' + k + '-v').textContent = '0';
  });
}

async function saveDraftRoute() {
  var starts   = creatorHolds.filter(function(h) { return h.role === 'start'; }).length;
  var finishes = creatorHolds.filter(function(h) { return h.role === 'finish'; }).length;
  var hands    = creatorHolds.filter(function(h) { return h.role === 'hand'; }).length;

  if (starts < 1)   { showToast('Add at least 1 start hold.', 'warn'); return; }
  if (finishes < 1) { showToast('Add a finish hold.', 'warn'); return; }
  if (hands < 1)    { showToast('Add at least 1 hand hold.', 'warn'); return; }

  var grade = document.getElementById('pred-grade').textContent.replace('~', '');
  var conf  = parseFloat((document.getElementById('pred-conf').textContent || '0')) / 100 || 0;

  var name = prompt('Name this route:', 'My Route ' + new Date().toLocaleDateString());
  if (name === null) return;  // user cancelled

  try {
    var res = await fetch(API + '/api/routes/saved', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name:             name || 'Untitled Route',
        angle:            creatorAngle,
        predicted_grade:  grade,
        confidence:       conf,
        holds:            creatorHolds,
      }),
    });
    var data = await res.json();
    if (data.error) { showToast('Save failed: ' + data.error, 'error'); return; }
    showToast('Saved "' + name + '" — ' + creatorHolds.length + ' holds · ' + grade, 'success');
  } catch(e) {
    showToast('Could not save — is the API running?', 'error');
  }
}
