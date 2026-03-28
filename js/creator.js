// ── Creator tab — route building with live grade prediction ───────────────────

var creatorHolds = [];
var activeRole   = 'start';
var suggestMode  = false;
var predictTimer = null;
var _creatorAnimator = null;
var _lastPredictedGrade = null;  // for grade delta flash

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
    if (activeRole === 'finish' && creatorHolds.filter(function(h) { return h.role === 'finish'; }).length >= 2) {
      document.getElementById('pf-status').textContent = 'Routes can have at most 2 finish holds (Kilter standard).';
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
    if (window.telem) window.telem.track('hold_placed', { role: activeRole, holdCount: creatorHolds.length });
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

  // Assign hand sequences by Y position — y_cm=0 is bottom of wall, y_cm increases upward.
  // Sequence 1 = lowest hold (smallest y_cm = start hold near floor).
  var seq = 1;
  creatorHolds
    .filter(function(h) { return ['start','hand','finish'].includes(h.role); })
    .sort(function(a, b) { return a.y_cm - b.y_cm; }) // ascending: lower on wall = earlier in sequence
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
    var currentGrade = (document.getElementById('pred-grade')?.textContent || '').replace('~', '');
    var res = await fetch(API + '/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        holds: creatorHolds.map(function(h) { return { x_cm: h.x_cm, y_cm: h.y_cm, role: h.role }; }),
        angle: creatorAngle,
        count: 8,
        target_grade: _selectedAutoGrade || currentGrade || '',
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
  if (window.telem) window.telem.track('auto_generate_clicked', { holdCount: creatorHolds.length, angle: creatorAngle });

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
      // auto_generate returns x_cm/y_cm (position coords), not x/y (board grid).
      // Match against boardHolds by position to recover holdId so refreshCreatorBoard
      // can highlight the correct DOM element.
      var bh = boardHolds.find(function(b) { return b.x === h.x_cm && b.y === h.y_cm; });
      return {
        holdId:        bh ? bh.id : null,
        x_cm:          h.x_cm,
        y_cm:          h.y_cm,
        role:          h.role,
        hold_type:     (bh && bh.hold_type !== 'unknown') ? bh.hold_type : (h.hold_type || null),
        hand_sequence: h.hand_sequence || null,
      };
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
  if (window.telem) window.telem.track('predict_fired', { holdCount: creatorHolds.length, angle: creatorAngle });
  var handHolds = creatorHolds.filter(function(h) { return ['start','hand','finish'].includes(h.role); });
  if (handHolds.length < 2) {
    document.getElementById('pred-grade').textContent = '—';
    document.getElementById('pred-conf').textContent  = '';
    document.getElementById('grade-bar-fill').style.width = '0%';
    var stEl = document.getElementById('pred-style-tags');
    if (stEl) { stEl.innerHTML = ''; stEl.style.display = 'none'; }
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
    if (savedProfile) {
      reqBody.profile = savedProfile;
    } else {
      // Send weight alone for weight-adjusted grade even without full body dims
      try {
        var _p = JSON.parse(localStorage.getItem('climberProfile') || '{}');
        var _wt = _p.weight;
        if (_wt) {
          if (_p.units === 'imperial') _wt = _wt * 0.453592;
          reqBody.profile = { weight_kg: _wt };
        }
      } catch(e) {}
    }

    var res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody),
    });
    var data = await res.json();
    if (data.error) return;

    var gradeEl = document.getElementById('pred-grade');
    var confPct = Math.round(data.confidence * 100);
    var newGradeText = confPct < 15 ? '~' + data.grade : data.grade;
    // Animate grade reveal only when grade changes
    if (gradeEl.textContent !== newGradeText) {
      gradeEl.classList.remove('updating');
      void gradeEl.offsetWidth; // force reflow to restart animation
      gradeEl.classList.add('updating');
    }
    if (window.telem) window.telem.track('predict_result', { grade: data.grade, confidence: confPct, holdCount: creatorHolds.length, style_tags: data.style_tags });
    // Below 15% confidence the model is uncertain — show grade with a tilde
    gradeEl.textContent = newGradeText;
    var gc = confPct < 15 ? 'var(--mist)' : gradeColor(data.grade);
    gradeEl.style.color = gc;
    gradeEl.style.webkitTextFillColor = gc;
    gradeEl.style.background = 'none';

    // Grade delta flash — show "+1V" / "-2V" when grade changes
    var deltaEl = document.getElementById('pred-grade-delta');
    if (deltaEl && _lastPredictedGrade && _lastPredictedGrade !== data.grade) {
      var prevN = parseInt(_lastPredictedGrade.replace(/[^0-9]/g, ''), 10);
      var newN  = parseInt(data.grade.replace(/[^0-9]/g, ''), 10);
      if (!isNaN(prevN) && !isNaN(newN) && prevN !== newN) {
        var diff = newN - prevN;
        deltaEl.textContent = (diff > 0 ? '+' : '') + diff + 'V';
        deltaEl.style.color  = diff > 0 ? 'var(--coral)' : 'var(--green)';
        deltaEl.style.opacity = '1';
        clearTimeout(deltaEl._fadeTimer);
        deltaEl._fadeTimer = setTimeout(function() {
          deltaEl.style.opacity = '0';
        }, 2000);
      }
    }
    _lastPredictedGrade = data.grade;

    // Grade band from quantile models (e.g. "V5–V7" or hidden when same grade)
    var bandEl = document.getElementById('pred-grade-band');
    if (bandEl) {
      if (data.grade_low && data.grade_high && data.grade_low !== data.grade_high) {
        bandEl.textContent = data.grade_low + '–' + data.grade_high;
        bandEl.style.display = '';
      } else {
        bandEl.style.display = 'none';
      }
    }

    document.getElementById('pred-conf').textContent = confPct < 15
      ? confPct + '% — uncertain'
      : confPct + '% conf';

    // Style tags (dynamic, technical, endurance, compression, slab, span)
    var styleTagsEl = document.getElementById('pred-style-tags');
    if (styleTagsEl) {
      styleTagsEl.innerHTML = '';
      var tagColors = {
        dynamic:     '#f97316',
        technical:   '#3b82f6',
        endurance:   '#10b981',
        compression: '#8b5cf6',
        slab:        '#eab308',
        span:        '#06b6d4',
      };
      var tags = data.style_tags || [];
      if (tags.length > 0) {
        styleTagsEl.style.display = 'flex';
        tags.forEach(function(tag) {
          var pill = document.createElement('span');
          pill.textContent = tag;
          var color = tagColors[tag] || '#888';
          pill.style.cssText = 'font-size:9px;text-transform:uppercase;letter-spacing:0.06em;padding:2px 6px;border-radius:10px;border:1px solid ' + color + ';color:' + color + ';font-weight:600;';
          styleTagsEl.appendChild(pill);
        });
      } else {
        styleTagsEl.style.display = 'none';
      }
    }
    document.getElementById('grade-bar-fill').style.width = (data.score * 100).toFixed(1) + '%';

    // Personal difficulty score (1.0–10.0 scale) — only shown when profile has body dims
    var pdEl = document.getElementById('pred-pd-grade');
    if (pdEl) {
      if (data.pd_personal != null) {
        pdEl.style.display = '';
        var pdInner = document.getElementById('pred-pd-grade-val');
        if (pdInner) {
          pdInner.textContent = data.pd_personal.toFixed(1);
          // Colour: green <4, teal 4–6, amber 6–8, coral >8
          pdInner.style.color = data.pd_personal < 4 ? 'var(--green)' :
                                 data.pd_personal < 6 ? 'var(--teal)' :
                                 data.pd_personal < 8 ? 'var(--amber)' : 'var(--coral)';
        }
      } else {
        pdEl.style.display = 'none';
      }
    }

    // Weight-adjusted grade (only if API returned it)
    var wtAdjEl = document.getElementById('pred-weight-adj');
    if (wtAdjEl) {
      if (data.grade_weight_adjusted) {
        wtAdjEl.style.display = '';
        var wtVal = document.getElementById('pred-weight-adj-val');
        if (wtVal) wtVal.textContent = data.grade_weight_adjusted;
      } else {
        wtAdjEl.style.display = 'none';
      }
    }

    var dna = data.dna || {};
    ['reach','sustained','dynamic','lateral','complexity','power'].forEach(function(k) {
      document.getElementById('dna-' + k).style.width = (dna[k] || 0) + '%';
      document.getElementById('dna-' + k + '-v').textContent = dna[k] || 0;
    });

    // SHAP explanation: "Why this grade?"
    var explainEl = document.getElementById('pred-explanation');
    if (explainEl) {
      var expl = data.explanation || [];
      if (expl.length > 0) {
        explainEl.innerHTML = '<div style="font-size:9px;text-transform:uppercase;letter-spacing:0.06em;color:var(--mist);margin-bottom:6px;">Why this grade?</div>' +
          expl.slice(0, 3).map(function(e) {
            var color = e.direction === 'harder' ? 'var(--coral)' : 'var(--teal)';
            var arrow = e.direction === 'harder' ? '↑' : '↓';
            return '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">' +
              '<span style="font-size:10px;color:var(--white);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + e.label + '</span>' +
              '<span style="font-size:10px;color:' + color + ';margin-left:8px;flex-shrink:0;font-family:var(--mono);">' + arrow + e.delta_grades.toFixed(1) + 'V</span>' +
              '</div>';
          }).join('');
        explainEl.style.display = '';
      } else {
        explainEl.style.display = 'none';
      }
    }
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
  var clearGradeEl = document.getElementById('pred-grade');
  clearGradeEl.textContent = '—';
  clearGradeEl.style.color = '';
  clearGradeEl.style.webkitTextFillColor = '';
  clearGradeEl.style.background = '';
  document.getElementById('pred-conf').textContent  = '';
  document.getElementById('grade-bar-fill').style.width = '0%';
  var pdEl = document.getElementById('pred-pd-grade');
  if (pdEl) pdEl.style.display = 'none';
  ['reach','sustained','dynamic','lateral','complexity','power'].forEach(function(k) {
    document.getElementById('dna-' + k).style.width = '0%';
    document.getElementById('dna-' + k + '-v').textContent = '0';
  });
}

function exportRouteText() {
  if (!creatorHolds.length) { showToast('Add some holds first.', 'warn'); return; }
  var grade = document.getElementById('pred-grade').textContent.replace('~', '') || '?';
  var lines = [
    'ClimbingML Route Export',
    'Grade: ' + grade + '  |  Angle: ' + creatorAngle + '°  |  Holds: ' + creatorHolds.length,
    '',
  ];
  var seq = 1;
  ['start', 'hand', 'finish', 'foot'].forEach(function(role) {
    var roleHolds = creatorHolds.filter(function(h) { return h.role === role; });
    roleHolds.forEach(function(h) {
      var label = role === 'start' ? 'START' : role === 'finish' ? 'FINISH' : role === 'foot' ? 'FOOT' : '#' + seq++;
      lines.push(label + '  x=' + Math.round(h.x_cm) + 'cm  y=' + Math.round(h.y_cm) + 'cm' + (h.hold_type ? '  [' + h.hold_type + ']' : ''));
    });
  });
  lines.push('', 'Generated by ClimbingML');

  var blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'route_' + grade + '_' + creatorAngle + 'deg.txt';
  a.click();
  URL.revokeObjectURL(a.href);
  if (window.telem) window.telem.track('route_saved', { holdCount: creatorHolds.length, angle: creatorAngle, type: 'export' });
}

async function saveDraftRoute() {
  var starts   = creatorHolds.filter(function(h) { return h.role === 'start'; }).length;
  var finishes = creatorHolds.filter(function(h) { return h.role === 'finish'; }).length;
  var hands    = creatorHolds.filter(function(h) { return h.role === 'hand'; }).length;

  if (starts < 1)   { showToast('Add at least 1 start hold.', 'warn'); return; }
  if (finishes < 1) { showToast('Add a finish hold.', 'warn'); return; }
  if (hands < 1)    { showToast('Add at least 1 hand hold.', 'warn'); return; }

  if (window.telem) window.telem.track('route_saved', { holdCount: creatorHolds.length, angle: creatorAngle });

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

// ── Circuit Builder (in-progress circuit) ────────────────────────────────────

var _circuitRoutes = []; // { name, grade, holds_count, angle, holds }

function renderCircuitBuildList() {
  var el = document.getElementById('circuit-build-list');
  if (!el) return;
  if (!_circuitRoutes.length) {
    el.innerHTML = '<div style="font-size:10px;color:var(--mist);padding:4px 0;">No routes added yet. Generate or place holds, then add.</div>';
    return;
  }
  el.innerHTML = _circuitRoutes.map(function(r, i) {
    return '<div style="display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);">' +
      '<span style="font-size:11px;color:var(--chalk);">' + (r.name||'Route '+(i+1)) + ' · ' + (r.grade||'?') + '</span>' +
      '<button onclick="_circuitRoutes.splice('+i+',1);renderCircuitBuildList()" style="background:none;border:none;color:var(--mist);cursor:pointer;font-size:12px;padding:0 4px;">×</button>' +
    '</div>';
  }).join('');
}

function addCurrentToCircuit() {
  var grade = document.getElementById('circuit-add-grade').value || document.getElementById('pred-grade')?.textContent?.replace('~','') || '';
  if (!creatorHolds.length) { if (typeof showToast==='function') showToast('Place some holds first'); return; }
  var name = 'Route ' + (_circuitRoutes.length + 1);
  _circuitRoutes.push({ name: name, grade: grade, holds_count: creatorHolds.length, angle: creatorAngle, holds: JSON.parse(JSON.stringify(creatorHolds)) });
  renderCircuitBuildList();
  if (typeof showToast==='function') showToast('Added to circuit: ' + grade + ' · ' + creatorHolds.length + ' holds');
}

function saveCurrentCircuit() {
  if (!_circuitRoutes.length) { if (typeof showToast==='function') showToast('Add routes to the circuit first'); return; }
  var name = document.getElementById('circuit-name-input').value.trim() || 'Circuit ' + new Date().toLocaleDateString();
  if (typeof createCircuit === 'function') {
    var savedCount = _circuitRoutes.length;
    createCircuit(name, _circuitRoutes);
    _circuitRoutes = [];
    renderCircuitBuildList();
    document.getElementById('circuit-name-input').value = '';
    if (typeof showToast==='function') showToast('Circuit "' + name + '" saved — ' + savedCount + ' routes');
  }
}

// Init circuit list on page load
document.addEventListener('DOMContentLoaded', function() { renderCircuitBuildList(); });
