// ── Profile tab ────────────────────────────────────────────────────────────────

var _profileLoaded = false;

async function loadProfileStats() {
  if (_profileLoaded) return;
  _profileLoaded = true;

  loadClimberProfile();
  renderDefaultFigure('beta-figure-svg');
  buildJointRefGrid();

  // Load body mechanics stats (aggregate across all scraped climbs)
  try {
    var res = await fetch(API + '/api/pose/stats');
    if (res.ok) {
      var d = await res.json();
      _updateStatBox('ps-total-frames',   (d.total_frames   || 0).toLocaleString());
      _updateStatBox('ps-total-attempts', (d.total_attempts || 0).toLocaleString());
      if (d.avg_tension   != null) _updateStatBox('ps-avg-tension', Math.round(d.avg_tension * 100) + '%');
      if (d.avg_hip_angle != null) _updateStatBox('ps-avg-hip',     Math.round(d.avg_hip_angle) + '°');
      if (d.avg_arm_reach != null) _updateStatBox('ps-avg-reach',   Math.round(d.avg_arm_reach * 100) + '%');
      if (d.avg_com_height!= null) _updateStatBox('ps-avg-com',     Math.round(d.avg_com_height * 100) + '%');
    }
  } catch(e) {}

  // Load personalised content based on saved profile
  var profile = _readProfileInputs();
  if (profile.onsight) {
    loadWeakPoints(profile.onsight);
    loadRecommendations(profile.onsight);
  }
}

function _updateStatBox(id, val) {
  var el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Technique Gaps ─────────────────────────────────────────────────────────────

async function loadWeakPoints(currentGrade) {
  var el = document.getElementById('weak-points-content');
  if (!el) return;

  // Derive target grade (one above current)
  var grades = ['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14'];
  var idx    = grades.indexOf(currentGrade);
  var target = idx >= 0 && idx < grades.length - 1 ? grades[idx + 1] : currentGrade;

  el.innerHTML = '<div style="font-size:11px;color:var(--mist);">Loading…</div>';
  try {
    var res  = await fetch(API + '/api/climber/weak-points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_grade: currentGrade, target_grade: target }),
    });
    var data = await res.json();

    if (data.error) {
      el.innerHTML = '<div style="font-size:11px;color:var(--mist);">' + data.error + '</div>';
      return;
    }

    var gaps = (data.weak_points || []).slice(0, 5);
    if (!gaps.length) {
      el.innerHTML = '<div style="font-size:11px;color:var(--mist);">Not enough pose data for ' + currentGrade + '/' + target + ' yet.</div>';
      return;
    }

    el.innerHTML = '<div style="font-size:10px;color:var(--mist);margin-bottom:10px;">' +
      currentGrade + ' → ' + target + ' — what your body needs to change</div>' +
      gaps.map(function(wp) {
        var color = wp.severity > 50 ? 'var(--coral)' : wp.severity > 25 ? 'var(--amber, #f59e0b)' : 'var(--mist)';
        var bar   = Math.round(wp.severity);
        var arrow = wp.needs_more ? '↑' : '↓';
        return '<div style="margin-bottom:12px;">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">' +
            '<span style="font-size:11px;color:var(--chalk);font-weight:600;">' + wp.label + '</span>' +
            '<span style="font-size:10px;color:' + color + ';">' + arrow + ' ' + wp.severity + '</span>' +
          '</div>' +
          '<div style="height:4px;background:var(--border);border-radius:2px;">' +
            '<div style="height:4px;width:' + bar + '%;background:' + color + ';border-radius:2px;transition:width 0.4s;"></div>' +
          '</div>' +
          '<div style="font-size:10px;color:var(--mist);margin-top:4px;">' + wp.description + '</div>' +
        '</div>';
      }).join('');
  } catch(e) {
    el.innerHTML = '<div style="font-size:11px;color:var(--mist);">Could not load — is the API running?</div>';
  }
}

// ── Personalised Route Recommendations ────────────────────────────────────────

async function loadRecommendations(currentGrade) {
  var el = document.getElementById('profile-recs-content');
  if (!el) return;

  el.innerHTML = '<div style="font-size:11px;color:var(--mist);">Loading…</div>';
  try {
    var res  = await fetch(API + '/api/climber/recommendations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_grade: currentGrade, limit: 8 }),
    });
    var data = await res.json();

    if (data.error || !data.routes || !data.routes.length) {
      el.innerHTML = '<div style="font-size:11px;color:var(--mist);">No recommendations available yet.</div>';
      return;
    }

    el.innerHTML = '<div style="font-size:10px;color:var(--mist);margin-bottom:10px;">' +
      'Targeting ' + data.target_grade + ' · sorted by community quality rating</div>' +
      '<div style="display:flex;flex-direction:column;gap:8px;">' +
      data.routes.map(function(r) {
        var gc = gradeColor(r.grade);
        return '<div style="display:flex;align-items:center;gap:10px;padding:8px 10px;' +
          'background:var(--panel);border:1px solid var(--border);border-radius:6px;cursor:pointer;"' +
          'onclick="switchTab(\'explore\')">' +
          '<span style="width:32px;height:20px;background:' + gc + ';border-radius:3px;font-size:10px;' +
            'font-weight:700;display:flex;align-items:center;justify-content:center;color:#000;flex-shrink:0;">' +
            r.grade + '</span>' +
          '<div style="flex:1;min-width:0;">' +
            '<div style="font-size:11px;color:var(--chalk);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + r.name + '</div>' +
            '<div style="font-size:10px;color:var(--mist);">' + r.why + '</div>' +
          '</div>' +
          (r.has_pose ? '<span style="font-size:9px;background:rgba(139,92,246,0.15);border:1px solid var(--purple);' +
            'color:var(--purple);border-radius:3px;padding:1px 5px;flex-shrink:0;">BETA</span>' : '') +
          '<span style="font-size:10px;color:var(--mist);flex-shrink:0;">' + r.angle + '°</span>' +
        '</div>';
      }).join('') +
      '</div>';
  } catch(e) {
    el.innerHTML = '<div style="font-size:11px;color:var(--mist);">Could not load — is the API running?</div>';
  }
}

// ── Climber Profile (localStorage) ───────────────────────────────────────────

var _profileUnits = 'metric'; // 'metric' | 'imperial'

function loadClimberProfile() {
  var saved = localStorage.getItem('climberProfile');
  if (!saved) return;
  try {
    var p = JSON.parse(saved);
    _profileUnits = p.units || 'metric';
    var unitsSel = document.getElementById('p-units');
    if (unitsSel) unitsSel.value = _profileUnits;
    _setInputIfPresent('p-height',     p.height);
    _setInputIfPresent('p-wingspan',   p.wingspan);
    _setInputIfPresent('p-weight',     p.weight);
    _setInputIfPresent('p-experience', p.experience);
    _setInputIfPresent('p-onsight',    p.onsight);
    updateDerived();
  } catch(e) {}
}

function saveClimberProfile() {
  var p = _readProfileInputs();
  localStorage.setItem('climberProfile', JSON.stringify(p));
  updateDerived();
  var msg = document.getElementById('profile-saved-msg');
  if (msg) {
    msg.style.opacity = '1';
    setTimeout(function() { msg.style.opacity = '0'; }, 1800);
  }
}

function updateDerived() {
  var p       = _readProfileInputs();
  var apeEl   = document.getElementById('pd-ape');
  var bmiEl   = document.getElementById('pd-bmi');
  var reachEl = document.getElementById('pd-reach');

  // Convert to metric for computation
  var heightCm   = p.height;
  var wingspanCm = p.wingspan;
  var weightKg   = p.weight;
  if (_profileUnits === 'imperial') {
    heightCm   = p.height   ? p.height   * 2.54 : null;
    wingspanCm = p.wingspan ? p.wingspan * 2.54 : null;
    weightKg   = p.weight   ? p.weight   * 0.453592 : null;
  }

  if (apeEl) {
    if (heightCm && wingspanCm) {
      var ape = Math.round(wingspanCm - heightCm);
      apeEl.textContent = (ape >= 0 ? '+' : '') + ape + ' cm';
    } else {
      apeEl.textContent = '—';
    }
  }

  if (bmiEl) {
    if (heightCm && weightKg) {
      bmiEl.textContent = (weightKg / Math.pow(heightCm / 100, 2)).toFixed(1);
    } else {
      bmiEl.textContent = '—';
    }
  }

  if (reachEl) {
    if (heightCm && wingspanCm) {
      var apeVal = wingspanCm - heightCm;
      reachEl.textContent = apeVal >= 8  ? 'Gorilla' :
                            apeVal >= 3  ? 'Long'    :
                            apeVal >= -2 ? 'Average' :
                            apeVal >= -7 ? 'Short'   : 'Compact';
    } else {
      reachEl.textContent = '—';
    }
  }

  // Auto-save (don't show flash on oninput calls)
  var p2 = _readProfileInputs();
  localStorage.setItem('climberProfile', JSON.stringify(p2));
}

function toggleUnits() {
  var sel = document.getElementById('p-units');
  var newUnits = sel ? sel.value : 'metric';
  if (newUnits === _profileUnits) return;

  var hEl = document.getElementById('p-height');
  var wEl = document.getElementById('p-wingspan');
  var wtEl = document.getElementById('p-weight');
  var h  = hEl  && hEl.value  ? parseFloat(hEl.value)  : null;
  var ws = wEl  && wEl.value  ? parseFloat(wEl.value)  : null;
  var wt = wtEl && wtEl.value ? parseFloat(wtEl.value) : null;

  if (newUnits === 'imperial' && _profileUnits === 'metric') {
    if (h  && hEl)  hEl.value  = (h  / 2.54).toFixed(1);
    if (ws && wEl)  wEl.value  = (ws / 2.54).toFixed(1);
    if (wt && wtEl) wtEl.value = (wt / 0.453592).toFixed(1);
    if (hEl)  hEl.placeholder  = 'in';
    if (wEl)  wEl.placeholder  = 'in';
    if (wtEl) wtEl.placeholder = 'lbs';
  } else if (newUnits === 'metric' && _profileUnits === 'imperial') {
    if (h  && hEl)  hEl.value  = (h  * 2.54).toFixed(0);
    if (ws && wEl)  wEl.value  = (ws * 2.54).toFixed(0);
    if (wt && wtEl) wtEl.value = (wt * 0.453592).toFixed(1);
    if (hEl)  hEl.placeholder  = 'cm';
    if (wEl)  wEl.placeholder  = 'cm';
    if (wtEl) wtEl.placeholder = 'kg';
  }

  _profileUnits = newUnits;
  saveClimberProfile();
}

function _setInputIfPresent(id, val) {
  var el = document.getElementById(id);
  if (el && val != null) el.value = val;
}

function _readProfileInputs() {
  return {
    units:      (document.getElementById('p-units')      || {}).value || 'metric',
    height:     _numVal('p-height'),
    wingspan:   _numVal('p-wingspan'),
    weight:     _numVal('p-weight'),
    experience: (document.getElementById('p-experience') || {}).value || '',
    onsight:    (document.getElementById('p-onsight')    || {}).value || '',
  };
}

function _numVal(id) {
  var el = document.getElementById(id);
  if (!el || el.value === '') return null;
  var n = parseFloat(el.value);
  return isNaN(n) ? null : n;
}

// ── Joint Reference Grid ──────────────────────────────────────────────────────

function buildJointRefGrid() {
  var grid = document.getElementById('joint-ref-grid');
  if (!grid) return;
  var groups = [
    { label: 'Head',      joints: ['nose'] },
    { label: 'Left Arm',  joints: ['left_shoulder','left_elbow','left_wrist','left_index'] },
    { label: 'Right Arm', joints: ['right_shoulder','right_elbow','right_wrist','right_index'] },
    { label: 'Core',      joints: ['left_hip','right_hip'] },
    { label: 'Left Leg',  joints: ['left_knee','left_ankle','left_foot_index'] },
    { label: 'Right Leg', joints: ['right_knee','right_ankle','right_foot_index'] },
  ];
  grid.innerHTML = groups.map(function(g) {
    var isLeft  = g.label.includes('Left');
    var isRight = g.label.includes('Right');
    var color   = isLeft ? 'var(--teal)' : isRight ? 'var(--coral)' : 'var(--chalk)';
    return '<div style="background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px;">' +
      '<div style="font-size:10px;font-weight:700;color:' + color + ';margin-bottom:6px;">' + g.label + '</div>' +
      g.joints.map(function(j) {
        return '<div style="font-family:\'Space Mono\',monospace;font-size:9px;color:var(--mist);line-height:1.7;">' + j + '</div>';
      }).join('') +
      '</div>';
  }).join('');
}
