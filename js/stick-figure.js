// ── Stick Figure — SVG skeleton renderer ──────────────────────────────────────

var JOINTS = [
  'nose',
  'left_shoulder', 'right_shoulder',
  'left_elbow',    'right_elbow',
  'left_wrist',    'right_wrist',
  'left_hip',      'right_hip',
  'left_knee',     'right_knee',
  'left_ankle',    'right_ankle',
  'left_index',    'right_index',
  'left_foot_index','right_foot_index',
];

var LIMBS = [
  ['left_shoulder',  'left_elbow',      'left'],
  ['left_elbow',     'left_wrist',      'left'],
  ['left_wrist',     'left_index',      'left'],
  ['right_shoulder', 'right_elbow',     'right'],
  ['right_elbow',    'right_wrist',     'right'],
  ['right_wrist',    'right_index',     'right'],
  ['left_shoulder',  'right_shoulder',  'spine'],
  ['left_shoulder',  'left_hip',        'left'],
  ['right_shoulder', 'right_hip',       'right'],
  ['left_hip',       'right_hip',       'spine'],
  ['left_hip',       'left_knee',       'left'],
  ['left_knee',      'left_ankle',      'left'],
  ['left_ankle',     'left_foot_index', 'left'],
  ['right_hip',      'right_knee',      'right'],
  ['right_knee',     'right_ankle',     'right'],
  ['right_ankle',    'right_foot_index','right'],
  ['nose',           'left_shoulder',   'spine'],
  ['nose',           'right_shoulder',  'spine'],
];

var LIMB_COLOR = { left: '#00D4B8', right: '#E05C3A', spine: '#EEF0ED' };
var LIMB_W     = { left: 2.5,       right: 2.5,       spine: 2.0       };

// Compute a climbing pose from route-level or hold-level parameters.
// Returns joints dict: { joint_name: {x, y} } in normalized [0,1] coords.
function computeClimbingPose(opts) {
  var angle          = (opts && opts.angle)          || 40;
  var avg_arm_reach  = (opts && opts.avg_arm_reach)  || 0.85;
  var com_height     = (opts && opts.com_height)     || 0.5;
  var tension        = (opts && opts.tension)        || 0.55;
  var lateral_spread = (opts && opts.lateral_spread) || 0.45;

  var profHeight = (opts && opts.heightCm)   || 175;
  var profWing   = (opts && opts.wingspanCm) || 180;
  var hScale = Math.max(0.8, Math.min(1.2, profHeight / 175));
  var wScale = Math.max(0.8, Math.min(1.3, profWing   / 180));

  var lean    = angle / 70;
  var compact = tension;
  var armExt  = Math.min(avg_arm_reach, 1.2) / 1.2;
  var spread  = Math.max(0.15, Math.min(lateral_spread, 0.65));

  // hipY: 0 = top of SVG, 1 = bottom. Clamp so feet never fall below 0.93.
  var hipY      = Math.min(0.72 - com_height * 0.3 * hScale, 0.58);
  var shoulderY = hipY - (0.18 - lean * 0.04) * hScale;
  var midX      = 0.5;
  var hipX_off  = lean * 0.04;

  var j = {};

  j.left_hip       = { x: midX - 0.08 + hipX_off, y: hipY };
  j.right_hip      = { x: midX + 0.08 + hipX_off, y: hipY };
  j.left_shoulder  = { x: midX - 0.10,             y: shoulderY };
  j.right_shoulder = { x: midX + 0.10,             y: shoulderY };
  j.nose           = { x: midX + lean * 0.03,       y: shoulderY - 0.10 };

  var elbowSpreadX = spread * (0.5 + armExt * 0.3) * wScale;
  var elbowY       = shoulderY - 0.06 * (1 - compact * 0.4);
  j.left_elbow     = { x: midX - elbowSpreadX * 0.65, y: elbowY };
  j.right_elbow    = { x: midX + elbowSpreadX * 0.65, y: elbowY };

  var wristY    = shoulderY - 0.22 * hScale * armExt;
  var wristSprd = spread * (0.8 + armExt * 0.25) * wScale;
  j.left_wrist  = { x: midX - wristSprd * 0.70, y: wristY };
  j.right_wrist = { x: midX + wristSprd * 0.55, y: wristY + 0.03 };

  j.left_index  = { x: j.left_wrist.x  - 0.02, y: j.left_wrist.y  - 0.03 };
  j.right_index = { x: j.right_wrist.x + 0.02, y: j.right_wrist.y - 0.03 };

  var kneeBend = 0.12 + compact * 0.08 + lean * 0.05;
  var kneeY    = hipY + 0.18 * hScale;
  j.left_knee  = { x: midX - 0.09 - lean * 0.02, y: kneeY };
  j.right_knee = { x: midX + 0.09 - lean * 0.02, y: kneeY };

  var ankleY    = kneeY + 0.16 * hScale;
  var footSprd  = spread * 0.55;
  j.left_ankle  = { x: midX - footSprd - lean * 0.04,         y: ankleY };
  j.right_ankle = { x: midX + footSprd * 0.8 - lean * 0.04,  y: ankleY - 0.02 };

  j.left_foot_index  = { x: j.left_ankle.x  - 0.04, y: j.left_ankle.y  + 0.04 };
  j.right_foot_index = { x: j.right_ankle.x + 0.04, y: j.right_ankle.y + 0.04 };

  return j;
}

// Compute pose oriented toward a specific hold position (for hold preview / move viewer).
// holdXPct and holdYPct are 0-100 percentages of the board.
function computeHoldReachPose(holdXPct, holdYPct, boardAngle) {
  var lateralOffset = Math.abs(holdXPct - 50) / 50;
  var lateral_spread = 0.20 + lateralOffset * 0.45;
  // Lower hold on board → lower COM; board y_pct=0 is top, 100 is bottom
  var com_height = Math.max(0.1, Math.min(0.95, 1 - (holdYPct / 100) - 0.15));
  var j = computeClimbingPose({
    angle:          boardAngle || 40,
    avg_arm_reach:  0.92,
    com_height:     com_height,
    tension:        0.55,
    lateral_spread: lateral_spread,
  });
  // Bias the appropriate wrist toward the hold
  var normX = holdXPct / 100;
  var normY = holdYPct / 100;
  if (holdXPct >= 50) {
    j.right_wrist = { x: normX * 0.6 + j.right_wrist.x * 0.4, y: normY * 0.6 + j.right_wrist.y * 0.4 };
    j.right_index = { x: j.right_wrist.x + 0.02, y: j.right_wrist.y - 0.02 };
  } else {
    j.left_wrist  = { x: normX * 0.6 + j.left_wrist.x * 0.4,  y: normY * 0.6 + j.left_wrist.y  * 0.4 };
    j.left_index  = { x: j.left_wrist.x - 0.02,                y: j.left_wrist.y - 0.02 };
  }
  return j;
}

// Render a stick figure into an SVG element.
// joints: { joint_name: {x, y} } in normalized 0-1 coords
function renderStickFigure(svg, joints, W, H, pad, labels) {
  W   = W   || 140;
  H   = H   || 210;
  pad = pad || 12;

  while (svg.firstChild) svg.removeChild(svg.firstChild);

  var scaleX = function(n) { return pad + n * (W - 2 * pad); };
  var scaleY = function(n) { return pad + n * (H - 2 * pad); };
  var ns = 'http://www.w3.org/2000/svg';

  var mkEl = function(tag, attrs) {
    var el = document.createElementNS(ns, tag);
    Object.keys(attrs).forEach(function(k) { el.setAttribute(k, attrs[k]); });
    return el;
  };

  LIMBS.forEach(function(limb) {
    var a = limb[0], b = limb[1], side = limb[2];
    if (!joints[a] || !joints[b]) return;
    svg.appendChild(mkEl('line', {
      x1: scaleX(joints[a].x), y1: scaleY(joints[a].y),
      x2: scaleX(joints[b].x), y2: scaleY(joints[b].y),
      stroke: LIMB_COLOR[side],
      'stroke-width': LIMB_W[side],
      'stroke-linecap': 'round',
      opacity: 0.9,
    }));
  });

  if (joints.nose) {
    svg.appendChild(mkEl('circle', {
      cx: scaleX(joints.nose.x), cy: scaleY(joints.nose.y) - 8,
      r: 8, fill: 'none', stroke: '#EEF0ED', 'stroke-width': 2, opacity: 0.9,
    }));
  }

  JOINTS.forEach(function(name) {
    if (!joints[name] || name === 'nose') return;
    var isLeft  = name.startsWith('left');
    var isRight = name.startsWith('right');
    var color   = isLeft ? LIMB_COLOR.left : isRight ? LIMB_COLOR.right : LIMB_COLOR.spine;
    var isEnd   = name.includes('wrist') || name.includes('index') || name.includes('foot_index');
    svg.appendChild(mkEl('circle', {
      cx: scaleX(joints[name].x),
      cy: scaleY(joints[name].y),
      r:  isEnd ? 3.5 : 4,
      fill: color, opacity: 0.85,
      'data-joint': name,
    }));
  });
}

function getClimberPhysFromProfile() {
  try {
    var p = JSON.parse(localStorage.getItem('climberProfile') || '{}');
    var h = p.height, ws = p.wingspan;
    if (p.units === 'imperial') { h = h ? h * 2.54 : null; ws = ws ? ws * 2.54 : null; }
    return { heightCm: h || 175, wingspanCm: ws || 180 };
  } catch(e) { return { heightCm: 175, wingspanCm: 180 }; }
}

function renderDefaultFigure(svgId) {
  var svg = document.getElementById(svgId);
  if (!svg) return;
  var W = +svg.getAttribute('width');
  var H = +svg.getAttribute('height');
  var prof = getClimberPhysFromProfile();
  renderStickFigure(svg, computeClimbingPose(prof), W, H);
}

// Render static figure from route metadata + start hold positions
function renderRouteFigure(route) {
  var svg = document.getElementById('stick-figure-svg');
  if (!svg) return;
  var W = +svg.getAttribute('width');
  var H = +svg.getAttribute('height');

  var angle   = route.board_angle_deg || 40;
  var score   = route.difficulty_score || 0.4;
  var tension = 0.4 + score * 0.5;
  var armReach = 0.65 + score * 0.35;

  // Sort start holds left→right so leftmost → left hand, rightmost → right hand
  var startHolds = (route.holds || []).filter(function(h) {
    return h.role === 'start' && h.x_pct != null;
  }).sort(function(a, b) { return a.x_pct - b.x_pct; });

  var com_height = 0.25;
  var lateral_spread = 0.45;
  var lHold = null, rHold = null;

  if (startHolds.length > 0) {
    lHold = startHolds[0];
    rHold = startHolds[startHolds.length - 1];
    var avgY = (lHold.y_pct + rHold.y_pct) / 2;
    // lateral_spread from actual horizontal distance between holds
    var holdSpread = Math.abs(rHold.x_pct - lHold.x_pct) / 100;
    lateral_spread = Math.max(0.30, Math.min(0.65, 0.18 + holdSpread * 0.75));
    com_height = Math.max(0.05, Math.min(0.50, 1 - (avgY / 100) - 0.10));
    // arm reach: holds high on board (low y_pct) → arms more extended
    armReach = Math.max(0.70, Math.min(1.10, 1.05 - avgY / 160));
  }

  var joints = computeClimbingPose(Object.assign({
    angle: angle, avg_arm_reach: armReach,
    com_height: com_height, tension: tension, lateral_spread: lateral_spread,
  }, getClimberPhysFromProfile()));

  // Pin wrists to actual hold positions.
  // Y: relative to shoulder — y_pct=0 (top of board) = arms fully up above shoulder;
  //    y_pct=55 ≈ horizontal; y_pct=100 (bottom) = arms reaching down.
  // Formula: wristY = shoulderY - 0.22 + (y_pct/100) * 0.40
  var shoulderY = (joints.left_shoulder.y + joints.right_shoulder.y) / 2;

  if (lHold) {
    var lWY = shoulderY - 0.22 + (lHold.y_pct / 100) * 0.40;
    joints.left_wrist = { x: lHold.x_pct / 100, y: lWY };
    joints.left_index = { x: lHold.x_pct / 100 - 0.025, y: lWY - 0.025 };
    joints.left_elbow = {
      x: (joints.left_wrist.x  + joints.left_shoulder.x)  / 2 + 0.01,
      y: (joints.left_wrist.y  + joints.left_shoulder.y)  / 2 - 0.02,
    };
  }
  if (rHold) {
    var rWY = shoulderY - 0.22 + (rHold.y_pct / 100) * 0.40;
    joints.right_wrist = { x: rHold.x_pct / 100, y: rWY };
    joints.right_index = { x: rHold.x_pct / 100 + 0.025, y: rWY - 0.025 };
    joints.right_elbow = {
      x: (joints.right_wrist.x + joints.right_shoulder.x) / 2 - 0.01,
      y: (joints.right_wrist.y + joints.right_shoulder.y) / 2 - 0.02,
    };
  }

  // Pin ankles to actual foot hold positions.
  var footHolds = (route.holds || []).filter(function(h) {
    return h.role === 'foot' && h.x_pct != null;
  }).sort(function(a, b) { return a.x_pct - b.x_pct; });

  var lFoot = null, rFoot = null;
  if (footHolds.length === 1) {
    if (footHolds[0].x_pct >= 50) { rFoot = footHolds[0]; }
    else                           { lFoot = footHolds[0]; }
  } else if (footHolds.length >= 2) {
    lFoot = footHolds[0];
    rFoot = footHolds[footHolds.length - 1];
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

  document.getElementById('route-figure-panel').style.display = 'flex';

  var metricsEl = document.getElementById('figure-metrics');
  if (metricsEl) {
    metricsEl.innerHTML = [
      ['Angle',    angle + '°'],
      ['Reach',    Math.round(armReach * 100) + '%'],
      ['Tension',  Math.round(tension * 100) + '%'],
      ['COM',      Math.round(com_height * 100) + '%'],
    ].map(function(pair) {
      return '<div class="figure-metric"><span class="figure-metric-label">' + pair[0] +
             '</span><span class="figure-metric-val">' + pair[1] + '</span></div>';
    }).join('');
  }
}

function _drawHoldMarkers(svg, joints, lHold, rHold, lFoot, rFoot, W, H) {
  var ns  = 'http://www.w3.org/2000/svg';
  var pad = 12;
  var sx  = function(n) { return pad + n * (W - 2*pad); };
  var sy  = function(n) { return pad + n * (H - 2*pad); };
  var glow = function(joint, color) {
    if (!joints[joint]) return;
    var c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', sx(joints[joint].x));
    c.setAttribute('cy', sy(joints[joint].y));
    c.setAttribute('r', 7);
    c.setAttribute('fill', 'none');
    c.setAttribute('stroke', color);
    c.setAttribute('stroke-width', 1.5);
    c.setAttribute('opacity', 0.55);
    svg.appendChild(c);
    var c2 = document.createElementNS(ns, 'circle');
    c2.setAttribute('cx', sx(joints[joint].x));
    c2.setAttribute('cy', sy(joints[joint].y));
    c2.setAttribute('r', 4.5);
    c2.setAttribute('fill', color);
    c2.setAttribute('opacity', 0.9);
    svg.appendChild(c2);
  };
  if (lHold)  glow('left_wrist',  '#00D4B8');
  if (rHold)  glow('right_wrist', '#E05C3A');
  if (lFoot)  glow('left_ankle',  '#F5A623');
  if (rFoot)  glow('right_ankle', '#F5A623');
}
