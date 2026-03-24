// ── Physics-based Stick Figure Animator ───────────────────────────────────────
//
// Animates a stick figure climbing through a route's hold sequence directly
// overlaid on the Kilter Board.
//
// Physics model:
//   - Two-joint IK (cosine rule) for arms and legs
//   - COM tracks a damped arc between stable positions (pendulum)
//   - Gravity pulls the body down during free-hand reach
//   - Weight shift before each move (lean toward stationary hand)
//   - Eased timing: ease-in-out cubic for smooth acceleration
//
// Usage:
//   var anim = new PoseAnimator(boardEl, svgOverlayEl);
//   anim.loadRoute(route);
//   anim.play();

'use strict';

// ── Anatomical proportions (normalized to body height = 1.0) ──────────────────
var PHYS = {
  upperArm:    0.148,
  forearm:     0.137,
  thigh:       0.240,
  shin:        0.222,
  torso:       0.310,   // shoulder-to-hip
  shoulderW:   0.190,   // half shoulder width
  hipW:        0.120,   // half hip width
  headR:       0.082,
};

// ── Climber Profile ───────────────────────────────────────────────────────────

function _readClimberProfile() {
  try {
    var p = JSON.parse(localStorage.getItem('climberProfile') || '{}');
    var h = p.height, ws = p.wingspan;
    if (p.units === 'imperial') { h = h ? h * 2.54 : null; ws = ws ? ws * 2.54 : null; }
    return { heightCm: h || 175, wingspanCm: ws || 180 };
  } catch(e) { return { heightCm: 175, wingspanCm: 180 }; }
}

// ── Easing ───────────────────────────────────────────────────────────────────

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}
function easeOut(t) {
  return 1 - Math.pow(1 - t, 3);
}
function easeIn(t) {
  return t * t * t;
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function lerpPt(a, b, t) {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

// ── Two-joint IK ──────────────────────────────────────────────────────────────
// Given root (shoulder/hip) and target (wrist/ankle) + limb lengths,
// return intermediate joint (elbow/knee) position.
// bendDir: +1 = bend forward/out, -1 = bend back/in
function solveIK(root, target, len1, len2, bendDir) {
  var dx = target.x - root.x;
  var dy = target.y - root.y;
  var d  = Math.sqrt(dx * dx + dy * dy);

  // Clamp reach to max extension (small epsilon to avoid NaN)
  d = Math.min(d, len1 + len2 - 0.001);
  if (d < 0.001) d = 0.001;

  // Angle from root to target
  var baseAngle = Math.atan2(dy, dx);

  // Cosine rule: angle at root
  var cosA = (len1 * len1 + d * d - len2 * len2) / (2 * len1 * d);
  cosA = Math.max(-1, Math.min(1, cosA));
  var A = Math.acos(cosA);

  var jointAngle = baseAngle + bendDir * A;
  return {
    x: root.x + len1 * Math.cos(jointAngle),
    y: root.y + len1 * Math.sin(jointAngle),
  };
}

// ── PoseAnimator ──────────────────────────────────────────────────────────────

function PoseAnimator(boardEl, overlayId) {
  this._board    = boardEl;
  this._overlayId = overlayId;
  this._svg      = null;
  this._route    = null;
  this._seq      = [];      // hand hold sequence [{x,y} in board-px space]
  this._feet     = [];      // foot holds
  this._raf      = null;
  this._playing  = false;
  this._t        = 0;       // global animation time in seconds
  this._moveIdx  = 0;       // current move index (0 = approaching hold 0)
  this._angle    = 40;
  this._bh       = 1;       // body height in px (set from board size)

  // Physics state
  this._com      = { x: 0, y: 0 };   // center of mass, board-px
  this._comVel   = { x: 0, y: 0 };   // velocity
  this._lastTime = null;

  // Contact points (board-px)
  this._lWrist   = null;
  this._rWrist   = null;
  this._lAnkle   = null;
  this._rAnkle   = null;

  // Move state machine
  this._moveState = 'idle';   // idle | weight_shift | reach | settle
  this._moveDur   = { weight_shift: 0.35, reach: 0.55, settle: 0.30 };
  this._moveElapsed = 0;
  this._movingHand  = null;  // 'left' | 'right'
  this._reachFrom   = null;
  this._reachTo     = null;
  this._comFrom     = null;
  this._comTo       = null;

  this._ns = 'http://www.w3.org/2000/svg';
}

PoseAnimator.prototype.loadRoute = function(route) {
  this._route = route;
  this._angle = route.board_angle_deg || 40;

  var holds = route.holds || [];

  // Non-finish holds in sequence, finish holds at the very end
  var nonFinish = holds
    .filter(function(h) { return h.role !== 'foot' && h.role !== 'finish' && h.x_pct != null; })
    .sort(function(a, b) { return (a.hand_sequence || 99) - (b.hand_sequence || 99); });
  var finish = holds
    .filter(function(h) { return h.role === 'finish' && h.x_pct != null; });

  this._handHolds = nonFinish.concat(finish);
  this._seqHolds  = this._handHolds;  // set on play() too, but init here
  this._moveIdx   = 0;
  this._moveState = 'idle';
  this._t         = 0;
};

// Convert board percentage coords to SVG pixel coords
PoseAnimator.prototype._toSVG = function(xPct, yPct) {
  var svg = this._getSVG();
  var W   = svg.viewBox.baseVal.width  || parseFloat(svg.getAttribute('width'))  || 400;
  var H   = svg.viewBox.baseVal.height || parseFloat(svg.getAttribute('height')) || 400;
  return {
    x: (xPct / 100) * W,
    y: (yPct / 100) * H,
  };
};

PoseAnimator.prototype._getSVG = function() {
  if (!this._svg) {
    this._svg = document.getElementById(this._overlayId);
    if (!this._svg) {
      // Create overlay SVG on top of the board
      var svg = document.createElementNS(this._ns, 'svg');
      svg.id  = this._overlayId;
      svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;';
      svg.setAttribute('viewBox', '0 0 400 400');
      svg.setAttribute('preserveAspectRatio', 'none');
      if (this._board) {
        this._board.style.position = 'relative';
        this._board.appendChild(svg);
      } else {
        document.body.appendChild(svg);
      }
      this._svg = svg;
    }
  }
  return this._svg;
};

PoseAnimator.prototype._computeBodyHeight = function() {
  var svg = this._getSVG();
  var H   = svg.viewBox.baseVal.height || 400;
  var prof = _readClimberProfile();
  return H * 0.42 * Math.max(0.75, Math.min(1.25, prof.heightCm / 175));
};

// Compute stable joint positions given both hand contact points
PoseAnimator.prototype._stableJoints = function(lWrist, rWrist, lAnkle, rAnkle) {
  var bh  = this._computeBodyHeight();
  var prof  = _readClimberProfile();
  var hScale = Math.max(0.75, Math.min(1.25, prof.heightCm  / 175));
  var wScale = Math.max(0.75, Math.min(1.35, prof.wingspanCm / 180));
  var ua  = PHYS.upperArm  * wScale * bh;
  var fa  = PHYS.forearm   * wScale * bh;
  var th  = PHYS.thigh     * hScale * bh;
  var sh  = PHYS.shin      * hScale * bh;
  var tor = PHYS.torso     * hScale * bh;
  var sw  = PHYS.shoulderW * hScale * bh;
  var hw  = PHYS.hipW      * hScale * bh;

  // COM: midpoint between hands, offset downward by torso length
  var handMid = {
    x: (lWrist.x + rWrist.x) / 2,
    y: (lWrist.y + rWrist.y) / 2,
  };
  var lean = (this._angle / 70) * 0.08;   // lean into wall
  var shoulderY = handMid.y + tor * 0.45;
  var shoulderX = handMid.x + lean * bh;
  var hipX = shoulderX + lean * bh * 0.5;
  var hipY = shoulderY + tor;

  // Clamp: keep hips within SVG with margin
  var svg = this._getSVG();
  var svgH = svg.viewBox.baseVal.height || 400;
  var svgW = svg.viewBox.baseVal.width  || 400;
  if (hipY > svgH - hw) {
    var overflow = hipY - (svgH - hw);
    hipY      -= overflow;
    shoulderY -= overflow;
  }

  // Shoulders
  var lShoulder = { x: shoulderX - sw, y: shoulderY };
  var rShoulder = { x: shoulderX + sw, y: shoulderY };

  // Hips
  var lHip = { x: hipX - hw, y: hipY };
  var rHip = { x: hipX + hw, y: hipY };

  // IK: arms — elbow bends outward (left arm: bend left, right: bend right)
  var lElbow = solveIK(lShoulder, lWrist, ua, fa, -1);
  var rElbow = solveIK(rShoulder, rWrist, ua, fa, +1);

  // Finger tips (extend slightly past wrist)
  var lIndex = {
    x: lWrist.x - 0.03 * bh,
    y: lWrist.y - 0.02 * bh,
  };
  var rIndex = {
    x: rWrist.x + 0.03 * bh,
    y: rWrist.y - 0.02 * bh,
  };

  // IK: legs — knees bend outward (left knee bends left, right knee bends right)
  var lKnee  = lAnkle ? solveIK(lHip, lAnkle, th, sh, -1) : null;
  var rKnee  = rAnkle ? solveIK(rHip, rAnkle, th, sh, +1) : null;

  var lFoot  = lAnkle ? { x: lAnkle.x - 0.02 * bh, y: lAnkle.y + 0.03 * bh } : null;
  var rFoot  = rAnkle ? { x: rAnkle.x + 0.02 * bh, y: rAnkle.y + 0.03 * bh } : null;

  // Nose — lean toward top of wall
  var nose = { x: shoulderX - lean * bh * 0.3, y: shoulderY - PHYS.headR * bh * 2.2 };

  return {
    nose:              nose,
    left_shoulder:     lShoulder,
    right_shoulder:    rShoulder,
    left_elbow:        lElbow,
    right_elbow:       rElbow,
    left_wrist:        lWrist,
    right_wrist:       rWrist,
    left_index:        lIndex,
    right_index:       rIndex,
    left_hip:          lHip,
    right_hip:         rHip,
    left_knee:         lKnee,
    right_knee:        rKnee,
    left_ankle:        lAnkle,
    right_ankle:       rAnkle,
    left_foot_index:   lFoot,
    right_foot_index:  rFoot,
    _bh:               bh,
  };
};

// Draw joints to overlay SVG
PoseAnimator.prototype._draw = function(joints) {
  var svg = this._getSVG();
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  var ns  = this._ns;
  var bh  = joints._bh || this._computeBodyHeight();

  var mkLine = function(a, b, color, w) {
    if (!a || !b) return;
    var el = document.createElementNS(ns, 'line');
    el.setAttribute('x1', a.x); el.setAttribute('y1', a.y);
    el.setAttribute('x2', b.x); el.setAttribute('y2', b.y);
    el.setAttribute('stroke', color);
    el.setAttribute('stroke-width', w || 3);
    el.setAttribute('stroke-linecap', 'round');
    el.setAttribute('opacity', '0.92');
    svg.appendChild(el);
  };

  var mkCirc = function(p, r, fill, opacity) {
    if (!p) return;
    var el = document.createElementNS(ns, 'circle');
    el.setAttribute('cx', p.x); el.setAttribute('cy', p.y);
    el.setAttribute('r', r);
    el.setAttribute('fill', fill);
    el.setAttribute('opacity', opacity || 0.9);
    svg.appendChild(el);
  };

  var mkGlow = function(p, color) {
    if (!p) return;
    mkCirc(p, bh * 0.040, color, 0.25);
    mkCirc(p, bh * 0.022, color, 0.9);
  };

  var W  = bh * 0.025;   // limb stroke width
  var tl = '#00D4B8';    // teal (left)
  var co = '#E05C3A';    // coral (right)
  var wh = '#EEF0ED';    // white/spine

  // Skeleton
  mkLine(joints.left_shoulder,  joints.left_elbow,      tl, W);
  mkLine(joints.left_elbow,     joints.left_wrist,      tl, W);
  mkLine(joints.left_wrist,     joints.left_index,      tl, W * 0.7);
  mkLine(joints.right_shoulder, joints.right_elbow,     co, W);
  mkLine(joints.right_elbow,    joints.right_wrist,     co, W);
  mkLine(joints.right_wrist,    joints.right_index,     co, W * 0.7);
  mkLine(joints.left_shoulder,  joints.right_shoulder,  wh, W);
  mkLine(joints.left_shoulder,  joints.left_hip,        tl, W * 0.85);
  mkLine(joints.right_shoulder, joints.right_hip,       co, W * 0.85);
  mkLine(joints.left_hip,       joints.right_hip,       wh, W);
  mkLine(joints.left_hip,       joints.left_knee,       tl, W);
  mkLine(joints.left_knee,      joints.left_ankle,      tl, W);
  mkLine(joints.left_ankle,     joints.left_foot_index, tl, W * 0.7);
  mkLine(joints.right_hip,      joints.right_knee,      co, W);
  mkLine(joints.right_knee,     joints.right_ankle,     co, W);
  mkLine(joints.right_ankle,    joints.right_foot_index,co, W * 0.7);
  mkLine(joints.nose,           joints.left_shoulder,   wh, W * 0.6);
  mkLine(joints.nose,           joints.right_shoulder,  wh, W * 0.6);

  // Head
  if (joints.nose) {
    mkCirc(joints.nose, bh * PHYS.headR, 'none', 0);
    var head = document.createElementNS(ns, 'circle');
    head.setAttribute('cx', joints.nose.x);
    head.setAttribute('cy', joints.nose.y - bh * PHYS.headR);
    head.setAttribute('r', bh * PHYS.headR);
    head.setAttribute('fill', 'none');
    head.setAttribute('stroke', wh);
    head.setAttribute('stroke-width', W * 0.7);
    head.setAttribute('opacity', '0.9');
    svg.appendChild(head);
  }

  // Joint dots
  var jNames = ['left_elbow','right_elbow','left_knee','right_knee',
                'left_hip','right_hip','left_shoulder','right_shoulder'];
  jNames.forEach(function(n) {
    if (!joints[n]) return;
    var c = n.startsWith('left') ? tl : n.startsWith('right') ? co : wh;
    mkCirc(joints[n], W * 1.4, c, 0.85);
  });

  // Contact glow rings at wrists + ankles
  mkGlow(joints.left_wrist,  tl);
  mkGlow(joints.right_wrist, co);
  mkGlow(joints.left_ankle,  '#F5A623');
  mkGlow(joints.right_ankle, '#F5A623');
};

// Compute natural foot position hanging below current hand/COM position.
// Feet are NOT locked to specific holds — they swing freely with the body.
PoseAnimator.prototype._naturalFoot = function(side) {
  var bh  = this._computeBodyHeight();
  var th  = PHYS.thigh * bh;
  var sh  = PHYS.shin  * bh;
  var tor = PHYS.torso * bh;
  var hw  = PHYS.hipW  * bh;

  // Hip position derived from current hands
  var lW = this._lWrist || { x: 0, y: 0 };
  var rW = this._rWrist || { x: 0, y: 0 };
  var midX = (lW.x + rW.x) / 2;
  var midY = (lW.y + rW.y) / 2;
  var hipX = midX + (side === 'left' ? -hw : hw);
  var hipY = midY + tor;

  // Foot hangs at ~45° below hip for a natural resting position on a steep wall
  var wallLean = (this._angle / 90) * 0.25;  // more lean on steeper walls
  var footOffX = (side === 'left' ? -1 : 1) * (hw * 0.8 + wallLean * bh * 0.15);
  var legLen   = th + sh;
  var footY    = hipY + legLen * 0.75;  // feet don't hang fully straight

  var svg  = this._getSVG();
  var svgH = svg.viewBox.baseVal.height || 400;
  footY = Math.min(footY, svgH - 10);

  return { x: hipX + footOffX, y: footY };
};

// ── Play / Pause / Reset ──────────────────────────────────────────────────────

PoseAnimator.prototype.play = function() {
  if (this._playing) return;
  if (!this._handHolds || this._handHolds.length < 2) {
    console.warn('PoseAnimator: need at least 2 hand holds');
    return;
  }

  this._playing  = true;
  this._lastTime = null;
  this._moveIdx  = 0;

  // ── Bootstrap: pick the two LOWEST holds (highest y_pct = bottom of board)
  // as the starting hand positions, regardless of sequence order.
  // Finish holds are moved to the very end — they should only be grabbed last.
  var nonFinish = this._handHolds.filter(function(h) { return h.role !== 'finish'; });
  var finish    = this._handHolds.filter(function(h) { return h.role === 'finish'; });

  // Re-order: non-finish holds in sequence, then finish holds
  this._seqHolds = nonFinish.concat(finish);

  // The two starting holds are the two bottommost (highest y_pct)
  var bottomTwo = nonFinish.slice().sort(function(a, b) { return b.y_pct - a.y_pct; }).slice(0, 2);
  bottomTwo.sort(function(a, b) { return a.x_pct - b.x_pct; }); // left to right

  var h0 = bottomTwo[0];
  var h1 = bottomTwo[1] || h0;
  this._lWrist = this._toSVG(h0.x_pct, h0.y_pct);
  this._rWrist = this._toSVG(h1.x_pct, h1.y_pct);

  // Feet hang naturally — no hold locking
  this._lAnkle = this._naturalFoot('left');
  this._rAnkle = this._naturalFoot('right');

  // Start sequencing from hold index 2 (after the two starts)
  this._moveIdx = 2;
  this._startNextMove();
  this._loop();
};

PoseAnimator.prototype.pause = function() {
  this._playing = false;
  if (this._raf) cancelAnimationFrame(this._raf);
};

PoseAnimator.prototype.reset = function() {
  this.pause();
  this._moveIdx  = 0;
  this._moveState = 'idle';
  if (this._svg) while (this._svg.firstChild) this._svg.removeChild(this._svg.firstChild);
};

PoseAnimator.prototype.isPlaying = function() { return this._playing; };

// ── Move state machine ────────────────────────────────────────────────────────

PoseAnimator.prototype._startNextMove = function() {
  var seq = this._seqHolds || this._handHolds;
  if (this._moveIdx >= seq.length) {
    // Done — draw final stable pose and stop
    this._moveState = 'done';
    this._playing   = false;
    return;
  }

  var nextHold = seq[this._moveIdx];
  var nextPt   = this._toSVG(nextHold.x_pct, nextHold.y_pct);

  // Decide which hand moves: the one farther from the next hold
  var dl = Math.pow(this._lWrist.x - nextPt.x, 2) + Math.pow(this._lWrist.y - nextPt.y, 2);
  var dr = Math.pow(this._rWrist.x - nextPt.x, 2) + Math.pow(this._rWrist.y - nextPt.y, 2);
  this._movingHand = dl > dr ? 'left' : 'right';

  this._reachFrom = this._movingHand === 'left' ? this._lWrist : this._rWrist;
  this._reachTo   = nextPt;

  // COM moves toward stationary hand during weight shift, then centers at new position
  var stationary  = this._movingHand === 'left' ? this._rWrist : this._lWrist;
  var bh          = this._computeBodyHeight();
  this._comFrom   = {
    x: (this._lWrist.x + this._rWrist.x) / 2,
    y: (this._lWrist.y + this._rWrist.y) / 2 + bh * 0.30,
  };
  this._comTarget = {
    x: (stationary.x + nextPt.x) / 2,
    y: (stationary.y + nextPt.y) / 2 + bh * 0.30,
  };

  this._moveState   = 'weight_shift';
  this._moveElapsed = 0;
};

PoseAnimator.prototype._advanceMoveState = function(dt) {
  if (this._moveState === 'idle' || this._moveState === 'done') return;

  this._moveElapsed += dt;
  var dur    = this._moveDur[this._moveState] || 0.4;
  var rawT   = Math.min(this._moveElapsed / dur, 1.0);

  if (this._moveState === 'weight_shift') {
    var t = easeInOut(rawT);
    // COM shifts toward stationary hand
    this._com = lerpPt(this._comFrom, {
      x: this._movingHand === 'left'
        ? this._rWrist.x + (this._comFrom.x - this._rWrist.x) * 0.6
        : this._lWrist.x + (this._comFrom.x - this._lWrist.x) * 0.6,
      y: this._comFrom.y - this._computeBodyHeight() * 0.04,
    }, t);
    if (rawT >= 1) { this._moveState = 'reach'; this._moveElapsed = 0; }

  } else if (this._moveState === 'reach') {
    var t = easeOut(rawT);
    // Moving hand arcs toward next hold; slight gravity sag in the arc
    var arcSag = Math.sin(rawT * Math.PI) * this._computeBodyHeight() * 0.10;
    var mid    = lerpPt(this._reachFrom, this._reachTo, t);
    var reaching = { x: mid.x, y: mid.y - arcSag };
    if (this._movingHand === 'left') {
      this._lWrist = reaching;
    } else {
      this._rWrist = reaching;
    }
    // COM also glides toward new center
    this._com = lerpPt(this._comFrom, this._comTarget, easeInOut(rawT));
    if (rawT >= 1) {
      // Snap hand to exact hold position
      if (this._movingHand === 'left') this._lWrist = this._reachTo;
      else                             this._rWrist = this._reachTo;
      this._moveState = 'settle'; this._moveElapsed = 0;
    }

  } else if (this._moveState === 'settle') {
    var t = easeOut(rawT);
    // Slight oscillation as body settles (damped spring)
    var osc = Math.exp(-rawT * 5) * Math.sin(rawT * 12) * 0.015 * this._computeBodyHeight();
    this._com = {
      x: this._comTarget.x + osc,
      y: this._comTarget.y + Math.abs(osc) * 0.5,
    };
    if (rawT >= 1) {
      this._moveIdx++;
      // Feet follow body naturally — recompute from new COM
      this._lAnkle = this._naturalFoot('left');
      this._rAnkle = this._naturalFoot('right');
      this._comFrom = this._comTarget;
      this._startNextMove();
    }
  }
};

// ── Main animation loop ───────────────────────────────────────────────────────

PoseAnimator.prototype._loop = function() {
  var self = this;
  if (!this._playing) return;

  this._raf = requestAnimationFrame(function(ts) {
    if (!self._lastTime) self._lastTime = ts;
    var dt = Math.min((ts - self._lastTime) / 1000, 0.05);   // cap at 50ms
    self._lastTime = ts;

    self._advanceMoveState(dt);

    // Feet are always free-hanging — recompute each frame from current wrist positions
    // so they follow the body up the wall naturally instead of freezing between moves.
    var lAnkle = self._naturalFoot('left');
    var rAnkle = self._naturalFoot('right');

    // Compute full joint set from current contact points
    var joints = self._stableJoints(
      self._lWrist, self._rWrist,
      lAnkle, rAnkle,
    );
    self._draw(joints);

    if (self._playing) self._loop();
  });
};

// ── Global helpers ────────────────────────────────────────────────────────────

// Singleton animator instance per board
var _boardAnimator = null;

function getBoardAnimator() {
  var board = document.getElementById('board');
  if (!board) return null;
  if (!_boardAnimator) {
    _boardAnimator = new PoseAnimator(board, 'board-figure-overlay');
  }
  return _boardAnimator;
}

function playRouteAnimation(route) {
  var anim = getBoardAnimator();
  if (!anim) return;
  anim.reset();
  anim.loadRoute(route);
  anim.play();
}

function stopRouteAnimation() {
  if (_boardAnimator) _boardAnimator.reset();
}

function toggleRouteAnimation(route) {
  var anim = getBoardAnimator();
  if (!anim) return;
  var btn = document.getElementById('pose-play-btn');
  if (anim.isPlaying()) {
    anim.reset();
    if (btn) btn.textContent = '▶ Play Beta';
  } else {
    anim.reset();
    anim.loadRoute(route);
    anim.play();
    if (btn) btn.textContent = '⏹ Stop';
  }
}
