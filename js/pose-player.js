// ── Pose Animation — real MediaPipe data playback ─────────────────────────────

var poseFrames     = [];
var poseFrameIdx   = 0;
var poseAnimTimer  = null;
var posePlayActive = false;

// ── Load ──────────────────────────────────────────────────────────────────────

async function loadPoseAnimation(climbUuid) {
  stopPosePlay();
  poseFrames   = [];
  poseFrameIdx = 0;

  var btn  = document.getElementById('pose-play-btn');
  var ctr  = document.getElementById('pose-frame-counter');
  var sec  = document.getElementById('pose-phases-section');
  var scWr = document.getElementById('scrubber-wrap');
  var scEl = document.getElementById('pose-scrubber');
  var spWr = document.getElementById('sparkline-wrap');
  var kbh  = document.getElementById('kbd-hint');

  if (btn)  { btn.disabled = true; btn.textContent = '▶ Play Beta'; }
  if (ctr)  ctr.textContent = 'Loading pose data…';
  if (sec)  sec.style.display = 'none';
  if (scWr) scWr.style.display = 'none';
  if (spWr) spWr.style.display = 'none';
  if (kbh)  kbh.style.display = 'none';
  if (scEl) { scEl.disabled = true; scEl.value = 0; }

  try {
    var res = await fetch(API + '/api/pose/frames/' + climbUuid);
    if (!res.ok) { if (ctr) ctr.textContent = ''; return; }
    var data = await res.json();
    if (!data.frames || !data.frames.length) { if (ctr) ctr.textContent = ''; return; }

    poseFrames = data.frames;
    if (btn) btn.disabled = false;
    if (ctr) ctr.textContent = poseFrames.length + ' frames · ' + data.total_frames + ' total';
    if (scEl) { scEl.max = poseFrames.length - 1; scEl.value = 0; scEl.disabled = false; }
    if (scWr) scWr.style.display = 'flex';
    if (kbh)  kbh.style.display = '';

    buildSparkline(poseFrames);
    renderPosePhases(poseFrames);

    // Show the first frame immediately so the figure updates
    _renderPoseFrame(0);
  } catch(e) {
    if (ctr) ctr.textContent = '';
  }
}

// ── Playback ──────────────────────────────────────────────────────────────────

function togglePosePlay() {
  if (posePlayActive) stopPosePlay();
  else startPosePlay();
}

function startPosePlay() {
  if (!poseFrames.length) return;
  posePlayActive = true;
  var btn = document.getElementById('pose-play-btn');
  if (btn) btn.textContent = '⏸ Pause';
  stepPoseFrame();
}

function stopPosePlay() {
  posePlayActive = false;
  if (poseAnimTimer) { clearTimeout(poseAnimTimer); poseAnimTimer = null; }
  var btn = document.getElementById('pose-play-btn');
  if (btn && poseFrames.length) btn.textContent = '▶ Play Beta';
}

function stepPoseFrame() {
  if (!posePlayActive) return;
  if (!poseFrames.length) { stopPosePlay(); return; }
  _renderPoseFrame(poseFrameIdx);
  poseFrameIdx  = (poseFrameIdx + 1) % poseFrames.length;
  poseAnimTimer = setTimeout(stepPoseFrame, POSE_FRAME_MS);
}

function seekPoseFrame(idx) {
  poseFrameIdx = Math.max(0, Math.min(idx, poseFrames.length - 1));
  _renderPoseFrame(poseFrameIdx);
}

// ── Frame Renderer ────────────────────────────────────────────────────────────

function _renderPoseFrame(idx) {
  var svg   = document.getElementById('stick-figure-svg');
  var frame = poseFrames[idx];
  if (!svg || !frame) return;

  var W = +svg.getAttribute('width');
  var H = +svg.getAttribute('height');
  renderStickFigure(svg, frame.landmarks, W, H);

  // Scrubber
  var scEl = document.getElementById('pose-scrubber');
  var scTs = document.getElementById('scrubber-ts');
  if (scEl) scEl.value = idx;
  if (scTs) scTs.textContent = frame.ts != null ? frame.ts.toFixed(1) + 's' : idx;

  // Sparkline cursor
  var cursor = document.getElementById('sparkline-cursor');
  var spWr   = document.getElementById('sparkline-wrap');
  if (cursor && spWr && poseFrames.length > 1) {
    var pct = idx / (poseFrames.length - 1);
    cursor.style.left = (pct * spWr.offsetWidth).toFixed(1) + 'px';
  }

  // Frame counter
  var ctr = document.getElementById('pose-frame-counter');
  if (ctr) {
    ctr.textContent = [
      (idx + 1) + '/' + poseFrames.length,
      frame.tension    != null ? 'T:' + Math.round(frame.tension * 100) + '%' : '',
      frame.hip_angle  != null ? 'Hip:' + Math.round(frame.hip_angle) + '°'   : '',
      frame.com_height != null ? 'COM:' + Math.round(frame.com_height * 100) + '%' : '',
    ].filter(Boolean).join('  ');
  }

  // Metrics grid
  var metricsEl = document.getElementById('figure-metrics');
  if (metricsEl) {
    metricsEl.innerHTML = [
      ['Hip',     frame.hip_angle != null ? Math.round(frame.hip_angle) + '°'     : '—'],
      ['Elbow L', frame.elbow_l   != null ? Math.round(frame.elbow_l)   + '°'     : '—'],
      ['Elbow R', frame.elbow_r   != null ? Math.round(frame.elbow_r)   + '°'     : '—'],
      ['Tension', frame.tension   != null ? Math.round(frame.tension * 100) + '%' : '—'],
    ].map(function(pair) {
      return '<div class="figure-metric"><span class="figure-metric-label">' + pair[0] +
             '</span><span class="figure-metric-val">' + pair[1] + '</span></div>';
    }).join('');
  }
}

// ── COM Sparkline ─────────────────────────────────────────────────────────────

function buildSparkline(frames) {
  var spWr = document.getElementById('sparkline-wrap');
  var svg  = document.getElementById('com-sparkline');
  if (!spWr || !svg || !frames.length) return;
  spWr.style.display = '';

  var W   = spWr.offsetWidth || 260;
  var H   = 32;
  var PAD = 2;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('width', W);

  var heights = frames.map(function(f) { return f.com_height != null ? f.com_height : 0; });
  var lo = Math.min.apply(null, heights);
  var hi = Math.max.apply(null, heights);
  var range = hi - lo || 1;

  var pts = heights.map(function(h, i) {
    var x = PAD + (i / (frames.length - 1)) * (W - PAD * 2);
    var y = H - PAD - ((h - lo) / range) * (H - PAD * 2);
    return x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');

  while (svg.firstChild) svg.removeChild(svg.firstChild);

  var ns = 'http://www.w3.org/2000/svg';
  var area = document.createElementNS(ns, 'polygon');
  area.setAttribute('points', PAD + ',' + H + ' ' + pts + ' ' + (W - PAD) + ',' + H);
  area.setAttribute('fill', 'rgba(0,212,184,0.12)');
  svg.appendChild(area);

  var line = document.createElementNS(ns, 'polyline');
  line.setAttribute('points', pts);
  line.setAttribute('fill', 'none');
  line.setAttribute('stroke', 'var(--teal)');
  line.setAttribute('stroke-width', '1.5');
  line.setAttribute('stroke-linecap', 'round');
  line.setAttribute('stroke-linejoin', 'round');
  svg.appendChild(line);
}

// ── Phase Breakdown ───────────────────────────────────────────────────────────

function renderPosePhases(frames) {
  var sec  = document.getElementById('pose-phases-section');
  var grid = document.getElementById('pose-phase-grid');
  if (!sec || !grid || !frames.length) return;

  var sorted = frames.slice().sort(function(a, b) {
    return (a.com_height || 0) - (b.com_height || 0);
  });
  var q = Math.ceil(sorted.length / 4);
  var phases = [
    { label: 'Bottom',   frames: sorted.slice(0, q) },
    { label: 'Mid-Low',  frames: sorted.slice(q, q * 2) },
    { label: 'Mid-High', frames: sorted.slice(q * 2, q * 3) },
    { label: 'Top',      frames: sorted.slice(q * 3) },
  ];

  function avg(arr, key) {
    var vals = arr.map(function(f) { return f[key]; }).filter(function(v) { return v != null; });
    return vals.length ? vals.reduce(function(s, v) { return s + v; }, 0) / vals.length : null;
  }
  function fmt(v, d) { return v != null ? v.toFixed(d || 0) : '—'; }

  grid.innerHTML = phases.map(function(p) {
    var hip     = avg(p.frames, 'hip_angle');
    var tension = avg(p.frames, 'tension');
    var elbowL  = avg(p.frames, 'elbow_l');
    var elbowR  = avg(p.frames, 'elbow_r');
    return '<div class="pose-phase-card">' +
      '<div class="pose-phase-label">' + p.label + '</div>' +
      '<div class="pose-phase-stat">' +
        '<span class="pose-phase-sub">Hip  </span>' + fmt(hip) + '°<br>' +
        '<span class="pose-phase-sub">Tens </span>' + (tension != null ? Math.round(tension * 100) + '%' : '—') + '<br>' +
        '<span class="pose-phase-sub">Elbow</span>' + fmt(elbowL) + '° / ' + fmt(elbowR) + '°<br>' +
      '</div></div>';
  }).join('');

  sec.style.display = '';
}

// ── Keyboard Controls ─────────────────────────────────────────────────────────

document.addEventListener('keydown', function(e) {
  if (!poseFrames.length) return;
  var panel = document.getElementById('route-figure-panel');
  if (!panel || panel.style.display === 'none') return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.code === 'Space') {
    e.preventDefault();
    togglePosePlay();
  } else if (e.code === 'ArrowRight') {
    e.preventDefault();
    stopPosePlay();
    poseFrameIdx = (poseFrameIdx + 1) % poseFrames.length;
    _renderPoseFrame(poseFrameIdx);
  } else if (e.code === 'ArrowLeft') {
    e.preventDefault();
    stopPosePlay();
    poseFrameIdx = (poseFrameIdx - 1 + poseFrames.length) % poseFrames.length;
    _renderPoseFrame(poseFrameIdx);
  }
});
