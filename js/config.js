// ── Config & shared utilities ─────────────────────────────────────────────────
var API           = 'http://localhost:5001';
var POSE_FRAME_MS = 110; // ~9 fps playback

function gradeColor(grade) {
  if (!grade) return 'var(--teal)';
  var n = parseInt(grade.replace(/[^0-9]/g, ''), 10);
  if (isNaN(n))  return 'var(--teal)';
  if (n <= 3)    return 'var(--green)';
  if (n <= 6)    return 'var(--teal)';
  if (n <= 9)    return 'var(--amber)';
  return 'var(--coral)';
}
