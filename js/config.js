// ── Config & shared utilities ─────────────────────────────────────────────────
//
// API_BASE resolution:
//   1. window.API_BASE — set this in index.html before scripts load for production deploys
//   2. Same origin on port 5001 — when index.html is served by Flask directly
//   3. localhost:5001 — default for local development with a separate file server
//
var API = (function () {
  if (typeof window.API_BASE !== 'undefined') return window.API_BASE;
  if (window.location.port === '5001') return window.location.origin;
  return 'http://localhost:5001';
})();

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
