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

// ── Supabase client ───────────────────────────────────────────────────────────
// Initialised asynchronously from /api/config so credentials never sit in JS.
// SUPABASE_CLIENT resolves to the client once ready; use _withSupabase(cb) for
// anything that needs it at startup.
var SUPABASE_CLIENT = null;
var _supabaseReady  = false;
var _supabaseCbs    = [];

(async function _initSupabase() {
  try {
    var res  = await fetch(API + '/api/config');
    var cfg  = await res.json();
    if (cfg.supabase_url && cfg.supabase_key && typeof supabase !== 'undefined') {
      SUPABASE_CLIENT = supabase.createClient(cfg.supabase_url, cfg.supabase_key);
    }
  } catch(e) {
    // API offline or Supabase not configured — client stays null, app still works
  } finally {
    _supabaseReady = true;
    _supabaseCbs.forEach(function(cb) { try { cb(SUPABASE_CLIENT); } catch(e) {} });
    _supabaseCbs = [];
  }
})();

// Call cb(supabaseClient) once the client is initialised (or immediately if ready).
// cb receives null if Supabase is not configured.
function _withSupabase(cb) {
  if (_supabaseReady) { cb(SUPABASE_CLIENT); return; }
  _supabaseCbs.push(cb);
}

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
