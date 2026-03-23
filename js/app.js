// ── App orchestration — init, tab switching, global utilities ─────────────────

var _creatorInitialized = false;

// ── Toast Notification ────────────────────────────────────────────────────────

function showToast(msg, type) {
  var toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.className   = 'show' + (type ? ' ' + type : '');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(function() {
    toast.className = toast.className.replace('show', '').trim();
  }, 3000);
}

// ── Tab Switching ─────────────────────────────────────────────────────────────

function switchTab(tab) {
  document.querySelectorAll('[data-tab-view]').forEach(function(v) { v.classList.remove('active'); });
  document.querySelectorAll('.nav-tab').forEach(function(t) { t.classList.remove('active'); });

  var view = document.querySelector('[data-tab-view="' + tab + '"]');
  if (view) view.classList.add('active');
  document.querySelectorAll('.nav-tab').forEach(function(t) {
    if (t.textContent.toLowerCase() === tab) t.classList.add('active');
  });

  if (tab === 'explore' && boardLoaded) {
    setTimeout(function() {
      renderBoard('board', false);
      if (currentRoute) renderRouteOnBoard(currentRoute);
    }, 50);
  }

  if (tab === 'creator') {
    setTimeout(function() {
      if (!boardLoaded) return;
      renderBoard('creator-board', true);
      if (!_creatorInitialized) {
        _creatorInitialized = true;
        refreshCreatorBoard();
        updatePathfinderStatus();
        _updateCreatorFigure();
      } else {
        refreshCreatorBoard();
      }
    }, 50);
  }

  if (tab === 'profile') {
    loadProfileStats();
  }
}

// ── Nav Stats ─────────────────────────────────────────────────────────────────

async function loadStats() {
  try {
    var res  = await fetch(API + '/api/stats');
    var data = await res.json();
    document.getElementById('nav-stats').innerHTML =
      '<b>' + data.total_routes.toLocaleString() + '</b> routes · <b>' + data.model_accuracy + '</b>';
  } catch(e) {
    document.getElementById('nav-stats').textContent = 'API offline';
  }
}

// ── Resize ────────────────────────────────────────────────────────────────────

window.addEventListener('resize', function() {
  if (!boardLoaded) return;
  renderBoard('board', false);
  if (_creatorInitialized) renderBoard('creator-board', true);
  if (currentRoute) renderRouteOnBoard(currentRoute);
  if (_creatorInitialized) refreshCreatorBoard();
});

// ── Init ──────────────────────────────────────────────────────────────────────

(async function init() {
  await Promise.all([
    loadBoardMeta(),
    loadBoardHolds(),
    loadRoutes(true),
    loadStats(),
  ]);
})();

window.addEventListener('DOMContentLoaded', function() {
  renderDefaultFigure('stick-figure-svg');
  // Live label for auto-hold-count slider
  var countSlider = document.getElementById('auto-hold-count');
  var countLbl    = document.getElementById('auto-count-lbl');
  if (countSlider && countLbl) {
    countSlider.addEventListener('input', function() { countLbl.textContent = countSlider.value; });
  }
});
// Also render now in case DOMContentLoaded already fired
if (document.readyState !== 'loading') {
  renderDefaultFigure('stick-figure-svg');
}
