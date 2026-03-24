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

// ── Profile Modal ─────────────────────────────────────────────────────────────

function showProfileModal() {
  var modal = document.getElementById('profile-modal');
  if (modal) modal.style.display = 'flex';
}

function dismissProfileModal() {
  var modal = document.getElementById('profile-modal');
  if (modal) modal.style.display = 'none';
  localStorage.setItem('profileModalDismissed', '1');
}

function submitProfileModal() {
  var name     = (document.getElementById('modal-name')     || {}).value || '';
  var height   = parseFloat((document.getElementById('modal-height')   || {}).value) || null;
  var wingspan = parseFloat((document.getElementById('modal-wingspan') || {}).value) || null;
  var onsight  = (document.getElementById('modal-onsight')  || {}).value || '';

  var existing = {};
  try { existing = JSON.parse(localStorage.getItem('climberProfile') || '{}'); } catch(e) {}
  var p = Object.assign(existing, { name: name, height: height, wingspan: wingspan, onsight: onsight, units: 'metric' });
  localStorage.setItem('climberProfile', JSON.stringify(p));

  // Mirror to profile tab inputs
  _setInputIfPresent('p-height',   height);
  _setInputIfPresent('p-wingspan', wingspan);
  _setInputIfPresent('p-onsight',  onsight);
  if (typeof updateDerived === 'function') updateDerived();

  _updateNavProfileLabel();
  dismissProfileModal();
  showToast('Profile saved — beta animator updated', 'success');

  // Refresh figure with new proportions
  if (typeof renderDefaultFigure === 'function') renderDefaultFigure('stick-figure-svg');
  if (typeof currentRoute !== 'undefined' && currentRoute && typeof renderRouteFigure === 'function') {
    renderRouteFigure(currentRoute);
  }
}

function _updateNavProfileLabel() {
  var btn = document.getElementById('nav-profile-btn');
  var lbl = document.getElementById('nav-profile-label');
  if (!btn || !lbl) return;
  try {
    var p = JSON.parse(localStorage.getItem('climberProfile') || '{}');
    if (p.height || p.name) {
      var parts = [];
      if (p.name)     parts.push(p.name);
      if (p.height)   parts.push(p.height + 'cm');
      if (p.wingspan) {
        var ape = Math.round(p.wingspan - p.height);
        parts.push((ape >= 0 ? '+' : '') + ape + ' ape');
      }
      lbl.textContent = parts.join(' · ');
      btn.classList.add('has-profile');
    } else {
      lbl.textContent = 'Set up profile';
      btn.classList.remove('has-profile');
    }
  } catch(e) {}
}

// Check on load whether to show the modal
(function _checkProfileOnLoad() {
  setTimeout(function() {
    _updateNavProfileLabel();
    var saved = localStorage.getItem('climberProfile');
    var dismissed = localStorage.getItem('profileModalDismissed');
    if (!dismissed) {
      try { var p = JSON.parse(saved || '{}'); if (!p.height && !p.name) showProfileModal(); }
      catch(e) { showProfileModal(); }
    }
  }, 800);
})();
