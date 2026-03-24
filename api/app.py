"""
api/app.py
----------
Climbing Intelligence API — application factory.

Usage:
    python3 api/app.py
    python3 api/app.py --port 5001 --no-debug

Endpoints (see individual blueprints for full documentation):
    GET  /api/board                        Board holds with type annotations
    GET  /api/boards                       Available board types
    GET  /api/routes                       Route listing (filterable)
    GET  /api/route/<id>                   Single route with holds
    POST /api/predict                      Live grade prediction
    POST /api/suggest                      Next hold suggestions
    POST /api/auto_generate                Auto-generate a route
    GET  /api/stats                        DB counts + pose coverage

    GET  /api/climber/benchmarks           Body mechanics by grade
    POST /api/climber/recommendations      Personalised route recommendations
    POST /api/climber/weak-points          Technique gap analysis

    GET  /api/gym/dashboard                Grade distribution + top routes
    GET  /api/gym/setting-recommendations  What grades/styles to set next
    GET  /api/gym/route-performance        Send counts + quality analytics

    GET  /api/pose/stats                   Pose coverage summary
    GET  /api/pose/correlations            Pearson r: pose metric ↔ grade
    GET  /api/pose/<climb_uuid>            Aggregated pose metrics for a route
    GET  /api/pose/frames/<climb_uuid>     Per-frame landmarks for animation
    GET  /api/pose/predictions             Pose-model predictions vs actual grade

    GET  /api/routes/saved                 User-saved routes
    POST /api/routes/saved                 Save a route
    DELETE /api/routes/saved/<id>          Delete a saved route
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path so `api.*` and `ml.*` imports resolve
# regardless of which directory the user runs from.
_API_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_API_DIR)
for _p in (_PROJECT_ROOT, _API_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Flask, jsonify
from flask_cors import CORS

from api.blueprints.board   import bp as board_bp
from api.blueprints.routes  import bp as routes_bp
from api.blueprints.predict import bp as predict_bp
from api.blueprints.stats   import bp as stats_bp
from api.blueprints.climber import bp as climber_bp
from api.blueprints.gym     import bp as gym_bp
from api.blueprints.pose    import bp as pose_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    for bp in (board_bp, routes_bp, predict_bp, stats_bp, climber_bp, gym_bp, pose_bp):
        app.register_blueprint(bp)

    @app.route("/")
    def index():
        return jsonify({"status": "ok", "version": "2.0", "endpoints": [
            "/api/board", "/api/boards",
            "/api/routes", "/api/route/<id>",
            "/api/predict", "/api/suggest", "/api/auto_generate",
            "/api/stats",
            "/api/climber/benchmarks", "/api/climber/recommendations", "/api/climber/weak-points",
            "/api/gym/dashboard", "/api/gym/setting-recommendations", "/api/gym/route-performance",
            "/api/pose/stats", "/api/pose/correlations",
            "/api/pose/<climb_uuid>", "/api/pose/frames/<climb_uuid>",
            "/api/pose/predictions",
            "/api/routes/saved",
        ]})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climbing Intelligence API")
    parser.add_argument("--port",     type=int, default=5001)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    app = create_app()

    # Warm up caches before first request
    from api.board_config import get_board_holds
    from api.ml_engine    import get_model
    print("\nClimbing Intelligence API")
    print("  Warming board hold cache…")
    get_board_holds()
    print("  Loading ML model…")
    get_model()
    print(f"  Listening on http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=not args.no_debug)
