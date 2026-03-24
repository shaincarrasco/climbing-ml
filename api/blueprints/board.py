from flask import Blueprint, jsonify, request
from api.board_config import BOARD_CONFIGS, get_board_holds

bp = Blueprint("board", __name__)


@bp.route("/api/board")
def board():
    """
    Return board holds with Climbology types and display positions.
    Query params: board_type ('original'|'homewall'), set_filter ('all'|'bolt-ons'|etc.)
    """
    board_type = request.args.get("board_type", "original")
    set_filter  = request.args.get("set_filter", "all")
    return jsonify(get_board_holds(board_type, set_filter))


@bp.route("/api/boards")
def boards():
    """Return metadata for all available board types."""
    return jsonify({
        k: {
            "name":           v["name"],
            "description":    v["description"],
            "hold_character": v["hold_character"],
            "sets":           list(v["sets"].keys()),
            "default_set":    v["default_set"],
        }
        for k, v in BOARD_CONFIGS.items()
    })
