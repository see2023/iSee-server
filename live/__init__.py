from flask import Blueprint

live_bp = Blueprint('live', __name__, url_prefix='/api/v1/live')
from . import live_api
