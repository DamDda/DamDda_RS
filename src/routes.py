from flask import Blueprint
from .TFIDF_Model import TFIDF_Model

main_bp = Blueprint("main", __name__)
model = TFIDF_Model()

@main_bp.route("/api/recommend/<int:member_id>", methods=["GET"])
def getRecommendation(member_id):
    try:
        return model.getRecommendationFromMemberId(member_id)
    except Exception as e:
        return str(e), 500