from flask import Blueprint
from TFIDF_Model import TFIDF_Model

main_bp = Blueprint("main", __name__)
model = TFIDF_Model()

@main_bp.route("/api/recommend/<member_id>", methods=["GET"])
def getRecommendation(member_id):
    return model.getRecommendationFromMemberId(member_id)