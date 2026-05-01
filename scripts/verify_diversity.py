
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.api.routers.recommendations import predict_cmf_sync
from src.schemas.requests import RecommendationRequest

def print_recs(title, recs):
    print(f"\n--- {title} ---")
    for i, r in enumerate(recs[:5]):
        print(f"{i+1}. {r.zone_name} ({r.borough}) - Score: {r.opportunity_score:.4f}")

# 1. Healthy Indian (should favor campus/lunch areas with high bias)
req_indian = RecommendationRequest(
    concept_subtype="healthy_indian",
    max_results=10
)
recs_indian = predict_cmf_sync(req_indian).recommendations
print_recs("Healthy Indian Recommendations", recs_indian)

# 2. Pizza (should favor business districts or have different ordering)
req_pizza = RecommendationRequest(
    concept_subtype="pizza",
    max_results=10
)
recs_pizza = predict_cmf_sync(req_pizza).recommendations
print_recs("Pizza Recommendations", recs_pizza)

# 3. Check for Borough Diversity
boroughs_indian = [r.borough for r in recs_indian[:5]]
print(f"\nBoroughs (Indian): {boroughs_indian}")
boroughs_pizza = [r.borough for r in recs_pizza[:5]]
print(f"Boroughs (Pizza): {boroughs_pizza}")

# Verification logic
if recs_indian[0].zone_id == recs_pizza[0].zone_id:
    # They might still be the same if one spot is TRULY dominant, but check for variation in the list
    ids_indian = [r.zone_id for r in recs_indian[:5]]
    ids_pizza = [r.zone_id for r in recs_pizza[:5]]
    if ids_indian == ids_pizza:
        print("\nFAILURE: Recommendations are identical across categories.")
    else:
        print("\nSUCCESS: Recommendations lists differ across categories.")
else:
    print("\nSUCCESS: Top recommendation differs across categories.")

# Check for diversity (should not be all Manhattan/Brooklyn)
if len(set(boroughs_indian)) > 1:
    print("SUCCESS: Borough diversity achieved in Indian recs.")
else:
    print("WARNING: Low borough diversity in Indian recs.")

if len(set(boroughs_pizza)) > 1:
    print("SUCCESS: Borough diversity achieved in Pizza recs.")
else:
    print("WARNING: Low borough diversity in Pizza recs.")
