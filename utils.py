import numpy as np


def estimate_skill(test_scores, review_ratio):
    interval_range = 100 / len(test_scores)

    def _estimate_skill(test_score, lower_bound, max_points=100):
        points_no_review = test_score - max_points * review_ratio

        # if passed the review or this was a first-level test (= no review)
        if points_no_review >= 0 or lower_bound == 0:
            return lower_bound + (test_score / 100) * interval_range
        # if didn't pass the review, go back one level
        else:
            return _estimate_skill(test_score, lower_bound - interval_range,
                                   max_points * review_ratio)

    lower_bounds = np.linspace(0, 100, num=len(test_scores), endpoint=False)
    result = [_estimate_skill(score, bound)
              for score, bound in zip(test_scores, lower_bounds)]
    return max(result)
