from hw_as.metric.base_metric import BaseMetric
from hw_as.metric.eer_calc import compute_eer

class EERMetric(BaseMetric):
    def __init__(self, args, kwargs):
        super.__init__(*args, **kwargs)

    def __call__(self, prediction, target, **kwargs):
        bona_cm = cm_scores[target == 1, 1].detach().cpu()
        spoof_cm = cm_scores[target == 0, 1].detach().cpu()
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
        return eer_cm