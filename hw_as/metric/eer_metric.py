from hw_as.metric.base_metric import BaseMetric
from hw_as.metric.eer_calc import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prediction, target, **kwargs):
        detached_prediction = prediction.detach().cpu().numpy()
        detached_target = target.detach().cpu().numpy()
        bona_cm = detached_prediction[detached_target == 1, 1]
        spoof_cm = detached_prediction[detached_target == 0, 1]
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
        return eer_cm