import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio = []
    target = []
    audio_path = []

    for item in dataset_items:
        audio.append(item["audio"].T)
        target.append(item["target"])
        audio_path.append(item["audio_path"])

    return {
        "audio_path": audio_path,
        "target": torch.tensor(target),
        "audio": pad_sequence(audio, batch_first=True).transpose(1, 2)
    }