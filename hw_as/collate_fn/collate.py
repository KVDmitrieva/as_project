import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectrogram, audio = [], []
    target = []

    for item in dataset_items:
        audio.append(item["audio"].T)
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        target.append(item["target"])

    return {
        "target": torch.tensor(target),
        "audio": pad_sequence(audio, batch_first=True).transpose(1, 2),
        "mel":  pad_sequence(spectrogram, batch_first=True, padding_value=config.pad_value).transpose(1, 2)
    }