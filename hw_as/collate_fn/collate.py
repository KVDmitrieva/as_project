import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    target = []
    audio_path = []
    spectrogram = []

    for item in dataset_items:
        target.append(item["target"])
        audio.append(item["audio_path"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)

    return {
        "audio_path": audio_path,
        "target": torch.tensor(target),
        "mel":  pad_sequence(spectrogram, batch_first=True, padding_value=config.pad_value).transpose(1, 2)
    }