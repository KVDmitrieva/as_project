import logging
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from hw_as.datasets.base_dataset import BaseDataset
from hw_as.utils import ROOT_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ASVspoofDataset(BaseDataset):
    def __init__(self, part, data_dir, *args, **kwargs):
        assert part in ["train", "dev", "eval"]
        self._data_dir = Path(data_dir)
        self._index_dir = ROOT_PATH / "data" / "datasets" / "asvspoof"
        self._index_dir.mkdir(exist_ok=True, parents=True)
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        suff = "trn" if part == "train" else "trl"
        protocols_src = self._data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.{suff}.txt"
        audio_src_dir = self._data_dir / f"ASVspoof2019_LA_{part}" / "flac"
        with protocols_src.open() as f:
            for line in tqdm(f, desc=f"Prepare {part} dataset"):
                speaker, utterance, ut_type, spoof_alg, target = line.strip().split(' ')
                flac_path = audio_src_dir / f"{utterance}.flac"
                t_info = torchaudio.info(str(flac_path))
                length = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": str(flac_path.absolute().resolve()),
                        "audio_len": length,
                        "spoof_algorithm": spoof_alg,
                        "target": int(target == "bonafide")
                    }
                )
        return index

