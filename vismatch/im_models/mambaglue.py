import os
import shutil
import tempfile
from pathlib import Path

from torch.hub import download_url_to_file

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MambaGlue"))

from mambaglue import ALIKED, DISK, DoGHardNet, MambaGlue, SIFT, SuperPoint, match_pair


MAMBAGLUE_VERSION = "v0.1"
MAMBAGLUE_RELEASE_URL = "https://github.com/url-kaist/MambaGlue/releases/download/{version}/{weights}.tar"
MAMBAGLUE_WEIGHTS = {
    "sift": "sift_mambaglue",
    "superpoint": "superpoint_mambaglue",
    "disk": "disk_mambaglue",
    "aliked": "aliked_mambaglue",
    "doghardnet": "doghardnet_mambaglue",
}


def _load_pretrained_mambaglue(feature: str) -> MambaGlue:
    """Load official MambaGlue release weights without modifying the submodule."""
    weights = MAMBAGLUE_WEIGHTS[feature]
    cache_dir = Path.home() / ".cache" / "vismatch" / "mambaglue"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_checkpoint = cache_dir / f"{weights}.tar"

    if not cached_checkpoint.exists():
        url = MAMBAGLUE_RELEASE_URL.format(version=MAMBAGLUE_VERSION, weights=weights)
        download_url_to_file(url, cached_checkpoint)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copyfile(cached_checkpoint, tmp_path / "checkpoint_best.tar")
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            return MambaGlue(features=feature)
        finally:
            os.chdir(original_cwd)


class MambaGlueBase(BaseMatcher):
    """Base wrapper for MambaGlue matcher with different local feature extractors."""

    def _forward(self, img0, img1):
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, img0, img1, device=self.device)

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]
        matches = matches01["matches"]

        desc0 = feats0["descriptors"]
        desc1 = feats1["descriptors"]

        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1


class SiftMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = _load_pretrained_mambaglue("sift").eval().to(self.device)


class SuperpointMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = _load_pretrained_mambaglue("superpoint").eval().to(self.device)


class DiskMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = _load_pretrained_mambaglue("disk").eval().to(self.device)


class AlikedMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = _load_pretrained_mambaglue("aliked").eval().to(self.device)


class DognetMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = _load_pretrained_mambaglue("doghardnet").eval().to(self.device)
