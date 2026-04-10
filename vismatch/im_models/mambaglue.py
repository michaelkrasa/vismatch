from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MambaGlue"))

from mambaglue import ALIKED, DISK, DoGHardNet, MambaGlue, SIFT, SuperPoint, match_pair


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
        self.matcher = MambaGlue(features="sift").eval().to(self.device)


class SuperpointMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="superpoint").eval().to(self.device)


class DiskMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="disk").eval().to(self.device)


class AlikedMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="aliked").eval().to(self.device)


class DognetMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="doghardnet").eval().to(self.device)
