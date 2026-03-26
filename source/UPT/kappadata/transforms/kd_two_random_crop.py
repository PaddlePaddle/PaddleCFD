from kappadata.compat.image import get_image_size, crop

from kappadata.utils.bounding_box_utils import intersection_area_ijkl
from .kd_random_crop import KDRandomCrop


class KDTwoRandomCrop(KDRandomCrop):
    def __init__(self, *args, overlap_min=None, overlap_max=None, tries=20, **kwargs):
        super().__init__(*args, **kwargs)
        overlap_min = overlap_min or 0.
        overlap_max = overlap_max or 1.
        assert 0. <= overlap_min <= 1., overlap_min
        assert 0. <= overlap_max <= 1., overlap_max
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        self.tries = tries

    def __call__(self, img, ctx=None):
        img = self._pad_image(img)

        # make initial crop
        i0, j0, h0, w0 = self.get_params(img)
        crop0 = crop(img, i0, j0, h0, w0)

        # make second crop
        out_of_tries = False
        k0 = i0 + h0
        l0 = j0 + w0
        area0 = h0 * w0
        tries = 0
        while True:
            i1, j1, h1, w1 = self.get_params(img)
            k1 = i1 + h1
            l1 = j1 + w1
            area1 = h1 * w1
            area_intersection = intersection_area_ijkl(
                i0=i0, j0=j0, k0=k0, l0=l0,
                i1=i1, j1=j1, k1=k1, l1=l1,
            )
            area_union = area0 + area1 - area_intersection
            overlap = area_intersection / area_union
            if self.overlap_min <= overlap <= self.overlap_max:
                break
            tries += 1
            if self.tries is not None and tries >= self.tries:
                out_of_tries = True
                break
        crop1 = crop(img, i1, j1, h1, w1)

        if ctx is not None:
            ctx["two_random_crop"] = dict(
                i0=i0, j0=j0, h0=h0, w0=w0,
                i1=i1, j1=j1, h1=h1, w1=w1,
                out_of_tries=out_of_tries,
                overlap=overlap,
            )
        return [crop0, crop1]

    def get_params(self, img):
        w, h = get_image_size(img)
        th, tw = self.size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = int(self.rng.integers(0, h - th + 1))
        j = int(self.rng.integers(0, w - tw + 1))
        return i, j, th, tw
