# import einops
# import paddleaudio
# from paddle.nn.functional import pad
#
# from kappadata.transforms.base.kd_transform import KDTransform
#
#
# class KDFbank(KDTransform):
#     def __init__(self, target_length, htk_compat=False, num_mel_bins=23, window_type="povey", **kwargs):
#         super().__init__(**kwargs)
#         self.target_length = target_length
#         self.htk_compat = htk_compat
#         self.num_mel_bins = num_mel_bins
#         self.window_type = window_type
#
#     def __call__(self, x, ctx=None):
#         assert ctx is not None and "sampling_frequency" in ctx
#         x = paddleaudio.compliance.kaldi.fbank(
#             x,
#             htk_compat=self.htk_compat,
#             num_mel_bins=self.num_mel_bins,
#             sample_frequency=ctx["sampling_frequency"],
#             window_type=self.window_type,
#         )
#
#         # pad/cut to target_length
#         delta = self.target_length - x.shape[0]
#         if delta > 0:
#             x = pad(x, (0, 0, 0, delta), "constant", 0.)
#         elif delta < 0:
#             x = x[:self.target_length]
#
#         # to image format
#         x = einops.rearrange(x, "time freq -> 1 time freq")
#
#         return x
