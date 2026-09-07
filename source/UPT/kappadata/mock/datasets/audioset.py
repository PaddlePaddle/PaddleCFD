# import paddle
# import yaml
# import paddleaudio
# from kappadata.utils.param_checking import to_2tuple
# from .utils import setup
#
#
# def generate_mock_audioset(
#         dst,
#         num_unbalanced_samples=2000,
#         num_balanced_samples=1000,
#         num_eval_samples=1000,
#         labels_per_sample_max=5,
#         seed=0,
#         log_fn=None,
# ):
#     log, dst, generator = setup(log_fn=log_fn, dst=dst, seed=seed)
#
#     with open("kappadata/res/audioset_class_ids.yaml") as f:
#         class_ids = yaml.safe_load(f)
#
#     log(f"generating mock AudioSet into '{dst.as_posix()}'")
#     for split, num_samples in [
#         ("unbalanced_train_segments", num_unbalanced_samples),
#         ("balanced_train_segments", num_balanced_samples),
#         ("eval_segments", num_eval_samples),
#     ]:
#         split_uri = dst / split
#         split_uri.mkdir()
#
#         log(f"generating {num_samples} {split} samples")
#         sample_rows = []
#         num_positive_labels_total = 0
#         for i in range(num_samples):
#             # generate id
#             ytid = "".join([chr(ord("a") + v) for v in paddle.randint(26, size=(11,), generator=generator)])
#             # generate waveform
#             waveform = paddle.randn(1, 16000, generator=generator)
#             paddleaudio.save(split_uri / f"{ytid}.wav", waveform, sample_rate=16000)
#             # generate labels
#             num_positive_labels = paddle.randint(1, labels_per_sample_max, size=(1,), generator=generator)
#             num_positive_labels_total += num_positive_labels.item()
#             positive_class_ids = [class_ids[i] for i in paddle.randperm(len(class_ids))[:num_positive_labels]]
#             row = f'{ytid}, 0.000, 1.000, "{",".join(positive_class_ids)}"'
#             sample_rows.append(row)
#
#         # write csv
#         meta_rows = [
#             "# Segments csv created",
#             (
#                 f"# num_ytids={len(sample_rows)}, num_segs={len(sample_rows)}, "
#                 f"num_unique_labels={len(class_ids)}, num_positive_labels={num_positive_labels_total}"
#             ),
#             "# YTID, start_seconds, end_seconds, positive_labels",
#         ]
#         with open(split_uri.with_suffix(".csv"), "w") as f:
#             f.writelines([f"{row}\n" for row in meta_rows + sample_rows])
