# Cross-scan fine-tuning: pre-training implementation amendment 04

Status: **discovered after v2 materialization and preprocessing, but before any model training,
pilot prediction, experiment prediction, score, or outcome inspection**.

The replacement execution lock with content SHA-256
`f2006d6e435c50aaaf6b3b63e2654e90d0c9c71e714c52d1b8f367ae2b8cb8bd` completed all 384
case materializations and preprocessing of all 288 training/validation cases. Its preprocessing
receipt has content SHA-256
`59c4a045620da0f2179f383fa8d3767d01a67faafeaa956281f9b9af4b98f88c`.
An independent audit re-hashed all 1,158 preprocessing files and found all 1,156 non-lock-bound
files byte-identical to v1. The subsequent synthetic-only GPU memory gate stopped while loading
the frozen released checkpoint, before optimizer construction or a synthetic optimization step.
It produced no PASS receipt. The data root contained no training directory, training receipt,
checkpoint, prediction, score, pilot result, or final result when this amendment was written.

The affected lock is retained at
`results/crossscan_finetune/execution_lock.superseded-20260811-pretraining-v2.json`. The completed
v2 data root is retained as failed pre-training provenance and is not rebound to a replacement
lock. A fresh data root must materialize and preprocess under the next replacement lock before
training may begin.

## Restrict the trusted checkpoint load explicitly

Villa's pinned `load_pretrained_weights` calls `torch.load` without specifying `weights_only`.
The locked PyTorch 2.13 runtime therefore uses its restricted weights-only default. The released
m7 checkpoint contains NumPy scalar metadata, so the restricted loader rejected it before
transferring any parameter.

The runtime now verifies the checkpoint's frozen byte length and SHA-256 before deserialization,
requires PyTorch's static unsafe-global scan to equal exactly `numpy.dtype` and
`numpy._core.multiarray.scalar`, and scopes a safe-global allowlist to those two types plus the
three concrete NumPy dtype classes actually constructed by the checkpoint: float32, float64, and
int64. A standalone restricted load of the exact frozen checkpoint succeeds and exposes 956
network tensors. No unrestricted pickle load is introduced for the released checkpoint.

The same helper is used by the public synthetic GPU memory smoke and every training fold. Its
policy is recorded in the smoke receipt and every training receipt, and receipt validation
requires the exact policy. Unit tests reject checkpoint hash drift and any added static unsafe
global. The replacement lock binds this amendment, the corrected runtime, the public memory
smoke, the regression test, and the repository attributes that force both locked smoke scripts
to LF bytes on every checkout.

This compatibility correction changes no label mapping, case, fold, seed, duration, optimizer
value, augmentation, normalization, endpoint, effect threshold, statistical test, safety gate,
visual case, or permitted pilot adaptation.
