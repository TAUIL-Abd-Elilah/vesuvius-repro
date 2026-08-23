# PHerc0139 fiber benchmark — transport-only amendment

The first and only outcome attempt began from public preregistration commit
`b309c4c4d23686e14fe8b82faa6413e29789275d` and wrote its exclusive marker
before prediction access. It completed all 32,490 planned requests, but 22
requests failed with transient TLS `DECRYPTION_FAILED_OR_BAD_RECORD_MAC`
errors. The runner aborted before `zarr.open`, sampling, metrics, or panels, as
required by the preregistration. This is a technical failure, not a null.

The machine-readable failure record is
`pherc0139_fiber_transport_failure.json`. It fixes the exact 22 failed object
keys, the 14 HTTP-404 zero-fill keys inferred from the completed task set, the
original marker receipt, and a receipt-set digest covering every byte in the
6.78 GB partial mirror.

## Permitted recovery

One transport-only resume is allowed after this amendment is committed and
published under annotated tag
`pherc0139-fiber-sanity-transport-amendment-v1`:

1. Keep the original lock, manifest, samples, offsets, gates, and visual IDs
   byte-for-byte unchanged.
2. Keep and verify the original `OUTCOME_STARTED` marker. Do not create a
   second marker or outcome slot.
3. Recompute the receipt-set digest of every existing mirror object and abort
   unless it equals the failure record.
4. Rehash every cached object, keep the 14 first-attempt HTTP 404 objects as
   pinned zero fills without requesting them again, and retry only the exact
   22 TLS-failed objects. Any new non-404 failure aborts.
5. Open the arrays and compute the frozen analysis only after the mirror has
   no non-404 error.
6. Record both the original preregistration and this amendment in the result.
   Publish a positive, null, or second technical failure without tuning.

The amended runner writes an exclusive `TRANSPORT_RESUME_STARTED` marker
before new requests and persists `TRANSPORT_RESUME_FAILURE.json` if that
single resume fails before producing `result.json`; another retry would need a
new public amendment.

No scientific choice is changed by this amendment.
