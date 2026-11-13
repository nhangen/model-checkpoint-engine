# Repairs exposed by downstream enforcing CI

This repair uses existing issues #7, #9–#11, #14, #21, #27, and #38–#42.
It preserves the engine integration; it does not delete the Phase2/3 modules or
claim that every feature or security issue in those modules is resolved.

The downstream pinned revision was `190c351`. This branch starts from current
upstream main `7273cde`, preserving its intervening changes. Upstream commit
`001f151` had already removed the domain-specific SystemValidator and its tests;
the two corresponding old failing nodes are absent upstream, not counted as
passing repairs here. The downstream pose repository keeps its own validator.

The repair covers:

- Idempotent migration003 column additions, populated upgrades through the
  historical004 index, and a new005 migration replacing the non-deterministic
  timestamp index. Historical SQL files and their recorded checksums are retained.
- Valid standalone experiment rows, a single best-metric decision per save,
  best-state restoration after reopen, and retention that preserves actual best
  loss/validation checkpoints plus recent history. Retention deletes matching
  database rows and backend-owned files. Protected best/latest records can exceed
  a configured limit smaller than the number of distinct protected records.
- JSON serialization of bulk checkpoint dictionaries and accurate CSV output-file
  metadata. Fixture foreign keys stay enabled.
- Restricted `weights_only=True` reads in the PyTorch backend and verifier. New
  PyTorch writes use protocol2 and normalize NumPy scalar metadata without
  changing model/optimizer/scheduler state or state-dict version metadata.
- Real cloud-provider and CloudManager tests using a stateful in-memory S3
  transport. These execute upload/download/list/metadata/retention paths; they
  do not establish live AWS compatibility or configure credentials.
- Current analytics/plugin/monitor APIs, missing training lifecycle events
  appended after all existing enum IDs, actual REST parameter dispatch, and
  backend metadata/delete methods. REST tests use the real temporary manager.
- Package exports point to existing implementations rather than fictional files.

## Authorization and checkpoint compatibility

The previously declared `auth_required` flag is enforced before a handler or
cached response runs. Protected REST endpoints return401 unless the application
supplies an `authorize(headers) -> bool` callback to `RestAPI`. Health/info remain
public. This repair configures no credentials and starts no servers. Applications
must supply their own authorization policy before using protected endpoints;
tests inject an explicit test-only authorizer. This is not a claim of a complete
HTTP/authentication security audit.

Existing arbitrary Python objects and protocol4/raw-pickle checkpoints cannot
be made safe by switching an unrestricted load's keyword alone. The standard
restricted reader rejects representative engine-written protocol4 files. This
repair makes future tensor/state-dict writes restricted-readable and leaves the
legacy-format workflow decision explicit; it does not add an unrestricted
fallback, broad safe-global allowlist, scanner suppression or security exception.
The legacy core/migration adapters retain separate known security issues; this
change must not be described as eliminating all engine deserialization findings.

Safetensors tests are required for this repair, not optional skips. The backend
retains state-dict version metadata alongside weights and deletion/retention
handle its deterministic adjacent weight sidecar without unpickling metadata
for path discovery.

## Verification

Install CPU PyTorch separately for the target platform, then install the source
requirements and `requirements-test.txt`. Run `python -m pytest tests/ -n 0`.
The explicit `-n 0` respects the downstream local runner's CPU allocation.
The downstream pose workflow pins the reviewed engine commit and installs these
test dependencies before running this entire suite. No assertion failure is
converted to success.

Focused regression coverage includes populated/already004 upgrades, invalid
migration failure propagation, optimizer/scheduler/scalar-metadata roundtrips,
BatchNorm state metadata, best-loss/validation retention, reopen behavior,
corruption detection, rejected code-execution payloads, all62 original enum IDs,
real REST metadata/delete/auth/cache behavior, and safetensors retention/deletion.
Model smoke tests and transport stubs establish execution, not model accuracy,
generalization, live AWS interoperability or a global security audit.

## Explicit legacy conversion

Ordinary pose resume/inference uses the restricted reader and never calls this
converter automatically. For a supported legacy tensor/state-dict checkpoint
whose provenance you trust:

```sh
python -m model_checkpoint.migration.convert_legacy \
  --input /path/to/trusted-legacy-file \
  --output /path/to/new-checkpoint.pt --trust-input
```

**This opt-in conversion can execute arbitrary Python code from the input.**
It does not make untrusted pickle safe. The guard runs before input I/O or
unpickling and the warning is printed before deserialization. There is no
environment variable, remembered trust decision or automatic fallback. Originals
and existing outputs are preserved. The supported converted tensor, optimizer,
scheduler and primitive metadata are verified with restricted loading before an
atomic, exclusive output publication. Unsupported objects/mapping attributes fail
instead of being silently discarded; the result must not be claimed to preserve
arbitrary Python object semantics.

The converter intentionally contains unrestricted read operations behind this
explicit opt-in. Their static security findings are not suppressed; this is not
a claim that a global engine Bandit scan is clean. Normal pose loading has no
unrestricted read or converter call. Existing generic legacy core/adapters have
separate security issues that this bounded downstream repair does not conceal.

The restricted checkpoint feature requires stable **Torch>=2.15.0** (and thus
Python3.11+ for current Torch wheels). Metadata APIs are not given a new global
Python floor. Install `requirements-restricted.txt` or the
`restricted-checkpoints` extra for this feature. A runtime guard rejects older
or unrecognized/prerelease versions before deserialization, including conversion
preflight. The dependency constraint alone is insufficient for existing installs.
The floor addresses the official [restricted-loader RCE advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
and [file-disclosure advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-v32g-mr63-3j83).
This is a known-patch minimum, not a promise that arbitrary model files are free
of resource-exhaustion or future library risks.
