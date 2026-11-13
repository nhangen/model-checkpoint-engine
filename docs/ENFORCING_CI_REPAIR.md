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
