# Contributing

Small, evidence-backed improvements are welcome.

Before opening a pull request:

1. Keep model weights, generated media, credentials, and private endpoints out
   of the repository.
2. Run `make audit`, `make test`, and `bash -n scripts/*.sh`.
3. State the exact hardware, image digest, network transport, command, and
   observed result for runtime changes.
4. Keep modifications to third-party Apache-2.0 interfaces narrow and clearly
   attributed.
5. Do not generalize two-machine measurements into vendor benchmarks or
   upstream support claims.
6. Re-run a cold start, health/model identity checks, an NCCL transport check,
   and a real decoded audio-video request for executor changes.

Runtime pull requests should explain the failure being addressed, why the
change is scoped narrowly, and whether both ranks were observed computing the
same request.
