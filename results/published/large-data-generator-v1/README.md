# Tested larger-corpus generator and completed benchmark

This snapshot publishes the tested generator/configuration and completed small
benchmark. It is not a completed large dataset or a new model result. The main
48-worker generation was launched separately; its accepted counts remain pending.

Benchmark:77 admitted roots from42 independent source families,2,464 R32 policy
continuations,83.32s including game generation/selection/audit, zero unresolved.
The same-root board-copy benchmark exactly matched every trajectory and label in
58.29s including worker startup; the scopes differ, so this is not a direct
speedup ratio. All17 new generator tests pass.

Read docs/LARGE_DATASET.md and configs/v2_large_dataset_v1.json. The frozen main
request is256/64/64 source families and128/64/64 demo families per rule,13 variants
plus standard, R32: at most10,752 roots and344,064 continuations before conservative
geometry/short-source exclusions. No main query count is claimed in this snapshot.
The raw benchmark and all eventual main traces/shards stay in local deliverables.
