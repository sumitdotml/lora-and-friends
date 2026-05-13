# Figure 8: Throughput by request shape

## Caption

Wall-clock seconds per optimizer step for Qwen3-8B attention-only LoRA at effective batch size 8. Batching cuts the step time from 20.2 s to 5.2 s, and pipelined batching lowers it to 2.4 s (8.4× faster than sequential).

## Marker encoding

| element | color | shape | what is plotted |
| --- | --- | --- | --- |
| sequential | `#d95f02` | horizontal rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| batched | `#1b9e77` | horizontal rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| pipelined | `#7570b3` | horizontal rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| value labels | `#333333` | text at bar end | seconds per optimizer step and speedup versus sequential |

## Source files

- Plotted points: `fig_08_throughput_probe.data.csv` (one row per request shape).
- Provenance and input SHA-256 hashes: `fig_08_throughput_probe.provenance.json`.
- Rendered: `fig_08_throughput_probe.pdf`, `fig_08_throughput_probe.png`, `fig_08_throughput_probe.grayscale.png`.
