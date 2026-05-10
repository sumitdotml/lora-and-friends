# Figure 8: Throughput probe: seconds per optimizer step by request shape

## Caption

Pipelined batched training calls deliver a 8.4× throughput improvement over sequential single-datum calls on Qwen3-8B with attention-only LoRA (effective batch size 8). Batching alone — submitting all 8 examples in one API call — accounts for 3.9× of the speedup; pipelining, which submits the next batch before awaiting the prior optimizer step, contributes an additional 2.2× by overlapping communication with computation. Measured wall-clock seconds per optimizer step: sequential 20.2 s, batched 5.2 s, pipelined 2.4 s. The pipelined shape was adopted for all main training runs.

## Marker encoding

| element | color | shape | what is plotted |
| --- | --- | --- | --- |
| bars | `#1f77b4` | filled rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| value labels | `#333333` | text above bar | seconds per optimizer step rounded to two decimals |

## Source files

- Plotted points: `fig_08_throughput_probe.data.csv` (one row per request shape).
- Provenance and input SHA-256 hashes: `fig_08_throughput_probe.provenance.json`.
- Rendered: `fig_08_throughput_probe.pdf`, `fig_08_throughput_probe.png`, `fig_08_throughput_probe.grayscale.png`.
