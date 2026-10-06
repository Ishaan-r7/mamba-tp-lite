"""
Quantization-error evaluation harness (Phase 6).

Measures only the error introduced by quantizing AllReduce payloads: every
config is compared against the fp32 TP model's own logits on the same
passages, position by position (KL(fp32 || quantized), Top-1, Top-5 ordered
and unordered), then averaged per passage so the spread across passages can
be reported.

`EvalPool` keeps two gloo workers (and the loaded model shards) alive and
swaps the per-site AllReduce spec between evaluations, so a sweep over many
configs pays process start-up and checkpoint load once.
"""
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

WIKI_FILE = ("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet")


def load_passages(tokenizer, n_passages: int, length: int, seed: int = 0, min_chars: int = 3000):
    """n_passages token windows of `length` tokens from distinct Simple English
    Wikipedia articles (long enough to fill the window), as [n, length] ids."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKI_FILE[0], WIKI_FILE[1], repo_type="dataset")
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    texts = [t for t in texts if len(t) >= min_chars]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=g).tolist()
    rows = []
    for i in order:
        ids = tokenizer(texts[i], return_tensors="pt").input_ids[0]
        if ids.numel() >= length:
            rows.append(ids[:length])
        if len(rows) == n_passages:
            break
    assert len(rows) == n_passages, f"only found {len(rows)} long-enough articles"
    return torch.stack(rows)


def quant_metrics(ref: torch.Tensor, q: torch.Tensor) -> dict:
    """Per-passage means over positions. ref/q: [b, l, V] logits."""
    lp_r = F.log_softmax(ref, dim=-1)
    lp_q = F.log_softmax(q, dim=-1)
    kl = (lp_r.exp() * (lp_r - lp_q)).sum(-1)
    tr = ref.topk(5, dim=-1).indices
    tq = q.topk(5, dim=-1).indices
    return {
        "kl": kl.mean(1),
        "top1": (tr[..., 0] == tq[..., 0]).float().mean(1),
        "top5_ordered": (tr == tq).all(-1).float().mean(1),
        "top5_unordered": (tr.sort(-1).values == tq.sort(-1).values).all(-1).float().mean(1),
    }


def summarize(per_passage: dict) -> dict:
    out = {}
    for k, v in per_passage.items():
        v = torch.tensor(v)
        out[k] = {"mean": v.mean().item(), "std": v.std().item() if v.numel() > 1 else 0.0}
    return out


def _worker(rank, world_size, init_file, model_path, ids, batch, threads, cmd_q, res_q):
    from .tp_model import TPMambaLM

    torch.set_num_threads(threads)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    model = TPMambaLM.from_pretrained_shard(model_path, rank=rank, world_size=world_size).eval()

    def run(spec):
        model.set_allreduce_dtype(spec)
        outs = []
        with torch.no_grad():
            for i in range(0, ids.shape[0], batch):
                logits, _ = model.prefill(ids[i:i + batch])
                outs.append(logits if rank == 0 else None)
        return outs

    ref = run("fp32")
    if rank == 0:
        res_q.put("ready")
    while True:
        spec = cmd_q.get()
        if spec is None:
            break
        t0 = time.time()
        outs = run(spec)
        if rank == 0:
            parts = [quant_metrics(r, o) for r, o in zip(ref, outs)]
            res = {k: torch.cat([p[k] for p in parts]).tolist() for k in parts[0]}
            res["seconds"] = time.time() - t0
            res_q.put(res)
    dist.destroy_process_group()


class EvalPool:
    def __init__(self, model_path: str, ids: torch.Tensor, batch: int = 8, world_size: int = 2, threads: int = 4):
        ctx = mp.get_context("spawn")
        self.world_size = world_size
        self.cmd_qs = [ctx.Queue() for _ in range(world_size)]
        self.res_q = ctx.Queue()
        self._tmp = tempfile.TemporaryDirectory()
        init_file = os.path.join(self._tmp.name, "rendezvous")
        self.procs = [
            ctx.Process(target=_worker, args=(r, world_size, init_file, model_path, ids, batch, threads,
                                              self.cmd_qs[r], self.res_q))
            for r in range(world_size)
        ]
        for p in self.procs:
            p.start()
        assert self.res_q.get(timeout=1800) == "ready"

    def evaluate(self, spec, timeout: float = 7200) -> dict:
        """spec: dtype name or {(layer_idx, site): dtype}. Returns per-passage metric lists + seconds."""
        for q in self.cmd_qs:
            q.put(spec)
        return self.res_q.get(timeout=timeout)

    def close(self):
        for q in self.cmd_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=60)
        self._tmp.cleanup()
