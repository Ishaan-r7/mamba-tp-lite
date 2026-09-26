"""
Verifies mamba_lite.MambaLM against HF's MambaForCausalLM:
  1. prefill logits match closely (allclose)
  2. greedy-decoded tokens match exactly, using our own cache (prefill + step)
  3. our own cached decode reproduces what a single full prefill would compute
"""
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, MambaForCausalLM

from mamba_lite.model import MambaLM

MODEL_ID = "state-spaces/mamba-130m-hf"


def _load_both():
    path = snapshot_download(MODEL_ID)
    ours = MambaLM.from_pretrained(path, dtype=torch.float32).eval()
    hf = MambaForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return ours, hf, tok


def test_prefill_logits_match():
    ours, hf, tok = _load_both()
    ids = tok("The quick brown fox jumps over", return_tensors="pt").input_ids

    with torch.no_grad():
        our_logits, _ = ours.prefill(ids)
        hf_logits = hf(ids).logits

    torch.testing.assert_close(our_logits, hf_logits, atol=1e-3, rtol=1e-3)


def test_greedy_generation_matches():
    ours, hf, tok = _load_both()
    ids = tok("Once upon a time", return_tensors="pt").input_ids
    n_new = 20

    with torch.no_grad():
        our_tokens = ours.generate(ids, max_new_tokens=n_new, greedy=True)
        hf_out = hf.generate(ids, max_new_tokens=n_new, do_sample=False)

    assert torch.equal(our_tokens, hf_out), (
        f"ours: {tok.decode(our_tokens[0])}\nhf:   {tok.decode(hf_out[0])}"
    )


def test_cached_step_matches_full_prefill_logits():
    """
    Feed the same known token sequence two ways:
      (a) one full prefill() over the whole sequence
      (b) prefill() on token 0, then step() one token at a time for the rest,
          using the *actual* tokens (teacher forcing, not self-generated)
    Logits at every position should match closely -> the cache carries exactly
    the state a full prefill would have produced at that point.
    """
    ours, _, tok = _load_both()
    ids = tok("Once upon a time there was a", return_tensors="pt").input_ids
    seq_len = ids.shape[1]

    with torch.no_grad():
        full_logits, _ = ours.prefill(ids)  # [1, seq_len, vocab]

        _, cache = ours.prefill(ids[:, :1])
        cached_logits = [full_logits[:, :1]]  # position 0 comes from the same prefill call
        for t in range(1, seq_len):
            logits_t = ours.step(ids[:, t : t + 1], cache)  # teacher-forced actual token
            cached_logits.append(logits_t)
        cached_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(cached_logits, full_logits, atol=1e-3, rtol=1e-3)
