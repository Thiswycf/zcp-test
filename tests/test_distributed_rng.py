import json
import os
import subprocess
import sys
from pathlib import Path


def test_two_rank_checkpoint_round_trip_restores_rank_local_rng(tmp_path):
    script = tmp_path / "distributed_rng_round_trip.py"
    result_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    script.write_text(
        """import json
import random
import sys

import numpy as np
import torch

from zcp_test.training.checkpoint import atomic_torch_save, load_checkpoint
from zcp_test.training.trainer import _collect_checkpoint_rng, _restore_checkpoint_rng


result_path, checkpoint_path = sys.argv[1:]
torch.distributed.init_process_group(backend="gloo")
rank = torch.distributed.get_rank()
random.seed(1000 + rank)
np.random.seed(2000 + rank)
torch.manual_seed(3000 + rank)

checkpoint_rng = _collect_checkpoint_rng(True, rank)
expected = [random.random(), float(np.random.random()), float(torch.rand(()))]
if rank == 0:
    atomic_torch_save(checkpoint_rng, checkpoint_path)
torch.distributed.barrier()

loaded = load_checkpoint(checkpoint_path, trusted=True)
random.seed(9000)
np.random.seed(9000)
torch.manual_seed(9000)
_restore_checkpoint_rng(loaded, True, rank)
actual = [random.random(), float(np.random.random()), float(torch.rand(()))]
gathered = [None] * torch.distributed.get_world_size()
torch.distributed.all_gather_object(gathered, {"rank": rank, "expected": expected, "actual": actual})

if rank == 0:
    states = loaded["rng_by_rank"]
    payload = {
        "world_size": len(states),
        "states_distinct": not torch.equal(states[0]["torch"], states[1]["torch"]),
        "ranks": gathered,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
torch.distributed.barrier()
torch.distributed.destroy_process_group()
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(script),
            str(result_path),
            str(checkpoint_path),
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=60,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["world_size"] == 2
    assert result["states_distinct"] is True
    assert [record["rank"] for record in result["ranks"]] == [0, 1]
    for record in result["ranks"]:
        assert record["actual"] == record["expected"]
