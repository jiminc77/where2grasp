import torch
import genesis as gs
from _common import probe_main

def body():
    gs.init(seed=0, precision="64", logging_level="warning", backend=gs.gpu)
    assert gs.backend == gs.cuda, (gs.backend, gs.cuda)
    assert torch.version.cuda == "12.8", torch.version.cuda
    name = torch.cuda.get_device_name(0)
    assert "RTX PRO 6000" in name, name
    a = torch.randn((1024, 1024), device="cuda")
    b = torch.randn((1024, 1024), device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert torch.isfinite(c).all().item()
    print(f"backend={gs.backend} cuda={torch.version.cuda} device={name} matmul_sum={c.sum().item():.3f}")
    return f"backend=cuda; device={name}; cuda={torch.version.cuda}"
if __name__ == "__main__": raise SystemExit(probe_main("p1_backend", body))
