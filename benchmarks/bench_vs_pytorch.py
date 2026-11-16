"""Reproducible Oxideon vs PyTorch CPU & GPU benchmark suite."""

import time
import oxideon as ox

try:
    import torch
except ImportError:
    torch = None


def bench(fn, iters: int = 25) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    if torch is None:
        print("PyTorch not installed; running Oxideon-only benchmarks.")

    cases = []

    # GEMM
    a_ox, b_ox = ox.randn([512, 512]), ox.randn([512, 512])
    A_ox, B_ox = ox.randn([2048, 2048]), ox.randn([2048, 2048])
    x_ox = ox.randn([64, 128, 768])
    ln_ox = ox.nn.LayerNorm(768)
    img_ox = ox.randn([16, 64, 32, 32])
    c_ox = ox.nn.Conv2d(64, 64, kernel_size=3, padding=1)
    m_ox = ox.nn.Sequential(ox.nn.Linear(768, 3072), ox.nn.GELU(), ox.nn.Linear(3072, 768))
    inp_ox = ox.randn([128, 768])

    def step_ox() -> None:
        for p in m_ox.parameters():
            p.grad = None
        y = m_ox(inp_ox).sum()
        y.backward()

    if torch is not None:
        a_th, b_th = torch.randn(512, 512), torch.randn(512, 512)
        A_th, B_th = torch.randn(2048, 2048), torch.randn(2048, 2048)
        x_th = torch.randn(64, 128, 768)
        ln_th = torch.nn.LayerNorm(768)
        img_th = torch.randn(16, 64, 32, 32)
        c_th = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        m_th = torch.nn.Sequential(torch.nn.Linear(768, 3072), torch.nn.GELU(), torch.nn.Linear(3072, 768))
        inp_th = torch.randn(128, 768)

        def step_th() -> None:
            m_th.zero_grad(set_to_none=True)
            y = m_th(inp_th).sum()
            y.backward()

        cases = [
            ("matmul 512x512", lambda: a_ox @ b_ox, lambda: a_th @ b_th, 30),
            ("matmul 2048x2048", lambda: A_ox @ B_ox, lambda: A_th @ B_th, 5),
            ("add [64,128,768]", lambda: x_ox + x_ox, lambda: x_th + x_th, 25),
            ("mul [64,128,768]", lambda: x_ox * x_ox, lambda: x_th * x_th, 25),
            ("relu [64,128,768]", lambda: x_ox.relu(), lambda: torch.relu(x_th), 25),
            ("sigmoid [64,128,768]", lambda: x_ox.sigmoid(), lambda: torch.sigmoid(x_th), 25),
            ("silu [64,128,768]", lambda: x_ox.silu(), lambda: torch.nn.functional.silu(x_th), 25),
            ("gelu [64,128,768]", lambda: x_ox.gelu(), lambda: torch.nn.functional.gelu(x_th), 25),
            ("exp [64,128,768]", lambda: x_ox.exp(), lambda: x_th.exp(), 25),
            ("softmax [64,128,768]", lambda: x_ox.softmax(-1), lambda: x_th.softmax(-1), 25),
            ("sum(-1) [64,128,768]", lambda: x_ox.sum(dim=-1), lambda: x_th.sum(dim=-1), 25),
            ("sum(0) [64,128,768]", lambda: x_ox.sum(dim=0), lambda: x_th.sum(dim=0), 25),
            ("layer_norm [64,128,768]", lambda: ln_ox(x_ox), lambda: ln_th(x_th), 25),
            ("conv2d [16,64,32,32]", lambda: c_ox(img_ox), lambda: c_th(img_th), 15),
            ("MLP fwd+bwd [128,768]", step_ox, step_th, 15),
        ]
        print(f"{'Benchmark':28s} | {'Oxideon':>9s} | {'PyTorch':>9s} | {'Ox/Th':>7s}")
        print("-" * 62)
        for name, f_ox, f_th, it in cases:
            t_ox = bench(f_ox, it)
            t_th = bench(f_th, it)
            print(f"{name:28s} | {t_ox:7.3f}ms | {t_th:7.3f}ms | {t_ox / t_th:6.2f}x")
    else:
        for name, fn, it in [
            ("matmul 512x512", lambda: a_ox @ b_ox, 30),
            ("matmul 2048x2048", lambda: A_ox @ B_ox, 5),
            ("softmax [64,128,768]", lambda: x_ox.softmax(-1), 25),
            ("layer_norm [64,128,768]", lambda: ln_ox(x_ox), 25),
            ("conv2d [16,64,32,32]", lambda: c_ox(img_ox), 15),
            ("MLP fwd+bwd [128,768]", step_ox, 15),
        ]:
            print(f"{name:28s} | {bench(fn, it):7.3f}ms")

    print("\n=== GPU Info ===")
    print("ox.cuda.is_available():", ox.cuda.is_available())
    if ox.cuda.is_available():
        print("CUDA device 0:", ox.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
