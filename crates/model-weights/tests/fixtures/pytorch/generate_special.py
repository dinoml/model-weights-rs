"""Regenerate the sparse, quantized, and DEFLATE fixtures using PyTorch CPU."""

import io
import json
from pathlib import Path
import zipfile

import torch


def entries():
    matrix = torch.tensor(
        [[0.0, -4.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0],
         [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]
    )
    values = torch.arange(-31.0, 34.0).reshape(5, 13) / 8.3
    q_i8 = torch.quantize_per_tensor(values, 0.031234567891, -29, torch.qint8)
    return {
        "csr": matrix.to_sparse_csr(),
        "csc": matrix.to_sparse_csc(),
        "bsr": matrix.to_sparse_bsr((2, 2)),
        "bsc": matrix.to_sparse_bsc((2, 2)),
        "strided": values.T,
        "coo": torch.sparse_coo_tensor(
            [[2, 0, 2, 1, 2], [0, 1, 0, 1, 0]], [1.25, 3.5, -4.0, 2.0, 6.0], (3, 2)
        ),
        "coo_hybrid": torch.sparse_coo_tensor([[0, 2, 0]], [[-1.0, 2.0], [3.0, 4.0], [2.0, -4.0]], (4, 2)),
        "coo_half": torch.sparse_coo_tensor(
            [[0, 0, 0, 0, 1]], torch.tensor([1.0, 0.0004, 0.0004, 0.0004, 2.0], dtype=torch.float16), (2,)
        ),
        "coo_bool": torch.sparse_coo_tensor([[0, 0, 2]], torch.tensor([True, False, True]), (4,)),
        "coo_complex": torch.sparse_coo_tensor([[0, 0, 1]], torch.tensor([1j, 2 - 3j, 1 + 0j]), (3,)),
        "coalesced": matrix.to_sparse().coalesce(),
        "empty": torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (2, 3)),
        "csr_i32": torch.sparse_csr_tensor(
            torch.tensor([0, 1, 1, 2], dtype=torch.int32),
            torch.tensor([2, 1], dtype=torch.int32),
            torch.tensor([5, 7], dtype=torch.int32),
            size=(3, 4),
        ),
        "csr_batched": torch.stack([matrix, matrix * 2]).to_sparse_csr(),
        "csr_hybrid": torch.stack([matrix, matrix * 2], -1).to_sparse_csr(dense_dim=1),
        "q_i8": q_i8,
        "q_u8": torch.quantize_per_tensor(values, 0.017304892738, 130, torch.quint8),
        "q_i32": torch.quantize_per_tensor(values * 1234567, 0.032346276383, -1345814, torch.qint32),
        "q_i32_extrema": torch._make_per_tensor_quantized_tensor(
            torch.tensor([-(2**31), 2**31 - 1, -17_000_003, 17_000_003], dtype=torch.int32),
            0.031234567891, -1234567890,
        ),
        "q_u4": torch.quantize_per_tensor(values, 0.51, 8, torch.quint4x2),
        "q_u2": torch.quantize_per_tensor(values, 1.32, 1, torch.quint2x4),
        "q_channel": torch.quantize_per_channel(
            values, torch.linspace(0.03456789, 0.09123456, 13, dtype=torch.float64), torch.arange(-6, 7), 1, torch.qint8
        ),
        "q_float": torch.quantize_per_channel(
            values, torch.linspace(0.02738362, 0.08922173, 5), torch.linspace(112.4, 139.1, 5), 0, torch.quint8
        ),
        "q_float_u4": torch.quantize_per_channel(
            torch.arange(12.0).reshape(3, 4), torch.tensor([0.4, 0.6, 0.8]), torch.tensor([0.5, 1.1, 2.9]), 0, torch.quint4x2
        ),
        "linear._packed_params._packed_params": (q_i8, torch.arange(5.0)),
    }


def reference(path):
    def describe(tensor):
        dense = tensor.dequantize() if tensor.is_quantized else tensor.to_dense()
        array = dense.contiguous().numpy()
        return {
            "shape": list(dense.shape),
            "dtype": str(dense.dtype).removeprefix("torch."),
            "bytes": array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes().hex(),
        }

    out = {}
    for name, value in torch.load(path, map_location="cpu", weights_only=True)["state_dict"].items():
        if isinstance(value, tuple):
            out.update({f"{name}.{index}": describe(tensor) for index, tensor in enumerate(value)})
        else:
            out[name] = describe(value)
    return out


def main():
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parent
    archive = io.BytesIO()
    payload = {"state_dict": entries()}
    torch.save(payload, archive)
    legacy = directory / "special-legacy.bin"
    torch.save(payload, legacy, _use_new_zipfile_serialization=False)
    compressed = directory / "special-deflated.pt"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name in source.namelist():
            output.writestr(name, source.read(name))
    expected = reference(compressed)
    assert expected == reference(legacy)
    (directory / "special-reference.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")

    # This checkpoint is used only for rejection in Rust. Never densify it in PyTorch.
    out_of_bounds = torch.sparse_coo_tensor([[3], [0]], [1.0], (2, 2), check_invariants=False)
    torch.save({"out_of_bounds": out_of_bounds}, directory / "invalid-coo.pt")
    print(f"{torch.__version__}: {len(expected)} matching tensor references")


if __name__ == "__main__":
    main()
