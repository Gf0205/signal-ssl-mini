import torch


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {device_arg}")


def seed_cuda(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_device_report(device: torch.device) -> None:
    print(f"Device: {device}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated(idx) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(idx) / 1024**3
        print(f"CUDA device index: {idx}")
        print(f"CUDA device name: {torch.cuda.get_device_name(idx)}")
        print(f"CUDA capability: {props.major}.{props.minor}")
        print(f"CUDA total memory: {total_gb:.2f} GB")
        print(f"CUDA memory allocated: {allocated_gb:.3f} GB")
        print(f"CUDA memory reserved: {reserved_gb:.3f} GB")
