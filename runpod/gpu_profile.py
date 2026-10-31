"""Explicit GPU identities and budget for the user-selected BN diagnostic."""

ADA = "NVIDIA GeForce RTX 4090"
BLACKWELL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
MIG24 = BLACKWELL + " MIG 1g.24gb"


def profile(task):
    return (MIG24, 0.69) if task == "bn" else (ADA, 1.0)


def validate_runtime(runtime, expected):
    if expected not in {ADA, MIG24}:
        raise ValueError("Unapproved GPU profile")
    if runtime.get("requested_gpu") != expected or runtime.get("device_count") != 1:
        raise ValueError("GPU selection mismatch")
    names = {BLACKWELL, MIG24} if expected == MIG24 else {ADA}
    capability = [12, 0] if expected == MIG24 else [8, 9]
    memory = runtime.get("total_memory_bytes", 0)
    if (
        runtime.get("name") not in names
        or runtime.get("capability") != capability
        or isinstance(memory, bool)
        or not isinstance(memory, int)
        or not 22 * 1024**3 <= memory <= 25 * 1024**3
    ):
        raise ValueError("Unexpected GPU identity, architecture or visible memory")


def baseline_tolerance(expected):
    if expected not in {ADA, MIG24}:
        raise ValueError("Unapproved GPU profile")
    # Absolute dB tolerance, fixed before running on a different architecture.
    return 1e-3 if expected == MIG24 else 1e-5
