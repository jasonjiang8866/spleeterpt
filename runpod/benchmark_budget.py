"""Read account budget without persisting balances, payment data, or API credentials."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from control import request

RATE_CAP = 0.69
LEASE_SECONDS = 3600
RESERVE_USD = 1.0
MONTHLY_BUDGET_USD = 500.0


def nonnegative(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def check(api, *, now=None):
    now = now or datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = request(
        "https://api.runpod.io/graphql",
        token=api.key,
        method="POST",
        payload={
            "query": "query BenchmarkBudget($start: DateTime!, $end: DateTime!) { myself { clientBalance currentSpendPerHr dailyCharges(startDate: $start, endDate: $end) { amount } } }",
            "variables": {"start": start.isoformat(), "end": now.isoformat()},
        },
    )
    if not isinstance(result, dict) or result.get("errors"):
        raise RuntimeError("Account budget query failed; no compute authorized by this check")
    account = result.get("data", {}).get("myself") or {}
    balance, spend = account.get("clientBalance"), account.get("currentSpendPerHr")
    charges = account.get("dailyCharges")
    if not nonnegative(balance) or not nonnegative(spend) or not isinstance(charges, list):
        raise RuntimeError("Account budget fields unavailable")
    amounts = [c.get("amount") for c in charges if isinstance(c, dict)]
    if len(amounts) != len(charges) or not all(finite(x) for x in amounts):
        raise RuntimeError("Invalid account charge history")
    # Conservative with either API charge sign convention: refunds cannot expand budget.
    charged = sum(abs(x) for x in amounts)
    if balance < RESERVE_USD or charged + spend + RESERVE_USD > MONTHLY_BUDGET_USD:
        raise RuntimeError("Insufficient reserve or monthly budget headroom")
    return {
        "checked_at_utc": now.isoformat(),
        "month_basis": "UTC",
        "reserve_usd": RESERVE_USD,
        "monthly_budget_usd": MONTHLY_BUDGET_USD,
        "balance_sufficient": True,
        "month_headroom_sufficient": True,
        "history_scope": "reported account daily charges plus one hour current spend; billing may lag",
        "gpu_rate_cap_usd_per_hour": RATE_CAP,
        "lease_seconds": LEASE_SECONDS,
        "scope": "compute rate/deadline plus reserve; storage/request fees are not an all-in hard cap",
    }


if __name__ == "__main__":
    from control import Runpod, identity

    try:
        identity()
        result = check(Runpod())
        output = Path("runpod-results")
        output.mkdir(exist_ok=True)
        (output / "benchmark-budget.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except Exception as error:  # noqa: BLE001 -- no raw account/payment information
        print(f"ERROR: {type(error).__name__}; account budget check incomplete")
        raise SystemExit(1) from None
