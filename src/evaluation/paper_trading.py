"""Deterministic paper-trading evaluation over recorded predictions."""

from __future__ import annotations

from typing import Any, Iterable


def evaluate_paper_trades(
    predictions: Iterable[dict[str, Any]],
    *,
    initial_capital: float = 10_000.0,
    notional: float = 1_000.0,
    transaction_cost_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> dict[str, Any]:
    """Evaluate non-overlapping, signal-directed paper trades.

    Only evaluated predictions with both entry and future prices are eligible.
    Costs are applied on entry and exit; no real order is placed.
    """
    if initial_capital <= 0 or notional <= 0 or transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Capital, notional, and cost inputs must be positive/nonnegative.")
    cost_rate = (transaction_cost_bps + slippage_bps) / 10_000
    trades: list[dict[str, Any]] = []
    equity = float(initial_capital)
    last_exit = None
    for prediction in sorted(predictions, key=lambda item: str(item.get("candle_timestamp", ""))):
        analytical_signal = str(prediction.get("final_signal") or "").upper()
        direction = {"BULLISH": "UP", "BEARISH": "DOWN", "UP": "UP", "DOWN": "DOWN"}.get(analytical_signal)
        if direction is None and not prediction.get("final_signal"):
            # Legacy rows predate final_signal; preserve their historical result
            # but never override a present non-actionable analytical state.
            direction = prediction.get("predicted_direction")
        entry = prediction.get("close_price")
        exit_price = prediction.get("actual_price")
        if prediction.get("outcome_status") != "evaluated" or direction not in {"UP", "DOWN"}:
            continue
        if entry is None or exit_price is None or entry <= 0:
            continue
        timestamp = prediction.get("candle_timestamp")
        if last_exit is not None and timestamp <= last_exit:
            continue
        raw_return = (float(exit_price) - float(entry)) / float(entry)
        directional_return = raw_return if direction == "UP" else -raw_return
        gross_pnl = notional * directional_return
        costs = notional * cost_rate * 2
        net_pnl = gross_pnl - costs
        equity += net_pnl
        trades.append({
            "prediction_id": prediction.get("prediction_id"),
            "entry_timestamp": timestamp,
            "exit_timestamp": prediction.get("outcome_timestamp") or prediction.get("outcome_evaluated_at"),
            "direction": direction,
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
        })
        last_exit = prediction.get("outcome_evaluated_at") or timestamp
    net = equity - initial_capital
    return {
        "paper_only": True,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "gross_pnl": sum(t["gross_pnl"] for t in trades),
        "total_costs": sum(t["costs"] for t in trades),
        "net_pnl": net,
        "trade_count": len(trades),
        "win_count": sum(t["net_pnl"] > 0 for t in trades),
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "trades": trades,
    }


def paper_cost_sensitivity(
    predictions: Iterable[dict[str, Any]],
    assumptions: Iterable[tuple[float, float]] = ((0.0, 0.0), (2.0, 1.0), (5.0, 3.0), (10.0, 5.0)),
) -> list[dict[str, Any]]:
    """Run the same historical signals under several cost assumptions."""

    return [
        evaluate_paper_trades(
            predictions,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        for transaction_cost_bps, slippage_bps in assumptions
    ]


__all__ = ["evaluate_paper_trades", "paper_cost_sensitivity"]
