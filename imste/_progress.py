"""Timestamped progress reporting for verbose IMSTE fits."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _timestamp() -> str:
    return datetime.now().strftime("%a %b %d %H:%M:%S %Y")


class IMSTEProgressReporter:
    """Render stage logs and tqdm bars from IMSTE progress callbacks."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self._bars: dict[str, Any] = {"graph": None, "epochs": None}
        self._stage: str | None = None
        self._epoch_phase = "Embedding epochs"
        self._last_plain_percent = -1
        try:
            from tqdm import tqdm
        except ImportError:  # Keep verbose fitting useful without notebook extras.
            tqdm = None
        self._tqdm = tqdm

    def log(self, message: str) -> None:
        print(f"{_timestamp()} {message}", flush=True)

    def start(self) -> None:
        params = self.estimator.get_params(deep=False)
        params.pop("progress_callback", None)
        params.pop("mst_progress_callback", None)
        init_coordinates = params.pop("init_coordinates")
        params["init_coordinates"] = init_coordinates is not None
        visible = {key: value for key, value in params.items() if key != "verbose"}
        self.log(f"IteratedMinimumSpanningTreeEmbedder({visible}, verbose=True)")
        self.log("Construct MST graph")

    @staticmethod
    def _stage_name(detail: str) -> str:
        if detail == "FAMST MPS neighbor search":
            return "Construct MST graph: MPS neighbor search"
        if detail.startswith("Coarsening level"):
            return f"Recursive initialization: {detail}"
        if detail.startswith(("Expanding level", "Expanded level")):
            return f"Recursive initialization: expand level {detail.split()[-1]}"
        if detail.startswith(("Refining level", "Refined level")):
            return f"Recursive initialization: refine level {detail.split()[-1]}"
        if detail == "Final embedding optimization":
            return detail
        return "Construct MST graph"

    def _close_bar(self, key: str) -> None:
        bar = self._bars[key]
        if bar is not None:
            bar.close()
            self._bars[key] = None

    def mst_callback(self, completed: int, total: int, detail: str) -> None:
        stage = self._stage_name(detail)
        if stage == "Final embedding optimization":
            if self._stage != stage:
                self._close_bar("graph")
                self._stage = stage
                self._epoch_phase = "Final embedding epochs"
                self.log(stage)
            return

        if stage != self._stage:
            self._close_bar("graph")
            self._stage = stage
            self._epoch_phase = stage.replace("Recursive initialization: ", "") + " epochs"
            self.log(stage)
            if self._tqdm is not None:
                if "Coarsening" in stage:
                    unit = "nodes"
                elif "neighbor search" in stage:
                    unit = "chunks"
                else:
                    unit = "steps"
                self._bars["graph"] = self._tqdm(
                    total=max(int(total), 1),
                    desc=stage,
                    unit=unit,
                    leave=True,
                    dynamic_ncols=True,
                )
        bar = self._bars["graph"]
        if bar is not None:
            bar.total = max(int(total), 1)
            bar.n = min(int(completed), bar.total)
            bar.refresh()
        elif self._tqdm is None:
            percent = int(100 * completed / max(total, 1))
            if percent >= self._last_plain_percent + 10 or completed >= total:
                self.log(f"{stage}: {completed:,}/{total:,} ({percent}%)")
                self._last_plain_percent = percent

    def epoch_callback(self, completed: int, total: int) -> None:
        total = max(int(total), 1)
        completed = min(int(completed), total)
        bar = self._bars["epochs"]
        if bar is None or bar.total != total or completed <= bar.n:
            self._close_bar("epochs")
            if self._tqdm is not None:
                bar = self._tqdm(
                    total=total,
                    desc=self._epoch_phase,
                    unit="epoch",
                    leave=True,
                    dynamic_ncols=True,
                )
                self._bars["epochs"] = bar
        bar = self._bars["epochs"]
        if bar is not None:
            bar.n = completed
            bar.refresh()
        elif self._tqdm is None:
            percent = int(100 * completed / total)
            if percent >= self._last_plain_percent + 10 or completed >= total:
                self.log(f"{self._epoch_phase}: {completed}/{total} ({percent}%)")
                self._last_plain_percent = percent

    def finish(self) -> None:
        self._close_bar("graph")
        self._close_bar("epochs")
        estimator = self.estimator
        self.log("Finished embedding")
        self.log("=== Timing Breakdown ===")
        self.log(f"MST graph construction: {estimator.graph_construction_time_:.3f}s")
        self.log(
            "Multilevel initialization: "
            f"{estimator.multilevel_initialization_time_:.3f}s"
        )
        self.log(
            "Final embedding optimization: "
            f"{estimator.final_embedding_optimization_time_:.3f}s"
        )
        self.log(f"Embedding optimization: {estimator.embedding_optimization_time_:.3f}s")
        if estimator.device_ == "mps":
            self.log(
                "  Final MPS negative sampling and coordinate updates: "
                f"{estimator.optimizer_coordinate_update_time_:.3f}s combined"
            )
        else:
            self.log(f"  Negative sampling: {estimator.optimizer_negative_sampling_time_:.3f}s")
            self.log(f"  Coordinate updates: {estimator.optimizer_coordinate_update_time_:.3f}s")
        self.log(f"Total measured: {estimator.fit_time_:.3f}s")
        self.log("Multilevel initialization is included in embedding optimization.")
