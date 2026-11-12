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
        self._bars: dict[str, Any] = {
            "graph": None,
            "epochs": None,
            "coarsening": None,
            "refinement": None,
        }
        self._stage: str | None = None
        self._epoch_phase = "Embedding epochs"
        self._last_plain_percent = -1
        self._coarsening_stage: str | None = None
        self._coarsening_offset = 0
        self._coarsening_level_total = 0
        self._refinement_epochs = 0
        self._refinement_offset = 0
        self._refinement_level_index = -1
        self._refinement_level_count = 0
        try:
            from tqdm import tqdm
        except ImportError:  # Keep verbose fitting useful without notebook extras.
            tqdm = None
        self._tqdm = tqdm

    def log(self, message: str) -> None:
        print(f"{_timestamp()} {message}", flush=True)

    def phase_finished(self, phase: str, seconds: float) -> None:
        self.log(f"{phase} complete: {seconds:.3f}s")

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

    def begin_multilevel_refinement(self, level_count: int, epochs: int) -> None:
        """Prepare one epoch bar whose total spans every refinement level."""
        self._refinement_level_count = int(level_count) + 1
        self._refinement_epochs = max(int(epochs), 0)
        self._refinement_offset = 0
        self._refinement_level_index = -1

    def multilevel_finished(self) -> None:
        """Close multilevel bars after the hierarchy and refinements complete."""
        bar = self._bars["coarsening"]
        if bar is not None:
            bar.n = bar.total
            bar.refresh()
        self._close_bar("coarsening")
        self._close_bar("refinement")

    def mst_callback(self, completed: int, total: int, detail: str) -> None:
        stage = self._stage_name(detail)
        if detail.startswith("Coarsening level"):
            if self._coarsening_stage != detail:
                if self._coarsening_stage is not None:
                    self._coarsening_offset += self._coarsening_level_total
                self._coarsening_stage = detail
                self._coarsening_level_total = max(int(total), 1)
                if self._bars["coarsening"] is None:
                    if self._tqdm is not None:
                        self._bars["coarsening"] = self._tqdm(
                            total=self._coarsening_level_total,
                            desc="Multilevel coarsening",
                            unit="nodes",
                            leave=True,
                            dynamic_ncols=True,
                        )
                bar = self._bars["coarsening"]
                if bar is not None:
                    bar.total = self._coarsening_offset + self._coarsening_level_total
                    bar.set_description(f"Multilevel coarsening ({detail.split()[2]})")
            bar = self._bars["coarsening"]
            if bar is not None:
                bar.n = self._coarsening_offset + min(int(completed), max(int(total), 1))
                bar.refresh()
            elif self._tqdm is None:
                percent = int(100 * completed / max(total, 1))
                if percent >= self._last_plain_percent + 10 or completed >= total:
                    self.log(f"Multilevel coarsening: {detail} {completed:,}/{total:,}")
                    self._last_plain_percent = percent
            self._stage = stage
            return

        if detail.startswith("Refining level"):
            self._refinement_level_index += 1
            self._refinement_offset = self._refinement_level_index * self._refinement_epochs
            self._epoch_phase = "Multilevel refinement epochs"
            if self._bars["refinement"] is None and self._tqdm is not None:
                self._bars["refinement"] = self._tqdm(
                    total=max(self._refinement_level_count * self._refinement_epochs, 1),
                    desc="Multilevel refinement epochs",
                    unit="epoch",
                    leave=True,
                    dynamic_ncols=True,
                )
            if stage != self._stage:
                self.log("Recursive initialization: refining levels")
            self._stage = stage
            return

        if detail.startswith(("Expanding level", "Expanded level", "Refined level")):
            # Level boundaries are reflected by the global coarsening/refinement bars.
            self._stage = stage
            return

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
        if self._epoch_phase == "Multilevel refinement epochs":
            bar = self._bars["refinement"]
            global_total = max(self._refinement_level_count * self._refinement_epochs, 1)
            if bar is not None:
                bar.total = global_total
                bar.n = min(self._refinement_offset + completed, global_total)
                bar.refresh()
            elif self._tqdm is None:
                percent = int(100 * (self._refinement_offset + completed) / global_total)
                if percent >= self._last_plain_percent + 10 or completed >= total:
                    self.log(
                        "Multilevel refinement epochs: "
                        f"{self._refinement_offset + completed}/{global_total} ({percent}%)"
                    )
                    self._last_plain_percent = percent
            return
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
        self.multilevel_finished()
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
