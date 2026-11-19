"""Export explicitly allowlisted measured evidence without opening sealed data.

No experiment directory traversal, training, deletion, or source modification.
Existing destination directories are rejected. Raw data/checkpoints stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def specifications():
    """Return source, publication path, and export mode; never discover files."""
    rows = []

    def add(phase, source, *files, mode="copy"):
        for filename in files:
            rows.append((f"{source}/{filename}", f"{phase}/{filename}", mode))

    add("phase1", "outputs/phase1_final", "summary.json")
    add("phase1", "outputs", "validation_record.json")
    add("phase2", "phase2_outputs/paired_5seeds", "aggregates.json")
    add("phase2", "phase2_outputs", "validation_record_final.json")
    add("phase3a", "phase3a_outputs", "goal_demo_complete_metrics.json",
        "mappo_review_sources.json", "validation_record_complete.json")
    add("phase3b1", "phase3b1_outputs", "environment.json", "upstream_review.json",
        "learning_curves_measured.svg")
    add("phase3b1/audit", "phase3b1_outputs/audit_run1", "audit.json")
    for seed in (7, 11):
        src = f"phase3b1_outputs/debug_seed{seed}_steps4096"
        dst = f"phase3b1/seed_{seed}"
        add(dst, src, "config.json", "learning_curve.csv")
        add(dst, src, "evaluation.json", "baselines.json", mode="summary")
    add("phase3b2a", "phase3b2a_outputs", "A_reward_sensitivity_labeled.svg",
        "validation_curves.svg", "final_integrity.json")
    add("phase3b2a/audit", "phase3b2a_outputs/audit_v1", "audit.json")
    add("phase3b2a", "phase3b2a_outputs/pilot_v1", "predeclared_plan.json",
        "phase3b1_diagnostics.json")
    add("phase3b2a", "phase3b2a_outputs/pilot_v1", "test_results.json", mode="summary")
    add("phase3b2a_repair", "phase3b2a_repair_outputs/figures_v1",
        "learning_curves_labeled.svg", "slot_only_physical_task_outcomes.svg")
    add("phase3b2a_repair", "phase3b2a_repair_outputs", "final_integrity.json")
    add("phase3b2a_repair/audit", "phase3b2a_repair_outputs/audit_v1", "audit.json")
    add("phase3b2a_repair", "phase3b2a_repair_outputs/training_replay_v1", "replay_audit.json")
    add("phase3b2a_repair", "phase3b2a_repair_outputs/pilot_v1",
        "predeclared_plan.json", "selection_freeze.json", "coverage_window_diagnostics.json")
    add("phase3b2a_repair", "phase3b2a_repair_outputs/pilot_v1",
        "same_condition_comparisons.json", "final_checkpoint_comparisons.json",
        "information_comparisons.json", "B3_deterministic_stochastic.json",
        "baseline_ladder.json", "exact_feasibility.json", mode="summary")
    add("phase3b2a_repair/sampling_probe", "phase3b2a_repair_outputs/sampling_probe_v1",
        "probe_plan.json")
    add("phase3b2a_repair/sampling_probe", "phase3b2a_repair_outputs/sampling_probe_v1",
        "results.json", mode="summary")
    add("phase3b2b", "phase3b2b_outputs/main_v1", "frozen_plan.json",
        "pretraining_freeze.json", "source_manifest.json", "dependencies.json",
        "future_comparison_contract.json", "all_seeds_selection_freeze.json",
        "statistics.json", "parent_training_wall.json")
    for seed in (101, 211, 307, 401, 503):
        add(f"phase3b2b/seed_{seed}", f"phase3b2b_outputs/main_v1/seed_{seed}",
            "config.json", "selection.json", "learning_curve.csv")
    add("phase3b2b/analysis", "phase3b2b_outputs/analysis_v1", "all_metrics.csv",
        "validation_curves.csv", "training_summary.json", "handover_censoring.json",
        "per_task_outcomes.json", "stochastic_repeat_variability.json",
        "scenario_complete_world_intervals.json", "analysis_scope.json")
    add("phase3b2b/analysis", "phase3b2b_outputs/analysis_supplement_v1",
        "tail_responsibility_gaps.json", "paired_scenario_uncertainty.json")
    add("phase3b2b/figures", "phase3b2b_outputs/figures_v1", "learning_validation_curves.svg",
        "benchmark_comparison.svg", "coverage_response_tradeoff.svg")
    add("phase3b2b/audit", "phase3b2b_outputs/audit_v3", "audit.json")
    add("phase3b2b", "phase3b2b_outputs", "final_integrity.json")
    return rows


def summarize(value):
    """Retain stored aggregates, remove per-episode arrays; never invent metrics."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "episodes" and isinstance(item, list):
                result["omitted_episode_records"] = len(item)
            else:
                result[key] = summarize(item)
        return result
    if isinstance(value, list):
        return [summarize(item) for item in value]
    return value


def export(destination):
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Publication output must stay within research/cah_mappo")
    rows = specifications()
    payloads = []
    for source, relative, mode in rows:
        # Every file is explicitly named above. Never read a holdout manifest.
        if "holdout" in source.lower() or Path(source).suffix not in {".json", ".csv", ".svg"}:
            raise ValueError(source)
        original = (ROOT / source).read_bytes()
        if mode == "summary":
            exported = (json.dumps(summarize(json.loads(original)), indent=2,
                                   ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        else:
            exported = original
        payloads.append((relative, exported, {
            "source": source, "published": relative, "mode": mode,
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "published_sha256": hashlib.sha256(exported).hexdigest(),
            "bytes": len(exported),
        }))
    if len({p[0] for p in payloads}) != len(payloads):
        raise ValueError("Duplicate publication path")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, payload, _ in payloads:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = {
        "purpose": "Measured evidence snapshot; not complete raw experiment data",
        "sealed_holdout_opened": False,
        "raw_logs_and_checkpoints_included": False,
        "files": [entry for _, _, entry in payloads],
    }
    (destination / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(payloads),
                      "bytes": sum(len(p) for _, p, _ in payloads),
                      "output": str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "snapshot_20261119")
    export(parser.parse_args().output)
