from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_TIMESTAMP_DIR_RE = re.compile(r"^\d{8}-\d{6}$")


@dataclass(frozen=True)
class BaselineExperimentSpec:
    agent_config: str
    systems_file: str
    max_systems: int | None
    budget: int
    num_episodes: int
    stability_tolerance: float
    enable_reflexion: bool = False

    @property
    def systems_file_stem(self) -> str:
        return Path(self.systems_file).stem if self.systems_file else "default_systems"

    @property
    def experiment_dir_name(self) -> str:
        max_systems_str = (
            str(self.max_systems) if self.max_systems is not None else "allsystems"
        )
        return (
            f"{self.agent_config}_{self.systems_file_stem}_{max_systems_str}systems_"
            f"{self.budget}queries_{int(self.stability_tolerance * 1000)}stabilitymeV"
        )


def is_timestamp_dir(path: Path) -> bool:
    return _TIMESTAMP_DIR_RE.fullmatch(path.name) is not None


def load_systems_from_json(
    systems_file: Path, max_systems: int | None = None
) -> list[list[str]]:
    with systems_file.open() as f:
        systems = [list(system) for system in json.load(f)]
    if max_systems is not None:
        systems = systems[:max_systems]
    return systems


def system_id_from_elements(elements: list[str]) -> str:
    return "-".join(elements)


def get_remaining_systems(
    all_systems: list[list[str]], completed_systems: list[str]
) -> list[list[str]]:
    completed_set = set(completed_systems)
    return [
        system
        for system in all_systems
        if system_id_from_elements(system) not in completed_set
    ]


def load_experiment_metadata(output_dir: Path) -> dict[str, Any] | None:
    metadata_path = output_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open() as f:
            return json.load(f)
    except Exception:
        return None


def metadata_matches_spec(metadata: dict[str, Any], spec: BaselineExperimentSpec) -> bool:
    metadata_systems_file = str(metadata.get("systems_file", ""))
    metadata_enable_reflexion = bool(metadata.get("enable_reflexion", False))
    metadata_stability = metadata.get("stability_tolerance")
    if metadata_stability is not None:
        try:
            if abs(float(metadata_stability) - spec.stability_tolerance) >= 1e-12:
                return False
        except (TypeError, ValueError):
            return False

    return (
        metadata.get("agent_config") == spec.agent_config
        and Path(metadata_systems_file).stem == spec.systems_file_stem
        and metadata.get("max_systems") == spec.max_systems
        and metadata.get("budget") == spec.budget
        and metadata.get("num_episodes") == spec.num_episodes
        and metadata_enable_reflexion == spec.enable_reflexion
    )


def find_matching_experiment_dirs(
    base_output_dir: Path,
    spec: BaselineExperimentSpec,
) -> list[Path]:
    if not base_output_dir.exists():
        return []

    candidates: dict[Path, None] = {}
    direct = base_output_dir / spec.experiment_dir_name
    if direct.exists():
        candidates[direct] = None

    for child in base_output_dir.iterdir():
        if not child.is_dir():
            continue
        nested = child / spec.experiment_dir_name
        if nested.exists():
            candidates[nested] = None

    matching: list[Path] = []
    for candidate in candidates:
        metadata = load_experiment_metadata(candidate)
        if metadata and metadata_matches_spec(metadata, spec):
            matching.append(candidate)
    return matching


def count_trace_components(trace_file: Path) -> dict[str, int] | None:
    counts: dict[str, int] = {}
    try:
        with trace_file.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                component = record.get("component")
                if component is None:
                    counts["<missing_component>"] = (
                        counts.get("<missing_component>", 0) + 1
                    )
                    continue
                counts[str(component)] = counts.get(str(component), 0) + 1
    except (OSError, json.JSONDecodeError):
        return None
    return counts


def _is_system_complete_from_summary(
    system_dir: Path,
    num_episodes: int,
) -> bool:
    summary_dir = system_dir / "summary"
    summary_file = summary_dir / "summary.json"
    episodes_file = summary_dir / "episodes.json"
    if not summary_file.exists() or not episodes_file.exists():
        return False
    try:
        with episodes_file.open() as f:
            episodes = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(episodes, list) and len(episodes) == num_episodes


def _is_system_complete_from_llm_traces(
    output_dir: Path,
    system_id: str,
    budget: int,
    num_episodes: int,
    enable_reflexion: bool,
) -> bool:
    trace_dir = output_dir / "llm_traces" / system_id
    if not trace_dir.exists():
        return False

    expected_trace_files = [
        trace_dir / f"episode_{episode_idx:03d}.jsonl"
        for episode_idx in range(num_episodes)
    ]
    if any(not path.exists() for path in expected_trace_files):
        return False

    extra_files = list(trace_dir.glob("episode_*.jsonl"))
    if len(extra_files) != num_episodes:
        return False

    expected_reflections = 1 if enable_reflexion else 0
    for trace_file in expected_trace_files:
        counts = count_trace_components(trace_file)
        if counts is None:
            return False
        if counts.get("orchestrator", 0) != budget:
            return False
        if counts.get("self_reflection", 0) != expected_reflections:
            return False

    return True


def is_system_complete(
    output_dir: Path,
    system_id: str,
    spec: BaselineExperimentSpec,
) -> bool:
    system_dir = output_dir / "systems" / system_id
    if _is_system_complete_from_summary(system_dir, spec.num_episodes):
        return True

    if spec.agent_config == "llm_react_orchestrator":
        return _is_system_complete_from_llm_traces(
            output_dir=output_dir,
            system_id=system_id,
            budget=spec.budget,
            num_episodes=spec.num_episodes,
            enable_reflexion=spec.enable_reflexion,
        )
    return False


def get_completed_systems(
    output_dir: Path,
    all_systems: list[list[str]],
    spec: BaselineExperimentSpec,
) -> list[str]:
    completed: list[str] = []
    for system in all_systems:
        system_id = system_id_from_elements(system)
        if is_system_complete(output_dir, system_id, spec):
            completed.append(system_id)
    return completed


def resolve_experiment_output_dir(
    base_output_dir: Path,
    spec: BaselineExperimentSpec,
    *,
    resume: bool,
    all_systems: list[list[str]] | None = None,
) -> tuple[Path, bool]:
    if resume:
        candidates = find_matching_experiment_dirs(base_output_dir, spec)
        if candidates:
            if all_systems:
                total_systems = len(all_systems)

                def rank(candidate: Path) -> tuple[int, float]:
                    completed = len(get_completed_systems(candidate, all_systems, spec))
                    is_complete = completed >= total_systems
                    return (1 if is_complete else 0, -candidate.stat().st_mtime)

                chosen = min(candidates, key=rank)
            else:
                chosen = max(candidates, key=lambda path: path.stat().st_mtime)
            chosen.mkdir(parents=True, exist_ok=True)
            return chosen, True

    if is_timestamp_dir(base_output_dir):
        output_dir = base_output_dir / spec.experiment_dir_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = base_output_dir / timestamp / spec.experiment_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, False
