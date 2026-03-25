"""
LLM ReAct Orchestrator Agent - Flexible LLM agent that orchestrates generators, filters, and scorers.

This agent uses DSPy's ReAct to decide what actions to take at each iteration:
- Generate structures using various generators (with filtering)
- Score structures using various scorers
- Query buffer for candidates (topk, diverse, by composition)
- Select structures for oracle evaluation

Key features:
- Maintains a buffer of pre-validated structures from generators
- All generated structures are filtered before adding to buffer
- Uniqueness filter is always re-run on new generations
- Full history of oracle evaluations available to LLM
- Caching of structure hashes to avoid duplicate processing
- Flexible buffer queries (topk, bottomk, diverse compositions)
"""

import logging
from collections import defaultdict
from typing import Any
import time, random
import math
import dspy
import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from made.agents.base import Agent, Filter, Generator, Plan, Scorer
from made.utils.dspy_lm import build_dspy_lm
from made.utils.structure_hash import structure_hash
from made.utils.llm_trace import append_llm_trace


logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signature for ReAct Orchestration (default, can be overridden via config)
# ============================================================================


class OrchestratorReActSignature(dspy.Signature):
    """
    You are an autonomous materials discovery agent.

    OBJECTIVE: Find as many NOVEL, UNIQUE, STABLE (or metastable) structures as possible. Use the available tools, then select ONE composition + structure for oracle evaluation.

    - Structures with e_above_hull <= stability_tolerance are stable/metastable (SUCCESS!)
    - Entries marked [STABLE, NOVEL] in evaluation_history are successful discoveries
    - We want to MAXIMIZE the number of novel stable structures found

    IMPORTANT: Different structures for the SAME composition can have DIFFERENT stabilities.
    - One unstable structure for a composition does NOT mean all structures for that composition are unstable.
    - Generators produce many different structures for the same composition

    UNIT CELL SIZE MATTERS:
    - Compositions are stored by REDUCED formula (e.g., Li2O)
    - But you can generate different UNIT CELL SIZES: Li2O, Li4O2, Li6O3, etc.
    - These occupy the same position on the phase diagram but are different structures
    - Different unit cell sizes often have different stabilities

    BUFFER ORGANIZATION:
    - Buffer is organized by reduced formula: {composition: [structures]}
    - Each structure shows its full formula (unit cell size) and index
    - Selection is two-step: pick composition, then pick structure index

    WORKFLOW:
    1. Decide which composition(s) to explore based on evaluation history
    2. Generate/create candidate structures for those compositions
    3. Score candidates if needed to prioritize within each composition
    4. List compositions and query structures to decide what to evaluate
    5. Select ONE composition + structure for oracle evaluation

    STRATEGY GUIDANCE:
    - If buffer empty/small: Generate more structures (diverse compositions OR more for promising ones)
    - If buffer has candidates: Score them, query compositions, select best
    - Balance exploration (new compositions) vs exploitation (more structures for successful compositions)
    """

    chemical_system: str = dspy.InputField(desc="Allowed elements (e.g., 'Li, O')")
    stability_tolerance: float = dspy.InputField(
        desc="e_above_hull threshold for stability"
    )
    max_stoichiometry: int = dspy.InputField(desc="Maximum atoms per structure")
    buffer_summary: str = dspy.InputField(
        desc="Summary of unevaluated candidate buffer"
    )
    evaluation_history: str = dspy.InputField(
        desc="History of oracle evaluations with results"
    )
    known_stable_materials: str = dspy.InputField(
        desc="Known stable materials from phase diagram"
    )

    prior_reflections: str = dspy.InputField(
        desc="Lessons learned from previous episodes on this chemical system. Use these to avoid repeating mistakes and build on successful strategies from the start."
    )

    answer: str = dspy.OutputField(
        desc="Summary of actions taken and final selection rationale"
    )


class SelfReflectionSignature(dspy.Signature):
    """You are reviewing a completed materials discovery episode to extract lessons for the next episode on the same chemical system.

    Analyze what happened and produce specific, actionable insights.
    Focus on:
    - Which composition families were productive (stable/novel) vs. consistently wasteful
    - Whether the exploration/exploitation balance was right
    - Patterns in what succeeded: stoichiometries, unit cell sizes, element ratios
    - Mistakes or inefficiencies to avoid repeating
    - Concrete strategy changes for the next episode

    Metric interpretation guardrails:
    - "NOVEL" is local structural novelty within this run: not matching MP-database initial structures, 
    and not matching previously discovered structures in this episode.
    - "Recall" is recovery of ground-truth stable formulas missing at initialization.
    - Novel counts and recall can diverge; do not assume "high novel + low recall" implies a bug.

    Be specific and concise. Write 3-5 bullet points. Avoid vague advice like "explore more".
    """

    chemical_system: str = dspy.InputField(
        desc="The chemical system being explored (e.g., 'Li, O')"
    )
    episode_trajectory: str = dspy.InputField(
        desc="All oracle evaluations from the completed episode: compositions, e_above_hull, stability, and novelty status"
    )
    episode_outcome: str = dspy.InputField(
        desc="Quantitative summary: novel stable structures found, recall achieved, oracle queries used"
    )
    prior_reflections: str = dspy.InputField(
        desc="Lessons from previous episodes on this system (empty if this is the first episode)"
    )

    reflection: str = dspy.OutputField(
        desc="Concise, actionable lessons for the next episode as bullet points identifying what to do differently."
    )


# ============================================================================
# Tool Classes for the Orchestrator
# ============================================================================


class OrchestratorTools:
    """Tools for the LLM ReAct Orchestrator to use.

    Buffer is organized by composition: {composition: [structure_entries]}
    This enables natural two-step selection: pick composition, then pick structure.
    """

    def __init__(
        self,
        generators: dict[str, Generator],
        static_filter: Filter | None,
        uniqueness_filter: Filter | None,
        scorers: dict[str, Scorer],
        chemical_system_elements: list[str],
        max_stoichiometry: int,
        buffer: dict[
            str, list[dict[str, Any]]
        ],  # composition -> list of structure entries
        structure_cache: dict[str, dict[str, Any]],
        state: dict[str, Any],
    ):
        self.generators = generators
        self.static_filter = static_filter
        self.uniqueness_filter = uniqueness_filter
        self.scorers = scorers
        self.chemical_system_elements = chemical_system_elements
        self.max_stoichiometry = max_stoichiometry
        self.buffer = buffer  # Now a dict: composition -> list of entries
        self.structure_cache = structure_cache
        self.state = state
        self._selected_structure: Structure | None = None
        self._selection_reason: str = ""
        self._telemetry: dict[str, Any] = {
            "generate_calls": 0,
            "generated_total": 0,
            "generated_added_total": 0,
            "generated_duplicate_total": 0,
            "generated_cached_total": 0,
            "generated_static_filtered_total": 0,
            "created_added_total": 0,
            # Per-query structure additions grouped by reduced composition.
            "added_by_composition": {},
        }

    def generate_structures(
        self,
        generator_name: str,
        compositions: str = "",
        num_candidates: int = 10,
    ) -> str:
        """Generate new candidate structures using a specified generator.

        Structures are automatically filtered (distance, SMACT, etc.) before adding to buffer.
        Uniqueness filter is always applied to remove duplicates of existing structures.

        Args:
            generator_name: Name of generator to use - one of 'chemeleon' (a generative model trained on stable structures), 'random' (a random structure generator)
            compositions: Comma-separated compositions with unit cell size (e.g., 'Li2O, Li4O2, LiO2').
                         Different unit cell sizes (Li2O vs Li4O2) represent different structure sizes
                         but occupy the same position on the phase diagram (both reduce to Li2O).
                         Generators create different structures for different unit cell sizes.
            num_candidates: Number of candidates to generate per composition

        Returns:
            Summary of generated and filtered structures
        """
        if generator_name not in self.generators:
            available = list(self.generators.keys())
            return (
                f"Error: Generator '{generator_name}' not found. Available: {available}"
            )
        self._telemetry["generate_calls"] += 1

        generator = self.generators[generator_name]

        # Parse compositions
        comp_list = []
        if compositions.strip():
            for comp_str in compositions.split(","):
                comp_str = comp_str.strip()
                if comp_str:
                    try:
                        comp = Composition(comp_str)
                        for el in comp.elements:
                            if str(el) not in self.chemical_system_elements:
                                return f"Error: Element {el} not in allowed elements {self.chemical_system_elements}"
                        comp_list.append(comp)
                    except Exception as e:
                        return f"Error parsing composition '{comp_str}': {e}"

        if not comp_list:
            logger.warning(f"No compositions provided for generator {generator_name}")

            return f"No compositions provided for generator {generator_name}"

        plan = Plan(
            compositions=comp_list,
            num_candidates=num_candidates,
            constraints={
                "elements": self.chemical_system_elements,
                "max_stoichiometry": self.max_stoichiometry,
            },
        )

        try:
            MAX_TRIES = 5
            structures = None
            last_err = None
            
            for attempt in range(1, MAX_TRIES + 1):
                try:
                    structures = generator.generate(plan, self.state)
                    last_err = None
                    break  # success
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
            
                    # Only retry on transient file/corruption-like errors
                    if "empty file" in msg or "unexpected end of file" in msg:
                        # backoff: 0.5, 1, 2, 4, 8 (capped) + small jitter
                        sleep_s = min(8.0, 0.5 * (2 ** (attempt - 1))) + random.random() * 0.2
                        logger.warning(
                            f"[Tool] {generator_name} generate() failed ({attempt}/{MAX_TRIES}): {e}. "
                            f"Retrying in {sleep_s:.2f}s"
                        )
                        time.sleep(sleep_s)
                        continue
            
                    # Non-transient: don't hide it
                    raise
            
            if last_err is not None and structures is None:
                raise last_err
            
            if not structures:
                return f"Generator {generator_name} produced no structures."
            self._telemetry["generated_total"] += len(structures)

            # Filter and add to buffer
            added_count = 0
            filtered_out = {"duplicate": 0, "cached": 0, "static_filter": 0}

            for structure in structures:
                struct_hash = structure_hash(structure)

                # Check cache
                if struct_hash in self.structure_cache:
                    filtered_out["cached"] += 1
                    continue

                # Run static filter chain
                if self.static_filter is not None:
                    try:
                        passed, _ = self.static_filter.filter(
                            [structure], self.state, return_results=True
                        )
                        if not passed:
                            filtered_out["static_filter"] += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Static filter failed: {e}")
                        filtered_out["static_filter"] += 1
                        continue

                # Run uniqueness filter (always re-run)
                if self.uniqueness_filter is not None:
                    try:
                        passed, _ = self.uniqueness_filter.filter(
                            [structure], self.state, return_results=True
                        )
                        if not passed:
                            filtered_out["duplicate"] += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Uniqueness filter failed: {e}")

                # Add to buffer and cache
                comp = structure.composition.reduced_formula
                full_formula = structure.composition.formula.replace(" ", "")
                entry = {
                    "structure": structure,
                    "hash": struct_hash,
                    "composition": comp,  # Reduced formula (e.g., Li2O)
                    "full_formula": full_formula,  # Full formula (e.g., Li4O2)
                    "source": generator_name,
                    "scores": {},
                    "num_sites": len(structure),
                }
                # Add to composition-based buffer (keyed by reduced formula)
                if comp not in self.buffer:
                    self.buffer[comp] = []
                self.buffer[comp].append(entry)
                self.structure_cache[struct_hash] = entry
                added_count += 1
                added_by_comp = self._telemetry["added_by_composition"]
                added_by_comp[comp] = int(added_by_comp.get(comp, 0)) + 1

            # Build summary
            total_count = sum(len(entries) for entries in self.buffer.values())
            msg_parts = [f"Generated {len(structures)} using {generator_name}."]
            msg_parts.append(
                f"Added {added_count} to buffer ({total_count} total, {len(self.buffer)} compositions)."
            )

            if filtered_out["cached"]:
                msg_parts.append(f"Cached: {filtered_out['cached']}.")
            if filtered_out["duplicate"]:
                msg_parts.append(f"Duplicates: {filtered_out['duplicate']}.")
            if filtered_out["static_filter"]:
                msg_parts.append(f"Filter failures: {filtered_out['static_filter']}.")
            self._telemetry["generated_added_total"] += added_count
            self._telemetry["generated_duplicate_total"] += filtered_out["duplicate"]
            self._telemetry["generated_cached_total"] += filtered_out["cached"]
            self._telemetry["generated_static_filtered_total"] += filtered_out[
                "static_filter"
            ]

            msg = " ".join(msg_parts)
            logger.info(f"[Tool] {msg}")
            return msg

        except Exception as e:
            logger.error(f"[Tool] Generation failed in '{generator_name}': {e}")
            return f"Error during generation in '{generator_name}': {e}"

    def score_buffer(self, scorer_name: str, composition: str = "") -> str:
        """Score candidates in the buffer using a specified scorer.

        Args:
            scorer_name: Name of scorer to use (e.g., 'diversity', 'oracle')
            composition: Optional composition to score (if empty, scores all)

        Returns:
            Summary of scoring results with top candidates per composition
        """
        if scorer_name not in self.scorers:
            available = list(self.scorers.keys())
            return f"Error: Scorer '{scorer_name}' not found. Available: {available}"

        if not self.buffer:
            return "Buffer is empty, nothing to score."

        scorer = self.scorers[scorer_name]

        try:
            # Collect structures to score
            to_score_comps = [composition] if composition else list(self.buffer.keys())
            total_scored = 0

            for comp in to_score_comps:
                if comp not in self.buffer:
                    continue

                entries = self.buffer[comp]
                structures = [e["structure"] for e in entries]

                scores, results = scorer.score_candidates(
                    structures, self.state, return_results=True
                )

                # Store scores in entries
                for entry, score in zip(entries, scores, strict=True):
                    entry["scores"][scorer_name] = score

                # Sort entries by score (descending) within composition
                self.buffer[comp].sort(
                    key=lambda e: e["scores"].get(scorer_name, float("-inf")),
                    reverse=True,
                )
                total_scored += len(entries)

            # Build summary showing top structures per composition
            lines = [f"Scored {total_scored} structures with '{scorer_name}'."]
            lines.append("Top structures per composition:")
            for comp in sorted(to_score_comps):
                if comp not in self.buffer:
                    continue
                entries = self.buffer[comp]
                if entries and scorer_name in entries[0].get("scores", {}):
                    top_score = entries[0]["scores"][scorer_name]
                    lines.append(
                        f"  {comp}: {len(entries)} structures, best score={top_score:.4f}"
                    )

            msg = "\n".join(lines)
            logger.info(f"[Tool] {msg}")
            return msg

        except Exception as e:
            logger.error(f"[Tool] Scoring failed: {e}")
            return f"Error during scoring: {e}"

    def list_compositions(
        self,
        k: int = 10,
        mode: str = "top",
        scorer_name: str = "",
    ) -> str:
        """List compositions in the buffer with structure counts and optional scoring.

        Use this to see what compositions are available and their characteristics.

        Args:
            k: Number of compositions to show
            mode: Query mode:
                - 'top': Highest scoring compositions (requires scorer_name, default)
                - 'bottom': Lowest scoring compositions (requires scorer_name)
                - 'random': Random sample (no scorer needed)
                - 'count': By structure count (most structures first)
            scorer_name: Scorer to rank compositions by best structure score (required for top/bottom)

        Returns:
            List of compositions with counts and optional best scores

        Examples:
            - list_compositions(k=5, mode='count')
              → Top 5 compositions by structure count
            - list_compositions(k=10, mode='top', scorer_name='oracle')
              → Top 10 compositions by best oracle score
            - list_compositions(k=3, mode='bottom', scorer_name='diversity')
              → Bottom 3 (least diverse) compositions
            - list_compositions(k=5, mode='random')
              → Random 5 compositions
        """
        if not self.buffer:
            return "Buffer is empty."

        comp_info = []
        for comp, entries in self.buffer.items():
            info = {
                "composition": comp,
                "count": len(entries),
                "best_score": None,
            }

            if scorer_name and entries:
                # Get best score for this composition
                scores = [
                    e["scores"].get(scorer_name)
                    for e in entries
                    if scorer_name in e.get("scores", {})
                ]
                if scores:
                    info["best_score"] = max(scores)

            comp_info.append(info)

        # Sort or sample based on mode
        if mode in ["top", "bottom"]:
            if not scorer_name:
                return f"Error: mode '{mode}' requires scorer_name."
            # Check if any composition has scores
            if not any(info["best_score"] is not None for info in comp_info):
                return f"No compositions scored with '{scorer_name}'. Run score_buffer first."
            reverse = mode == "top"
            comp_info.sort(
                key=lambda x: x["best_score"]
                if x["best_score"] is not None
                else float("-inf"),
                reverse=reverse,
            )
        elif mode == "random":
            np.random.shuffle(comp_info)
        else:  # mode == "count" or any other value
            comp_info.sort(key=lambda x: x["count"], reverse=True)

        # Format output
        total_structures = sum(info["count"] for info in comp_info)
        mode_desc = f"by {mode}" if mode != "count" else "by count"
        if mode in ["top", "bottom"] and scorer_name:
            mode_desc = f"{mode} by {scorer_name}"

        lines = [
            f"Buffer: {total_structures} structures across {len(self.buffer)} compositions"
        ]
        lines.append(f"Showing {min(k, len(comp_info))} compositions ({mode_desc}):")

        for i, info in enumerate(comp_info[:k]):
            score_str = (
                f", best_{scorer_name}={info['best_score']:.4f}"
                if info["best_score"] is not None
                else ""
            )
            lines.append(
                f"  {i + 1}. {info['composition']}: {info['count']} structures{score_str}"
            )

        msg = "\n".join(lines)
        logger.info(f"[Tool] {msg}")
        return msg

    def query_structures(
        self,
        composition: str,
        k: int = 5,
        mode: str = "top",
        scorer_name: str = "",
        include_structure_details: bool = False,
    ) -> str:
        """Query structures within a specific composition.

        Retrieve structures to help decide which one to select for evaluation.
        The index shown can be used directly in select_for_evaluation.

        Args:
            composition: Composition to query (e.g., 'Li2O')
            k: Number of structures to return
            mode: Query mode:
                - 'top': Highest scores (requires scorer_name, default)
                - 'bottom': Lowest scores (requires scorer_name)
                - 'random': Random sample (no scorer needed)
                - 'all': All structures in order (no sorting)
            scorer_name: Scorer to use for ranking (required for top/bottom modes)
            include_structure_details: If True, includes full structure (lattice, species, positions)

        Returns:
            List of structures with indices, scores, and optional structural details

        Examples:
            - query_structures('Li2O', k=3, mode='top', scorer_name='oracle')
              → Top 3 structures for Li2O by oracle score
            - query_structures('Li2O', k=5, mode='bottom', scorer_name='diversity')
              → Bottom 5 (least diverse) Li2O structures
            - query_structures('LiO2', k=2, mode='random')
              → Random 2 structures from LiO2
            - query_structures('Li2O', k=10, mode='all', include_structure_details=True)
              → All Li2O structures with lattice info
        """
        if not self.buffer:
            return "Buffer is empty."

        comp = composition.strip()
        if comp not in self.buffer:
            available = list(self.buffer.keys())[:10]
            return f"No structures for composition '{comp}'. Available: {available}"

        entries = self.buffer[comp].copy()

        # Sort or sample based on mode
        if mode in ["top", "bottom"]:
            if not scorer_name:
                return f"Error: mode '{mode}' requires scorer_name."
            if not entries[0].get("scores", {}).get(scorer_name):
                return f"Structures not scored with '{scorer_name}'. Run score_buffer first."
            # Sort by score
            reverse = mode == "top"
            sorted_entries = sorted(
                entries,
                key=lambda e: e["scores"].get(scorer_name, float("-inf")),
                reverse=reverse,
            )
        elif mode == "random":
            sorted_entries = entries.copy()
            np.random.shuffle(sorted_entries)
        else:  # mode == "all" or any other value
            sorted_entries = entries

        # Format output - show full formula to indicate unit cell size
        lines = [f"Structures for {comp} (reduced formula): {len(entries)} total"]
        for i, entry in enumerate(sorted_entries[:k]):
            full_formula = entry.get("full_formula", comp)
            score_str = ", ".join(
                f"{s}={v:.4f}" for s, v in entry.get("scores", {}).items()
            )

            structure_info = f"  {i}. {full_formula} [{entry['num_sites']} sites, {entry['source']}]{': ' + score_str if score_str else ''}"

            # Add full structural details if requested
            if include_structure_details:
                structure = entry["structure"]
                # Use pymatgen's string representation for complete structural info
                struct_str = str(structure)
                # Indent each line for better formatting
                indented_struct = "\n      ".join(struct_str.split("\n"))
                structure_info += f"\n      {indented_struct}"

            lines.append(structure_info)

        msg = "\n".join(lines)
        logger.info(f"[Tool] {msg}")
        return msg

    def get_buffer_stats(self) -> str:
        """Get detailed statistics about the current buffer.

        Returns:
            Summary: total count, compositions, sources, score ranges
        """
        if not self.buffer:
            return "Buffer is empty. Use generate_structures or create_structure to add candidates."

        # Count totals
        total_structures = sum(len(entries) for entries in self.buffer.values())
        num_compositions = len(self.buffer)

        # Sources
        sources = {}
        for entries in self.buffer.values():
            for entry in entries:
                src = entry.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

        # Score stats
        score_stats = {}
        for entries in self.buffer.values():
            for entry in entries:
                for scorer_name, score in entry.get("scores", {}).items():
                    if scorer_name not in score_stats:
                        score_stats[scorer_name] = []
                    score_stats[scorer_name].append(score)

        lines = [
            f"Buffer: {total_structures} structures across {num_compositions} compositions",
            f"  Sources: {sources}",
            "  Top compositions by count:",
        ]

        # Sort compositions by count
        sorted_comps = sorted(
            self.buffer.items(), key=lambda x: len(x[1]), reverse=True
        )[:5]
        for comp, entries in sorted_comps:
            lines.append(f"    {comp}: {len(entries)} structures")
        if len(self.buffer) > 5:
            lines.append(f"    ... +{len(self.buffer) - 5} more compositions")

        if score_stats:
            lines.append("  Score ranges:")
            for scorer_name, scores in score_stats.items():
                lines.append(
                    f"    {scorer_name}: min={min(scores):.4f}, max={max(scores):.4f}, mean={np.mean(scores):.4f}"
                )

        msg = "\n".join(lines)
        logger.info(f"[Tool] {msg}")
        return msg

    def select_for_evaluation(
        self,
        composition: str,
        structure_index: int = 0,
        scorer_name: str = "",
        reason: str = "",
    ) -> str:
        """Select a structure for oracle evaluation (two-step: composition then structure).

        This is the final action - selects ONE structure for ground-truth evaluation.

        Args:
            composition: Composition to select from (e.g., 'Li2O'). Required.
            structure_index: Index within the composition's structures (0-based).
                           If scorer_name provided, 0 = best scored structure.
            scorer_name: Optional scorer to use for ranking structures within composition.
                        If provided, structure_index refers to rank (0=best).
            reason: Reason for selection (for logging)

        Returns:
            Confirmation of selection with composition and scores
        """
        if not self.buffer:
            return "Error: Buffer is empty. Use generate_structures or create_structure first."

        comp = composition.strip()
        if not comp:
            return "Error: Must provide composition. Use list_compositions to see available."

        if comp not in self.buffer:
            available = list(self.buffer.keys())[:10]
            return (
                f"Error: No structures for composition '{comp}'. Available: {available}"
            )

        entries = self.buffer[comp]

        # Sort by scorer if specified
        if scorer_name:
            # Check if structures are scored
            if not entries[0].get("scores", {}).get(scorer_name):
                return f"Error: Structures for '{comp}' not scored with '{scorer_name}'. Run score_buffer first."
            # Already sorted by score_buffer, but ensure correct order
            entries = sorted(
                entries,
                key=lambda e: e["scores"].get(scorer_name, float("-inf")),
                reverse=True,
            )

        # Check index bounds
        if structure_index < 0 or structure_index >= len(entries):
            return f"Error: Index {structure_index} out of range for {comp} (has {len(entries)} structures)."

        # Select structure
        entry = entries[structure_index]
        self._selected_structure = entry["structure"]
        rank_str = (
            f"rank {structure_index} by {scorer_name}"
            if scorer_name
            else f"index {structure_index}"
        )
        self._selection_reason = reason or f"Selected {comp} ({rank_str})"

        # Remove from buffer
        self.buffer[comp].pop(structure_index)
        if not self.buffer[comp]:  # Remove composition if no more structures
            del self.buffer[comp]

        # Remove from cache
        if entry["hash"] in self.structure_cache:
            del self.structure_cache[entry["hash"]]

        # Format message
        full_formula = entry.get("full_formula", entry["composition"])
        score_str = ", ".join(
            f"{s}={v:.4f}" for s, v in entry.get("scores", {}).items()
        )
        total_remaining = sum(len(e) for e in self.buffer.values())
        msg = (
            f"SELECTED: {full_formula} (reduced: {entry['composition']}, {entry['num_sites']} sites, {entry['source']})"
            f"{': ' + score_str if score_str else ''}. "
            f"{self._selection_reason}. "
            f"Buffer: {total_remaining} structures remaining."
        )
        logger.info(f"[Tool] {msg}")
        return msg

    def create_structure(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        species: str = "",
        frac_coords: str = "",
    ) -> str:
        """Create a new structure from scratch with specified lattice and sites.

        This allows direct structure creation without using a generator.
        The structure is validated and added to the buffer if it passes filters.

        Args:
            a: Lattice parameter a (Angstroms)
            b: Lattice parameter b (Angstroms)
            c: Lattice parameter c (Angstroms)
            alpha: Lattice angle alpha (degrees, default 90)
            beta: Lattice angle beta (degrees, default 90)
            gamma: Lattice angle gamma (degrees, default 90)
            species: Comma-separated element symbols for each site (e.g., "Li, Li, O")
            frac_coords: Semicolon-separated fractional coordinates, each as "x,y,z"
                        (e.g., "0,0,0; 0.5,0.5,0.5; 0.25,0.25,0.25")

        Returns:
            Result message indicating success or failure
        """
        from pymatgen.core.lattice import Lattice

        # Parse species
        if not species.strip():
            return "Error: Must provide species (comma-separated element symbols)"
        species_list = [s.strip() for s in species.split(",") if s.strip()]

        # Parse fractional coordinates
        if not frac_coords.strip():
            return (
                "Error: Must provide frac_coords (semicolon-separated, each as 'x,y,z')"
            )

        coords_list = []
        for coord_str in frac_coords.split(";"):
            coord_str = coord_str.strip()
            if not coord_str:
                continue
            parts = [float(x.strip()) for x in coord_str.split(",")]
            if len(parts) != 3:
                return f"Error: Each coordinate must be 'x,y,z', got '{coord_str}'"
            coords_list.append(parts)

        # Validate lengths match
        if len(species_list) != len(coords_list):
            return f"Error: species count ({len(species_list)}) != coords count ({len(coords_list)})"

        # Validate elements
        for element in species_list:
            if element not in self.chemical_system_elements:
                return f"Error: Element {element} not in allowed elements: {self.chemical_system_elements}"

        # Check stoichiometry
        if len(species_list) > self.max_stoichiometry:
            return f"Error: Too many atoms ({len(species_list)}), max is {self.max_stoichiometry}"

        try:
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            structure = Structure(lattice, species_list, coords_list)

            # Check for duplicates
            struct_hash = structure_hash(structure)
            if struct_hash in self.structure_cache:
                return f"Structure already in buffer (composition: {structure.composition.reduced_formula})"

            # Run static filter
            if self.static_filter is not None:
                try:
                    passed, results = self.static_filter.filter(
                        [structure], self.state, return_results=True
                    )
                    if not passed:
                        reasons = [
                            r.rejection_reason
                            for r in results
                            if not r.passed and r.rejection_reason
                        ]
                        return f"Structure failed filters: {'; '.join(reasons) if reasons else 'failed validation'}"
                except Exception as e:
                    return f"Filter error: {e}"

            # Run uniqueness filter
            if self.uniqueness_filter is not None:
                try:
                    passed, results = self.uniqueness_filter.filter(
                        [structure], self.state, return_results=True
                    )
                    if not passed:
                        return (
                            "Structure is not unique (duplicate of existing structure)"
                        )
                except Exception as e:
                    logger.warning(f"Uniqueness filter error: {e}")

            # Add to buffer
            comp = structure.composition.reduced_formula
            full_formula = structure.composition.formula.replace(" ", "")
            entry = {
                "structure": structure,
                "hash": struct_hash,
                "composition": comp,  # Reduced formula
                "full_formula": full_formula,  # Full formula
                "source": "llm_created",
                "scores": {},
                "num_sites": len(structure),
            }
            # Add to composition-based buffer (keyed by reduced formula)
            if comp not in self.buffer:
                self.buffer[comp] = []
            comp_idx = len(self.buffer[comp])
            self.buffer[comp].append(entry)
            self.structure_cache[struct_hash] = entry

            total_count = sum(len(entries) for entries in self.buffer.values())
            self._telemetry["created_added_total"] += 1
            added_by_comp = self._telemetry["added_by_composition"]
            added_by_comp[comp] = int(added_by_comp.get(comp, 0)) + 1
            msg = f"Created and added: {full_formula} (reduced: {comp}, {len(structure)} sites). Buffer: {total_count} structures, {len(self.buffer)} compositions. {comp}: {comp_idx + 1} structures."
            logger.info(f"[Tool] {msg}")
            return msg

        except Exception as e:
            return f"Error creating structure: {e}"

    def get_selected_structure(self) -> Structure | None:
        """Get the selected structure (internal use)."""
        return self._selected_structure

    def get_telemetry(self) -> dict[str, Any]:
        """Get tool telemetry collected during this agent call."""
        return dict(self._telemetry)


# ============================================================================
# Main Agent Class
# ============================================================================


class LLMReActOrchestratorAgent(Agent):
    """
    LLM ReAct Orchestrator Agent that uses DSPy ReAct to decide actions.

    Maintains a buffer of pre-validated structures and uses an LLM
    to decide: generate, score, query, or select.
    """

    def __init__(
        self,
        # LLM configuration
        llm_config: dict[str, Any] | None = None,
        # Tool components
        generators: dict[str, Generator] | None = None,
        static_filters: Filter | None = None,  # FilterChain or single filter
        uniqueness_filter: Filter | None = None,  # Separate uniqueness filter
        scorers: dict[str, Scorer] | None = None,
        # Context and prompts (from config)
        context_config: dict[str, Any] | None = None,
        # Tool access
        enabled_tools: list[str] | None = None,
        # ReAct config
        max_iters: int = 10,
        # Constraints
        max_stoichiometry: int = 20,
        # History
        max_history_length: int = 20,
        # Reflexion
        enable_reflexion: bool = False,
        max_reflections: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize the LLM ReAct Orchestrator Agent.

        Args:
            llm_config: DSPy LM configuration (model, cache, etc.)
            generators: Dict of named generators
            static_filters: Filter or FilterChain for static filtering (cached)
            uniqueness_filter: Uniqueness filter (always re-run)
            scorers: Dict of named scorers
            context_config: Prompts and context configuration
                - orchestration_prompt: Custom prompt (overrides default)
                - include_structure_in_history: Include full structure (lattice, species, positions) in evaluation history
                - include_structure_in_known_materials: Include full structure in known stable materials
            enabled_tools: List of tool names to enable
            max_iters: Maximum ReAct iterations per step
            max_stoichiometry: Maximum atoms per structure
            max_history_length: Maximum evaluation history entries
            enable_reflexion: If True, agent generates a verbal self-reflection after each
                episode and uses accumulated reflections to guide subsequent episodes.
            max_reflections: Sliding window size for the reflection memory buffer (default 3).
        """
        # Initialize base Agent (sets self.last_step = 0)
        super().__init__()

        # Store components
        self.generators = generators or {}
        self.static_filter = static_filters
        self.uniqueness_filter = uniqueness_filter
        self.scorers_dict = scorers or {}

        # Context and prompts
        self.context_config = context_config or {}

        # Configuration
        self.max_iters = max_iters
        self.max_stoichiometry = max_stoichiometry
        self.max_history_length = max_history_length

        # Reflexion
        self.enable_reflexion = enable_reflexion
        self.max_reflections = max_reflections
        self.reflections: list[str] = []
        self.self_reflection_module = (
            dspy.ChainOfThought(SelfReflectionSignature) if enable_reflexion else None
        )

        # Enabled tools
        all_tools = [
            "generate_structures",
            "create_structure",
            "score_buffer",
            "list_compositions",
            "query_structures",
            "get_buffer_stats",
            "select_for_evaluation",
        ]
        self.enabled_tools = enabled_tools or all_tools

        # State - buffer is now composition-based: {composition: [entries]}
        self.buffer: dict[str, list[dict[str, Any]]] = {}
        self.structure_cache: dict[str, dict[str, Any]] = {}
        self.evaluation_history: list[dict[str, Any]] = []
        self.chemical_system_elements: list[str] = []
        self.selection_history: list[str] = []
        self.selection_counts: dict[str, int] = {}
        self.selection_new_count = 0
        self.selection_switch_count = 0
        self.selection_max_streak = 0
        self.selection_current_streak = 0
        self.selection_last_comp = ""

        self.targeted_compositions_seen: set[str] = set()
        self.targeted_new_total = 0
        self.targeted_revisit_total = 0

        self.element_tool_arg_call_counts: dict[str, int] = {}
        self.element_tool_arg_total_calls = 0

        self.generated_total = 0
        self.generated_added_total = 0
        self.generated_duplicate_total = 0
        self.generated_cached_total = 0
        self.generated_static_filtered_total = 0
        self.created_added_total = 0

        self.revisit_payoff_stats: dict[int, dict[str, int]] = {}
        self.revisit_total_count = 0
        self.revisit_stable_count = 0
        self.revisit_novel_stable_count = 0
        self.recent_yield_window = 10
        self._pending_selected_comp: str | None = None
        self._pending_selected_visit_index: int | None = None

        # Per-oracle-query family-exploration tracking.
        self.query_count = 0
        self.family_first_seen_query_idx: dict[str, int] = {}
        self.new_families_cumulative_set: set[str] = set()
        self.last_new_family_query_idx: int | None = None

        # Latest query-level metrics (updated once per oracle query).
        self.num_new_families_last_query = 0
        self.num_new_family_structures_added_last_query = 0
        self.selected_family_is_new_last_query = 0
        self.selected_family_age_queries_last_query = -1
        self.queries_since_last_new_family_last_query = 0
        self.new_family_generation_share_last_query = 0.0

        self.latest_behavior_metrics: dict[str, float] = {}
        self.behavior_metrics_history: list[dict[str, float]] = []

        # Initialize DSPy LM
        self.llm_config = llm_config or {}
        self._setup_dspy()

        # Prepare signature with optional prompt override (mirrors LLMScorer/LLMPlanner approach)
        signature = OrchestratorReActSignature
        if self.context_config.get("orchestration_prompt"):
            # Override the signature instructions with custom prompt from config
            signature.instructions = self.context_config["orchestration_prompt"]
        self.signature_class = signature

        logger.info(
            f"[LLMReActOrchestrator] Initialized with generators={list(self.generators.keys())}, "
            f"static_filter={self.static_filter is not None}, uniqueness_filter={self.uniqueness_filter is not None}, "
            f"scorers={list(self.scorers_dict.keys())}"
        )

    def _setup_dspy(self):
        """Setup DSPy LM."""
        try:
            self.lm = build_dspy_lm(self.llm_config)
            model = self.llm_config.get("model", "unknown")
            base_url = self.llm_config.get("base_url") or self.llm_config.get("api_base")
            if base_url:
                logger.info(
                    f"[LLMReActOrchestrator] DSPy LM: {model} (api_base={base_url})"
                )
            else:
                logger.info(f"[LLMReActOrchestrator] DSPy LM: {model}")
        except Exception as e:
            logger.error(f"[LLMReActOrchestrator] Failed to initialize DSPy LM: {e}")
            raise

    def _normalize_composition(self, formula: str) -> str | None:
        """Normalize composition formula text to reduced formula."""
        token = formula.strip()
        if not token:
            return None
        try:
            return Composition(token).reduced_composition.alphabetical_formula
        except Exception:
            return None

    def _extract_tool_steps(self, result_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ordered tool call steps from ReAct output trajectory."""
        trajectory = result_output.get("trajectory", {})
        if not isinstance(trajectory, dict):
            return []

        steps: list[dict[str, Any]] = []
        for key, value in trajectory.items():
            if not key.startswith("tool_name_"):
                continue
            try:
                idx = int(key.split("_")[-1])
            except ValueError:
                continue
            args = trajectory.get(f"tool_args_{idx}", {})
            if not isinstance(args, dict):
                args = {}
            steps.append(
                {
                    "index": idx,
                    "tool_name": str(value),
                    "tool_args": args,
                }
            )

        steps.sort(key=lambda x: x["index"])
        return steps

    def _extract_compositions_from_tool_args(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> set[str]:
        """Extract referenced compositions from tool arguments."""
        compositions: set[str] = set()

        comp_val = tool_args.get("composition")
        if isinstance(comp_val, str) and comp_val.strip():
            norm = self._normalize_composition(comp_val)
            if norm:
                compositions.add(norm)

        if tool_name == "generate_structures":
            comp_list = tool_args.get("compositions", "")
            if isinstance(comp_list, str):
                for token in comp_list.split(","):
                    norm = self._normalize_composition(token)
                    if norm:
                        compositions.add(norm)

        return compositions

    def _extract_elements_from_tool_args(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> set[str]:
        """Extract referenced elements from tool arguments."""
        elements: set[str] = set()
        allowed = set(self.chemical_system_elements)

        for comp in self._extract_compositions_from_tool_args(tool_name, tool_args):
            try:
                parsed = Composition(comp)
                for el in parsed.elements:
                    symbol = str(el)
                    if symbol in allowed:
                        elements.add(symbol)
            except Exception:
                continue

        if tool_name == "create_structure":
            species = tool_args.get("species", "")
            if isinstance(species, str):
                for token in species.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    if token in allowed:
                        elements.add(token)
                        continue
                    try:
                        parsed = Composition(token)
                    except Exception:
                        continue
                    for el in parsed.elements:
                        symbol = str(el)
                        if symbol in allowed:
                            elements.add(symbol)
        return elements

    def _update_tool_arg_tracking(self, tool_steps: list[dict[str, Any]]) -> None:
        """Accumulate tool-argument element/composition exploration counters."""
        query_targeted: set[str] = set()
        for step in tool_steps:
            tool_name = step.get("tool_name", "")
            tool_args = step.get("tool_args", {})
            if not isinstance(tool_args, dict):
                continue

            query_targeted.update(
                self._extract_compositions_from_tool_args(tool_name, tool_args)
            )

            call_elements = self._extract_elements_from_tool_args(tool_name, tool_args)
            if call_elements:
                self.element_tool_arg_total_calls += 1
                for element in call_elements:
                    self.element_tool_arg_call_counts[element] = (
                        self.element_tool_arg_call_counts.get(element, 0) + 1
                    )

        if query_targeted:
            new_targets = [c for c in query_targeted if c not in self.targeted_compositions_seen]
            self.targeted_new_total += len(new_targets)
            self.targeted_revisit_total += len(query_targeted) - len(new_targets)
            self.targeted_compositions_seen.update(query_targeted)

    def _update_generation_tracking(self, telemetry: dict[str, Any]) -> None:
        """Accumulate generation/buffer churn telemetry."""
        self.generated_total += telemetry.get("generated_total", 0)
        self.generated_added_total += telemetry.get("generated_added_total", 0)
        self.generated_duplicate_total += telemetry.get("generated_duplicate_total", 0)
        self.generated_cached_total += telemetry.get("generated_cached_total", 0)
        self.generated_static_filtered_total += telemetry.get(
            "generated_static_filtered_total", 0
        )
        self.created_added_total += telemetry.get("created_added_total", 0)

    def _update_query_family_metrics(
        self,
        *,
        query_idx: int,
        query_start_buffer_comps: set[str],
        selected_comp: str,
        tool_telemetry: dict[str, Any],
    ) -> None:
        """Update per-query family-introduction metrics after one oracle selection."""
        added_by_comp_raw = tool_telemetry.get("added_by_composition", {})
        added_by_comp = (
            {
                str(k): int(v)
                for k, v in added_by_comp_raw.items()
                if isinstance(k, str)
            }
            if isinstance(added_by_comp_raw, dict)
            else {}
        )

        # Ensure families already present at query start have a first-seen timestamp.
        for comp in query_start_buffer_comps:
            self.family_first_seen_query_idx.setdefault(comp, query_idx)

        new_families_this_query = {
            comp for comp in added_by_comp if comp not in query_start_buffer_comps
        }
        self.num_new_families_last_query = len(new_families_this_query)
        self.num_new_family_structures_added_last_query = int(
            sum(added_by_comp.get(comp, 0) for comp in new_families_this_query)
        )

        if new_families_this_query:
            self.new_families_cumulative_set.update(new_families_this_query)
            self.last_new_family_query_idx = query_idx
            for comp in new_families_this_query:
                self.family_first_seen_query_idx.setdefault(comp, query_idx)

        self.selected_family_is_new_last_query = int(
            selected_comp in new_families_this_query
        )
        selected_first_seen = self.family_first_seen_query_idx.get(selected_comp)
        self.selected_family_age_queries_last_query = (
            int(query_idx - selected_first_seen)
            if selected_first_seen is not None
            else -1
        )

        if self.num_new_families_last_query > 0:
            self.queries_since_last_new_family_last_query = 0
        elif self.last_new_family_query_idx is None:
            # No introduction yet: number of queries elapsed so far.
            self.queries_since_last_new_family_last_query = query_idx + 1
        else:
            self.queries_since_last_new_family_last_query = (
                query_idx - self.last_new_family_query_idx
            )

        total_structures_added_this_query = int(
            tool_telemetry.get("generated_added_total", 0)
            + tool_telemetry.get("created_added_total", 0)
        )
        self.new_family_generation_share_last_query = (
            float(
                self.num_new_family_structures_added_last_query
                / total_structures_added_this_query
            )
            if total_structures_added_this_query > 0
            else 0.0
        )

    def _record_selected_composition(self, composition: str) -> None:
        """Update selection concentration/switch/streak counters."""
        comp = composition.strip()
        if not comp:
            return

        is_new = comp not in self.selection_counts
        if is_new:
            self.selection_new_count += 1

        if self.selection_history:
            if self.selection_history[-1] != comp:
                self.selection_switch_count += 1
                self.selection_current_streak = 1
            else:
                self.selection_current_streak += 1
        else:
            self.selection_current_streak = 1

        self.selection_max_streak = max(
            self.selection_max_streak, self.selection_current_streak
        )
        self.selection_last_comp = comp
        self.selection_history.append(comp)
        self.selection_counts[comp] = self.selection_counts.get(comp, 0) + 1

        self._pending_selected_comp = comp
        self._pending_selected_visit_index = self.selection_counts[comp]

    def _update_revisit_payoff(self, composition: str, is_stable: bool, is_novel: bool) -> None:
        """Update revisit payoff statistics after an oracle result is observed."""
        visit_idx = self._pending_selected_visit_index
        if visit_idx is None or self._pending_selected_comp != composition:
            visit_idx = self.selection_counts.get(composition, 0)
            if visit_idx <= 0:
                visit_idx = 1

        bucket = self.revisit_payoff_stats.setdefault(
            visit_idx, {"count": 0, "stable": 0, "novel_stable": 0}
        )
        bucket["count"] += 1
        if is_stable:
            bucket["stable"] += 1
        if is_stable and is_novel:
            bucket["novel_stable"] += 1

        if visit_idx > 1:
            self.revisit_total_count += 1
            if is_stable:
                self.revisit_stable_count += 1
            if is_stable and is_novel:
                self.revisit_novel_stable_count += 1

        self._pending_selected_comp = None
        self._pending_selected_visit_index = None

    def _get_recent_yield_metrics(self) -> tuple[float, float, float]:
        """Return recent stable and novel-stable yields over the rolling window."""
        if not self.evaluation_history:
            return 0.0, 0.0, 0.0

        window_entries = self.evaluation_history[-self.recent_yield_window :]
        if not window_entries:
            return 0.0, 0.0, 0.0

        stable_recent = sum(1 for entry in window_entries if entry.get("is_stable", False))
        novel_stable_recent = sum(
            1
            for entry in window_entries
            if entry.get("is_stable", False) and entry.get("is_newly_discovered", False)
        )
        denom = float(len(window_entries))
        return (
            float(stable_recent),
            float(novel_stable_recent),
            float(novel_stable_recent / denom if denom > 0 else 0.0),
        )

    def _build_behavior_metrics(self) -> dict[str, float]:
        """Build a flat dict of agent behavior metrics for live logging."""
        metrics: dict[str, float] = {}
        total_selected = len(self.selection_history)
        unique_selected = len(self.selection_counts)

        metrics["selection_total"] = float(total_selected)
        metrics["selection_unique_count"] = float(unique_selected)
        metrics["selection_new_composition_count"] = float(self.selection_new_count)
        metrics["selection_new_composition_rate"] = float(
            self.selection_new_count / total_selected if total_selected > 0 else 0.0
        )

        if total_selected > 0:
            top_count = max(self.selection_counts.values())
            probs = [count / total_selected for count in self.selection_counts.values()]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 0.0
            entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
            metrics["selection_top1_share"] = float(top_count / total_selected)
            metrics["selection_entropy"] = float(entropy)
            metrics["selection_entropy_normalized"] = float(entropy_norm)
        else:
            metrics["selection_top1_share"] = 0.0
            metrics["selection_entropy"] = 0.0
            metrics["selection_entropy_normalized"] = 0.0

        transitions_denom = total_selected - 1
        metrics["selection_switch_rate"] = float(
            self.selection_switch_count / transitions_denom
            if transitions_denom > 0
            else 0.0
        )
        metrics["selection_max_same_composition_streak"] = float(
            self.selection_max_streak
        )

        metrics["targeted_unique_count"] = float(len(self.targeted_compositions_seen))
        metrics["targeted_new_total"] = float(self.targeted_new_total)
        metrics["targeted_revisit_total"] = float(self.targeted_revisit_total)
        metrics["explore_vs_exploit_ratio"] = float(
            self.targeted_new_total / self.targeted_revisit_total
            if self.targeted_revisit_total > 0
            else float(self.targeted_new_total)
        )

        metrics["tool_arg_element_calls_total"] = float(self.element_tool_arg_total_calls)
        for element in sorted(self.chemical_system_elements):
            count = float(self.element_tool_arg_call_counts.get(element, 0))
            ratio = (
                count / self.element_tool_arg_total_calls
                if self.element_tool_arg_total_calls > 0
                else 0.0
            )
            metrics[f"element_arg_count_{element}"] = count
            metrics[f"element_arg_presence_ratio_{element}"] = float(ratio)

        metrics["buffer_churn_generated_total"] = float(self.generated_total)
        metrics["buffer_churn_added_total"] = float(self.generated_added_total)
        metrics["buffer_churn_duplicate_total"] = float(self.generated_duplicate_total)
        metrics["buffer_churn_cached_total"] = float(self.generated_cached_total)
        metrics["buffer_churn_static_filtered_total"] = float(
            self.generated_static_filtered_total
        )
        metrics["buffer_churn_created_added_total"] = float(self.created_added_total)
        metrics["buffer_churn_addition_rate"] = float(
            self.generated_added_total / self.generated_total
            if self.generated_total > 0
            else 0.0
        )
        metrics["buffer_churn_duplicate_rejection_rate"] = float(
            (self.generated_duplicate_total + self.generated_cached_total)
            / self.generated_total
            if self.generated_total > 0
            else 0.0
        )

        metrics["revisit_payoff_revisit_count"] = float(self.revisit_total_count)
        metrics["revisit_payoff_revisit_stable_rate"] = float(
            self.revisit_stable_count / self.revisit_total_count
            if self.revisit_total_count > 0
            else 0.0
        )
        metrics["revisit_payoff_revisit_novel_stable_rate"] = float(
            self.revisit_novel_stable_count / self.revisit_total_count
            if self.revisit_total_count > 0
            else 0.0
        )

        overflow_count = 0
        overflow_stable = 0
        overflow_novel_stable = 0
        for visit_idx, stats in sorted(self.revisit_payoff_stats.items()):
            if visit_idx <= 5:
                count = stats["count"]
                metrics[f"revisit_payoff_visit{visit_idx}_count"] = float(count)
                metrics[f"revisit_payoff_visit{visit_idx}_stable_rate"] = float(
                    stats["stable"] / count if count > 0 else 0.0
                )
                metrics[f"revisit_payoff_visit{visit_idx}_novel_stable_rate"] = float(
                    stats["novel_stable"] / count if count > 0 else 0.0
                )
            else:
                overflow_count += stats["count"]
                overflow_stable += stats["stable"]
                overflow_novel_stable += stats["novel_stable"]

        metrics["revisit_payoff_visit_gt5_count"] = float(overflow_count)
        metrics["revisit_payoff_visit_gt5_stable_rate"] = float(
            overflow_stable / overflow_count if overflow_count > 0 else 0.0
        )
        metrics["revisit_payoff_visit_gt5_novel_stable_rate"] = float(
            overflow_novel_stable / overflow_count if overflow_count > 0 else 0.0
        )

        # Step-level family-introduction metrics (updated once per oracle query).
        # num_new_families: new unique families added this query that were absent at query start.
        metrics["num_new_families"] = float(self.num_new_families_last_query)
        # num_new_family_structures_added: structures added for those new families this query.
        metrics["num_new_family_structures_added"] = float(
            self.num_new_family_structures_added_last_query
        )
        # selected_family_is_new: whether the selected oracle candidate is from this query's new families.
        metrics["selected_family_is_new"] = float(self.selected_family_is_new_last_query)
        # selected_family_age_queries: query-age of selected family since first seen in buffer.
        metrics["selected_family_age_queries"] = float(
            self.selected_family_age_queries_last_query
        )
        # queries_since_last_new_family: number of queries since most recent new-family introduction.
        metrics["queries_since_last_new_family"] = float(
            self.queries_since_last_new_family_last_query
        )
        # new_families_cumulative: running unique count of introduced families across the episode.
        metrics["new_families_cumulative"] = float(
            len(self.new_families_cumulative_set)
        )
        # new_family_generation_share: share of this query's added structures belonging to new families.
        metrics["new_family_generation_share"] = float(
            self.new_family_generation_share_last_query
        )

        stable_recent, novel_stable_recent, novel_stable_rate_recent = (
            self._get_recent_yield_metrics()
        )
        metrics[f"recent_yield_stable_last{self.recent_yield_window}"] = stable_recent
        metrics[f"recent_yield_novel_stable_last{self.recent_yield_window}"] = (
            novel_stable_recent
        )
        metrics[f"recent_yield_novel_stable_rate_last{self.recent_yield_window}"] = (
            novel_stable_rate_recent
        )

        return metrics

    def _refresh_behavior_metrics(self) -> dict[str, float]:
        """Refresh latest metrics snapshot and append to metric history."""
        metrics = self._build_behavior_metrics()
        self.latest_behavior_metrics = metrics
        self.behavior_metrics_history.append(dict(metrics))
        return metrics

    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Composition, Structure]:
        """
        Propose a structure using the ReAct loop.
        """
        self.query_count += 1
        current_query_idx = self.query_count - 1
        query_start_buffer_comps = set(self.buffer.keys())

        self.chemical_system_elements = state.get("elements", [])
        stability_tolerance = state.get("stability_tolerance", 1e-8)

        logger.info(
            f"[LLMReActOrchestrator] Starting proposal (buffer={len(self.buffer)}, history={len(self.evaluation_history)})"
        )

        # Build context
        buffer_summary = self._format_buffer_summary()
        history_str = self._format_evaluation_history(stability_tolerance)
        stable_materials_str = self._format_known_stable_materials(state)
        prior_reflections_str = self._format_prior_reflections()

        # Create tools
        tools = OrchestratorTools(
            generators=self.generators,
            static_filter=self.static_filter,
            uniqueness_filter=self.uniqueness_filter,
            scorers=self.scorers_dict,
            chemical_system_elements=self.chemical_system_elements,
            max_stoichiometry=self.max_stoichiometry,
            buffer=self.buffer,
            structure_cache=self.structure_cache,
            state=state,
        )

        # Build tool list
        tool_map = {
            "generate_structures": tools.generate_structures,
            "create_structure": tools.create_structure,
            "score_buffer": tools.score_buffer,
            "list_compositions": tools.list_compositions,
            "query_structures": tools.query_structures,
            "get_buffer_stats": tools.get_buffer_stats,
            "select_for_evaluation": tools.select_for_evaluation,
        }
        enabled_tool_functions = [
            tool_map[name] for name in self.enabled_tools if name in tool_map
        ]

        # Create ReAct module
        react_module = dspy.ReAct(
            self.signature_class,
            tools=enabled_tool_functions,
            max_iters=self.max_iters,
        )

        logger.info(
            f"[LLMReActOrchestrator] ReAct: elements={self.chemical_system_elements}, tools={self.enabled_tools}"
        )

        # Run ReAct
        try:
            history_before = len(getattr(self.lm, "history", []))
            with dspy.context(lm=self.lm):
                result = react_module(
                    chemical_system=", ".join(self.chemical_system_elements),
                    stability_tolerance=stability_tolerance,
                    max_stoichiometry=self.max_stoichiometry,
                    buffer_summary=buffer_summary,
                    evaluation_history=history_str,
                    known_stable_materials=stable_materials_str,
                    prior_reflections=prior_reflections_str,
                )
            history_after = len(getattr(self.lm, "history", []))
            lm_calls = getattr(self.lm, "history", [])[history_before:history_after]

            logger.info(f"[LLMReActOrchestrator] Result: {result.answer}")
            result_output = getattr(
                result,
                "toDict",
                lambda: {"answer": getattr(result, "answer", str(result))},
            )()
            tool_steps = self._extract_tool_steps(result_output)
            self._update_tool_arg_tracking(tool_steps)
            tool_telemetry = tools.get_telemetry()
            self._update_generation_tracking(tool_telemetry)

            selected = tools.get_selected_structure()

            if selected is None:
                logger.warning(
                    "[LLMReActOrchestrator] No structure selected. Falling back."
                )
                selected = self._fallback_selection(state)

            selected_comp = selected.composition.reduced_composition.alphabetical_formula
            self._update_query_family_metrics(
                query_idx=current_query_idx,
                query_start_buffer_comps=query_start_buffer_comps,
                selected_comp=selected_comp,
                tool_telemetry=tool_telemetry,
            )
            self._record_selected_composition(selected_comp)
            behavior_metrics = self._refresh_behavior_metrics()

            append_llm_trace(
                component="orchestrator",
                llm_config=self.llm_config,
                output=result_output,
                inputs={
                    "chemical_system": self.chemical_system_elements,
                    "max_stoichiometry": self.max_stoichiometry,
                    "max_iters": self.max_iters,
                    "enabled_tools": self.enabled_tools,
                },
                extra={
                    "buffer_compositions": len(self.buffer),
                    "evaluation_history_len": len(self.evaluation_history),
                    "num_lm_calls": len(lm_calls),
                    "lm_calls": lm_calls,
                    "tool_telemetry": tool_telemetry,
                    "behavior_metrics": behavior_metrics,
                },
            )

            return selected.composition, selected

        except Exception as e:
            logger.error(f"[LLMReActOrchestrator] ReAct failed: {e}")
            selected = self._fallback_selection(state)
            selected_comp = (
                selected.composition.reduced_composition.alphabetical_formula
            )
            self._update_query_family_metrics(
                query_idx=current_query_idx,
                query_start_buffer_comps=query_start_buffer_comps,
                selected_comp=selected_comp,
                tool_telemetry={},
            )
            self._record_selected_composition(selected_comp)
            self._refresh_behavior_metrics()
            return selected.composition, selected

    def _fallback_selection(self, state: dict[str, Any]) -> Structure:
        """Fallback selection if ReAct fails."""
        if self.buffer:
            # Pick first composition and first structure from it
            comp = next(iter(self.buffer.keys()))
            entry = self.buffer[comp].pop(0)
            if not self.buffer[comp]:  # Remove composition if empty
                del self.buffer[comp]
            logger.info(f"[LLMReActOrchestrator] Fallback: {entry['composition']}")
            return entry["structure"]

        if self.generators:
            gen_name, generator = next(iter(self.generators.items()))
            plan = Plan(
                compositions=[
                    Composition(dict.fromkeys(self.chemical_system_elements, 1))
                ],
                num_candidates=1,
                constraints={"elements": self.chemical_system_elements},
            )
            try:
                structures = generator.generate(plan, state)
                if structures:
                    logger.info(
                        f"[LLMReActOrchestrator] Fallback: generated via {gen_name}"
                    )
                    return structures[0]
            except Exception as e:
                logger.error(f"[LLMReActOrchestrator] Fallback generation failed: {e}")

        raise RuntimeError("No structure available")

    def _format_buffer_summary(self) -> str:
        """Format buffer summary for LLM context."""
        if not self.buffer:
            return "Buffer is empty. Use generate_structures or create_structure to add candidates."

        total_structures = sum(len(entries) for entries in self.buffer.values())
        num_compositions = len(self.buffer)

        sources = {}
        for entries in self.buffer.values():
            for entry in entries:
                src = entry.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

        lines = [
            f"Buffer: {total_structures} structures across {num_compositions} compositions",
            f"  Sources: {sources}",
            "  Top compositions:",
        ]

        # Sort compositions by count
        sorted_comps = sorted(
            self.buffer.items(), key=lambda x: len(x[1]), reverse=True
        )[:3]
        for comp, entries in sorted_comps:
            lines.append(f"    {comp}: {len(entries)} structures")

        return "\n".join(lines)

    def _format_evaluation_history(self, stability_tolerance: float) -> str:
        """Format evaluation history as an aggregated summary for the ReAct agent.

        Produces three sections:
        1. High-level summary stats (total queries, stable count, avg e_above_hull)
        2. Per-composition aggregated table sorted by best e_above_hull — makes
           diminishing-returns and winning compositions immediately obvious
        3. Compact recent query log (last 5) with optional structure details
           for recency context

        This replaces the previous flat list of the last N entries, which forced
        the LLM to scan and mentally aggregate repeated compositions. The
        aggregated format covers *all* prior queries without a sliding window,
        uses similar or fewer tokens, and surfaces clearer signals.
        """
        if not self.evaluation_history:
            return "No evaluations yet."

        include_structures = self.context_config.get(
            "include_structure_in_history", False
        )

        # --- Collect per-entry data ---
        entries = []
        for entry in self.evaluation_history:
            comp = entry.get("composition", "?")
            e_hull = entry.get("e_above_hull", float("inf"))
            is_stable = entry.get("is_stable", False)
            is_novel = entry.get("is_newly_discovered", False)
            entries.append({
                "comp": comp,
                "e_hull": e_hull,
                "is_stable": is_stable,
                "is_novel": is_novel,
                "structure": entry.get("structure"),
            })

        # --- Section 1: Summary stats ---
        total = len(entries)
        n_stable = sum(1 for e in entries if e["is_stable"])
        n_novel = sum(1 for e in entries if e["is_novel"])
        e_hulls = [e["e_hull"] for e in entries if e["e_hull"] != float("inf")]
        avg_e = sum(e_hulls) / len(e_hulls) if e_hulls else float("inf")
        best_e = min(e_hulls) if e_hulls else float("inf")

        lines = [
            "=== SUMMARY ===",
            f"Queries: {total} | Stable: {n_stable}/{total} | Novel: {n_novel} | "
            f"Avg e_above_hull: {avg_e:.3f} | Best: {best_e:.3f}",
        ]

        # --- Section 2: Per-composition aggregation ---
        comp_groups: dict[str, list[dict]] = {}
        for e in entries:
            comp_groups.setdefault(e["comp"], []).append(e)

        comp_lines = ["", "=== BY COMPOSITION (sorted by best e_above_hull) ==="]
        sorted_comps = sorted(comp_groups.items(), key=lambda x: min(e["e_hull"] for e in x[1]))
        for comp, group in sorted_comps:
            count = len(group)
            stable_count = sum(1 for e in group if e["is_stable"])
            novel_count = sum(1 for e in group if e["is_novel"])
            best = min(e["e_hull"] for e in group)
            worst = max(e["e_hull"] for e in group)
            parts = [f"{count}q", f"{stable_count} stable", f"{novel_count} novel"]
            if count == 1:
                parts.append(f"e={best:.3f}")
            else:
                parts.append(f"best={best:.3f} worst={worst:.3f}")
            comp_lines.append(f"  {comp}: {', '.join(parts)}")

        # --- Section 3: Compact recent log (last 5) ---
        n_recent = 5
        recent = entries[-n_recent:]
        start_idx = total - len(recent) + 1
        recent_lines = ["", "=== LAST 5 QUERIES ==="]
        for i, e in enumerate(recent, start_idx):
            status = []
            if e["is_stable"]:
                status.append("S")
            if e["is_novel"]:
                status.append("N")
            tag = ",".join(status) if status else "-"
            line = f"  {i}. {e['comp']} [{tag}] e={e['e_hull']:.3f}"

            if include_structures and e.get("structure"):
                struct_str = str(e["structure"])
                indented = "\n     ".join(struct_str.split("\n"))
                line += f"\n     {indented}"

            recent_lines.append(line)

        return "\n".join(lines + comp_lines + recent_lines)

    def _format_evaluation_history_for_reflection(
        self, stability_tolerance: float
    ) -> str:
        """Format evaluation history as an aggregated summary for the reflection LLM.

        Produces three sections:
        1. High-level summary stats
        2. Per-composition aggregated table (sorted by best e_above_hull)
        3. Compact chronological query log (no structure details)
        """
        if not self.evaluation_history:
            return "No evaluations yet."

        # --- Collect per-entry data ---
        entries_data = []
        for entry in self.evaluation_history:
            comp = entry.get("composition", "?")
            e_hull = entry.get("e_above_hull", float("inf"))
            is_stable = entry.get("is_stable", False)
            is_novel = entry.get("is_newly_discovered", False)

            status = []
            if is_stable:
                status.append("STABLE")
            elif e_hull <= stability_tolerance:
                status.append("METASTABLE")
            if is_novel:
                status.append("NOVEL")
            status_str = ", ".join(status) if status else "unstable"

            entries_data.append(
                {
                    "comp": comp,
                    "e_hull": e_hull,
                    "is_stable": is_stable,
                    "is_novel": is_novel,
                    "status_str": status_str,
                }
            )

        # --- Section 1: Summary stats ---
        total = len(entries_data)
        n_stable = sum(1 for e in entries_data if e["is_stable"])
        n_novel = sum(1 for e in entries_data if e["is_novel"])
        n_novel_stable = sum(
            1 for e in entries_data if e["is_stable"] and e["is_novel"]
        )
        e_hulls = [e["e_hull"] for e in entries_data if e["e_hull"] != float("inf")]
        avg_e = sum(e_hulls) / len(e_hulls) if e_hulls else float("inf")
        best_e = min(e_hulls) if e_hulls else float("inf")
        worst_e = max(e_hulls) if e_hulls else float("inf")
        best_comp = next(
            (e["comp"] for e in entries_data if e["e_hull"] == best_e), "?"
        )
        worst_comp = next(
            (e["comp"] for e in entries_data if e["e_hull"] == worst_e), "?"
        )

        summary_lines = [
            "=== SUMMARY ===",
            f"Queries: {total} | Stable: {n_stable} | Novel: {n_novel} | Novel+Stable: {n_novel_stable}",
            f"Avg e_above_hull: {avg_e:.3f} | Best: {best_e:.3f} ({best_comp}) | Worst: {worst_e:.3f} ({worst_comp})",
        ]

        # --- Section 2: Per-composition aggregation ---
        comp_groups: dict[str, list[dict]] = defaultdict(list)
        for e in entries_data:
            comp_groups[e["comp"]].append(e)

        comp_rows = []
        for comp, group in comp_groups.items():
            count = len(group)
            stable_count = sum(1 for e in group if e["is_stable"])
            novel_count = sum(1 for e in group if e["is_novel"])
            best = min(e["e_hull"] for e in group)
            worst = max(e["e_hull"] for e in group)
            comp_rows.append(
                {
                    "comp": comp,
                    "count": count,
                    "stable": stable_count,
                    "novel": novel_count,
                    "best": best,
                    "worst": worst,
                }
            )
        comp_rows.sort(key=lambda r: r["best"])

        comp_lines = ["", "=== BY COMPOSITION (sorted by best e_above_hull) ==="]
        for r in comp_rows:
            parts = [f"{r['count']} {'query' if r['count'] == 1 else 'queries'}"]
            parts.append(f"{r['stable']} stable")
            if r["novel"] > 0:
                parts.append(f"{r['novel']} novel")
            if r["count"] == 1:
                parts.append(f"e={r['best']:.3f}")
            else:
                parts.append(f"best={r['best']:.3f}, worst={r['worst']:.3f}")
            comp_lines.append(f"  {r['comp']}: {', '.join(parts)}")

        # --- Section 3: Compact chronological log ---
        log_lines = ["", "=== CHRONOLOGICAL LOG ==="]
        for i, e in enumerate(entries_data, 1):
            log_lines.append(
                f"  {i:>2}. {e['comp']} [{e['status_str']}] e={e['e_hull']:.4f}"
            )

        return "\n".join(summary_lines + comp_lines + log_lines)

    def _format_known_stable_materials(self, state: dict[str, Any]) -> str:
        """Format known stable materials from phase diagram."""
        entries_raw = state.get("phase_diagram_all_entries", [])
        if not entries_raw:
            return "No known materials."

        include_structures = self.context_config.get(
            "include_structure_in_known_materials", False
        )

        entries = []
        for e in entries_raw:
            if isinstance(e, dict):
                entries.append(PDEntry.from_dict(e))
            elif isinstance(e, PDEntry):
                entries.append(e)

        if not entries:
            return "No known materials."

        try:
            pd = PhaseDiagram(entries)
            stable = pd.stable_entries

            # Filter to compounds only
            stable_compounds = [e for e in stable if len(e.composition.elements) >= 2]

            if not stable_compounds:
                return "No known compound stable materials."

            if not include_structures:
                # Simple format - just formulas
                formulas = [e.composition.reduced_formula for e in stable_compounds]
                return f"Known stable: {', '.join(sorted(set(formulas)))}"
            else:
                # Detailed format with structure info
                lines = ["Known stable materials:"]
                for entry in stable_compounds:
                    comp = entry.composition.reduced_formula
                    e_form = pd.get_form_energy_per_atom(entry)

                    line = f"  {comp} (formation_energy={e_form:.4f} eV/atom)"

                    # Add full structure info if available
                    structure = None
                    if hasattr(entry, "structure") and entry.structure:
                        structure = entry.structure
                    elif hasattr(entry, "attribute") and entry.attribute:
                        structure = entry.attribute.get("structure")

                    if structure:
                        struct_str = str(structure)
                        # Indent each line for better formatting
                        indented_struct = "\n    ".join(struct_str.split("\n"))
                        line += f"\n    {indented_struct}"

                    lines.append(line)

                return "\n".join(lines)
        except Exception as e:
            logger.warning(
                f"[LLMReActOrchestrator] Failed to compute stable materials: {e}"
            )
            return "Unable to compute stable materials."

    def _format_prior_reflections(self) -> str:
        """Format accumulated reflections for injection into the ReAct context."""
        if not self.enable_reflexion or not self.reflections:
            return "No prior episode reflections available."
        lines = []
        for i, r in enumerate(self.reflections):
            lines.append(f"Episode {i + 1} reflection:\n{r}")
        return "\n\n---\n\n".join(lines)

    def generate_reflection(
        self, episode_metrics: dict[str, Any], stability_tolerance: float
    ) -> str:
        """Generate a verbal self-reflection on the completed episode and store it.

        Should be called by the episode runner after each episode completes.
        The reflection is appended to self.reflections (bounded by max_reflections).

        Args:
            episode_metrics: Final metrics dict from env.get_latest_metrics().
            stability_tolerance: e_above_hull threshold used in this episode.

        Returns:
            The generated reflection string (empty string if reflexion disabled).
        """
        if not self.enable_reflexion or self.self_reflection_module is None:
            return ""

        episode_summary = self._format_evaluation_history_for_reflection(
            stability_tolerance
        )

        # Prefer the current environment metric key, with backward-compat fallback.
        num_stable = episode_metrics.get(
            "num_newly_discovered_stable",
            episode_metrics.get("num_novel_stable_discovered", 0),
        )
        recall = episode_metrics.get("recall_formula", 0.0)
        recall_num = episode_metrics.get("num_correct_stable_formulas", 0)
        recall_den = episode_metrics.get("num_gt_formulas_missing_initial", 0)
        num_queries = len(self.evaluation_history)
        outcome = (
            f"Novel stable structures found: {num_stable}\n"
            f"Recall (fraction of known stable phases discovered): {recall:.3f}\n"
            f"Recall numerator/denominator: {recall_num}/{recall_den}\n"
            f"Oracle queries used: {num_queries}\n"
            "Definitions:\n"
            "- Novel stable structures found = stable structures that are locally novel in this run (vs initial structures and prior discoveries).\n"
            "- Recall = fraction of ground-truth stable formulas (missing at initialization) that were recovered.\n"
            "- Novel and recall are different metrics and can diverge."
        )

        prior = (
            "\n---\n".join(self.reflections)
            if self.reflections
            else "None (this is the first episode)."
        )

        try:
            lm = build_dspy_lm(self.llm_config)
            with dspy.context(lm=lm):
                pred = self.self_reflection_module(
                    chemical_system=", ".join(self.chemical_system_elements),
                    episode_trajectory=episode_summary,
                    episode_outcome=outcome,
                    prior_reflections=prior,
                )
            raw_reflection = pred.reflection

            # parsing logic here to strip the </think> part, so that DSPy parses cleanly
            
            if "</think>" in raw_reflection:
                reflection = raw_reflection.split("</think>", 1)[1]
            else:
                reflection = raw_reflection
            
            lm_call = lm.history[-1] if getattr(lm, "history", None) else None
            append_llm_trace(
                component="self_reflection",
                llm_config=self.llm_config,
                output={"reflection": reflection},
                inputs={
                    "chemical_system": self.chemical_system_elements,
                    "episode_outcome": outcome,
                },
                extra={"lm_call": lm_call},
            )
        except Exception as e:
            logger.error(f"[Reflexion] Failed to generate reflection: {e}")
            return ""

        # Sliding window
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_reflections:
            self.reflections = self.reflections[-self.max_reflections :]

        logger.info(f"[Reflexion] Generated reflection: {reflection[:300]}...")
        return reflection

    def update_state(self, state: dict[str, Any]) -> None:
        """Update agent state from environment observation (uses base Agent step tracking)."""
        action = self._update_last_step(state)

        # Update chemical system on init
        if action == "init":
            self.chemical_system_elements = state.get("elements", [])
            logger.info(
                f"[LLMReActOrchestrator] Initialized: {self.chemical_system_elements}"
            )

        if action == "skip":
            return

        # Update component states (on init or update)
        for gen in self.generators.values():
            gen.update_state(state)
        if self.static_filter:
            self.static_filter.update_state(state)
        if self.uniqueness_filter:
            self.uniqueness_filter.update_state(state)
        for scorer in self.scorers_dict.values():
            scorer.update_state(state)

        # Only record evaluation on update (not init)
        if action != "update":
            return

        last_obs = state.get("last_observation")
        e_hull = last_obs.get("e_above_hull", float("inf"))
        stability_tolerance = state.get("stability_tolerance", 1e-8)

        # Store structure for optional context inclusion
        structure = None
        if "proposal" in last_obs:
            try:
                proposal = last_obs["proposal"]
                if isinstance(proposal, dict):
                    structure = Structure.from_dict(proposal)
                else:
                    structure = proposal
            except Exception:
                pass

        history_entry = {
            "composition": last_obs.get("reduced_formula", "?"),
            "e_above_hull": e_hull,
            "is_stable": last_obs.get("is_stable", e_hull <= stability_tolerance),
            "is_newly_discovered": last_obs.get("is_newly_discovered", False),
            "energy_per_atom": last_obs.get("energy_per_atom"),
            "structure": structure,  # Store for optional context
        }
        self.evaluation_history.append(history_entry)

        # Remove from cache
        if "proposal" in last_obs:
            try:
                proposal = last_obs["proposal"]
                if isinstance(proposal, dict):
                    proposal = Structure.from_dict(proposal)
                proposal_hash = structure_hash(proposal)
                if proposal_hash in self.structure_cache:
                    del self.structure_cache[proposal_hash]
            except Exception:
                pass

        status = "STABLE" if history_entry["is_stable"] else "unstable"
        if history_entry["is_newly_discovered"]:
            status += ", NOVEL"

        logger.info(
            f"[LLMReActOrchestrator] Recorded: {history_entry['composition']} [{status}, e={e_hull:.4f}]"
        )
        self._update_revisit_payoff(
            composition=history_entry["composition"],
            is_stable=bool(history_entry["is_stable"]),
            is_novel=bool(history_entry["is_newly_discovered"]),
        )
        self._refresh_behavior_metrics()

    def get_state(self) -> dict[str, Any]:
        """Get agent state for checkpointing."""
        # Serialize composition-based buffer
        buffer_dict = {}
        for comp, entries in self.buffer.items():
            buffer_dict[comp] = [
                {
                    "structure_dict": entry["structure"].as_dict(),
                    "hash": entry["hash"],
                    "composition": entry["composition"],
                    "full_formula": entry.get("full_formula", entry["composition"]),
                    "source": entry.get("source"),
                    "scores": entry.get("scores", {}),
                    "num_sites": entry.get("num_sites", 0),
                }
                for entry in entries
            ]

        # Serialize evaluation history with structures
        serialized_history = []
        for entry in self.evaluation_history:
            serialized_entry = {
                "composition": entry["composition"],
                "e_above_hull": entry["e_above_hull"],
                "is_stable": entry["is_stable"],
                "is_newly_discovered": entry["is_newly_discovered"],
                "energy_per_atom": entry.get("energy_per_atom"),
            }
            if entry.get("structure"):
                serialized_entry["structure_dict"] = entry["structure"].as_dict()
            serialized_history.append(serialized_entry)

        behavior_tracking = {
            "selection_history": list(self.selection_history),
            "selection_counts": dict(self.selection_counts),
            "selection_new_count": self.selection_new_count,
            "selection_switch_count": self.selection_switch_count,
            "selection_max_streak": self.selection_max_streak,
            "selection_current_streak": self.selection_current_streak,
            "selection_last_comp": self.selection_last_comp,
            "targeted_compositions_seen": sorted(self.targeted_compositions_seen),
            "targeted_new_total": self.targeted_new_total,
            "targeted_revisit_total": self.targeted_revisit_total,
            "element_tool_arg_call_counts": dict(self.element_tool_arg_call_counts),
            "element_tool_arg_total_calls": self.element_tool_arg_total_calls,
            "generated_total": self.generated_total,
            "generated_added_total": self.generated_added_total,
            "generated_duplicate_total": self.generated_duplicate_total,
            "generated_cached_total": self.generated_cached_total,
            "generated_static_filtered_total": self.generated_static_filtered_total,
            "created_added_total": self.created_added_total,
            "revisit_payoff_stats": {
                str(k): dict(v) for k, v in self.revisit_payoff_stats.items()
            },
            "revisit_total_count": self.revisit_total_count,
            "revisit_stable_count": self.revisit_stable_count,
            "revisit_novel_stable_count": self.revisit_novel_stable_count,
            "recent_yield_window": self.recent_yield_window,
            "query_count": self.query_count,
            "family_first_seen_query_idx": dict(self.family_first_seen_query_idx),
            "new_families_cumulative_set": sorted(self.new_families_cumulative_set),
            "last_new_family_query_idx": self.last_new_family_query_idx,
            "num_new_families_last_query": self.num_new_families_last_query,
            "num_new_family_structures_added_last_query": self.num_new_family_structures_added_last_query,
            "selected_family_is_new_last_query": self.selected_family_is_new_last_query,
            "selected_family_age_queries_last_query": self.selected_family_age_queries_last_query,
            "queries_since_last_new_family_last_query": self.queries_since_last_new_family_last_query,
            "new_family_generation_share_last_query": self.new_family_generation_share_last_query,
            "latest_behavior_metrics": dict(self.latest_behavior_metrics),
            "behavior_metrics_history": list(self.behavior_metrics_history),
            "pending_selected_comp": self._pending_selected_comp,
            "pending_selected_visit_index": self._pending_selected_visit_index,
        }

        return {
            "buffer": buffer_dict,
            "structure_cache_hashes": list(self.structure_cache.keys()),
            "evaluation_history": serialized_history,
            "chemical_system_elements": self.chemical_system_elements,
            "last_step": self.last_step,
            "reflections": self.reflections,
            "behavior_tracking": behavior_tracking,
        }

    def get_latest_behavior_metrics(self) -> dict[str, float]:
        """Return latest behavior metrics snapshot."""
        if not self.latest_behavior_metrics:
            return self._refresh_behavior_metrics()
        return dict(self.latest_behavior_metrics)

    def get_behavior_metrics_history(self) -> list[dict[str, float]]:
        """Return behavior metrics history snapshots."""
        return [dict(x) for x in self.behavior_metrics_history]

    def load_state(self, state: dict[str, Any]) -> None:
        """Load agent state from checkpoint."""
        if "buffer" in state:
            self.buffer = {}
            self.structure_cache = {}
            buffer_data = state["buffer"]

            # Handle both old (list) and new (dict) formats
            if isinstance(buffer_data, list):
                # Old format - convert to new format
                for entry_dict in buffer_data:
                    structure = Structure.from_dict(entry_dict["structure_dict"])
                    comp = entry_dict["composition"]
                    full_formula = entry_dict.get(
                        "full_formula", structure.composition.formula.replace(" ", "")
                    )
                    entry = {
                        "structure": structure,
                        "hash": entry_dict["hash"],
                        "composition": comp,
                        "full_formula": full_formula,
                        "source": entry_dict.get("source"),
                        "scores": entry_dict.get("scores", {}),
                        "num_sites": entry_dict.get("num_sites", len(structure)),
                    }
                    if comp not in self.buffer:
                        self.buffer[comp] = []
                    self.buffer[comp].append(entry)
                    self.structure_cache[entry["hash"]] = entry
            else:
                # New format (dict of compositions)
                for comp, entries_list in buffer_data.items():
                    self.buffer[comp] = []
                    for entry_dict in entries_list:
                        structure = Structure.from_dict(entry_dict["structure_dict"])
                        full_formula = entry_dict.get(
                            "full_formula",
                            structure.composition.formula.replace(" ", ""),
                        )
                        entry = {
                            "structure": structure,
                            "hash": entry_dict["hash"],
                            "composition": entry_dict["composition"],
                            "full_formula": full_formula,
                            "source": entry_dict.get("source"),
                            "scores": entry_dict.get("scores", {}),
                            "num_sites": entry_dict.get("num_sites", len(structure)),
                        }
                        self.buffer[comp].append(entry)
                        self.structure_cache[entry["hash"]] = entry

        if "evaluation_history" in state:
            # Deserialize structures in history
            self.evaluation_history = []
            for entry_dict in state["evaluation_history"]:
                entry = {
                    "composition": entry_dict["composition"],
                    "e_above_hull": entry_dict["e_above_hull"],
                    "is_stable": entry_dict["is_stable"],
                    "is_newly_discovered": entry_dict["is_newly_discovered"],
                    "energy_per_atom": entry_dict.get("energy_per_atom"),
                }
                # Deserialize structure if present
                if "structure_dict" in entry_dict:
                    entry["structure"] = Structure.from_dict(
                        entry_dict["structure_dict"]
                    )
                else:
                    entry["structure"] = None
                self.evaluation_history.append(entry)

        if "chemical_system_elements" in state:
            self.chemical_system_elements = state["chemical_system_elements"]

        if "last_step" in state:
            self.last_step = state["last_step"]

        if "reflections" in state:
            self.reflections = state["reflections"]

        behavior_tracking = state.get("behavior_tracking")
        if isinstance(behavior_tracking, dict):
            self.selection_history = [
                str(x) for x in behavior_tracking.get("selection_history", [])
            ]
            self.selection_counts = {
                str(k): int(v)
                for k, v in behavior_tracking.get("selection_counts", {}).items()
            }
            self.selection_new_count = int(
                behavior_tracking.get("selection_new_count", 0)
            )
            self.selection_switch_count = int(
                behavior_tracking.get("selection_switch_count", 0)
            )
            self.selection_max_streak = int(
                behavior_tracking.get("selection_max_streak", 0)
            )
            self.selection_current_streak = int(
                behavior_tracking.get("selection_current_streak", 0)
            )
            self.selection_last_comp = str(
                behavior_tracking.get("selection_last_comp", "")
            )

            self.targeted_compositions_seen = set(
                behavior_tracking.get("targeted_compositions_seen", [])
            )
            self.targeted_new_total = int(behavior_tracking.get("targeted_new_total", 0))
            self.targeted_revisit_total = int(
                behavior_tracking.get("targeted_revisit_total", 0)
            )

            self.element_tool_arg_call_counts = {
                str(k): int(v)
                for k, v in behavior_tracking.get(
                    "element_tool_arg_call_counts", {}
                ).items()
            }
            self.element_tool_arg_total_calls = int(
                behavior_tracking.get("element_tool_arg_total_calls", 0)
            )

            self.generated_total = int(behavior_tracking.get("generated_total", 0))
            self.generated_added_total = int(
                behavior_tracking.get("generated_added_total", 0)
            )
            self.generated_duplicate_total = int(
                behavior_tracking.get("generated_duplicate_total", 0)
            )
            self.generated_cached_total = int(
                behavior_tracking.get("generated_cached_total", 0)
            )
            self.generated_static_filtered_total = int(
                behavior_tracking.get("generated_static_filtered_total", 0)
            )
            self.created_added_total = int(
                behavior_tracking.get("created_added_total", 0)
            )

            raw_revisit = behavior_tracking.get("revisit_payoff_stats", {})
            if isinstance(raw_revisit, dict):
                self.revisit_payoff_stats = {}
                for k, v in raw_revisit.items():
                    try:
                        visit_idx = int(k)
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        self.revisit_payoff_stats[visit_idx] = {
                            "count": int(v.get("count", 0)),
                            "stable": int(v.get("stable", 0)),
                            "novel_stable": int(v.get("novel_stable", 0)),
                        }
            self.revisit_total_count = int(
                behavior_tracking.get("revisit_total_count", 0)
            )
            self.revisit_stable_count = int(
                behavior_tracking.get("revisit_stable_count", 0)
            )
            self.revisit_novel_stable_count = int(
                behavior_tracking.get("revisit_novel_stable_count", 0)
            )
            self.recent_yield_window = int(
                behavior_tracking.get("recent_yield_window", 10)
            )
            self.query_count = int(behavior_tracking.get("query_count", 0))
            self.family_first_seen_query_idx = {
                str(k): int(v)
                for k, v in behavior_tracking.get(
                    "family_first_seen_query_idx", {}
                ).items()
            }
            self.new_families_cumulative_set = set(
                behavior_tracking.get("new_families_cumulative_set", [])
            )
            last_new_family_query_idx = behavior_tracking.get(
                "last_new_family_query_idx"
            )
            self.last_new_family_query_idx = (
                int(last_new_family_query_idx)
                if last_new_family_query_idx is not None
                else None
            )
            self.num_new_families_last_query = int(
                behavior_tracking.get("num_new_families_last_query", 0)
            )
            self.num_new_family_structures_added_last_query = int(
                behavior_tracking.get(
                    "num_new_family_structures_added_last_query", 0
                )
            )
            self.selected_family_is_new_last_query = int(
                behavior_tracking.get("selected_family_is_new_last_query", 0)
            )
            self.selected_family_age_queries_last_query = int(
                behavior_tracking.get("selected_family_age_queries_last_query", -1)
            )
            self.queries_since_last_new_family_last_query = int(
                behavior_tracking.get("queries_since_last_new_family_last_query", 0)
            )
            self.new_family_generation_share_last_query = float(
                behavior_tracking.get("new_family_generation_share_last_query", 0.0)
            )

            self.latest_behavior_metrics = {
                str(k): float(v)
                for k, v in behavior_tracking.get("latest_behavior_metrics", {}).items()
            }
            history = behavior_tracking.get("behavior_metrics_history", [])
            self.behavior_metrics_history = []
            if isinstance(history, list):
                for row in history:
                    if isinstance(row, dict):
                        clean = {}
                        for k, v in row.items():
                            try:
                                clean[str(k)] = float(v)
                            except Exception:
                                continue
                        if clean:
                            self.behavior_metrics_history.append(clean)

            self._pending_selected_comp = behavior_tracking.get("pending_selected_comp")
            pending_idx = behavior_tracking.get("pending_selected_visit_index")
            self._pending_selected_visit_index = (
                int(pending_idx) if pending_idx is not None else None
            )
