"""
step01_descriptors.py
---------------------
Compute SOAP descriptors from structure files (vasprun.xml, .traj, …).

SOAP (Smooth Overlap of Atomic Positions) turns each atomic structure into a
fixed-length numeric vector that captures its local atomic environments.
Structures that are geometrically similar get similar SOAP vectors — this is
what steps 02 (dimensionality reduction) and 03 (sampling) rely on to measure
"how different" two structures are, without ever looking at raw coordinates
again.

What this step does, in order
------------------------------
    1. Discover structure files under `descriptors.input_roots`, optionally
       keep only a subset of files (`file_selection_mode`) and/or a subset
       of frames per file (`frame_stride`).
    2. Load them into ASE Atoms objects.
    3. (Optional, off by default) Collapse every chemical species to a single
       one — see `modify_species` below.
    4. Compute one SOAP vector per structure, optionally appending selected
       force components so the descriptor also reflects force diversity.
    5. Save the descriptor matrix (.npy) alongside a provenance table
       (which row came from which file/frame) and the exact run config, so a
       later step (or you, in 6 months) can trace any row back to its source.

Usage
-----
    python step01_descriptors.py                         # uses config.yaml in cwd
    python step01_descriptors.py --config my_cfg.yaml
    python step01_descriptors.py --override descriptors.soap.r_cut=8.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from dscribe.descriptors import SOAP
from tqdm import tqdm

from pipeline_utils import (
    apply_overrides,
    base_parser,
    load_config,
    resolve_work_dir,
    setup_logging,
)

# These imports mirror the original notebooks — adjust paths if your src/ layout differs
from src.desc_comp_utils import fixed_mask, force_components
from src.workflow_config import discover_structure_files, ensure_directory
from src.workflow_io import (
    build_filemap,
    build_provenance_table,
    clear_directory_outputs,
    load_structure_sets,
    save_descriptor_run,
    select_structure_files,
    unique_species,
)


# ---------------------------------------------------------------------------
# Core logic (importable by run_pipeline.py)
# ---------------------------------------------------------------------------

def run(cfg: dict, logger: logging.Logger) -> dict:
    """
    Execute the descriptor computation step.

    Parameters
    ----------
    cfg:    Full pipeline config dict.
    logger: Configured logger.

    Returns
    -------
    dict with paths to the saved artifacts (npy, provenance, filemap, …).
    """
    dcfg = cfg["descriptors"]
    work_dir = resolve_work_dir(cfg)

    # ------------------------------------------------------------------
    # 1. Discover and select structure files
    # ------------------------------------------------------------------
    input_roots = [str(work_dir / Path(r)) for r in dcfg["input_roots"]]
    input_patterns = dcfg.get("input_patterns", ["vasprun*.xml", "*.traj"])

    all_files = discover_structure_files(input_roots, patterns=input_patterns, recursive=True)
    selected_files = select_structure_files(
        all_files,
        mode=dcfg.get("file_selection_mode", "all"),
        random_count=dcfg.get("random_file_count"),
        file_stride=dcfg.get("file_stride"),
        random_seed=dcfg.get("selection_seed", 0),
    )
    frame_stride = dcfg.get("frame_stride", 1)

    logger.info("Discovered %d structure files across %d input roots.", len(all_files), len(input_roots))
    logger.info("Selected %d files (mode='%s').", len(selected_files), dcfg.get("file_selection_mode", "all"))
    logger.info("Frame stride: %d", frame_stride)

    # ------------------------------------------------------------------
    # 2. Load structures
    # ------------------------------------------------------------------
    structures, source_struct_ids = load_structure_sets(
        selected_files,
        frame_stride=frame_stride,
        return_frame_indices=True,
        show_progress=True,
    )
    total_structs = sum(len(s) for s in structures)
    logger.info("Total structures loaded: %d", total_structs)

    species = unique_species(structures)
    logger.info("Species found: %s", species)

    # ------------------------------------------------------------------
    # 3. Optional: collapse all species to a single element
    # ------------------------------------------------------------------
    # Not exposed in config.yaml (defaults off) — makes SOAP blind to which
    # element sits where, so descriptors compare pure geometry/structural
    # diversity instead of chemical identity. Enable via
    # --override descriptors.modify_species=true descriptors.target_species=X
    # if you need structure-only sampling regardless of composition.
    modify_species = dcfg.get("modify_species", False)
    new_species = list(species)

    if modify_species:
        target_species = dcfg.get("target_species", "C")
        logger.info("Collapsing all species → '%s'", target_species)
        for struct_list in structures:
            for atoms in struct_list:
                for atom in atoms:
                    if atom.symbol != target_species:
                        atom.symbol = target_species
        new_species = [target_species]

    # ------------------------------------------------------------------
    # 4. Build SOAP descriptor
    # ------------------------------------------------------------------
    soap_cfg = dcfg.get("soap", {})
    compression = soap_cfg.get("compression")

    soap_params = {
        "species": new_species,
        "periodic": True,
        "r_cut": float(soap_cfg.get("r_cut", 6.0)),
        "n_max": int(soap_cfg.get("n_max", 4)),
        "l_max": int(soap_cfg.get("l_max", 4)),
        "sigma": float(soap_cfg.get("sigma", 1.0)),
    }
    if compression:
        soap_params["compression"] = compression

    logger.info("Initialising SOAP with params: %s", soap_params)
    soap = SOAP(**soap_params)

    # ------------------------------------------------------------------
    # 5. Compute descriptors
    # ------------------------------------------------------------------
    # include_forces appends selected force components (0=Fx, 1=Fy, 2=Fz, by
    # atom, flattened) to the SOAP vector. Two structures with near-identical
    # geometry but very different forces (e.g. right before a bond breaks)
    # then land further apart in descriptor space — useful when the sampler
    # (step 03) should also chase force diversity, not just geometry.
    include_forces = dcfg.get("include_forces", False)
    force_components_idx = tuple(dcfg.get("force_components", [0, 1, 2]))
    missing_force_structures = 0

    desc_blocks, force_blocks = [], []

    pbar = tqdm(total=total_structs, desc="Computing SOAP descriptors")
    for struct_list in structures:
        for atoms in struct_list:
            D = soap.create(atoms, n_jobs=1)
            desc_blocks.append(D)
            if include_forces:
                F = force_components(atoms, components=force_components_idx)
                if np.isnan(F).any():
                    missing_force_structures += 1
                force_blocks.append(F)
            pbar.update(1)
    pbar.close()

    all_soap_descriptors = np.vstack(desc_blocks)

    if include_forces:
        all_forces = np.vstack(force_blocks)
        if np.isnan(all_forces).any():
            logger.warning(
                "Forces missing for %d/%d structures — skipping force augmentation.",
                missing_force_structures, total_structs,
            )
            include_forces = False
            all_descriptors = all_soap_descriptors
        else:
            all_descriptors = np.hstack([all_soap_descriptors, all_forces])
            logger.info("Force components appended. Descriptor dim: %d", all_descriptors.shape[1])
    else:
        all_descriptors = all_soap_descriptors

    logger.info("Descriptor matrix shape: %s", all_descriptors.shape)

    # ------------------------------------------------------------------
    # 6. Build provenance table
    # ------------------------------------------------------------------
    metadata_df = build_provenance_table(
        structures,
        selected_files,
        fixed_mask_fn=fixed_mask,
        source_struct_ids=source_struct_ids,
    )
    logger.info("Provenance table shape: %s, columns: %s", metadata_df.shape, metadata_df.columns.tolist())

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    output_dir = work_dir / dcfg.get("output_dir", "desc")
    ensure_directory(str(output_dir))

    # Default ON: wipes previous *.npy/provenance/filemap/config files in
    # output_dir before writing the new run's artifacts, so stale descriptors
    # from an earlier config never get silently mixed with the current ones.
    # Set descriptors.clean_output_dir=false to accumulate runs instead.
    clean = dcfg.get("clean_output_dir", True)
    if clean:
        removed = clear_directory_outputs(
            output_dir,
            patterns=("*.npy", "*_provenance.parquet", "*_provenance.csv",
                      "*_filemap.json", "*_params.txt", "*_config.json"),
        )
        if removed:
            logger.info("Removed %d previous descriptor files from %s", len(removed), output_dir)

    base = f"SOAP_{'-'.join(new_species)}"
    run_config = {
        "input_roots": dcfg["input_roots"],
        "input_patterns": input_patterns,
        "file_selection_mode": dcfg.get("file_selection_mode", "all"),
        "random_file_count": dcfg.get("random_file_count"),
        "file_stride": dcfg.get("file_stride"),
        "selection_seed": dcfg.get("selection_seed", 0),
        "frame_stride": frame_stride,
        "discovered_file_count": len(all_files),
        "selected_file_count": len(selected_files),
        "soap_params": soap_params,
        "include_forces": include_forces,
        "force_components": list(force_components_idx),
        "soap_dim": int(all_soap_descriptors.shape[1]),
        "force_dim": int(all_descriptors.shape[1] - all_soap_descriptors.shape[1]) if include_forces else 0,
        "total_structures": total_structs,
        "clean_output_dir": clean,
    }

    artifacts = save_descriptor_run(
        output_dir,
        all_descriptors,
        metadata_df,
        build_filemap(selected_files),
        run_config,
        base_name=base,
        clear_existing=True,
    )
    logger.info("Saved descriptor artifacts:")
    for key, path in artifacts.items():
        logger.info("  %-15s %s", key, path)

    return artifacts


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = base_parser(
        "step01_descriptors",
        description=__doc__,
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])

    work_dir = resolve_work_dir(cfg)
    log_file = work_dir / "logs" / "step01_descriptors.log"
    logger = setup_logging(
        log_file=log_file,
        level=getattr(logging, args.log_level),
        step_name="step01",
    )

    logger.info("=== Step 01 — SOAP Descriptor Computation ===")
    logger.info("Config: %s", args.config)

    try:
        run(cfg, logger)
        logger.info("=== Step 01 completed successfully ===")
    except Exception:
        logger.exception("Step 01 failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
