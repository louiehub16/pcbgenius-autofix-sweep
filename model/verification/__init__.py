"""PCBGenius — Real KiCad CLI verification engine (run_erc / run_drc)."""
from .kicad_engine import (
    kicad_cli_available,
    run_erc,
    run_drc,
    engine_used,
    build_kicad_sch,
    build_kicad_pcb,
    parse_kicad_erc_report,
    parse_kicad_drc_report,
    KICAD_CLI,
)

__all__ = [
    "kicad_cli_available",
    "run_erc",
    "run_drc",
    "engine_used",
    "build_kicad_sch",
    "build_kicad_pcb",
    "parse_kicad_erc_report",
    "parse_kicad_drc_report",
    "KICAD_CLI",
]