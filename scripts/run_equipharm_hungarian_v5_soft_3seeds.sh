#!/usr/bin/env bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_equipharm_seeded.sh" EquiPharm_Hungarian_v5_soft "$@"
