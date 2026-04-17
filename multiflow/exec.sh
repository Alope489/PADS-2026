#!/usr/bin/env bash
unset DISPLAY
export OMPI_MCA_btl_vader_single_copy_mechanism=none
set -euo pipefail
CODES_BIN="${CODES_REPLAY:-../../codes/build/bin/bin/model-net-mpi-replay}"
exec "$CODES_BIN" --synch=1 --workload_type=online \
    --workload_conf_file=../conf/work.load \
    --alloc_file=../conf/alloc.conf \
    --lp-io-dir=riodir --lp-io-use-suffix=1 \
    --payload_sz=64 \
    --end=70000 \
    --workload_period_file=period.file \
    -- ../conf/modsim-dfdally72-min.conf \
    &>> 3.flows.out
