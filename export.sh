#!/usr/bin/env bash
# DAOCP, HPIPM, BLASFEO, qpOASES binaries
export LD_LIBRARY_PATH="$HOME/Desktop/daocp/lib:/opt/hpipm/lib:$HOME/Desktop/qpOASES/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# DAOCP python interface
export PYTHONPATH="$HOME/Desktop/daocp/lib${PYTHONPATH:+:$PYTHONPATH}"
# HPIPM python interface
export PYTHONPATH="$HOME/Desktop/hpipm/interfaces/python/hpipm_python${PYTHONPATH:+:$PYTHONPATH}"
# qpOASES python interface
export PYTHONPATH="$HOME/Desktop/qpOASES/bin${PYTHONPATH:+:$PYTHONPATH}"
