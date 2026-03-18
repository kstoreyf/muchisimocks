#!/bin/bash

# Request interactive job and start Jupyter in one go
srun --pty \
  --cpus-per-task=8 \
  --mem=20G \
  --time=04:00:00 \
  --job-name=jupyter \
  bash -c '
    # Activate your environment
    source activate benv
    # OR: source /path/to/your/venv/bin/activate
    
    # Get connection info
    HOST=$(hostname)
    PORT=8888
    
    echo "========================================"
    echo "Jupyter starting on ${HOST}:${PORT}"
    echo "========================================"
    echo ""
    
    # Start Jupyter
    jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0

    echo "type this ssh tunnel command into a diff terminal (e.g. in vscode) (may need to change the first port; e.g. 8889?)"
    echo "echo "ssh -N -L ${PORT}:${HOST}:${PORT} $(echo $SLURM_SUBMIT_HOST)"
'

