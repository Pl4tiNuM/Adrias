This step concerns the simulation of random execution scenarios to capture system and application dynamics in "real-world" situations.

You should run the scripts sequentially, as they are listed above.

## Scripts description
`00-simulate.sh`:  This script performs a simulation on the disaggregated system. What it does is that it spawns random jobs at random time intervals (can be altered inside the script) over a given period (can be altered inside the script). It also utilizes the `Watcher` component to monitor and report perf events and FPGA metrics throughout execution (requires sudo privileges). Once completed, it creates a `results` folder and a `wrapper.log` file which contain all the information of the simulation.

Usage:
```bash
sudo ./00-simulate.sh
```

`01-system-simulation-dataset.sh`: This script creates a dataset containing the system metrics throughout the simulation execution, so that they can be plotted with the `01-system-simulation-plot.py` script. It assumes that the `results` folder and `wrapper.log` file are inside a parent folder on the current directory (e.g., `5-60/results/` and `5-60/wrapper.log`).

Usage:
```
sudo ./01-system-simulation-dataset.sh <scenario>
```

where scenario is the parent folder mentioned before (`5-60`). Once completed, it creates a `system-simulation.dat` file inside the parent folder.

`01-system-simulation-plot.py`: This script iterates over all parent directories and searches for `system-simulation.dat` files. It plots the respective events of the simulated scenario.

Usage
```
python3 01-system-simulation-plot.py
```