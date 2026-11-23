TLTDRP-Reproducibility
Complete implementation code and benchmark instances for Watchtower–Drone–Truck Cooperative Network Optimization for Forest Fire Patrol.

Repository: https://github.com/Lcy0507/TLTDRP-Reproducibility
1. Overview
This repository accompanies a research study on the Tower Location and Truck–Drone Routing Problem (TLTDRP) for forest-fire patrol. The proposed patrol system integrates stationary watchtowers, road-based patrol trucks, and drones launched from and recovered at selected watchtowers. The optimization model jointly determines watchtower locations, monitoring assignments, resource usage, and truck/drone patrol routes while minimizing the total system cost subject to coverage, capacity, and endurance constraints.

The repository provides implementations for exact mixed-integer optimization with Gurobi and an Improved Adaptive Large Neighborhood Search (IALNS) algorithm without a restart mechanism. It also contains code for Taguchi parameter tuning, comparison with genetic algorithm (GA), tabu search algorithm (TSA), and standard ALNS, ablation studies, four patrol-mode comparisons, and one-/two-factor sensitivity analysis.
2. Repository contents
The uploaded code package is organized under TLTDRP code/. Original Chinese filenames are retained so that they can be matched directly to the supplied scripts and data.
TLTDRP-Reproducibility/
├── README.md
├── LICENSE
├── .gitignore
├── TLTDRP code/
│   ├── Gurobi/
│   │   └── gurobi.py
│   ├── IALNS算法参数调优实验/
│   │   ├── TLTDRP_IALNS_Taguchi_无重启_最终单算法_12因子L27完整版.py
│   │   └── TLTDRP_IALNS_无重启_单算法_支持Taguchi参数JSON.py
│   ├── instances/
│   │   └── [15 simulated Excel instances and one 68-node Xinlin case]
│   ├── 四算法对比实验/
│   │   ├── TLTDRP_四算法_独立实现_调参与正式比较_完整代码.py
│   │   └── Taguchi_最终最优参数.json
│   ├── 消融实验/
│   │   ├── TLTDRP_IALNS_Taguchi新参数_无重启消融实验_最终完整代码.py
│   │   └── Taguchi_最终最优参数.json
│   ├── 四模式对比实验/
│   │   ├── TLTDRP_四场景_最新Taguchi无重启IALNS_内存修复断点续跑版.py
│   │   ├── Taguchi_最终最优参数.json
│   │   └── 四场景_运行说明.txt
│   └── 敏感性分析/
│       ├── TLTDRP_IALNS_Taguchi无重启_新林案例_单因素双因素交互敏感性_300km_2行3列完整版.py
│       ├── Taguchi_最终最优参数.json
│       └── 运行说明.txt
├── TLTDRP code.zip
└── 代码运行指南.docx
The source package contains 7 Python scripts, 16 Excel workbooks (15 simulated instances and the Xinlin 68-node case), four Taguchi parameter JSON files, and two text run guides. The ZIP is provided as a convenience download; the browsable source files are the preferred reference.
3. Data and benchmark instances
The instances/ directory contains 15 synthetic instances of increasing size: 4, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, and 65 monitoring nodes, and one real-world case with 68 monitoring nodes in Xinlin, Greater Khingan Mountains, Heilongjiang, China. The Xinlin case includes 41 roadside and 27 non-roadside monitoring nodes as described in the manuscript.

Input workbooks supply the case data used by the scripts. See the script-specific input routines and 代码运行指南.docx for workbook conventions. The case is an optimization benchmark based on the research inputs; it should not be interpreted as a field-validated operational deployment or as a complete geographical road network.
4. Environment and dependencies
The implementations are written in Python 3. Common Python packages include:
- numpy
- pandas
- matplotlib
- openpyxl (Excel input/output support)
- statsmodels (two-factor statistical analysis in the sensitivity script)
- gurobipy and a working Gurobi Optimizer license (exact solver only)
Install the needed packages in your Python environment, for example:
python -m pip install numpy pandas matplotlib openpyxl statsmodels
For the exact solution program, install Gurobi separately according to its official instructions: https://www.gurobi.com/documentation/ . Gurobi is proprietary and requires appropriate licensing. Exact tested Python/package/Gurobi version numbers have not yet been documented in this repository; users should record the versions of their own environment when reporting reproduction results.

Some scripts include absolute Windows paths as defaults. Always pass the --base-dir argument where supported or update the documented path variables before running. Commands below assume the current working directory is the repository root and the package folder is named TLTDRP code.
5. Reference experimental settings
The manuscript's baseline patrol parameters are:
Parameter	Baseline value
Watchtower coverage radius	10 km
Maximum drones per watchtower	2
Truck endurance	300 km
Drone endurance	50 km
Patrol frequency	360 patrols/year
Road-distance approximation factor	1.3


For the no-restart IALNS, the reported frozen search settings include 500 iterations, initial simulated annealing temperature 2500, cooling coefficient 0.992, minimum temperature 0.05, weight reaction factor 0.20, and operator-weight update interval 40 iterations. Use the corresponding Taguchi_最终最优参数.json rather than silently changing tuned parameters.

Different experiment scripts may employ different random seed bases or run protocols. Reproduction should follow the settings in the specific experiment and retain the generated configuration files. In particular, results from the four-algorithm and four-patrol-mode runs should not be mixed as though they came from the same experimental batch.
6. How to reproduce the experiments
All commands below are illustrative entry points using options verified in the uploaded Python scripts. Depending on your local directory names and platform, adjust the paths. Start with a single-case or quick run to check the environment; a quick run is not a replacement for full paper experiments.
6.1 Standalone IALNS, single instance
python "TLTDRP code/IALNS算法参数调优实验/TLTDRP_IALNS_无重启_单算法_支持Taguchi参数JSON.py" \
  --base-dir "TLTDRP code/instances" \
  --instance "小规模算例1_4个节点" \
  --parameter-file "TLTDRP code/四算法对比实验/Taguchi_最终最优参数.json" \
  --truck-range 300 --runs 10
The script also accepts --excel for an explicit workbook path and --quick for a short functionality check.
6.2 Taguchi tuning of no-restart IALNS
python "TLTDRP code/IALNS算法参数调优实验/TLTDRP_IALNS_Taguchi_无重启_最终单算法_12因子L27完整版.py" \
  --stage taguchi --base-dir "TLTDRP code/instances" --truck-range 300
Supported stages include design, taguchi, validate, solve, and all. Refer to --help and the script's generated reports for the full sequence and confirmation settings.
6.3 Four-algorithm comparison (GA, TSA, ALNS, IALNS)
python "TLTDRP code/四算法对比实验/TLTDRP_四算法_独立实现_调参与正式比较_完整代码.py" \
  --stage compare --scope simulation \
  --base-dir "TLTDRP code/instances" \
  --ialns-json "TLTDRP code/四算法对比实验/Taguchi_最终最优参数.json" \
  --truck-range 300 --runs 10 --time-limit 15
For a real-world case, use --scope real and check that only available workbooks are selected. The script also supports parameter tuning through --stage tune.
6.4 Ablation studies
python "TLTDRP code/消融实验/TLTDRP_IALNS_Taguchi新参数_无重启消融实验_最终完整代码.py" \
  --scope simulation --base-dir "TLTDRP code/instances" \
  --parameter-file "TLTDRP code/消融实验/Taguchi_最终最优参数.json" \
  --truck-range 300 --runs 10
6.5 Four patrol modes
The four scenarios are watchtower-only, watchtower + truck, watchtower + drone, and watchtower + truck + drone.
python "TLTDRP code/四模式对比实验/TLTDRP_四场景_最新Taguchi无重启IALNS_内存修复断点续跑版.py" \
  --scope simulation --base-dir "TLTDRP code/instances" \
  --parameter-file "TLTDRP code/四模式对比实验/Taguchi_最终最优参数.json" \
  --truck-range 300 --runs 10
The program supports --resume for runs interrupted under the same configuration. Consult 四场景_运行说明.txt; the actual Python filename above is authoritative if the text guide refers to a different earlier filename.
6.6 Single-factor and two-factor sensitivity analysis
python "TLTDRP code/敏感性分析/TLTDRP_IALNS_Taguchi无重启_新林案例_单因素双因素交互敏感性_300km_2行3列完整版.py" \
  --mode all --base-dir "TLTDRP code/instances" \
  --parameter-file "TLTDRP code/敏感性分析/Taguchi_最终最优参数.json" \
  --runs 10
This code uses the 300 km truck-endurance baseline. It examines coverage radius, tower drone capacity, truck endurance, and drone endurance; the pairwise design contains six two-factor combinations. Supported modes are one, interaction, and all. The script also supports --resume and --plots-only for existing results. Consult 敏感性分析/运行说明.txt for requirements specific to these modes.
6.7 Gurobi exact solver
Run TLTDRP code/Gurobi/gurobi.py from an environment with a working Gurobi license. Before executing, update the script's BASE_INSTANCE_DIR setting to your local instances/ directory. This program does not expose the same --base-dir command-line option as the heuristic experiment scripts.
7. Outputs, cost units, and reporting conventions
The implementations compute costs in USD. The manuscript presents most monetary results in 10,000 USD (万美元), generally to one decimal place:
cost_in_10k_USD = raw_cost_in_USD / 10000
For reproducible percentages and averages, calculate the statistics from the unrounded original values, and round only the displayed manuscript values. Best denotes the minimum over independent runs, Avg their mean, and running time is measured in seconds. Gurobi upper bounds should be distinguished from certified optimal values when the optimality gap remains positive.

The scripts generate experiment-specific result directories containing CSV/Excel summaries and, where enabled, route visualizations and convergence plots. This repository snapshot primarily provides code, parameters, and benchmark inputs; a complete archive of all underlying experiment run outputs should be deposited separately or added here before claiming that raw experimental outputs are available.
8. Reproducibility notes and limitations
- Use the published, frozen parameters for formal experiments; do not use --quick or reduced iteration counts for paper reproduction.
- Keep the 300 km truck range for baseline comparisons and the updated sensitivity analysis, unless intentionally studying truck range as a factor.
- Use fixed seeds and document seed bases as reported by each experiment driver; numerical outcomes of stochastic solvers can differ across environments.
- The code performs approximate truck distance estimation where detailed road-network data are unavailable; results are model-based research outputs.
- Some scripts retain local Windows default paths or older explanatory comments; follow the explicit invocation paths and the reported configuration used for each run.
- Check the scope and availability of any non-public input data before redistributing them.
9. Citation and acknowledgments
If you use this software or benchmark data, please cite the accompanying study when its final bibliographic information is available:
Watchtower–Drone–Truck Cooperative Network Optimization for Forest Fire Patrol. Manuscript submitted to the Journal of Systems Engineering. Full authorship, year, volume, pages, and DOI: to be added upon publication.

Source repository: https://github.com/Lcy0507/TLTDRP-Reproducibility

No Zenodo DOI or archived release is asserted here until one has actually been created.
10. License and data rights
The GitHub repository currently contains an MIT LICENSE file. That license should be interpreted for original code only to the extent that the repository owner has the necessary rights. The licensing and redistribution permissions for the Xinlin case data, any third-party materials, and external dependencies should be checked separately; Gurobi remains subject to its own proprietary license.
11. Contact and issue reporting
For reproducibility questions, bug reports, and proposed corrections, please open an Issue in this repository. Include the script name, command, environment details, input instance, random seed, and relevant error message.
