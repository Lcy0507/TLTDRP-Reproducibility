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
│   ├── 敏感性分析/
│   │    ├── TLTDRP_IALNS_Taguchi无重启_新林案例_单因素双因素交互敏感性_300km_2行3列完整版.py
│   │    ├── Taguchi_最终最优参数.json
│   │    └── 运行说明.txt
│   ├──稳健性分析/
│   │    ├── 稳健性实验A_B_新林68节点
│   │    ├── IALNS_原版独立算法
│   │    ├── Taguchi_最终最优参数
│   │    ├── 新林案例_68个节点
│   │    ├── 运行说明
└── 代码运行指南.docx
The source package contains 7 Python scripts, 17 Excel workbooks (15 simulated instances and the Xinlin 68-node case), four Taguchi parameter JSON files, and two text run guides. The ZIP is provided as a convenience download; the browsable source files are the preferred reference.
4. Data and benchmark instances
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

5. Citation and acknowledgments
If you use this software or benchmark data, please cite the accompanying study when its final bibliographic information is available:
Watchtower–Drone–Truck Cooperative Network Optimization for Forest Fire Patrol. Manuscript submitted to the Journal of Systems Engineering. Full authorship, year, volume, pages, and DOI: to be added upon publication.

Source repository: https://github.com/Lcy0507/TLTDRP-Reproducibility

No Zenodo DOI or archived release is asserted here until one has actually been created.
6. License and data rights
The GitHub repository currently contains an MIT LICENSE file. That license should be interpreted for original code only to the extent that the repository owner has the necessary rights. The licensing and redistribution permissions for the Xinlin case data, any third-party materials, and external dependencies should be checked separately; Gurobi remains subject to its own proprietary license.
7. Contact and issue reporting
For reproducibility questions, bug reports, and proposed corrections, please open an Issue in this repository. Include the script name, command, environment details, input instance, random seed, and relevant error message.
