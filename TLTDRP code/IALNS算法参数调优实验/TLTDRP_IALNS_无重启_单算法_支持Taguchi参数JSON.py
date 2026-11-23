
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
import hashlib
from collections import Counter, defaultdict, OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # 批量保存；--show-best 时切换交互后端
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd

# =========================
# 全局显示设置
# =========================
plt.rcParams["font.sans-serif"] = ["SimSun", "宋体", "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 12
# EPS/PS 尽量将 TrueType 作为 Type42 嵌入；字体不兼容时由 PDF->EPS 兜底。
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "path"

# =========================
# 用户指定参数
# =========================
DEFAULT_BASE_DIR = r"E:\森林火灾巡检研究\Code\instances-2"
INSTANCE_NAMES = [
    "小规模算例1_4个节点",
    "小规模算例2_6个节点",
    "小规模算例3_9个节点",
    "小规模算例4_12个节点",
    "小规模算例5_15个节点",
    "中规模算例1_20个节点",
    "中规模算例2_25个节点",
    "中规模算例3_30个节点",
    "中规模算例4_35个节点",
    "中规模算例5_40个节点",
    "大规模算例1_45个节点",
    "大规模算例2_50个节点",
    "大规模算例3_55个节点",
    "大规模算例4_60个节点",
    "大规模算例5_65个节点",
    "新林案例_68个节点",
]

PATROLS_PER_YEAR = 360
TOWER_FIXED_COST = 150000.0
TRUCK_FIXED_COST = 42000.0
DRONE_FIXED_COST = 6000.0
TOWER_NODE_COST_PER_PATROL = 1.0
TRUCK_COST_PER_KM = 0.5
DRONE_COST_PER_KM = 0.22
TRUCK_DETOUR_FACTOR = 1.3
TOWER_RADIUS = 10.0
TOWER_DRONE_CAPACITY = 2
TRUCK_MAX_RANGE = 300.0
DRONE_MAX_RANGE = 50.0
MAX_RUNS = 10
MAX_ITERATIONS = 500
DEFAULT_INIT_TEMP = 2500.0
DEFAULT_COOLING = 0.982
DEFAULT_MIN_TEMP = 0.05

# 当前单算法基准参数（待田口重新调优）
WEIGHT_REACTION = 0.2
WEIGHT_UPDATE_INTERVAL = 40
LOCAL_SEARCH_PERIOD = 3
LOCAL_SEARCH_COST_RATIO = 1.02
RANDOM_DESTROY_RATIO = 0.15
WORST_DESTROY_RATIO = 0.18

# 当前单算法基准奖励参数（待田口重新调优）
REWARD_GLOBAL_BEST = 25
REWARD_IMPROVE_CURRENT = 12
REWARD_ACCEPTED_WORSE = 4
REWARD_REJECTED = 1.0
LOCAL_REWARD_IMPROVE = 20.0
LOCAL_REWARD_FAIL = 3.0
EMPTY_TOWER_REWARD_IMPROVE = 10.0
EMPTY_TOWER_REWARD_FAIL = 2.0
PENALTY_UNCOVERED = 5e7
PENALTY_CONSTRAINT = 5e6
EPS = 1e-9


@dataclass(frozen=True)
class IALNSParameters:
    max_iterations: int = 700
    initial_temperature: float = 2500.0
    cooling_rate: float = 0.985
    minimum_temperature: float = 0.1
    reaction_factor: float = 0.3
    weight_update_interval: int = 40
    local_search_period: int = 3
    local_search_cost_ratio: float = 1.01
    random_destroy_ratio: float = 0.15
    worst_destroy_ratio: float = 0.18
    reward_global_best: float = 60.0
    reward_improve_current: float = 25.0
    reward_accepted_worse: float = 8.0
    reward_rejected: float = 1.0
    local_reward_improve: float = 20.0
    local_reward_fail: float = 3.0
    empty_tower_reward_improve: float = 10.0
    empty_tower_reward_fail: float = 2.0

DEFAULT_IALNS_PARAMETERS = IALNSParameters()

# =========================
# 数据结构
# =========================
@dataclass
class ProblemData:
    instance_name: str
    excel_path: str
    n1: List[int]
    n2: List[int]
    n: List[int]
    towers: List[int]
    depot_start: int
    depot_end: int
    coords: Dict[int, Tuple[float, float]]
    node_types: Dict[int, str]
    coverage: Dict[int, List[int]]
    coverage_set: Dict[int, Set[int]]
    all_nodes_order: List[int]
    aerial_dist: Dict[Tuple[int, int], float]
    truck_dist: Dict[Tuple[int, int], float]
    truck_limit: int
    tower_limit: int
    drone_limit: int


@dataclass
class CostBreakdown:
    fixed_total: float = 0.0
    fixed_tower: float = 0.0
    fixed_truck: float = 0.0
    fixed_drone: float = 0.0
    patrol_total: float = 0.0
    patrol_tower: float = 0.0
    patrol_truck: float = 0.0
    patrol_drone: float = 0.0
    annual_operating_total: float = 0.0
    annual_tower: float = 0.0
    annual_truck: float = 0.0
    annual_drone: float = 0.0
    annual_total: float = 0.0
    truck_distance_per_patrol: float = 0.0
    truck_distance_annual: float = 0.0
    drone_distance_per_patrol: float = 0.0
    drone_distance_annual: float = 0.0


@dataclass
class Solution:
    built_towers: Set[int] = field(default_factory=set)
    tower_visits: Dict[int, List[int]] = field(default_factory=dict)
    truck_routes: Dict[int, List[int]] = field(default_factory=dict)
    drone_routes: Dict[int, List[int]] = field(default_factory=dict)
    node_assignment: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    cost: float = float("inf")
    feasible: bool = False
    penalty: float = 0.0
    status: str = "UNINITIALIZED"
    initial_summary: Dict[str, Any] = field(default_factory=dict)
    debug_notes: List[str] = field(default_factory=list)

    def copy(self) -> "Solution":
        return Solution(
            built_towers=set(self.built_towers),
            tower_visits={k: list(v) for k, v in self.tower_visits.items()},
            truck_routes={k: list(v) for k, v in self.truck_routes.items()},
            drone_routes={k: list(v) for k, v in self.drone_routes.items()},
            node_assignment=dict(self.node_assignment),
            cost=self.cost,
            feasible=self.feasible,
            penalty=self.penalty,
            status=self.status,
            initial_summary=copy.deepcopy(self.initial_summary),
            debug_notes=list(self.debug_notes),
        )


@dataclass
class OperatorStat:
    name: str
    weight: float = 1.0
    score: float = 0.0
    used: int = 0
    improved_global: int = 0
    improved_current: int = 0
    accepted_worse: int = 0
    rejected: int = 0


# =========================
# 工具函数
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = name
    for ch in bad:
        out = out.replace(ch, "_")
    return out.strip().strip(".")


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def euc(coords: Dict[int, Tuple[float, float]], i: int, j: int) -> float:
    x1, y1 = coords[i]
    x2, y2 = coords[j]
    return math.hypot(x1 - x2, y1 - y2)


def route_distance(route: Sequence[int], dist_matrix: Dict[Tuple[int, int], float]) -> float:
    if len(route) <= 1:
        return 0.0
    return sum(dist_matrix[(route[i], route[i + 1])] for i in range(len(route) - 1))


def unique_preserve(seq: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def mean_safe(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def roulette_select(stats: List[OperatorStat]) -> int:
    weights = [max(s.weight, 1e-6) for s in stats]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for idx, w in enumerate(weights):
        acc += w
        if r <= acc:
            return idx
    return len(stats) - 1


def update_operator_weights(stats: List[OperatorStat], reaction: float = WEIGHT_REACTION) -> None:
    for s in stats:
        avg_perf = s.score / s.used if s.used > 0 else 0.0
        s.weight = max(0.1, (1.0 - reaction) * s.weight + reaction * max(0.1, avg_perf))


def nearest_neighbor_order(
    start: int,
    nodes: List[int],
    end: Optional[int],
    dist_matrix: Dict[Tuple[int, int], float],
) -> List[int]:
    remaining = list(nodes)
    cur = start
    order: List[int] = []
    while remaining:
        nxt = min(remaining, key=lambda x: dist_matrix[(cur, x)])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    if end is None:
        return [start] + order
    return [start] + order + [end]


def two_opt_once(route: List[int], dist_matrix: Dict[Tuple[int, int], float]) -> Tuple[List[int], bool]:
    if len(route) <= 4:
        return route, False
    best = list(route)
    best_dist = route_distance(best, dist_matrix)
    improved = False
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            if j - i == 1:
                continue
            cand = route[:i] + route[i:j][::-1] + route[j:]
            cand_dist = route_distance(cand, dist_matrix)
            if cand_dist + 1e-9 < best_dist:
                best = cand
                best_dist = cand_dist
                improved = True
    return best, improved


# =========================
# 数据读取与预处理
# =========================
def find_excel_for_instance(base_dir: str, instance_name: str) -> str:
    direct_xlsx = os.path.join(base_dir, f"{instance_name}.xlsx")
    if os.path.isfile(direct_xlsx):
        return direct_xlsx

    instance_dir = os.path.join(base_dir, instance_name)
    if os.path.isdir(instance_dir):
        exact = os.path.join(instance_dir, f"{instance_name}.xlsx")
        if os.path.isfile(exact):
            return exact
        xlsx_files = [
            os.path.join(instance_dir, f)
            for f in os.listdir(instance_dir)
            if f.lower().endswith(".xlsx") and not f.startswith("~$")
        ]
        if len(xlsx_files) == 1:
            return xlsx_files[0]
        if len(xlsx_files) > 1:
            preferred = [p for p in xlsx_files if instance_name in os.path.basename(p)]
            return preferred[0] if preferred else sorted(xlsx_files)[0]

    raise FileNotFoundError(
        f"未找到算例文件：{instance_name}\n"
        f"已尝试：\n  {direct_xlsx}\n  {os.path.join(base_dir, instance_name, instance_name + '.xlsx')}"
    )


def read_instance(base_dir: str, instance_name: str) -> ProblemData:
    excel_path = find_excel_for_instance(base_dir, instance_name)
    df = pd.read_excel(excel_path)
    need = {"node", "x", "y", "Classification"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{instance_name} 缺少列：{sorted(miss)}")

    df = df.copy()
    df["node"] = df["node"].astype(int)
    coords = {int(r.node): (float(r.x), float(r.y)) for _, r in df.iterrows()}
    node_types = {int(r.node): str(r.Classification).strip() for _, r in df.iterrows()}

    n1 = df.loc[df["Classification"] == "沿线道路监测节点", "node"].astype(int).tolist()
    n2 = df.loc[df["Classification"] == "非沿线道路监测节点", "node"].astype(int).tolist()
    towers = df.loc[df["Classification"] == "候选瞭望塔", "node"].astype(int).tolist()
    depots = df.loc[df["Classification"] == "消防站", "node"].astype(int).tolist()

    if len(depots) == 0:
        raise ValueError(f"{instance_name} 未找到消防站节点")
    if len(depots) == 1:
        depot_start = depots[0]
        # 单消防站情形：复制同一 ID 作为起终点逻辑会引起字典键冲突，这里采用同一个节点 ID 做首尾
        depot_end = depots[0]
    else:
        depot_start, depot_end = depots[0], depots[1]

    n = n1 + n2
    coverage = {l: [i for i in n if euc(coords, l, i) <= TOWER_RADIUS + 1e-9] for l in towers}
    coverage_set = {l: set(v) for l, v in coverage.items()}

    all_nodes_order = unique_preserve([depot_start, depot_end] + n1 + n2 + towers)
    aerial_dist = {(i, j): euc(coords, i, j) for i in all_nodes_order for j in all_nodes_order}
    truck_dist = {(i, j): TRUCK_DETOUR_FACTOR * euc(coords, i, j) for i in all_nodes_order for j in all_nodes_order}

    tower_limit = math.ceil(len(towers) / 2) if towers else 0
    drone_limit = tower_limit * 2
    truck_limit = math.ceil(calc_potential_trucks(n1, depot_start, depot_end, coords, truck_dist) / 2) if n1 else 0
    if n1 and truck_limit <= 0:
        truck_limit = 1

    return ProblemData(
        instance_name=instance_name,
        excel_path=excel_path,
        n1=n1,
        n2=n2,
        n=n,
        towers=towers,
        depot_start=depot_start,
        depot_end=depot_end,
        coords=coords,
        node_types=node_types,
        coverage=coverage,
        coverage_set=coverage_set,
        all_nodes_order=all_nodes_order,
        aerial_dist=aerial_dist,
        truck_dist=truck_dist,
        truck_limit=truck_limit,
        tower_limit=tower_limit,
        drone_limit=drone_limit,
    )


def calc_potential_trucks(
    n1: List[int],
    depot_start: int,
    depot_end: int,
    coords: Dict[int, Tuple[float, float]],
    truck_dist_matrix: Dict[Tuple[int, int], float],
) -> int:
    if not n1:
        return 0
    remaining = set(n1)
    count = 0
    while remaining:
        count += 1
        route = [depot_start]
        cur = depot_start
        dist_used = 0.0
        while True:
            feasible_next = []
            for node in remaining:
                add = truck_dist_matrix[(cur, node)]
                back = truck_dist_matrix[(node, depot_end)]
                if dist_used + add + back <= TRUCK_MAX_RANGE + 1e-9:
                    feasible_next.append(node)
            if not feasible_next:
                break
            nxt = min(feasible_next, key=lambda x: truck_dist_matrix[(cur, x)])
            dist_used += truck_dist_matrix[(cur, nxt)]
            route.append(nxt)
            cur = nxt
            remaining.remove(nxt)
        route.append(depot_end)
        if len(route) <= 2:
            break
    return max(1, count)


# =========================
# 解的辅助函数
# =========================
def built_cover_nodes(sol: Solution, data: ProblemData) -> Set[int]:
    cover: Set[int] = set()
    for l in sol.built_towers:
        cover.update(data.coverage_set.get(l, set()))
    return cover


def drone_count_on_tower(sol: Solution, tower: int) -> int:
    return sum(1 for r in sol.drone_routes.values() if len(r) >= 2 and r[0] == tower and r[-1] == tower)


def next_resource_id(d: Dict[int, Any]) -> int:
    return (max(d.keys()) + 1) if d else 1


def assigned_nodes(sol: Solution) -> Set[int]:
    return set(sol.node_assignment.keys())


def unassigned_nodes(sol: Solution, data: ProblemData) -> List[int]:
    return [n for n in data.n if n not in sol.node_assignment]


# =========================
# 成本与可行性
# =========================
def cost_breakdown(sol: Solution, data: ProblemData) -> CostBreakdown:
    cb = CostBreakdown()
    cb.fixed_tower = len(sol.built_towers) * TOWER_FIXED_COST
    cb.fixed_truck = len(sol.truck_routes) * TRUCK_FIXED_COST
    cb.fixed_drone = len(sol.drone_routes) * DRONE_FIXED_COST
    cb.fixed_total = cb.fixed_tower + cb.fixed_truck + cb.fixed_drone

    cb.patrol_tower = sum(len(sol.tower_visits.get(l, [])) for l in sol.built_towers) * TOWER_NODE_COST_PER_PATROL
    cb.truck_distance_per_patrol = sum(route_distance(r, data.truck_dist) for r in sol.truck_routes.values())
    cb.drone_distance_per_patrol = sum(route_distance(r, data.aerial_dist) for r in sol.drone_routes.values())
    cb.patrol_truck = cb.truck_distance_per_patrol * TRUCK_COST_PER_KM
    cb.patrol_drone = cb.drone_distance_per_patrol * DRONE_COST_PER_KM
    cb.patrol_total = cb.patrol_tower + cb.patrol_truck + cb.patrol_drone

    cb.annual_tower = cb.patrol_tower * PATROLS_PER_YEAR
    cb.annual_truck = cb.patrol_truck * PATROLS_PER_YEAR
    cb.annual_drone = cb.patrol_drone * PATROLS_PER_YEAR
    cb.annual_operating_total = cb.annual_tower + cb.annual_truck + cb.annual_drone
    cb.annual_total = cb.fixed_total + cb.annual_operating_total
    cb.truck_distance_annual = cb.truck_distance_per_patrol * PATROLS_PER_YEAR
    cb.drone_distance_annual = cb.drone_distance_per_patrol * PATROLS_PER_YEAR
    return cb


def evaluate_solution(sol: Solution, data: ProblemData) -> float:
    violations = constraint_violations(sol, data)
    cb = cost_breakdown(sol, data)
    pen = 0.0
    missing = len([v for v in violations if v.startswith("存在未覆盖节点")])
    if missing > 0:
        pen += PENALTY_UNCOVERED * missing
    pen += PENALTY_CONSTRAINT * max(0, len(violations) - missing)
    sol.penalty = pen
    sol.feasible = len(violations) == 0
    sol.status = "FEASIBLE" if sol.feasible else "INFEASIBLE"
    sol.cost = cb.annual_total + pen
    return sol.cost


def constraint_violations(sol: Solution, data: ProblemData) -> List[str]:
    violations: List[str] = []
    assigned = assigned_nodes(sol)

    # 全覆盖
    missing_nodes = sorted(set(data.n) - assigned)
    for n in missing_nodes:
        violations.append(f"存在未覆盖节点: {n}")

    # 节点唯一分配
    all_assigned_list: List[int] = []
    for l, nodes in sol.tower_visits.items():
        all_assigned_list.extend(nodes)
        if l not in sol.built_towers:
            violations.append(f"瞭望塔 {l} 未建设但存在巡检节点")
        if len(nodes) != len(set(nodes)):
            violations.append(f"瞭望塔 {l} 巡检节点重复")
        for n in nodes:
            if n not in data.coverage_set.get(l, set()):
                violations.append(f"节点 {n} 不在塔 {l} 覆盖内却被塔巡检")
            if sol.node_assignment.get(n) != ("W", l):
                violations.append(f"节点 {n} 的 tower assignment 与 node_assignment 不一致")

    for k, route in sol.truck_routes.items():
        if len(route) < 3:
            violations.append(f"卡车 {k} 路径长度不足: {route}")
            continue
        if route[0] != data.depot_start or route[-1] != data.depot_end:
            violations.append(f"卡车 {k} 未从消防站出发返回: {route}")
        for n in route[1:-1]:
            all_assigned_list.append(n)
            if n not in data.n1:
                violations.append(f"卡车 {k} 巡检了非沿线节点 {n}")
            if sol.node_assignment.get(n) != ("T", k):
                violations.append(f"节点 {n} 的 truck assignment 与 node_assignment 不一致")
        if route_distance(route, data.truck_dist) > TRUCK_MAX_RANGE + 1e-9:
            violations.append(f"卡车 {k} 里程超限")

    for u, route in sol.drone_routes.items():
        if len(route) < 3:
            violations.append(f"无人机 {u} 路径长度不足: {route}")
            continue
        tower = route[0]
        if tower != route[-1]:
            violations.append(f"无人机 {u} 未返回同一塔: {route}")
        if tower not in sol.built_towers:
            violations.append(f"无人机 {u} 起降塔 {tower} 未建设")
        if route_distance(route, data.aerial_dist) > DRONE_MAX_RANGE + 1e-9:
            violations.append(f"无人机 {u} 里程超限")
        for n in route[1:-1]:
            all_assigned_list.append(n)
            if n not in data.n:
                violations.append(f"无人机 {u} 巡检了非法节点 {n}")
            if sol.node_assignment.get(n) != ("D", u):
                violations.append(f"节点 {n} 的 drone assignment 与 node_assignment 不一致")

    dup_counter = Counter(all_assigned_list)
    duplicates = [n for n, c in dup_counter.items() if c > 1]
    for n in duplicates:
        violations.append(f"节点 {n} 被重复巡检")

    # N2 不能被卡车访问
    for n in data.n2:
        if sol.node_assignment.get(n, (None, None))[0] == "T":
            violations.append(f"N2 节点 {n} 被卡车巡检")

    # 卡车禁止访问被已建塔覆盖的节点
    built_cover = built_cover_nodes(sol, data)
    for n in data.n1:
        if sol.node_assignment.get(n, (None, None))[0] == "T" and n in built_cover:
            violations.append(f"节点 {n} 被已建塔覆盖但仍由卡车巡检")

    # 资源数量上限
    if len(sol.built_towers) > data.tower_limit:
        violations.append("已建瞭望塔数量超过上限")
    if len(sol.truck_routes) > data.truck_limit:
        violations.append("已用卡车数量超过上限")
    if len(sol.drone_routes) > data.drone_limit:
        violations.append("已用无人机数量超过上限")
    for l in sol.built_towers:
        if drone_count_on_tower(sol, l) > TOWER_DRONE_CAPACITY:
            violations.append(f"塔 {l} 的无人机数量超过上限")

    return violations


# =========================
# 基础修改操作
# =========================
def remove_node(sol: Solution, node: int) -> None:
    if node not in sol.node_assignment:
        return
    mode, rid = sol.node_assignment.pop(node)
    if mode == "W":
        if rid in sol.tower_visits and node in sol.tower_visits[rid]:
            sol.tower_visits[rid].remove(node)
    elif mode == "T":
        if rid in sol.truck_routes and node in sol.truck_routes[rid]:
            sol.truck_routes[rid].remove(node)
            if len(sol.truck_routes[rid]) <= 2:
                del sol.truck_routes[rid]
    elif mode == "D":
        if rid in sol.drone_routes and node in sol.drone_routes[rid]:
            sol.drone_routes[rid].remove(node)
            if len(sol.drone_routes[rid]) <= 2:
                del sol.drone_routes[rid]


def cleanup_solution(sol: Solution, data: ProblemData) -> None:
    # 清除空 tower_visits
    for l in list(sol.tower_visits.keys()):
        sol.tower_visits[l] = [n for n in sol.tower_visits[l] if sol.node_assignment.get(n) == ("W", l)]
        if len(sol.tower_visits[l]) == 0 and l not in sol.built_towers:
            del sol.tower_visits[l]

    # 删除无节点且无无人机的空塔
    for l in list(sol.built_towers):
        has_visit = len(sol.tower_visits.get(l, [])) > 0
        has_drone = any(r[0] == l and r[-1] == l for r in sol.drone_routes.values())
        if not has_visit and not has_drone:
            sol.built_towers.remove(l)
            sol.tower_visits.pop(l, None)

    # 同步 node_assignment
    valid_assign: Dict[int, Tuple[str, int]] = {}
    for l, nodes in sol.tower_visits.items():
        for n in nodes:
            valid_assign[n] = ("W", l)
    for k, route in sol.truck_routes.items():
        for n in route[1:-1]:
            valid_assign[n] = ("T", k)
    for u, route in sol.drone_routes.items():
        for n in route[1:-1]:
            valid_assign[n] = ("D", u)
    sol.node_assignment = valid_assign


# =========================
# 可行插入与修复
# =========================
def candidate_insertions(sol: Solution, node: int, data: ProblemData) -> List[Tuple[str, Dict[str, Any], float]]:
    """为一个未分配节点生成所有可行插入动作。
    返回: (动作名, 参数字典, 估计增量成本)
    """
    cands: List[Tuple[str, Dict[str, Any], float]] = []
    built_cover = built_cover_nodes(sol, data)

    # 1) 现有塔直接巡检
    for l in sol.built_towers:
        if node in data.coverage_set.get(l, set()):
            delta = TOWER_NODE_COST_PER_PATROL * PATROLS_PER_YEAR
            cands.append(("tower_existing", {"tower": l}, delta))

    # 2) 新建塔直接巡检
    if len(sol.built_towers) < data.tower_limit:
        for l in data.towers:
            if l in sol.built_towers:
                continue
            if node in data.coverage_set.get(l, set()):
                delta = TOWER_FIXED_COST + TOWER_NODE_COST_PER_PATROL * PATROLS_PER_YEAR
                cands.append(("tower_new", {"tower": l}, delta))

    # 3) 插入现有无人机路线
    for u, route in sol.drone_routes.items():
        tower = route[0]
        for pos in range(1, len(route)):
            new_route = route[:pos] + [node] + route[pos:]
            d = route_distance(new_route, data.aerial_dist)
            if d <= DRONE_MAX_RANGE + 1e-9:
                delta = route_distance(new_route, data.aerial_dist) - route_distance(route, data.aerial_dist)
                cands.append(("drone_insert_existing", {"drone": u, "pos": pos}, delta * DRONE_COST_PER_KM * PATROLS_PER_YEAR))

    # 4) 从现有塔新增无人机单路线
    for l in sol.built_towers:
        if drone_count_on_tower(sol, l) >= TOWER_DRONE_CAPACITY:
            continue
        d = data.aerial_dist[(l, node)] + data.aerial_dist[(node, l)]
        if d <= DRONE_MAX_RANGE + 1e-9 and len(sol.drone_routes) < data.drone_limit:
            delta = DRONE_FIXED_COST + d * DRONE_COST_PER_KM * PATROLS_PER_YEAR
            cands.append(("drone_new_from_existing_tower", {"tower": l}, delta))

    # 5) 新建塔并从该塔发无人机巡检（即使该节点不在该塔覆盖中也允许）
    if len(sol.built_towers) < data.tower_limit and len(sol.drone_routes) < data.drone_limit:
        for l in data.towers:
            if l in sol.built_towers:
                continue
            d = data.aerial_dist[(l, node)] + data.aerial_dist[(node, l)]
            if d <= DRONE_MAX_RANGE + 1e-9:
                delta = TOWER_FIXED_COST + DRONE_FIXED_COST + d * DRONE_COST_PER_KM * PATROLS_PER_YEAR
                cands.append(("drone_new_with_new_tower", {"tower": l}, delta))

    # 6) 插入现有卡车路线（仅 N1 且不能被已建塔覆盖）
    if node in data.n1 and node not in built_cover:
        for k, route in sol.truck_routes.items():
            for pos in range(1, len(route)):
                new_route = route[:pos] + [node] + route[pos:]
                d = route_distance(new_route, data.truck_dist)
                if d <= TRUCK_MAX_RANGE + 1e-9:
                    delta = route_distance(new_route, data.truck_dist) - route_distance(route, data.truck_dist)
                    cands.append(("truck_insert_existing", {"truck": k, "pos": pos}, delta * TRUCK_COST_PER_KM * PATROLS_PER_YEAR))

    # 7) 新增卡车路线（仅 N1 且不能被已建塔覆盖）
    if node in data.n1 and node not in built_cover and len(sol.truck_routes) < data.truck_limit:
        route = [data.depot_start, node, data.depot_end]
        d = route_distance(route, data.truck_dist)
        if d <= TRUCK_MAX_RANGE + 1e-9:
            delta = TRUCK_FIXED_COST + d * TRUCK_COST_PER_KM * PATROLS_PER_YEAR
            cands.append(("truck_new", {}, delta))

    return sorted(cands, key=lambda x: x[2])


def apply_insertion(sol: Solution, node: int, action: str, payload: Dict[str, Any], data: ProblemData) -> bool:
    if node in sol.node_assignment:
        return True

    if action == "tower_existing":
        l = payload["tower"]
        sol.built_towers.add(l)
        sol.tower_visits.setdefault(l, []).append(node)
        sol.node_assignment[node] = ("W", l)
        return True

    if action == "tower_new":
        l = payload["tower"]
        sol.built_towers.add(l)
        sol.tower_visits.setdefault(l, []).append(node)
        sol.node_assignment[node] = ("W", l)
        return True

    if action == "drone_insert_existing":
        u = payload["drone"]
        pos = payload["pos"]
        if u not in sol.drone_routes:
            return False
        sol.drone_routes[u] = sol.drone_routes[u][:pos] + [node] + sol.drone_routes[u][pos:]
        sol.node_assignment[node] = ("D", u)
        return True

    if action == "drone_new_from_existing_tower":
        l = payload["tower"]
        if drone_count_on_tower(sol, l) >= TOWER_DRONE_CAPACITY or len(sol.drone_routes) >= data.drone_limit:
            return False
        u = next_resource_id(sol.drone_routes)
        sol.drone_routes[u] = [l, node, l]
        sol.node_assignment[node] = ("D", u)
        return True

    if action == "drone_new_with_new_tower":
        l = payload["tower"]
        if len(sol.built_towers) >= data.tower_limit or len(sol.drone_routes) >= data.drone_limit:
            return False
        sol.built_towers.add(l)
        sol.tower_visits.setdefault(l, [])
        u = next_resource_id(sol.drone_routes)
        sol.drone_routes[u] = [l, node, l]
        sol.node_assignment[node] = ("D", u)
        return True

    if action == "truck_insert_existing":
        k = payload["truck"]
        pos = payload["pos"]
        if k not in sol.truck_routes:
            return False
        sol.truck_routes[k] = sol.truck_routes[k][:pos] + [node] + sol.truck_routes[k][pos:]
        sol.node_assignment[node] = ("T", k)
        return True

    if action == "truck_new":
        if len(sol.truck_routes) >= data.truck_limit:
            return False
        k = next_resource_id(sol.truck_routes)
        sol.truck_routes[k] = [data.depot_start, node, data.depot_end]
        sol.node_assignment[node] = ("T", k)
        return True

    return False


def repair_greedy(sol: Solution, removed_nodes: List[int], data: ProblemData) -> Tuple[Solution, List[str]]:
    sol = sol.copy()
    log: List[str] = []
    targets = unique_preserve(removed_nodes + unassigned_nodes(sol, data))
    for n in targets:
        if n in sol.node_assignment:
            continue
        cands = candidate_insertions(sol, n, data)
        placed = False
        for action, payload, _delta in cands:
            test = sol.copy()
            if apply_insertion(test, n, action, payload, data):
                cleanup_solution(test, data)
                if len(constraint_violations(test, data)) == 0 or all(not v.endswith(str(n)) for v in constraint_violations(test, data)):
                    sol = test
                    log.append(f"节点 {n}: {action}")
                    placed = True
                    break
        if not placed:
            # 宽松回退：直接采用最便宜动作，即便总体仍可能暂时不可行，后续继续修复
            if cands:
                action, payload, _ = cands[0]
                if apply_insertion(sol, n, action, payload, data):
                    log.append(f"节点 {n}: {action}(宽松回退)")
                    placed = True
        if not placed:
            log.append(f"节点 {n}: 无可行插入")
    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, log


def repair_regret(sol: Solution, removed_nodes: List[int], data: ProblemData) -> Tuple[Solution, List[str]]:
    sol = sol.copy()
    log: List[str] = []
    pending = unique_preserve(removed_nodes + unassigned_nodes(sol, data))
    while pending:
        regrets: List[Tuple[float, int, List[Tuple[str, Dict[str, Any], float]]]] = []
        for n in pending:
            cands = candidate_insertions(sol, n, data)
            if not cands:
                regrets.append((-1e18, n, []))
                continue
            best = cands[0][2]
            second = cands[1][2] if len(cands) >= 2 else best + 1e6
            regret = second - best
            regrets.append((regret, n, cands))
        regrets.sort(reverse=True, key=lambda x: x[0])
        _, node, cands = regrets[0]
        if not cands:
            log.append(f"节点 {node}: 无候选动作")
            pending.remove(node)
            continue
        action, payload, _ = cands[0]
        apply_insertion(sol, node, action, payload, data)
        log.append(f"节点 {node}: {action}")
        pending.remove(node)
        cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, log


def repair_existing_first(sol: Solution, removed_nodes: List[int], data: ProblemData) -> Tuple[Solution, List[str]]:
    sol = sol.copy()
    log: List[str] = []
    targets = unique_preserve(removed_nodes + unassigned_nodes(sol, data))
    preferred_order = [
        "tower_existing",
        "drone_insert_existing",
        "truck_insert_existing",
        "drone_new_from_existing_tower",
        "truck_new",
        "tower_new",
        "drone_new_with_new_tower",
    ]
    rank = {name: i for i, name in enumerate(preferred_order)}
    for n in targets:
        if n in sol.node_assignment:
            continue
        cands = candidate_insertions(sol, n, data)
        cands.sort(key=lambda x: (rank.get(x[0], 999), x[2]))
        if cands:
            action, payload, _ = cands[0]
            apply_insertion(sol, n, action, payload, data)
            log.append(f"节点 {n}: {action}")
        else:
            log.append(f"节点 {n}: 无候选动作")
        cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, log


def repair_randomized(sol: Solution, removed_nodes: List[int], data: ProblemData) -> Tuple[Solution, List[str]]:
    sol = sol.copy()
    log: List[str] = []
    targets = unique_preserve(removed_nodes + unassigned_nodes(sol, data))
    random.shuffle(targets)
    for n in targets:
        if n in sol.node_assignment:
            continue
        cands = candidate_insertions(sol, n, data)
        if cands:
            choice_pool = cands[: min(3, len(cands))]
            action, payload, _ = random.choice(choice_pool)
            apply_insertion(sol, n, action, payload, data)
            log.append(f"节点 {n}: {action}")
        else:
            log.append(f"节点 {n}: 无候选动作")
        cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, log


# =========================
# 初始解
# =========================
def generate_initial_solution(data: ProblemData) -> Solution:
    sol = Solution()
    notes: List[str] = []

    # 阶段 1：优先为 N2 构建塔，减少不可达风险
    uncovered_n2 = set(data.n2)
    while uncovered_n2 and len(sol.built_towers) < data.tower_limit:
        best_tower = None
        best_cover = []
        best_score = -1.0
        for l in data.towers:
            if l in sol.built_towers:
                continue
            cover_nodes = list(uncovered_n2 & data.coverage_set.get(l, set()))
            score = len(cover_nodes)
            if score > best_score:
                best_score = score
                best_tower = l
                best_cover = cover_nodes
        if best_tower is None or best_score <= 0:
            break
        sol.built_towers.add(best_tower)
        sol.tower_visits.setdefault(best_tower, [])
        notes.append(f"阶段1: 建塔 {best_tower} 覆盖 N2 {best_cover}")
        for n in sorted(best_cover):
            sol.tower_visits[best_tower].append(n)
            sol.node_assignment[n] = ("W", best_tower)
            uncovered_n2.discard(n)

    # 阶段 2：对剩余 N2，优先用现有塔发无人机；若不行则补建塔或新塔+无人机
    if uncovered_n2:
        remaining = sorted(uncovered_n2)
        for n in remaining:
            cands = candidate_insertions(sol, n, data)
            cands = [c for c in cands if c[0] in {"tower_existing", "tower_new", "drone_new_from_existing_tower", "drone_new_with_new_tower"}]
            if cands:
                action, payload, _ = cands[0]
                apply_insertion(sol, n, action, payload, data)
                notes.append(f"阶段2: N2 节点 {n} -> {action}")

    # 阶段 3：对 N1，先尝试卡车（前提是不被已建塔覆盖）
    remain_n1 = [n for n in data.n1 if n not in sol.node_assignment]
    for n in remain_n1:
        cands = candidate_insertions(sol, n, data)
        cands_truck = [c for c in cands if c[0] in {"truck_insert_existing", "truck_new"}]
        if cands_truck:
            action, payload, _ = cands_truck[0]
            apply_insertion(sol, n, action, payload, data)
            notes.append(f"阶段3: N1 节点 {n} -> {action}")

    # 阶段 4：剩余全部节点做贪心修复
    sol, repair_log = repair_greedy(sol, unassigned_nodes(sol, data), data)
    notes.extend([f"阶段4: {x}" for x in repair_log])

    # 阶段 5：路线简单优化
    for k, route in list(sol.truck_routes.items()):
        better, improved = two_opt_once(route, data.truck_dist)
        if improved:
            sol.truck_routes[k] = better
            notes.append(f"阶段5: 卡车 {k} 2-opt")
    for u, route in list(sol.drone_routes.items()):
        better, improved = two_opt_once(route, data.aerial_dist)
        if improved:
            sol.drone_routes[u] = better
            notes.append(f"阶段5: 无人机 {u} 2-opt")

    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    sol.initial_summary = {
        "built_towers": sorted(sol.built_towers),
        "tower_visits": {k: list(v) for k, v in sol.tower_visits.items()},
        "truck_routes": {k: list(v) for k, v in sol.truck_routes.items()},
        "drone_routes": {k: list(v) for k, v in sol.drone_routes.items()},
        "notes": notes,
        "coverage_rate": len(sol.node_assignment) / len(data.n) * 100 if data.n else 100.0,
        "cost": sol.cost,
        "feasible": sol.feasible,
    }
    sol.debug_notes.extend(notes)
    return sol


# =========================
# Destroy operators
# =========================
def destroy_random_nodes(sol: Solution, data: ProblemData, ratio: float = RANDOM_DESTROY_RATIO) -> Tuple[Solution, List[int], str]:
    sol = sol.copy()
    assigned = list(sol.node_assignment.keys())
    if not assigned:
        return sol, [], "destroy_random_nodes"
    q = max(1, math.ceil(ratio * len(assigned)))
    removed = random.sample(assigned, min(q, len(assigned)))
    for n in removed:
        remove_node(sol, n)
    cleanup_solution(sol, data)
    return sol, removed, "destroy_random_nodes"


def destroy_worst_cost_nodes(sol: Solution, data: ProblemData, ratio: float = WORST_DESTROY_RATIO) -> Tuple[Solution, List[int], str]:
    sol = sol.copy()
    if not sol.node_assignment:
        return sol, [], "destroy_worst_cost_nodes"
    contributions: List[Tuple[float, int]] = []
    for n, (mode, rid) in sol.node_assignment.items():
        if mode == "W":
            contrib = TOWER_NODE_COST_PER_PATROL * PATROLS_PER_YEAR
        elif mode == "T":
            route = sol.truck_routes.get(rid, [data.depot_start, data.depot_end])
            contrib = route_distance(route, data.truck_dist) * TRUCK_COST_PER_KM * PATROLS_PER_YEAR / max(1, len(route) - 2)
        else:
            route = sol.drone_routes.get(rid, [0, 0])
            contrib = (DRONE_FIXED_COST + route_distance(route, data.aerial_dist) * DRONE_COST_PER_KM * PATROLS_PER_YEAR) / max(1, len(route) - 2)
        contributions.append((contrib, n))
    contributions.sort(reverse=True)
    q = max(1, math.ceil(ratio * len(contributions)))
    removed = [n for _, n in contributions[:q]]
    for n in removed:
        remove_node(sol, n)
    cleanup_solution(sol, data)
    return sol, removed, "destroy_worst_cost_nodes"


def destroy_one_tower(sol: Solution, data: ProblemData) -> Tuple[Solution, List[int], str]:
    sol = sol.copy()
    if not sol.built_towers:
        return sol, [], "destroy_one_tower"
    tower = random.choice(sorted(sol.built_towers))
    removed: List[int] = []
    for n in list(sol.tower_visits.get(tower, [])):
        removed.append(n)
        sol.node_assignment.pop(n, None)
    sol.tower_visits.pop(tower, None)
    # 删除该塔上无人机及其节点
    for u, route in list(sol.drone_routes.items()):
        if route and route[0] == tower:
            for n in route[1:-1]:
                removed.append(n)
                sol.node_assignment.pop(n, None)
            del sol.drone_routes[u]
    sol.built_towers.discard(tower)
    cleanup_solution(sol, data)
    return sol, unique_preserve(removed), "destroy_one_tower"


def destroy_one_truck(sol: Solution, data: ProblemData) -> Tuple[Solution, List[int], str]:
    sol = sol.copy()
    if not sol.truck_routes:
        return sol, [], "destroy_one_truck"
    k = random.choice(sorted(sol.truck_routes.keys()))
    removed = list(sol.truck_routes[k][1:-1])
    for n in removed:
        sol.node_assignment.pop(n, None)
    del sol.truck_routes[k]
    cleanup_solution(sol, data)
    return sol, removed, "destroy_one_truck"


def destroy_one_drone(sol: Solution, data: ProblemData) -> Tuple[Solution, List[int], str]:
    sol = sol.copy()
    if not sol.drone_routes:
        return sol, [], "destroy_one_drone"
    u = random.choice(sorted(sol.drone_routes.keys()))
    removed = list(sol.drone_routes[u][1:-1])
    for n in removed:
        sol.node_assignment.pop(n, None)
    del sol.drone_routes[u]
    cleanup_solution(sol, data)
    return sol, removed, "destroy_one_drone"


def destroy_overlap_tower(sol: Solution, data: ProblemData) -> Tuple[Solution, List[int], str]:
    sol = sol.copy()
    if len(sol.built_towers) < 2:
        return sol, [], "destroy_overlap_tower"
    best_pair = None
    best_overlap = -1
    towers = sorted(sol.built_towers)
    for i in range(len(towers)):
        for j in range(i + 1, len(towers)):
            overlap = len(data.coverage_set[towers[i]] & data.coverage_set[towers[j]])
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (towers[i], towers[j])
    if best_pair is None:
        return sol, [], "destroy_overlap_tower"
    tower = random.choice(best_pair)
    removed: List[int] = []
    for n in list(sol.tower_visits.get(tower, [])):
        removed.append(n)
        sol.node_assignment.pop(n, None)
    sol.tower_visits.pop(tower, None)
    for u, route in list(sol.drone_routes.items()):
        if route and route[0] == tower:
            for n in route[1:-1]:
                removed.append(n)
                sol.node_assignment.pop(n, None)
            del sol.drone_routes[u]
    sol.built_towers.discard(tower)
    cleanup_solution(sol, data)
    return sol, unique_preserve(removed), "destroy_overlap_tower"


# =========================
# Repair operator wrappers
# =========================
def repair_op_greedy(sol: Solution, removed: List[int], data: ProblemData) -> Tuple[Solution, List[str], str]:
    out, logs = repair_greedy(sol, removed, data)
    return out, logs, "repair_greedy"


def repair_op_regret(sol: Solution, removed: List[int], data: ProblemData) -> Tuple[Solution, List[str], str]:
    out, logs = repair_regret(sol, removed, data)
    return out, logs, "repair_regret"


def repair_op_existing(sol: Solution, removed: List[int], data: ProblemData) -> Tuple[Solution, List[str], str]:
    out, logs = repair_existing_first(sol, removed, data)
    return out, logs, "repair_existing_first"


def repair_op_randomized(sol: Solution, removed: List[int], data: ProblemData) -> Tuple[Solution, List[str], str]:
    out, logs = repair_randomized(sol, removed, data)
    return out, logs, "repair_randomized"


# =========================
# Local search operators
# =========================
def ls_two_opt(sol: Solution, data: ProblemData) -> Tuple[Solution, List[str], str]:
    sol = sol.copy()
    logs: List[str] = []
    for k, route in list(sol.truck_routes.items()):
        better, improved = two_opt_once(route, data.truck_dist)
        if improved:
            sol.truck_routes[k] = better
            logs.append(f"卡车{k}:2-opt")
    for u, route in list(sol.drone_routes.items()):
        better, improved = two_opt_once(route, data.aerial_dist)
        if improved:
            sol.drone_routes[u] = better
            logs.append(f"无人机{u}:2-opt")
    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, logs, "ls_two_opt"


def ls_shift_truck_to_tower_or_drone(sol: Solution, data: ProblemData) -> Tuple[Solution, List[str], str]:
    sol = sol.copy()
    logs: List[str] = []
    changed = True
    while changed:
        changed = False
        for n in list(data.n1):
            if sol.node_assignment.get(n, (None, None))[0] != "T":
                continue
            current = sol.copy()
            old_cost = evaluate_solution(current, data)
            current_mode, current_id = current.node_assignment[n]
            remove_node(current, n)
            cleanup_solution(current, data)
            cands = candidate_insertions(current, n, data)
            cands = [c for c in cands if c[0].startswith("tower") or c[0].startswith("drone")]
            if not cands:
                continue
            action, payload, _ = cands[0]
            apply_insertion(current, n, action, payload, data)
            cleanup_solution(current, data)
            new_cost = evaluate_solution(current, data)
            if current.feasible and new_cost + 1e-9 < old_cost:
                sol = current
                logs.append(f"节点{n}: T->{action}")
                changed = True
                break
    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, logs, "ls_shift_truck_to_tower_or_drone"


def ls_merge_drone_routes(sol: Solution, data: ProblemData) -> Tuple[Solution, List[str], str]:
    sol = sol.copy()
    logs: List[str] = []
    improved = True
    while improved:
        improved = False
        by_tower: Dict[int, List[int]] = defaultdict(list)
        for u, route in sol.drone_routes.items():
            if len(route) >= 3:
                by_tower[route[0]].append(u)
        for tower, drones in by_tower.items():
            if len(drones) < 2:
                continue
            drones = sorted(drones)
            for i in range(len(drones)):
                for j in range(i + 1, len(drones)):
                    u1, u2 = drones[i], drones[j]
                    nodes = sol.drone_routes[u1][1:-1] + sol.drone_routes[u2][1:-1]
                    merged = nearest_neighbor_order(tower, nodes, tower, data.aerial_dist)
                    if route_distance(merged, data.aerial_dist) <= DRONE_MAX_RANGE + 1e-9:
                        old = evaluate_solution(sol.copy(), data)
                        new_sol = sol.copy()
                        new_sol.drone_routes[u1] = merged
                        del new_sol.drone_routes[u2]
                        for n in merged[1:-1]:
                            new_sol.node_assignment[n] = ("D", u1)
                        cleanup_solution(new_sol, data)
                        new = evaluate_solution(new_sol, data)
                        if new_sol.feasible and new + 1e-9 < old:
                            sol = new_sol
                            logs.append(f"塔{tower}: 合并无人机 {u1}+{u2}")
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, logs, "ls_merge_drone_routes"


def ls_relocate_tower(sol: Solution, data: ProblemData) -> Tuple[Solution, List[str], str]:
    sol = sol.copy()
    logs: List[str] = []
    baseline = evaluate_solution(sol.copy(), data)
    for current_tower in sorted(sol.built_towers):
        visit_nodes = list(sol.tower_visits.get(current_tower, []))
        tower_drone_ids = [u for u, r in sol.drone_routes.items() if r[0] == current_tower]
        drone_nodes = [n for u in tower_drone_ids for n in sol.drone_routes[u][1:-1]]
        for cand in data.towers:
            if cand in sol.built_towers or cand == current_tower:
                continue
            # 候选塔要能覆盖当前塔直巡节点；对无人机节点要能在 50km 内完成各自路线
            if any(n not in data.coverage_set.get(cand, set()) for n in visit_nodes):
                continue
            ok = True
            for u in tower_drone_ids:
                nodes = sol.drone_routes[u][1:-1]
                new_route = nearest_neighbor_order(cand, nodes, cand, data.aerial_dist)
                if route_distance(new_route, data.aerial_dist) > DRONE_MAX_RANGE + 1e-9:
                    ok = False
                    break
            if not ok:
                continue
            new_sol = sol.copy()
            new_sol.built_towers.remove(current_tower)
            new_sol.built_towers.add(cand)
            new_sol.tower_visits.pop(current_tower, None)
            new_sol.tower_visits[cand] = visit_nodes[:]
            for n in visit_nodes:
                new_sol.node_assignment[n] = ("W", cand)
            for u in tower_drone_ids:
                nodes = sol.drone_routes[u][1:-1]
                new_sol.drone_routes[u] = nearest_neighbor_order(cand, nodes, cand, data.aerial_dist)
                for n in nodes:
                    new_sol.node_assignment[n] = ("D", u)
            cleanup_solution(new_sol, data)
            new_cost = evaluate_solution(new_sol, data)
            if new_sol.feasible and new_cost + 1e-9 < baseline:
                sol = new_sol
                baseline = new_cost
                logs.append(f"塔 {current_tower}->{cand}")
                break
        if logs:
            break
    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, logs, "ls_relocate_tower"


def ls_remove_empty_towers(sol: Solution, data: ProblemData) -> Tuple[Solution, List[str], str]:
    sol = sol.copy()
    logs: List[str] = []
    for l in list(sol.built_towers):
        has_visits = len(sol.tower_visits.get(l, [])) > 0
        has_drone = any(r[0] == l and r[-1] == l for r in sol.drone_routes.values())
        if not has_visits and not has_drone:
            sol.built_towers.remove(l)
            sol.tower_visits.pop(l, None)
            logs.append(f"移除空塔 {l}")
    cleanup_solution(sol, data)
    evaluate_solution(sol, data)
    return sol, logs, "ls_remove_empty_towers"


class IALNS:
    def __init__(
        self,
        data: ProblemData,
        max_iterations: Optional[int] = None,
        parameters: Optional[IALNSParameters] = None,
    ):
        self.data = data
        self.parameters = parameters or DEFAULT_IALNS_PARAMETERS
        self.max_iterations = int(max_iterations if max_iterations is not None else self.parameters.max_iterations)
        self.init_temp = float(self.parameters.initial_temperature)
        self.cooling = float(self.parameters.cooling_rate)
        self.min_temp = float(self.parameters.minimum_temperature)

        common_destroy = [
            ("destroy_random_nodes", destroy_random_nodes),
            ("destroy_worst_cost_nodes", destroy_worst_cost_nodes),
        ]
        problem_destroy = [
            ("destroy_one_tower", destroy_one_tower),
            ("destroy_one_truck", destroy_one_truck),
            ("destroy_one_drone", destroy_one_drone),
            ("destroy_overlap_tower", destroy_overlap_tower),
        ]
        self.destroy_entries = common_destroy + problem_destroy

        self.repair_entries = [
            ("repair_greedy", repair_op_greedy),
            ("repair_regret", repair_op_regret),
            ("repair_existing_first", repair_op_existing),
            ("repair_randomized", repair_op_randomized),
        ]

        self.destroy_stats = [OperatorStat(name) for name, _ in self.destroy_entries]
        self.repair_stats = [OperatorStat(name) for name, _ in self.repair_entries]

        self.local_entries = [
            ("ls_two_opt", ls_two_opt),
            ("ls_shift_truck_to_tower_or_drone", ls_shift_truck_to_tower_or_drone),
            ("ls_merge_drone_routes", ls_merge_drone_routes),
            ("ls_relocate_tower", ls_relocate_tower),
            ("ls_remove_empty_towers", ls_remove_empty_towers),
        ]
        self.local_stats = [OperatorStat(name) for name, _ in self.local_entries]

        self.iteration_history: List[Dict[str, Any]] = []
        self.initial_solution: Optional[Solution] = None

    def destroy(self, sol: Solution, idx: int) -> Tuple[Solution, List[int], str]:
        _name, func = self.destroy_entries[idx]
        if idx == 0:
            return func(sol, self.data, self.parameters.random_destroy_ratio)
        if idx == 1:
            return func(sol, self.data, self.parameters.worst_destroy_ratio)
        return func(sol, self.data)

    def repair(self, sol: Solution, removed: List[int], idx: int) -> Tuple[Solution, List[str], str]:
        _name, func = self.repair_entries[idx]
        return func(sol, removed, self.data)

    def _score_local(self, stat: OperatorStat, old_cost: float, new_cost: float, empty_tower: bool = False) -> None:
        stat.used += 1
        if new_cost + 1e-9 < old_cost:
            stat.score += self.parameters.empty_tower_reward_improve if empty_tower else self.parameters.local_reward_improve
            stat.improved_current += 1
        else:
            stat.score += self.parameters.empty_tower_reward_fail if empty_tower else self.parameters.local_reward_fail
            stat.rejected += 1

    def local_search(self, sol: Solution) -> Tuple[Solution, List[str]]:
        # 完整特色局部搜索：按权重选择一个主算子，再执行空塔删除。
        main_stats = self.local_stats[:-1]
        idx = roulette_select(main_stats)
        _name, func = self.local_entries[idx]
        old_cost = sol.cost
        new_sol, step_logs, name = func(sol, self.data)
        self._score_local(self.local_stats[idx], old_cost, new_sol.cost)
        logs = step_logs if step_logs else [name]

        tail_idx = len(self.local_entries) - 1
        old_tail_cost = new_sol.cost
        final_sol, tail_logs, tail_name = self.local_entries[tail_idx][1](new_sol, self.data)
        self._score_local(self.local_stats[tail_idx], old_tail_cost, final_sol.cost, empty_tower=True)
        logs.extend(tail_logs if tail_logs else [tail_name])
        return final_sol, logs

    def solve(self, seed: int = 1, verbose: bool = False) -> Tuple[Solution, Dict[str, Any]]:
        random.seed(seed)
        np.random.seed(seed)
        t0 = time.time()

        current = generate_initial_solution(self.data)
        self.initial_solution = current.copy()
        best = current.copy()
        temp = self.init_temp

        if verbose:
            print(
                f"  [IALNS] 初始解成本: {current.cost:.2f} | "
                f"可行: {current.feasible}"
            )

        for it in range(1, self.max_iterations + 1):
            d_idx = roulette_select(self.destroy_stats)
            r_idx = roulette_select(self.repair_stats)
            self.destroy_stats[d_idx].used += 1
            self.repair_stats[r_idx].used += 1

            destroyed, removed, destroy_name = self.destroy(current, d_idx)
            repaired, repair_logs, repair_name = self.repair(destroyed, removed, r_idx)

            ls_logs: List[str] = []
            do_ls = (
                (it % max(1, self.parameters.local_search_period) == 0)
                or (repaired.feasible and repaired.cost <= current.cost * self.parameters.local_search_cost_ratio)
            )
            candidate = repaired
            if do_ls:
                candidate, ls_logs = self.local_search(repaired)

            evaluate_solution(candidate, self.data)
            cur_cost = current.cost
            cand_cost = candidate.cost
            accepted = False
            accept_status = "rejected"

            if candidate.feasible and cand_cost + 1e-9 < best.cost:
                best = candidate.copy()
                current = candidate.copy()
                accepted = True
                accept_status = "global_best"
                self.destroy_stats[d_idx].score += self.parameters.reward_global_best
                self.repair_stats[r_idx].score += self.parameters.reward_global_best
                self.destroy_stats[d_idx].improved_global += 1
                self.repair_stats[r_idx].improved_global += 1
            elif candidate.feasible and cand_cost + 1e-9 < cur_cost:
                current = candidate.copy()
                accepted = True
                accept_status = "improved_current"
                self.destroy_stats[d_idx].score += self.parameters.reward_improve_current
                self.repair_stats[r_idx].score += self.parameters.reward_improve_current
                self.destroy_stats[d_idx].improved_current += 1
                self.repair_stats[r_idx].improved_current += 1
            else:
                delta = cand_cost - cur_cost
                prob = math.exp(-max(0.0, delta) / max(temp, 1e-6))
                if candidate.feasible and random.random() < prob:
                    current = candidate.copy()
                    accepted = True
                    accept_status = "accepted_worse"
                    self.destroy_stats[d_idx].score += self.parameters.reward_accepted_worse
                    self.repair_stats[r_idx].score += self.parameters.reward_accepted_worse
                    self.destroy_stats[d_idx].accepted_worse += 1
                    self.repair_stats[r_idx].accepted_worse += 1
                else:
                    self.destroy_stats[d_idx].score += self.parameters.reward_rejected
                    self.repair_stats[r_idx].score += self.parameters.reward_rejected
                    self.destroy_stats[d_idx].rejected += 1
                    self.repair_stats[r_idx].rejected += 1

            temp = max(self.min_temp, temp * self.cooling)
            if it % max(1, self.parameters.weight_update_interval) == 0:
                update_operator_weights(self.destroy_stats, self.parameters.reaction_factor)
                update_operator_weights(self.repair_stats, self.parameters.reaction_factor)
                if self.local_stats:
                    update_operator_weights(self.local_stats, self.parameters.reaction_factor)

            record = {
                "iteration": it,
                "destroy_op_idx": d_idx,
                "destroy_op_name": destroy_name,
                "repair_op_idx": r_idx,
                "repair_op_name": repair_name,
                "removed_count": len(removed),
                "repair_log": " | ".join(repair_logs[:12]),
                "local_search": " | ".join(ls_logs) if ls_logs else "none",
                "current_cost": current.cost,
                "candidate_cost": cand_cost,
                "best_cost": best.cost,
                "accepted": accepted,
                "accept_status": accept_status,
                "temperature": temp,
                "feasible_candidate": candidate.feasible,
                "violations_candidate": len(constraint_violations(candidate, self.data)),
            }
            self.iteration_history.append(record)

            if verbose and (it == 1 or it % 50 == 0 or it == self.max_iterations):
                print(
                    f"  [IALNS] 迭代 {it:>3d}/{self.max_iterations} | "
                    f"当前 {current.cost:,.2f} | 最优 {best.cost:,.2f} | "
                    f"破坏={destroy_name} | 修复={repair_name} | 接受={accept_status}"
                )


        elapsed = time.time() - t0
        result_meta = {
            "seed": seed,
            "elapsed_time": elapsed,
            "initial_solution": self.initial_solution,
            "destroy_stats": copy.deepcopy(self.destroy_stats),
            "repair_stats": copy.deepcopy(self.repair_stats),
            "local_stats": copy.deepcopy(self.local_stats),
            "iteration_history": list(self.iteration_history),
            "parameters": asdict(self.parameters),
            "max_iterations": self.max_iterations,
        }
        return best, result_meta


# =========================
# 结果输出
# =========================
def save_distance_matrices(data: ProblemData, out_dir: str) -> None:
    idx = data.all_nodes_order
    euc_df = pd.DataFrame(index=idx, columns=idx, dtype=float)
    truck_df = pd.DataFrame(index=idx, columns=idx, dtype=float)
    for i in idx:
        for j in idx:
            euc_df.loc[i, j] = data.aerial_dist[(i, j)]
            truck_df.loc[i, j] = data.truck_dist[(i, j)]
    euc_csv = os.path.join(out_dir, "距离矩阵_欧氏距离_无人机与瞭望塔覆盖使用.csv")
    truck_csv = os.path.join(out_dir, f"距离矩阵_卡车距离_欧氏距离乘以{TRUCK_DETOUR_FACTOR}.csv")
    node_csv = os.path.join(out_dir, "节点坐标与类型.csv")
    coverage_csv = os.path.join(out_dir, "塔覆盖关系.csv")
    try:
        euc_df.to_csv(euc_csv, encoding="utf-8-sig")
        truck_df.to_csv(truck_csv, encoding="utf-8-sig")
        pd.DataFrame(
            [{"node": n, "x": data.coords[n][0], "y": data.coords[n][1], "Classification": data.node_types[n]} for n in idx]
        ).to_csv(node_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [{"tower": l, "covered_nodes": ", ".join(map(str, data.coverage[l]))} for l in data.towers]
        ).to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"警告：距离矩阵保存失败：{e}")


def setup_chinese_fonts():
    """检测实际字体，避免Windows上EPS后端被不可转换的TTC字体阻塞。"""
    from matplotlib import font_manager as fm
    from fontTools.ttLib import TTFont
    import warnings
    preferred = ["SimHei", "Microsoft YaHei", "SimSun", "Noto Sans CJK SC",
                 "Noto Sans CJK JP", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    candidates = sorted(fm.fontManager.ttflist,
        key=lambda x: next((i for i,v in enumerate(preferred) if v.lower() in x.name.lower()), 999))
    fallback_cjk = None
    for f in candidates:
        if not any(v.lower() in f.name.lower() for v in preferred):
            continue
        try:
            font = TTFont(f.fname, fontNumber=0, lazy=True)
            tables = set(font.keys())
            cmap = font.getBestCmap() or {}
            works = all(ord(ch) in cmap for ch in "瞭望塔消防站无人机路线")
            # OpenType CFF/TTC 不必拦截: PDF/SVG兜底支持；优先glyf字体
            if works and fallback_cjk is None: fallback_cjk=f.name
            if works and {'glyf','loca','head','hmtx','maxp'}.issubset(tables):
                plt.rcParams['font.sans-serif'] = [f.name, 'DejaVu Sans']
                print(f"[字体] 使用EPS兼容中文字体: {f.name}")
                return f.name
        except Exception:
            pass
    if fallback_cjk:
        plt.rcParams["font.sans-serif"] = [fallback_cjk,"DejaVu Sans"]
        print(f"[字体] 中文字体 {fallback_cjk} 使用PDF->EPS矢量转换兼容路径")
        return fallback_cjk
    warnings.warn("未找到可用中文字体；请安装SimHei/微软雅黑或Noto CJK")
    return None


def save_eps_safely(fig, eps_path):
    """保留路线/节点的矢量图。EPS不透明，使用独立EPS外观，不影响PNG透明效果。

    优先直接EPS。发生字体异常时尝试先输出PDF，再通过pdftops转换；
    没有转换工具则保存SVG并报告，而不是让实验结果丢失。
    """
    import shutil, subprocess, warnings
    from pathlib import Path
    from matplotlib.patches import Circle
    eps_path = Path(eps_path)
    # PS不支持透明度，使用粉色实心淡填充保持视觉接近
    old=[]
    for ax in fig.axes:
        for artist in ax.patches:
            if isinstance(artist, Circle) and artist.get_alpha() not in (None,1):
                old.append((artist,artist.get_alpha(),artist.get_facecolor()))
                artist.set_alpha(1)
                artist.set_facecolor('#fcebf2')
    try:
        try:
            # 字体为OTF/TTC时绕开旧版本Matplotlib的PostScript转换缺陷
            use_direct = not any(('CJK' in n or 'WenQuanYi' in n) for n in plt.rcParams['font.sans-serif'])
            if use_direct:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='The PostScript backend does not support transparency')
                    fig.savefig(eps_path,format='eps',bbox_inches='tight')
                if eps_path.exists() and eps_path.stat().st_size>500:
                    return True
        except Exception as exc:
            print(f"[EPS字体兼容] 直接导出失败: {exc}; 尝试PDF矢量转换")
        pdf_path=eps_path.with_suffix('.pdf')
        try:
            fig.savefig(pdf_path,format='pdf',bbox_inches='tight')
            converter=shutil.which('pdftops')
            if converter:
                cp=subprocess.run([converter,'-eps',str(pdf_path),str(eps_path)],
                                  capture_output=True,text=True,timeout=60)
                if cp.returncode==0 and eps_path.exists() and eps_path.stat().st_size>500:
                    return True
                print('[EPS] PDF转换失败:',cp.stderr[:350])
        except Exception as exc:
            print('[EPS] PDF兜底失败:',exc)
        svg_path=eps_path.with_suffix('.svg')
        fig.savefig(svg_path,format='svg',bbox_inches='tight')
        print(f'[EPS] 无法输出可靠EPS，已保存SVG矢量图: {svg_path}')
        return False
    finally:
        for artist,alpha,face in old:
            artist.set_facecolor(face)
            artist.set_alpha(alpha)


def visualize_solution(sol, data, out_path, title=None, show=False):
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.patches import FancyArrowPatch, Circle
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(right=0.71)
    for l in sorted(sol.built_towers):
        ax.add_patch(Circle(data.coords[l], TOWER_RADIUS, facecolor="#ffd6e7",
                            edgecolor="#ec70a0", linestyle="--", lw=1.1, alpha=0.28, zorder=0))
    for l, nodes in sol.tower_visits.items():
        for n in nodes:
            a,b=data.coords[l],data.coords[n]
            ax.plot([a[0],b[0]],[a[1],b[1]], color="#e979a5",ls="--",lw=.85,zorder=2)
    def arrow_route(route,color,style):
        for a,b in zip(route[:-1],route[1:]):
            p,q=data.coords[a],data.coords[b]
            if math.dist(p,q)<1e-9: continue
            ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=10,
                       lw=1.05,color=color,linestyle=style,zorder=3))
    for route in sol.truck_routes.values(): arrow_route(route,"#8b38ae","-")
    for route in sol.drone_routes.values(): arrow_route(route,"#16864a","--")
    for nodes,color,marker,size in [(data.n1,"#f4a023","o",48),(data.n2,"#338ed0","o",48),
        (sorted(set(data.towers)-sol.built_towers),"#888888","^",92),
        (sorted(sol.built_towers),"#e33442","^",105)]:
        if nodes:
            ax.scatter([data.coords[n][0] for n in nodes],[data.coords[n][1] for n in nodes],
                       c=color,marker=marker,s=size,edgecolors="white",linewidths=.4,zorder=6)
    depots=unique_preserve([data.depot_start,data.depot_end])
    ax.scatter([data.coords[n][0] for n in depots],[data.coords[n][1] for n in depots],
               c="black",marker="s",s=85,zorder=8)
    for idx,n in enumerate(data.n1+data.n2):
        x,y=data.coords[n]
        ax.annotate(str(idx+1),(x,y),xytext=(7,1),textcoords="offset points",
                    fontsize=8,color="#a65d05" if n in data.n1 else "#175a8e",zorder=9)
    for idx,n in enumerate(data.towers):
        x,y=data.coords[n]
        ax.annotate(f"T{idx+1}",(x,y),xytext=(8,0),textcoords="offset points",
                    fontsize=8,color="#9d202b" if n in sol.built_towers else "#555555",zorder=9)
    for n in depots:
        ax.annotate("消防站",data.coords[n],xytext=(8,-11),textcoords="offset points",fontsize=9)
    coverage_legend=Circle((0,0),1,facecolor="#ffd6e7",edgecolor="#ec70a0",
                           linestyle="--",linewidth=1.4,label="瞭望塔覆盖范围")
    legend=[Line2D([],[],ls="none",marker="o",color="#f4a023",markersize=9,label="道路沿线监测节点"),
            Line2D([],[],ls="none",marker="o",color="#338ed0",markersize=9,label="非道路监测节点"),
            Line2D([],[],ls="none",marker="^",color="#e33442",markersize=10,label="已建设瞭望塔"),
            Line2D([],[],ls="none",marker="^",color="#888888",markersize=10,label="未建设瞭望塔"),
            coverage_legend,
            Line2D([],[],color="#e979a5",ls="--",lw=1,label="瞭望塔巡检路线"),
            FancyArrowPatch((0,0),(.9,0),arrowstyle="-|>",color="#8b38ae",lw=1.2,label="卡车巡检路线"),
            FancyArrowPatch((0,0),(.9,0),arrowstyle="-|>",color="#16864a",ls="--",lw=1.2,label="无人机巡检路线"),
            Line2D([],[],ls="none",marker="s",color="black",markersize=9,label="消防站")]
    class CoverageHandler(HandlerPatch):
        def create_artists(self,legend,orig,xdescent,ydescent,width,height,fontsize,trans):
            c=Circle((xdescent+width/2,ydescent+height/2),min(width,height)*0.38,
                facecolor="#ffd6e7",edgecolor="#ec70a0",linestyle="--",
                linewidth=1.4,transform=trans)
            return [c]
    class ArrowHandler(HandlerPatch):
        def create_artists(self,legend,orig,xdescent,ydescent,width,height,fontsize,trans):
            p=FancyArrowPatch((xdescent,ydescent+height/2),(xdescent+width,ydescent+height/2),
                arrowstyle="-|>",mutation_scale=12,lw=1.3,color=orig.get_edgecolor(),
                linestyle=orig.get_linestyle(),transform=trans)
            return [p]
    ax.legend(handles=legend,loc="upper left",bbox_to_anchor=(1.03,1.0),fontsize=12,
              frameon=True,handler_map={FancyArrowPatch:ArrowHandler(),Circle:CoverageHandler()})
    xs=[p[0] for p in data.coords.values()];ys=[p[1] for p in data.coords.values()]
    pad=TOWER_RADIUS+2
    ax.set_xlim(min(xs)-pad,max(xs)+pad);ax.set_ylim(min(ys)-pad,max(ys)+pad)
    ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("X 坐标 (km)",fontsize=12);ax.set_ylabel("Y 坐标 (km)",fontsize=12)
    ax.set_title(title or f"{data.instance_name} 最优巡检方案",fontsize=13)
    ax.grid(alpha=.15)
    fig.savefig(out_path,dpi=300,bbox_inches="tight")
    save_eps_safely(fig, Path(out_path).with_suffix(".eps"))
    if show:
        try:
            import matplotlib.image as mpimg
            image=mpimg.imread(out_path)
            viewer=plt.figure(figsize=(13,9));plt.imshow(image);plt.axis("off");plt.show(block=False)
        except Exception as exc: print("无法弹出图片，已保存PNG/EPS:",exc)
    plt.close(fig)


def plot_iteration_curve(iter_hist: List[Dict[str, Any]], out_curve: str, out_ops: str, title_prefix: str = "IALNS") -> None:
    if not iter_hist:
        return
    it = [r["iteration"] for r in iter_hist]
    cur = [r["current_cost"] for r in iter_hist]
    best = [r["best_cost"] for r in iter_hist]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(it, cur, label="当前解成本", linewidth=1.0, alpha=0.7)
    ax.plot(it, best, label="最优解成本", linewidth=2.0)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("总成本")
    ax.set_title("IALNS无重启算法迭代收敛曲线")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_curve, dpi=300, bbox_inches="tight")
    plt.close(fig)

    destroy_counter = Counter(r["destroy_op_name"] for r in iter_hist)
    repair_counter = Counter(r["repair_op_name"] for r in iter_hist)
    ls_counter = Counter(r["local_search"] for r in iter_hist)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, counter, title in [
        (axes[0], destroy_counter, "破坏算子使用频次"),
        (axes[1], repair_counter, "修复算子使用频次"),
        (axes[2], ls_counter, "局部搜索使用频次"),
    ]:
        names = list(counter.keys())
        vals = [counter[n] for n in names]
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_ops, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_operator_stats(meta: Dict[str, Any], out_dir: str) -> None:
    rows = []
    for group_name, stats in [
        ("destroy", meta["destroy_stats"]),
        ("repair", meta["repair_stats"]),
        ("local", meta["local_stats"]),
    ]:
        for s in stats:
            rows.append(
                {
                    "group": group_name,
                    "name": s.name,
                    "weight": s.weight,
                    "score": s.score,
                    "used": s.used,
                    "improved_global": s.improved_global,
                    "improved_current": s.improved_current,
                    "accepted_worse": s.accepted_worse,
                    "rejected": s.rejected,
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "算子统计.csv"), index=False, encoding="utf-8-sig")


def save_iteration_history(meta: Dict[str, Any], out_dir: str) -> None:
    df = pd.DataFrame(meta["iteration_history"])
    df.to_csv(os.path.join(out_dir, "迭代历史.csv"), index=False, encoding="utf-8-sig")


def save_solution_tables(sol: Solution, data: ProblemData, out_dir: str) -> None:
    cb = cost_breakdown(sol, data)

    assign_rows = []
    for n in data.n:
        mode, rid = sol.node_assignment.get(n, ("UNASSIGNED", -1))
        assign_rows.append(
            {
                "node": n,
                "mode": mode,
                "resource_id": rid,
                "x": data.coords[n][0],
                "y": data.coords[n][1],
                "node_type": data.node_types[n],
            }
        )
    pd.DataFrame(assign_rows).to_csv(os.path.join(out_dir, "节点分配表.csv"), index=False, encoding="utf-8-sig")

    tower_rows = []
    for l in sorted(sol.built_towers):
        tower_rows.append(
            {
                "tower": l,
                "visited_nodes": str(sol.tower_visits.get(l, [])),
                "num_visited": len(sol.tower_visits.get(l, [])),
                "drone_count": drone_count_on_tower(sol, l),
            }
        )
    pd.DataFrame(tower_rows).to_csv(os.path.join(out_dir, "瞭望塔巡检表.csv"), index=False, encoding="utf-8-sig")

    truck_rows = []
    for k, route in sorted(sol.truck_routes.items()):
        dist = route_distance(route, data.truck_dist)
        truck_rows.append(
            {
                "truck_id": k,
                "route": str(route),
                "distance_per_patrol": dist,
                "distance_annual": dist * PATROLS_PER_YEAR,
                "is_over_limit": "是" if dist > TRUCK_MAX_RANGE + 1e-9 else "否",
            }
        )
    pd.DataFrame(truck_rows).to_csv(os.path.join(out_dir, "卡车路径表.csv"), index=False, encoding="utf-8-sig")

    drone_rows = []
    for u, route in sorted(sol.drone_routes.items()):
        dist = route_distance(route, data.aerial_dist)
        drone_rows.append(
            {
                "drone_id": u,
                "route": str(route),
                "launch_tower": route[0] if route else None,
                "distance_per_patrol": dist,
                "distance_annual": dist * PATROLS_PER_YEAR,
                "is_over_limit": "是" if dist > DRONE_MAX_RANGE + 1e-9 else "否",
            }
        )
    pd.DataFrame(drone_rows).to_csv(os.path.join(out_dir, "无人机路径表.csv"), index=False, encoding="utf-8-sig")

    cost_df = pd.DataFrame(
        [
            {"item": "固定成本_总", "value": cb.fixed_total},
            {"item": "固定成本_塔", "value": cb.fixed_tower},
            {"item": "固定成本_卡车", "value": cb.fixed_truck},
            {"item": "固定成本_无人机", "value": cb.fixed_drone},
            {"item": "每次巡逻运营成本_总", "value": cb.patrol_total},
            {"item": "每次巡逻运营成本_塔", "value": cb.patrol_tower},
            {"item": "每次巡逻运营成本_卡车", "value": cb.patrol_truck},
            {"item": "每次巡逻运营成本_无人机", "value": cb.patrol_drone},
            {"item": "全年运营成本_总", "value": cb.annual_operating_total},
            {"item": "全年运营成本_塔", "value": cb.annual_tower},
            {"item": "全年运营成本_卡车", "value": cb.annual_truck},
            {"item": "全年运营成本_无人机", "value": cb.annual_drone},
            {"item": "总成本_年度总", "value": cb.annual_total},
            {"item": "卡车里程_每次", "value": cb.truck_distance_per_patrol},
            {"item": "卡车里程_全年", "value": cb.truck_distance_annual},
            {"item": "无人机里程_每次", "value": cb.drone_distance_per_patrol},
            {"item": "无人机里程_全年", "value": cb.drone_distance_annual},
        ]
    )
    cost_df.to_csv(os.path.join(out_dir, "成本分解表.csv"), index=False, encoding="utf-8-sig")


def save_detailed_report(sol: Solution, data: ProblemData, meta: Dict[str, Any], run_stats: Dict[str, Any], out_path: str) -> None:
    cb = cost_breakdown(sol, data)
    violations = constraint_violations(sol, data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"IALNS（无重启）详细结果报告 - {data.instance_name}\n")
        f.write("=" * 100 + "\n\n")
        f.write("【算法】无重启IALNS（单算法）\n")
        f.write(f"  瞭望塔覆盖半径: {TOWER_RADIUS:.1f} km\n\n")

        f.write("【模型求解状态】\n")
        f.write(f"  状态: {'HEURISTIC_FEASIBLE' if sol.feasible else 'HEURISTIC_INFEASIBLE'}\n")
        f.write(f"  运行时间: {run_stats['elapsed_time']:.4f} 秒\n")
        f.write(f"  总迭代次数: {meta.get('max_iterations', MAX_ITERATIONS)}\n")
        f.write(f"  目标值: {sol.cost:.2f}\n")
        f.write(f"  约束违反数: {len(violations)}\n")
        if violations:
            for v in violations[:20]:
                f.write(f"    - {v}\n")
        f.write("\n")

        f.write("【节点与巡检率】\n")
        f.write(f"  待巡检监测节点总数: {len(data.n)}\n")
        f.write(f"  沿线道路监测节点 N1 数量: {len(data.n1)}，节点: {data.n1}\n")
        f.write(f"  非道路监测节点 N2 数量: {len(data.n2)}，节点: {data.n2}\n")
        f.write(f"  巡检率: {len(sol.node_assignment) / len(data.n) * 100 if data.n else 100:.2f}%\n\n")

        f.write("【成本分解】\n")
        f.write(f"  年度总成本: {cb.annual_total:.2f}\n")
        f.write(f"  1) 固定成本总计: {cb.fixed_total:.2f}\n")
        f.write(f"     - 瞭望塔固定成本: {cb.fixed_tower:.2f}\n")
        f.write(f"     - 卡车固定成本: {cb.fixed_truck:.2f}\n")
        f.write(f"     - 无人机固定成本: {cb.fixed_drone:.2f}\n")
        f.write(f"  2) 每次巡逻运营成本总计: {cb.patrol_total:.2f}\n")
        f.write(f"     - 瞭望塔每次巡逻运营成本: {cb.patrol_tower:.2f}\n")
        f.write(f"     - 卡车每次巡逻运营成本: {cb.patrol_truck:.2f}\n")
        f.write(f"     - 无人机每次巡逻运营成本: {cb.patrol_drone:.2f}\n")
        f.write(f"  3) 全年 {PATROLS_PER_YEAR} 次巡逻运营成本总计: {cb.annual_operating_total:.2f}\n")
        f.write(f"     - 瞭望塔全年运营成本: {cb.annual_tower:.2f}\n")
        f.write(f"     - 卡车全年运营成本: {cb.annual_truck:.2f}\n")
        f.write(f"     - 无人机全年运营成本: {cb.annual_drone:.2f}\n")
        f.write(f"  4) 卡车里程: 每次 {cb.truck_distance_per_patrol:.2f} km，全年 {cb.truck_distance_annual:.2f} km\n")
        f.write(f"  5) 无人机里程: 每次 {cb.drone_distance_per_patrol:.2f} km，全年 {cb.drone_distance_annual:.2f} km\n\n")

        f.write("【资源方案】\n")
        f.write(f"  实际建设瞭望塔数量: {len(sol.built_towers)} / 上限 {data.tower_limit}\n")
        for l in sorted(sol.built_towers):
            f.write(f"    塔 {l}: 巡检节点 {sol.tower_visits.get(l, [])}\n")
        f.write(f"  实际使用卡车数量: {len(sol.truck_routes)} / 上限 {data.truck_limit}\n")
        for k, route in sorted(sol.truck_routes.items()):
            dist = route_distance(route, data.truck_dist)
            f.write(f"    卡车 {k}: {route} | 里程 {dist:.2f} km | 是否超限: {'是' if dist > TRUCK_MAX_RANGE else '否'}\n")
        f.write(f"  实际使用无人机数量: {len(sol.drone_routes)} / 上限 {data.drone_limit}\n")
        for u, route in sorted(sol.drone_routes.items()):
            dist = route_distance(route, data.aerial_dist)
            f.write(f"    无人机 {u}: {route} | 里程 {dist:.2f} km | 是否超限: {'是' if dist > DRONE_MAX_RANGE else '否'}\n")
        f.write("\n")

        init_sol: Solution = meta["initial_solution"]
        f.write("【初始解方案构成】\n")
        if init_sol is not None:
            f.write(f"  初始解成本: {init_sol.cost:.2f}\n")
            f.write(f"  初始解可行: {init_sol.feasible}\n")
            f.write(f"  初始建塔: {sorted(init_sol.built_towers)}\n")
            f.write(f"  初始卡车路径: {init_sol.truck_routes}\n")
            f.write(f"  初始无人机路径: {init_sol.drone_routes}\n")
            for note in init_sol.initial_summary.get("notes", [])[:80]:
                f.write(f"    - {note}\n")
        f.write("\n")

        f.write("【算子统计】\n")
        for label, stats in [("破坏算子", meta["destroy_stats"]), ("修复算子", meta["repair_stats"]), ("局部搜索算子", meta["local_stats"])]:
            f.write(f"  {label}:\n")
            for s in stats:
                f.write(
                    f"    - {s.name}: used={s.used}, score={s.score:.2f}, weight={s.weight:.4f}, "
                    f"global={s.improved_global}, current={s.improved_current}, worse={s.accepted_worse}, rejected={s.rejected}\n"
                )
        f.write("\n")

        f.write("【迭代信息（每10次摘录）】\n")
        for rec in meta["iteration_history"]:
            if rec["iteration"] % 10 == 0 or rec["iteration"] == 1:
                f.write(
                    f"  iter={rec['iteration']:>3d} | destroy={rec['destroy_op_name']} | repair={rec['repair_op_name']} | "
                    f"ls={rec['local_search']} | current={rec['current_cost']:.2f} | best={rec['best_cost']:.2f} | accept={rec['accept_status']}\n"
                )
        f.write("\n")

        f.write("【10 次统计口径】\n")
        f.write(f"  Best  = {run_stats['best_cost']:.2f}\n")
        f.write(f"  Avg   = {run_stats['avg_cost']:.2f}\n")
        f.write(f"  Worst = {run_stats['worst_cost']:.2f}\n")
        f.write(f"  CPU   = {run_stats['avg_cpu']:.4f} 秒\n")


def save_multi_run_stats(instance_dir: str, run_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(run_records)
    csv_path = os.path.join(instance_dir, "10次运行统计.csv")
    xlsx_path = os.path.join(instance_dir, "10次运行统计.xlsx")
    txt_path = os.path.join(instance_dir, "10次运行统计说明.txt")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"警告：{xlsx_path} 保存失败：{e}")

    best_cost = float(df["total_cost"].min()) if not df.empty else float("inf")
    avg_cost = float(df["total_cost"].mean()) if not df.empty else float("inf")
    worst_cost = float(df["total_cost"].max()) if not df.empty else float("inf")
    avg_cpu = float(df["elapsed_time"].mean()) if not df.empty else 0.0
    best_row = df.loc[df["total_cost"].idxmin()].to_dict() if not df.empty else None

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("IALNS-w=10 单算例 10 次独立运行统计\n")
        f.write("=" * 80 + "\n")
        f.write("统计口径：Best/Avg/Worst 为 10 次独立运行目标值；CPU 为 10 次平均运行时间。\n\n")
        f.write(f"Best  = {best_cost:.2f}\n")
        f.write(f"Avg   = {avg_cost:.2f}\n")
        f.write(f"Worst = {worst_cost:.2f}\n")
        f.write(f"CPU   = {avg_cpu:.4f} 秒\n")
        if best_row is not None:
            f.write(f"最优运行 = Run {int(best_row['run_id'])} (seed={int(best_row['seed'])})\n")

    return {
        "best_cost": best_cost,
        "avg_cost": avg_cost,
        "worst_cost": worst_cost,
        "avg_cpu": avg_cpu,
        "best_row": best_row,
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
        "txt_path": txt_path,
    }



# ======================== 单算法运行接口 ========================
def locate_excel(base_dir: str, instance: str | None = None, excel: str | None = None) -> Path:
    """优先直接接收任意Excel；否则在根目录及子文件夹中根据算例名查找。"""
    if excel:
        file=Path(excel).expanduser().resolve()
        if not file.is_file() or file.suffix.lower() not in ('.xlsx','.xls'):
            raise FileNotFoundError(f'Excel文件不存在或后缀错误: {file}')
        return file
    if not instance:
        raise ValueError('必须指定 --instance 或 --excel')
    possible=Path(instance).expanduser()
    if possible.is_file() and possible.suffix.lower() in ('.xlsx','.xls'):
        return possible.resolve()
    base=Path(base_dir).expanduser()
    if not base.exists():
        raise FileNotFoundError(f'算例数据目录不存在: {base}，请用 --base-dir 指定')
    matches=[p for p in base.rglob('*.xlsx') if not p.name.startswith('~$') and (p.stem==instance or p.stem.startswith(instance+'('))]
    matches.sort(key=lambda p:(p.stem!=instance,len(p.parts),str(p)))
    if not matches:
        raise FileNotFoundError(f'找不到算例: {instance}，搜索根目录: {base}')
    if len(matches)>1:
        print(f'[注意] 发现{len(matches)}个同名/带编号文件，优先选用: {matches[0]}')
    return matches[0].resolve()






def load_taguchi_parameters(filename):
    """直接读取配套田口调优程序输出的最终最优参数JSON。"""
    payload=json.loads(Path(filename).read_text(encoding='utf-8'))
    if payload.get('algorithm') not in (None,'IALNS_FULL_NO_RESTART'):
        raise ValueError('参数JSON并非无重启IALNS，禁止混用')
    values=payload.get('parameters',payload)
    known=set(asdict(DEFAULT_IALNS_PARAMETERS))
    if set(values)-known:
        raise ValueError('参数JSON存在不支持的字段：'+str(set(values)-known))
    return IALNSParameters(**{**asdict(DEFAULT_IALNS_PARAMETERS),**values})


def run_single():
    global TOWER_RADIUS, TRUCK_MAX_RANGE, DRONE_MAX_RANGE, find_excel_for_instance
    parser=argparse.ArgumentParser(description='TLTDRP 无重启IALNS独立单算法：单Excel求解（无消融）')
    parser.add_argument('--instance',default=None,help='算例名称，例如 小规模算例1_4个节点')
    parser.add_argument('--excel',default=None,help='任意单个Excel完整路径（优先于--instance）')
    parser.add_argument('--base-dir',default=DEFAULT_BASE_DIR,help='算例搜索根目录，默认E盘instances-2')
    parser.add_argument('--output-dir',default='IALNS_单算法求解结果')
    parser.add_argument('--runs',type=int,default=10)
    parser.add_argument('--parameter-file',default=None,help='田口调优输出的Taguchi_最终最优参数.json')
    parser.add_argument('--iterations',type=int,default=None,help='可覆盖田口JSON里的最大迭代次数；未给则采用JSON参数')
    parser.add_argument('--seed-base',type=int,default=1000)
    parser.add_argument('--radius',type=float,default=10.0,help='瞭望塔覆盖半径，km')
    parser.add_argument('--truck-range',type=float,default=TRUCK_MAX_RANGE,help='卡车最大续航，km')
    parser.add_argument('--drone-range',type=float,default=DRONE_MAX_RANGE,help='无人机最大续航，km')
    parser.add_argument('--quick',action='store_true',help='快速测试：2次×40迭代')
    parser.add_argument('--verbose',action='store_true',help='每50迭代打印状态')
    parser.add_argument('--skip-plots',action='store_true',help='跳过全部方案图和曲线绘制')
    args=parser.parse_args()
    p=load_taguchi_parameters(args.parameter_file) if args.parameter_file else DEFAULT_IALNS_PARAMETERS
    if args.iterations is None: args.iterations=int(p.max_iterations)
    if args.quick: args.runs,args.iterations=2,40
    if args.runs<1 or args.iterations<1 or min(args.radius,args.truck_range,args.drone_range)<=0:
        parser.error('运行次数、迭代次数及覆盖/续航参数必须大于0')
    if not args.excel and not args.instance:
        parser.error('必须提供 --instance "算例名" 或 --excel "Excel文件完整路径"')

    TOWER_RADIUS=float(args.radius)
    TRUCK_MAX_RANGE=float(args.truck_range)
    DRONE_MAX_RANGE=float(args.drone_range)
    file=locate_excel(args.base_dir,args.instance,args.excel)
    instance=(args.instance if args.instance else file.stem)
    old_find=find_excel_for_instance
    try:
        find_excel_for_instance=lambda _base,_name:str(file)
        data=read_instance(str(file.parent),instance)
    finally:
        find_excel_for_instance=old_find

    out=(Path(args.output_dir).expanduser() / safe_name(instance)).resolve()
    out.mkdir(parents=True,exist_ok=True)
    # 参数及原始Excel指纹，便于论文复现
    config={'algorithm':'IALNS_FULL_NO_RESTART_SINGLE','instance':instance,'excel':str(file),
            'excel_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'runs':args.runs,'iterations':args.iterations,'seed_base':args.seed_base,
            'tower_radius':TOWER_RADIUS,'truck_range':TRUCK_MAX_RANGE,'drone_range':DRONE_MAX_RANGE,
            'patrols_per_year':PATROLS_PER_YEAR,'truck_detour_factor':TRUCK_DETOUR_FACTOR,
            'parameter_source':args.parameter_file or 'standalone-default',
            'taguchi_parameters':asdict(p)}
    (out/'实验参数.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    setup_chinese_fonts()
    save_distance_matrices(data,str(out))
    records=[];histories=[]
    best_sol=None;best_meta=None;best_run=None
    print(f'\n[IALNS无重启单算法] {instance} | 监测点={len(data.n)} | 候选塔={len(data.towers)} | Excel={file}',flush=True)
    for run in range(1,args.runs+1):
        seed=args.seed_base+run
        solver=IALNS(data,max_iterations=args.iterations,parameters=p)
        sol,meta=solver.solve(seed=seed,verbose=args.verbose)
        evaluate_solution(sol,data)
        rec={'Run':run,'Seed':seed,'Feasible':int(sol.feasible),'Cost':sol.cost if sol.feasible else float('nan'),
             'CPU':meta['elapsed_time'],'NumTowers':len(sol.built_towers),
             'NumTrucks':len(sol.truck_routes),'NumDrones':len(sol.drone_routes)}
        records.append(rec)
        pd.DataFrame(records).to_csv(out/'全部运行记录.csv',index=False,encoding='utf-8-sig')
        hist=pd.DataFrame(meta['iteration_history'])
        hist.to_csv(out/f'迭代历史_Run{run:02d}.csv',index=False,encoding='utf-8-sig')
        histories.append(hist)
        print(f'Run={run:02d}/{args.runs} Seed={seed} Cost={rec["Cost"]:,.2f} CPU={rec["CPU"]:.2f}s 可行={sol.feasible}',flush=True)
        if sol.feasible and (best_sol is None or sol.cost<best_sol.cost-1e-9):
            best_sol,best_meta,best_run=sol.copy(),meta,run
    runs_df=pd.DataFrame(records)
    valid=runs_df[runs_df.Feasible==1]
    stats={'Instance':instance,'Runs':len(runs_df),'FeasibleRuns':len(valid),'Best':float(valid.Cost.min()) if len(valid) else None,
           'Avg':float(valid.Cost.mean()) if len(valid) else None,'Worst':float(valid.Cost.max()) if len(valid) else None,
           'Std':float(valid.Cost.std(ddof=1)) if len(valid)>1 else 0.0,
           'CPU':float(runs_df.CPU.mean()),'BestRun':best_run,'BestSeed':(args.seed_base+best_run) if best_run else None}
    pd.DataFrame([stats]).to_csv(out/'单算法_汇总.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(out/'单算法_求解结果.xlsx',engine='openpyxl') as w:
        runs_df.to_excel(w,sheet_name='全部运行记录',index=False)
        pd.DataFrame([stats]).to_excel(w,sheet_name='汇总统计',index=False)
    if best_sol is None:
        (out/'运行状态.txt').write_text('没有找到可行解。请核对算例约束和参数。',encoding='utf8')
        print('没有得到可行解，未生成方案图。输出目录:',out)
        return
    save_solution_tables(best_sol,data,str(out))
    save_iteration_history(best_meta,str(out))
    save_operator_stats(best_meta,str(out))
    save_detailed_report(best_sol,data,best_meta,
        {'best_cost':stats['Best'],'avg_cost':stats['Avg'],'worst_cost':stats['Worst'],
         'avg_cpu':stats['CPU'],'elapsed_time':best_meta['elapsed_time']},str(out/'最优方案详细报告.txt'))
    if not args.skip_plots:
        try:
            visualize_solution(best_sol,data,str(out/'最优巡检方案.png'),title=f'{instance} | IALNS无重启')
        except Exception as exc:
            print('[警告] 巡检方案绘图失败:',exc)
        try:
            plot_iteration_curve(best_meta['iteration_history'],str(out/'最佳运行_迭代收敛曲线.png'),
                                 str(out/'最佳运行_算子使用频次.png'))
            if histories:
                aligned=pd.concat([h.set_index('iteration')['best_cost'].rename(i) for i,h in enumerate(histories) if len(h)],axis=1)
                fig,ax=plt.subplots(figsize=(10,6))
                ax.plot(aligned.index,aligned.mean(axis=1),lw=2,label='多次运行最优成本均值')
                ax.set_xlabel('迭代次数');ax.set_ylabel('总成本（美元）');ax.grid(alpha=.2);ax.legend()
                fig.savefig(out/'多次运行_平均收敛曲线.png',dpi=300,bbox_inches='tight')
                fig.savefig(out/'多次运行_平均收敛曲线.pdf',bbox_inches='tight')
                plt.close(fig)
        except Exception as exc:
            print('[警告] 迭代曲线绘图失败:',exc)
    print(f'\n完成：Best={stats["Best"]:,.2f}美元 / {stats["Best"]/10000:.4f}万美元，Avg={stats["Avg"]:,.2f}美元，平均CPU={stats["CPU"]:.2f}秒')
    print('最优运行Run=',best_run,'| 输出路径：',out)


if __name__=='__main__':
    run_single()
