

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, fields
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
]
REAL_INSTANCE_NAMES = ("真实案例_大规模_61个节点", "新林案例_68个节点")
SIMULATION_INSTANCE_NAMES = tuple(INSTANCE_NAMES)


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
MAX_ITERATIONS = 700  # Taguchi 最优参数
DEFAULT_INIT_TEMP = 2500.0
DEFAULT_COOLING = 0.985
DEFAULT_MIN_TEMP = 0.1

# Taguchi 最优自适应与邻域参数
WEIGHT_REACTION = 0.3
WEIGHT_UPDATE_INTERVAL = 40
LOCAL_SEARCH_PERIOD = 3
LOCAL_SEARCH_COST_RATIO = 1.01
RANDOM_DESTROY_RATIO = 0.15
WORST_DESTROY_RATIO = 0.18

# Taguchi 最优奖励参数
REWARD_GLOBAL_BEST = 60.0
REWARD_IMPROVE_CURRENT = 25.0
REWARD_ACCEPTED_WORSE = 8.0
REWARD_REJECTED = 1.0
LOCAL_REWARD_IMPROVE = 20.0
LOCAL_REWARD_FAIL = 3.0
EMPTY_TOWER_REWARD_IMPROVE = 10.0
EMPTY_TOWER_REWARD_FAIL = 2.0
PENALTY_UNCOVERED = 5e7
PENALTY_CONSTRAINT = 5e6
EPS = 1e-9

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


# =========================
# 消融变体与 IALNS 主体
# =========================
@dataclass(frozen=True)
class IALNSParameters:
    """2026-11-20无重启L27确认胜出的参数，可由JSON动态覆盖。"""
    max_iterations: int = 500
    initial_temperature: float = 2500.0
    cooling_rate: float = 0.992
    minimum_temperature: float = 0.05
    reaction_factor: float = 0.2
    weight_update_interval: int = 40
    local_search_period: int = 3
    local_search_cost_ratio: float = 1.02
    random_destroy_ratio: float = 0.15
    worst_destroy_ratio: float = 0.18
    reward_global_best: float = 25.0
    reward_improve_current: float = 12.0
    reward_accepted_worse: float = 4.0
    reward_rejected: float = 1.0
    local_reward_improve: float = 20.0
    local_reward_fail: float = 3.0
    empty_tower_reward_improve: float = 10.0
    empty_tower_reward_fail: float = 2.0


DEFAULT_IALNS_PARAMETERS=IALNSParameters()
TAGUCHI_VERSION='NO_RESTART_IALNS_TAGUCHI_L27_20261120_V1'
TAGUCHI_SOURCE='L27排名试验_11'
ENGINE_VERSION="NO_RESTART_TAGUCHI_ABLATION_20261120_V2"


def load_taguchi_parameters(path: Path) -> IALNSParameters:
    """仅接受预期无重启IALNS参数方案，避免混入旧版重启参数。"""
    raw=json.loads(path.read_text(encoding='utf-8-sig'))
    if raw.get('algorithm')!='IALNS_FULL_NO_RESTART':
        raise ValueError(f'不是最终无重启IALNS的参数JSON: {raw.get("algorithm")}')
    values=raw.get('parameters')
    if not isinstance(values,dict):
        raise ValueError('参数JSON缺少parameters字典')
    expected={f.name for f in fields(IALNSParameters)}
    if set(values)!=expected:
        raise ValueError(f'参数字段不匹配；缺少: {sorted(expected-set(values))}；多出: {sorted(set(values)-expected)}')
    p=IALNSParameters(**values)
    if p.max_iterations<1 or p.weight_update_interval<1 or p.local_search_period<1:
        raise ValueError('迭代次数和周期必须为正整数')
    if not (0<p.cooling_rate<1 and p.initial_temperature>0 and p.minimum_temperature>0):
        raise ValueError('模拟退火参数范围不正确')
    if not (0<p.random_destroy_ratio<=1 and 0<p.worst_destroy_ratio<=1):
        raise ValueError('破坏比例必须处于(0,1]')
    return p


@dataclass(frozen=True)
class AblationVariant:
    code: str
    name_cn: str
    description: str
    use_problem_dr: bool = True
    local_search_mode: str = "full"  # full / basic
    use_adaptive_weights: bool = True
    use_restart: bool = False  # 最终版固定关闭重启，保留字段仅兼容既有辅助函数


ABLATION_VARIANTS: Dict[str, AblationVariant] = {
    "FULL": AblationVariant(
        code="FULL", name_cn="完整IALNS（无重启）",
        description="保留问题驱动破坏修复、特色局部搜索和自适应权重更新；完全不使用停滞重启。"),
    "NO_PD_DR": AblationVariant(
        code="NO_PD_DR", name_cn="去特色破坏修复",
        description="移除问题驱动破坏与修复算子，仅保留通用随机/高成本节点移除及随机/贪心/后悔插入。",
        use_problem_dr=False),
    "NO_PD_LS": AblationVariant(
        code="NO_PD_LS", name_cn="去特色局部搜索",
        description="移除卡车节点转移、无人机路径合并、瞭望塔重定位和空塔删除，仅保留标准2-opt。",
        local_search_mode="basic"),
}
DEFAULT_VARIANT_CODES = ["FULL", "NO_PD_DR", "NO_PD_LS"]


class IALNS:
    def __init__(
        self,
        data: ProblemData,
        variant: AblationVariant,
        max_iterations: Optional[int] = None,
        parameters: Optional[IALNSParameters] = None,
    ):
        self.data = data
        self.variant = variant
        self.parameters = parameters or DEFAULT_IALNS_PARAMETERS
        self.max_iterations = int(max_iterations if max_iterations is not None else self.parameters.max_iterations)
        self.init_temp = float(self.parameters.initial_temperature)
        self.cooling = float(self.parameters.cooling_rate)
        self.min_temp = float(self.parameters.minimum_temperature)
        self.restart_events = 0  # 最终版无重启，固定为0，仅兼容结果记录字段
        self.restart_log = []

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
        self.destroy_entries = common_destroy + (problem_destroy if variant.use_problem_dr else [])

        # FULL 的算子顺序保持与田口调优后的原算法一致。
        if variant.use_problem_dr:
            self.repair_entries = [
                ("repair_greedy", repair_op_greedy),
                ("repair_regret", repair_op_regret),
                ("repair_existing_first", repair_op_existing),
                ("repair_randomized", repair_op_randomized),
            ]
        else:
            self.repair_entries = [
                ("repair_greedy", repair_op_greedy),
                ("repair_regret", repair_op_regret),
                ("repair_randomized", repair_op_randomized),
            ]

        self.destroy_stats = [OperatorStat(name) for name, _ in self.destroy_entries]
        self.repair_stats = [OperatorStat(name) for name, _ in self.repair_entries]

        if variant.local_search_mode == "full":
            self.local_entries = [
                ("ls_two_opt", ls_two_opt),
                ("ls_shift_truck_to_tower_or_drone", ls_shift_truck_to_tower_or_drone),
                ("ls_merge_drone_routes", ls_merge_drone_routes),
                ("ls_relocate_tower", ls_relocate_tower),
                ("ls_remove_empty_towers", ls_remove_empty_towers),
            ]
        elif variant.local_search_mode == "basic":
            self.local_entries = [("ls_two_opt", ls_two_opt)]
        else:
            self.local_entries = []
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
        if self.variant.local_search_mode == "none" or not self.local_entries:
            return sol.copy(), []

        # 标准/基础局部搜索：仅执行2-opt。
        if self.variant.local_search_mode == "basic":
            old_cost = sol.cost
            new_sol, step_logs, name = self.local_entries[0][1](sol, self.data)
            self._score_local(self.local_stats[0], old_cost, new_sol.cost)
            return new_sol, step_logs if step_logs else [name]

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
                f"  [{self.variant.code}] 初始解成本: {current.cost:.2f} | "
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
            do_ls = self.variant.local_search_mode != "none" and (
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
            if self.variant.use_adaptive_weights and it % max(1, self.parameters.weight_update_interval) == 0:
                update_operator_weights(self.destroy_stats, self.parameters.reaction_factor)
                update_operator_weights(self.repair_stats, self.parameters.reaction_factor)
                if self.local_stats:
                    update_operator_weights(self.local_stats, self.parameters.reaction_factor)

            record = {
                "iteration": it,
                "variant_code": self.variant.code,
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
                "restart_count": self.restart_events,
            }
            self.iteration_history.append(record)

            if verbose and (it == 1 or it % 50 == 0 or it == self.max_iterations):
                print(
                    f"  [{self.variant.code}] 迭代 {it:>3d}/{self.max_iterations} | "
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
            "restart_events": self.restart_events,
            "restart_log": [],
            "variant_code": self.variant.code,
            "variant_name": self.variant.name_cn,
            "variant_description": self.variant.description,
            "max_iterations": self.max_iterations,
            "parameters": asdict(self.parameters),
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
    # 覆盖范围图例使用真正的浅粉色填充圆形 + 粉色虚线边界，不再使用单独虚线。
    coverage_legend_circle=Circle((0,0),radius=1,facecolor="#ffd6e7",
                                  edgecolor="#ec70a0",linestyle="--",linewidth=1.4,
                                  label="瞭望塔覆盖范围")
    legend=[Line2D([],[],ls="none",marker="o",color="#f4a023",markersize=9,label="道路沿线监测节点"),
            Line2D([],[],ls="none",marker="o",color="#338ed0",markersize=9,label="非道路监测节点"),
            Line2D([],[],ls="none",marker="^",color="#e33442",markersize=10,label="已建设瞭望塔"),
            Line2D([],[],ls="none",marker="^",color="#888888",markersize=10,label="未建设瞭望塔"),
            coverage_legend_circle,
            Line2D([],[],color="#e979a5",ls="--",lw=1,label="瞭望塔巡检路线"),
            FancyArrowPatch((0,0),(.9,0),arrowstyle="-|>",color="#8b38ae",lw=1.2,label="卡车巡检路线"),
            FancyArrowPatch((0,0),(.9,0),arrowstyle="-|>",color="#16864a",ls="--",lw=1.2,label="无人机巡检路线"),
            Line2D([],[],ls="none",marker="s",color="black",markersize=9,label="消防站")]
    class CoverageCircleHandler(HandlerPatch):
        def create_artists(self,legend,orig,xdescent,ydescent,width,height,fontsize,trans):
            radius=min(width,height)*0.38
            c=Circle((xdescent+width/2,ydescent+height/2),radius,
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
              frameon=True,handler_map={FancyArrowPatch:ArrowHandler(),Circle:CoverageCircleHandler()})
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
    ax.set_title("IALNS-w=10（Taguchi最优参数）迭代收敛曲线")
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
        f.write(f"{meta.get('variant_name', 'IALNS')} 详细结果报告 - {data.instance_name}\n")
        f.write("=" * 100 + "\n\n")
        f.write("【Taguchi最终冻结参数】\n")
        f.write(json.dumps(meta.get('parameters',{}),ensure_ascii=False,indent=2)+"\n\n")
        f.write("【消融变体】\n")
        f.write(f"  代码: {meta.get('variant_code', 'FULL')}\n")
        f.write(f"  名称: {meta.get('variant_name', '完整IALNS')}\n")
        f.write(f"  定义: {meta.get('variant_description', '')}\n")
        f.write(f"  瞭望塔覆盖半径: {TOWER_RADIUS:.1f} km\n")
        f.write(f"  重启触发次数: {meta.get('restart_events', 0)}\n\n")

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


def plot_global_summary(summary_df: pd.DataFrame, out_png: str) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    x = np.arange(len(summary_df))
    width = 0.25
    axes[0].bar(x - width, summary_df["Best"], width=width, label="Best")
    axes[0].bar(x, summary_df["Avg"], width=width, label="Avg")
    axes[0].bar(x + width, summary_df["Worst"], width=width, label="Worst")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df["Instance"], rotation=35, ha="right")
    axes[0].set_ylabel("总成本")
    axes[0].set_title("16 个算例 10 次运行目标值统计")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(x, summary_df["CPU"], marker="o")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["Instance"], rotation=35, ha="right")
    axes[1].set_ylabel("平均 CPU (秒)")
    axes[1].set_title("16 个算例平均运行时间")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 消融实验统计与绘图
# =========================
def parse_variant_codes(raw_values: Optional[List[str]]) -> List[str]:
    if not raw_values:
        return list(DEFAULT_VARIANT_CODES)
    codes: List[str] = []
    for raw in raw_values:
        for code in raw.split(","):
            code = code.strip().upper()
            if not code:
                continue
            if code not in ABLATION_VARIANTS:
                raise ValueError(
                    f"未知消融变体 {code}。可选：{', '.join(ABLATION_VARIANTS.keys())}"
                )
            if code not in codes:
                codes.append(code)
    if "FULL" not in codes:
        codes.insert(0, "FULL")
    return codes


def parse_radii(radius_arg: str) -> List[float]:
    text = str(radius_arg).strip().lower()
    if text == "all":
        return [10.0, 15.0]
    values = []
    for part in text.split(","):
        value = float(part.strip())
        if value <= 0:
            raise ValueError("覆盖半径必须为正数")
        values.append(value)
    return values


def append_record_csv(path: str, record: Dict[str, Any]) -> None:
    df = pd.DataFrame([record])
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    df.to_csv(path, mode="a", header=not exists, index=False, encoding="utf-8-sig")


def load_existing_records(path: str) -> pd.DataFrame:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def record_key(radius: float, instance: str, variant: str, run_id: int, seed: int) -> Tuple[Any, ...]:
    return (round(float(radius), 6), str(instance), str(variant), int(run_id), int(seed))


def make_run_record(
    radius: float,
    instance_name: str,
    variant: AblationVariant,
    run_id: int,
    seed: int,
    sol: Solution,
    meta: Dict[str, Any],
    data: ProblemData,
) -> Dict[str, Any]:
    cb = cost_breakdown(sol, data)
    return {
        "Radius": radius,
        "Instance": instance_name,
        "Variant": variant.code,
        "VariantName": variant.name_cn,
        "Run": run_id,
        "Seed": seed,
        "TotalCost": sol.cost,
        "Feasible": int(sol.feasible),
        "CPU": meta["elapsed_time"],
        "RestartEvents": meta.get("restart_events", 0),
        "NumTowers": len(sol.built_towers),
        "NumTrucks": len(sol.truck_routes),
        "NumDrones": len(sol.drone_routes),
        "FixedCost": cb.fixed_total,
        "AnnualOperatingCost": cb.annual_operating_total,
        "TowerFixedCost": cb.fixed_tower,
        "TruckFixedCost": cb.fixed_truck,
        "DroneFixedCost": cb.fixed_drone,
        "AnnualTowerCost": cb.annual_tower,
        "AnnualTruckCost": cb.annual_truck,
        "AnnualDroneCost": cb.annual_drone,
        "TruckDistancePerPatrol": cb.truck_distance_per_patrol,
        "DroneDistancePerPatrol": cb.drone_distance_per_patrol,
        "Iterations": meta.get("max_iterations", MAX_ITERATIONS),
    }


def summarize_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if raw_df.empty:
        return pd.DataFrame()
    numeric_cols = ["Radius", "Run", "Seed", "TotalCost", "Feasible", "CPU"]
    for col in numeric_cols:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    for (radius, instance, variant), g in raw_df.groupby(["Radius", "Instance", "Variant"], sort=False):
        g = g.dropna(subset=["TotalCost"])
        if g.empty:
            continue
        best_idx = g["TotalCost"].idxmin()
        best_row = g.loc[best_idx]
        avg = float(g["TotalCost"].mean())
        std = float(g["TotalCost"].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(
            {
                "Radius": float(radius),
                "Instance": instance,
                "Variant": variant,
                "VariantName": best_row.get("VariantName", variant),
                "Runs": int(len(g)),
                "Best": float(g["TotalCost"].min()),
                "Avg": avg,
                "Worst": float(g["TotalCost"].max()),
                "Std": std,
                "CV(%)": 100.0 * std / avg if abs(avg) > EPS else 0.0,
                "CPU": float(g["CPU"].mean()),
                "FeasibleRate(%)": 100.0 * float(g["Feasible"].mean()),
                "BestRun": int(best_row["Run"]),
                "BestSeed": int(best_row["Seed"]),
                "BestTowers": int(best_row.get("NumTowers", 0)),
                "BestTrucks": int(best_row.get("NumTrucks", 0)),
                "BestDrones": int(best_row.get("NumDrones", 0)),
                "AvgRestartEvents": float(g.get("RestartEvents", pd.Series([0])).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    full = summary[summary["Variant"] == "FULL"][["Radius", "Instance", "Best", "Avg", "CPU"]].rename(
        columns={"Best": "FullBest", "Avg": "FullAvg", "CPU": "FullCPU"}
    )
    summary = summary.merge(full, on=["Radius", "Instance"], how="left")
    summary["GapBest_vs_FULL(%)"] = 100.0 * (summary["Best"] - summary["FullBest"]) / summary["FullBest"]
    summary["GapAvg_vs_FULL(%)"] = 100.0 * (summary["Avg"] - summary["FullAvg"]) / summary["FullAvg"]
    summary["CPUChange_vs_FULL(%)"] = 100.0 * (summary["CPU"] - summary["FullCPU"]) / summary["FullCPU"]
    return summary


def paired_comparison(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    full = raw_df[raw_df["Variant"] == "FULL"][["Radius", "Instance", "Run", "Seed", "TotalCost"]].rename(
        columns={"TotalCost": "FullCost"}
    )
    merged = raw_df.merge(full, on=["Radius", "Instance", "Run", "Seed"], how="inner")
    merged = merged[merged["Variant"] != "FULL"].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["PairedGap(%)"] = 100.0 * (merged["TotalCost"] - merged["FullCost"]) / merged["FullCost"]
    merged["Outcome"] = np.where(
        merged["TotalCost"] < merged["FullCost"] - 1e-7,
        "Win",
        np.where(merged["TotalCost"] > merged["FullCost"] + 1e-7, "Loss", "Tie"),
    )
    rows = []
    for (radius, instance, variant), g in merged.groupby(["Radius", "Instance", "Variant"], sort=False):
        rows.append(
            {
                "Radius": radius,
                "Instance": instance,
                "Variant": variant,
                "Pairs": len(g),
                "MeanPairedGap(%)": g["PairedGap(%)"].mean(),
                "MedianPairedGap(%)": g["PairedGap(%)"].median(),
                "Win": int((g["Outcome"] == "Win").sum()),
                "Tie": int((g["Outcome"] == "Tie").sum()),
                "Loss": int((g["Outcome"] == "Loss").sum()),
            }
        )
    return pd.DataFrame(rows)


def overall_module_effect(summary_df: pd.DataFrame, paired_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    sub = summary_df[summary_df["Variant"] != "FULL"].copy()
    rows = []
    for (radius, variant), g in sub.groupby(["Radius", "Variant"], sort=False):
        paired_g = paired_df[(paired_df["Radius"] == radius) & (paired_df["Variant"] == variant)] if not paired_df.empty else pd.DataFrame()
        rows.append(
            {
                "Radius": radius,
                "Variant": variant,
                "VariantName": g["VariantName"].iloc[0],
                "Instances": len(g),
                "MeanGapBest_vs_FULL(%)": g["GapBest_vs_FULL(%)"].mean(),
                "MeanGapAvg_vs_FULL(%)": g["GapAvg_vs_FULL(%)"].mean(),
                "MedianGapAvg_vs_FULL(%)": g["GapAvg_vs_FULL(%)"].median(),
                "MeanCV(%)": g["CV(%)"].mean(),
                "MeanCPUChange_vs_FULL(%)": g["CPUChange_vs_FULL(%)"].mean(),
                "InstancesWorseOnAvg": int((g["GapAvg_vs_FULL(%)"] > 1e-7).sum()),
                "InstancesBetterOnAvg": int((g["GapAvg_vs_FULL(%)"] < -1e-7).sum()),
                "MeanPairedGap(%)": paired_g["MeanPairedGap(%)"].mean() if not paired_g.empty else np.nan,
                "TotalWins": int(paired_g["Win"].sum()) if not paired_g.empty else 0,
                "TotalTies": int(paired_g["Tie"].sum()) if not paired_g.empty else 0,
                "TotalLosses": int(paired_g["Loss"].sum()) if not paired_g.empty else 0,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Radius", "MeanGapAvg_vs_FULL(%)"], ascending=[True, False])
    return out


def plot_ablation_gap(summary_df: pd.DataFrame, out_path: str, radius: float) -> None:
    data = summary_df[(summary_df["Radius"] == radius) & (summary_df["Variant"] != "FULL")].copy()
    if data.empty:
        return
    instances = list(dict.fromkeys(data["Instance"].tolist()))
    x = np.arange(len(instances))
    fig, ax = plt.subplots(figsize=(16, 7))
    for variant, g in data.groupby("Variant", sort=False):
        mapping = dict(zip(g["Instance"], g["GapAvg_vs_FULL(%)"]))
        y = [mapping.get(inst, np.nan) for inst in instances]
        name = g["VariantName"].iloc[0]
        ax.plot(x, y, marker="o", linewidth=1.7, label=f"{variant}：{name}")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=35, ha="right")
    ax.set_xlabel("算例编号", labelpad=12)
    ax.set_ylabel("相对完整IALNS的平均成本退化率（%）")
    ax.set_title(f"IALNS关键模块消融结果（瞭望塔覆盖半径={radius:g} km）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_effect(overall_df: pd.DataFrame, out_path: str, radius: float) -> None:
    if overall_df.empty or "Radius" not in overall_df.columns:
        return
    data = overall_df[overall_df["Radius"] == radius].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{r.Variant}\n{r.VariantName}" for r in data.itertuples()]
    values = data["MeanGapAvg_vs_FULL(%)"].to_numpy()
    bars = ax.bar(np.arange(len(data)), values)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("消融变体", labelpad=12)
    ax.set_ylabel("16个算例平均成本退化率（%）")
    ax.set_title(f"IALNS改进模块总体贡献（覆盖半径={radius:g} km）")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variant_convergence(
    histories: Dict[str, List[Dict[str, Any]]],
    variants: Dict[str, AblationVariant],
    out_path: str,
    instance_name: str,
    radius: float,
) -> None:
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for code, hist in histories.items():
        if not hist:
            continue
        x = [r["iteration"] for r in hist]
        y = [r["best_cost"] for r in hist]
        ax.plot(x, y, linewidth=1.8, label=f"{code}：{variants[code].name_cn}")
    ax.set_xlabel("迭代次数", labelpad=10)
    ax.set_ylabel("历史最优总成本")
    ax.set_title(f"{instance_name}消融变体收敛曲线（R={radius:g} km）")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_experiment_definition(out_root: str, variant_codes: List[str], args: argparse.Namespace) -> None:
    definition = {
        "base_dir": args.base_dir,
        "instances": args.instance if args.instance else INSTANCE_NAMES,
        "radii": parse_radii(args.radius),
        "runs": args.runs,
        "iterations": args.iterations,
        "seed_base": args.seed_base,
        "variants": {
            code: {
                "name": ABLATION_VARIANTS[code].name_cn,
                "description": ABLATION_VARIANTS[code].description,
                "use_problem_dr": ABLATION_VARIANTS[code].use_problem_dr,
                "local_search_mode": ABLATION_VARIANTS[code].local_search_mode,
                "use_restart": ABLATION_VARIANTS[code].use_restart,
                "use_adaptive_weights": ABLATION_VARIANTS[code].use_adaptive_weights,
            }
            for code in variant_codes
        },
        "taguchi_parameters": {
            "initial_temperature": DEFAULT_INIT_TEMP,
            "cooling_rate": DEFAULT_COOLING,
            "minimum_temperature": DEFAULT_MIN_TEMP,
            "reaction_factor": WEIGHT_REACTION,
            "weight_update_interval": WEIGHT_UPDATE_INTERVAL,
            "local_search_period": LOCAL_SEARCH_PERIOD,
            "local_search_cost_ratio": LOCAL_SEARCH_COST_RATIO,
            "random_destroy_ratio": RANDOM_DESTROY_RATIO,
            "worst_destroy_ratio": WORST_DESTROY_RATIO,
        },
    }
    with open(os.path.join(out_root, "消融实验定义.json"), "w", encoding="utf-8") as f:
        json.dump(definition, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, "消融实验说明.txt"), "w", encoding="utf-8") as f:
        f.write("TLTDRP—IALNS 消融实验说明\n")
        f.write("=" * 100 + "\n")
        f.write("所有变体使用相同初始解、相同田口最优参数、相同迭代预算和相同随机种子。\n")
        f.write("主判断指标为Avg与相对FULL的GapAvg；Best用于反映寻优上限，Std/CV用于反映稳定性。\n")
        f.write("本最终版本不包含停滞重启机制。Gap为正表示移除该模块后成本升高，即该模块具有正向贡献。\n\n")
        for code in variant_codes:
            v = ABLATION_VARIANTS[code]
            f.write(f"{code}（{v.name_cn}）：{v.description}\n")


def export_all_results(out_root: str, raw_df: pd.DataFrame) -> None:
    raw_csv = os.path.join(out_root, "消融实验_原始运行记录.csv")
    raw_xlsx = os.path.join(out_root, "消融实验_原始运行记录.xlsx")
    summary = summarize_records(raw_df.copy())
    paired = paired_comparison(raw_df.copy())
    overall = overall_module_effect(summary, paired)

    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(out_root, "消融实验_算例变体汇总.csv"), index=False, encoding="utf-8-sig")
    paired.to_csv(os.path.join(out_root, "消融实验_同随机种子配对比较.csv"), index=False, encoding="utf-8-sig")
    overall.to_csv(os.path.join(out_root, "消融实验_模块总体贡献.csv"), index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(os.path.join(out_root, "消融实验_总结果.xlsx"), engine="openpyxl") as writer:
            raw_df.to_excel(writer, sheet_name="原始运行记录", index=False)
            summary.to_excel(writer, sheet_name="算例变体汇总", index=False)
            paired.to_excel(writer, sheet_name="配对比较", index=False)
            overall.to_excel(writer, sheet_name="模块总体贡献", index=False)
        raw_df.to_excel(raw_xlsx, index=False)
    except Exception as exc:
        print(f"警告：Excel结果保存失败：{exc}")

    for radius in sorted(summary["Radius"].dropna().unique()) if not summary.empty else []:
        plot_ablation_gap(summary, os.path.join(out_root, f"消融实验_R{radius:g}_各算例退化率.png"), float(radius))
        plot_overall_effect(overall, os.path.join(out_root, f"消融实验_R{radius:g}_模块总体贡献.png"), float(radius))


def run_ablation_experiment(args: argparse.Namespace) -> pd.DataFrame:
    global TOWER_RADIUS

    variant_codes = parse_variant_codes(args.variant)
    variants = {code: ABLATION_VARIANTS[code] for code in variant_codes}
    radii = parse_radii(args.radius)
    instances = args.instance if args.instance else list(INSTANCE_NAMES)
    output_root = args.output_dir
    if not os.path.isabs(output_root):
        output_root = os.path.join(script_dir(), output_root)
    output_root = ensure_dir(output_root)
    save_experiment_definition(output_root, variant_codes, args)

    master_raw_path = os.path.join(output_root, "消融实验_原始运行记录.csv")
    existing_df = load_existing_records(master_raw_path) if args.resume else pd.DataFrame()
    done_keys: Set[Tuple[Any, ...]] = set()
    if not existing_df.empty:
        for row in existing_df.itertuples(index=False):
            done_keys.add(record_key(row.Radius, row.Instance, row.Variant, row.Run, row.Seed))

    all_new_records: List[Dict[str, Any]] = []
    total_jobs = len(radii) * len(instances) * len(variant_codes) * args.runs
    job_counter = 0

    for radius in radii:
        TOWER_RADIUS = float(radius)
        radius_dir = ensure_dir(os.path.join(output_root, f"R{radius:g}"))
        for instance_idx, instance_name in enumerate(instances, start=1):
            data = read_instance(args.base_dir, instance_name)
            instance_dir = ensure_dir(os.path.join(radius_dir, safe_name(instance_name)))
            save_distance_matrices(data, instance_dir)
            best_histories: Dict[str, List[Dict[str, Any]]] = {}

            print("\n" + "=" * 110)
            print(
                f"半径 {radius:g} km | 算例 [{instance_idx}/{len(instances)}] {instance_name} | "
                f"N={len(data.n)}, L={len(data.towers)}"
            )
            print("=" * 110)

            for code in variant_codes:
                variant = variants[code]
                variant_dir = ensure_dir(os.path.join(instance_dir, code))
                current_records: List[Dict[str, Any]] = []
                best_solution: Optional[Solution] = None
                best_meta: Optional[Dict[str, Any]] = None
                best_cost = float("inf")

                for run_id in range(1, args.runs + 1):
                    job_counter += 1
                    seed = args.seed_base + run_id
                    key = record_key(radius, instance_name, code, run_id, seed)
                    if key in done_keys:
                        if not args.quiet:
                            print(f"[{job_counter}/{total_jobs}] {code} Run {run_id}: 已存在，跳过")
                        continue

                    if not args.quiet:
                        print(
                            f"[{job_counter}/{total_jobs}] {code}（{variant.name_cn}） "
                            f"Run {run_id}/{args.runs}, seed={seed}"
                        )
                    solver = IALNS(data, variant=variant, max_iterations=args.iterations)
                    sol, meta = solver.solve(seed=seed, verbose=args.verbose_iterations)
                    record = make_run_record(radius, instance_name, variant, run_id, seed, sol, meta, data)
                    current_records.append(record)
                    all_new_records.append(record)
                    append_record_csv(master_raw_path, record)
                    done_keys.add(key)

                    if sol.cost + EPS < best_cost:
                        best_cost = sol.cost
                        best_solution = sol.copy()
                        best_meta = meta

                # 读取该组合全部记录（含断点续跑前的记录）并确定最佳随机种子。
                temp_df = load_existing_records(master_raw_path)
                combo = temp_df[
                    (pd.to_numeric(temp_df["Radius"], errors="coerce") == float(radius))
                    & (temp_df["Instance"] == instance_name)
                    & (temp_df["Variant"] == code)
                ].copy()
                if combo.empty:
                    raise RuntimeError(f"{instance_name}-{code} 未获得运行记录")
                combo["TotalCost"] = pd.to_numeric(combo["TotalCost"], errors="coerce")
                combo["CPU"] = pd.to_numeric(combo["CPU"], errors="coerce")
                combo.to_csv(os.path.join(variant_dir, "独立运行记录.csv"), index=False, encoding="utf-8-sig")
                try:
                    combo.to_excel(os.path.join(variant_dir, "独立运行记录.xlsx"), index=False)
                except Exception:
                    pass

                best_row = combo.loc[combo["TotalCost"].idxmin()]
                best_seed = int(best_row["Seed"])

                # 中断续跑时可能没有内存中的最优解；此时仅为输出方案而复现最优seed。
                if best_solution is None or best_meta is None or abs(best_solution.cost - float(best_row["TotalCost"])) > 1e-6:
                    solver = IALNS(data, variant=variant, max_iterations=args.iterations)
                    best_solution, best_meta = solver.solve(seed=best_seed, verbose=False)

                best_histories[code] = best_meta["iteration_history"]
                if not args.skip_detail:
                    save_solution_tables(best_solution, data, variant_dir)
                    save_iteration_history(best_meta, variant_dir)
                    save_operator_stats(best_meta, variant_dir)
                    visualize_solution(
                        best_solution,
                        data,
                        os.path.join(variant_dir, "最优解可视化.png"),
                        title=f"{variant.name_cn}最优方案（{instance_name}, R={radius:g} km）",
                    )
                    plot_iteration_curve(
                        best_meta["iteration_history"],
                        os.path.join(variant_dir, "最优解迭代曲线.png"),
                        os.path.join(variant_dir, "最优解算子使用情况.png"),
                        title_prefix=f"{code}（{variant.name_cn}）",
                    )
                    save_detailed_report(
                        best_solution,
                        data,
                        best_meta,
                        {
                            "best_cost": float(combo["TotalCost"].min()),
                            "avg_cost": float(combo["TotalCost"].mean()),
                            "worst_cost": float(combo["TotalCost"].max()),
                            "avg_cpu": float(combo["CPU"].mean()),
                            "elapsed_time": float(best_row["CPU"]),
                        },
                        os.path.join(variant_dir, "最优解详细报告.txt"),
                    )
                    with open(os.path.join(variant_dir, "变体定义.txt"), "w", encoding="utf-8") as f:
                        f.write(f"代码：{code}\n名称：{variant.name_cn}\n定义：{variant.description}\n")

            plot_variant_convergence(
                best_histories,
                variants,
                os.path.join(instance_dir, "各消融变体最优运行收敛曲线.png"),
                instance_name,
                radius,
            )
            # 每完成一个算例即刷新总结果，降低意外中断损失。
            export_all_results(output_root, load_existing_records(master_raw_path))

    final_df = load_existing_records(master_raw_path)
    export_all_results(output_root, final_df)
    return final_df


def source_file(name, base):
    matches=sorted(Path(base).rglob(name + "*.xlsx"), key=lambda p:(p.stem != name, len(p.parts), str(p)))
    matches=[p for p in matches if not p.name.startswith("~$")]
    if not matches:
        raise FileNotFoundError(f"{name} 未找到Excel文件，请检查 --base-dir={base}。请确认文件名与算例清单一致。")
    return matches[0]


def load_instance(name, base):
    global find_excel_for_instance
    file_path=source_file(name,base)
    original=find_excel_for_instance
    try:
        find_excel_for_instance=lambda _base,_name: str(file_path)
        return read_instance(str(base),name)
    finally:
        find_excel_for_instance=original


# =============================================================================
# 最终版：无重启机制的批量消融实验 + 迭代收敛曲线 + 论文表5
# =============================================================================
def _save_curve_formats(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def _plot_run_curve(history, path_stem: Path, title: str):
    if not history:
        return
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['iteration'], df['current_cost'], lw=0.9, alpha=0.45, label='当前解成本')
    ax.plot(df['iteration'], df['best_cost'], lw=2.0, label='历史最优成本')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('总成本（美元）')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    _save_curve_formats(fig, path_stem)


def _plot_mean_best_curve(histories, path_stem: Path, title: str):
    """对同一算例、同一变体的多次运行按迭代位置求历史最优成本均值。"""
    if not histories:
        return
    arrays=[]
    for hist in histories:
        if hist:
            d=pd.DataFrame(hist).sort_values('iteration')
            arrays.append(d[['iteration','best_cost']].set_index('iteration')['best_cost'])
    if not arrays:
        return
    mat=pd.concat(arrays,axis=1)
    mean=mat.mean(axis=1)
    std=mat.std(axis=1).fillna(0)
    fig,ax=plt.subplots(figsize=(10,6))
    x=mean.index.to_numpy(); y=mean.to_numpy(); e=std.to_numpy()
    ax.plot(x,y,lw=2.0,label='多次运行历史最优成本均值')
    ax.fill_between(x,y-e,y+e,alpha=0.18,label='±1个标准差')
    ax.set_xlabel('迭代次数');ax.set_ylabel('历史最优成本（美元）')
    ax.set_title(title);ax.grid(alpha=.25);ax.legend()
    _save_curve_formats(fig,path_stem)


def _plot_three_variant_curve(histories_by_variant, variants, path_stem: Path, title: str):
    fig,ax=plt.subplots(figsize=(11,6))
    drew=False
    for code,histories in histories_by_variant.items():
        arrays=[]
        for hist in histories:
            if hist:
                d=pd.DataFrame(hist).sort_values('iteration')
                arrays.append(d[['iteration','best_cost']].set_index('iteration')['best_cost'])
        if not arrays: continue
        mean=pd.concat(arrays,axis=1).mean(axis=1)
        ax.plot(mean.index,mean.values,lw=2,label=f'{code}：{variants[code].name_cn}')
        drew=True
    if not drew:
        plt.close(fig);return
    ax.set_xlabel('迭代次数');ax.set_ylabel('历史最优成本（美元）')
    ax.set_title(title);ax.grid(alpha=.25);ax.legend(loc='best')
    _save_curve_formats(fig,path_stem)


def _make_final_table(raw: pd.DataFrame, out: Path, instances, runs: int):
    if raw.empty: raise RuntimeError('没有运行记录，无法生成论文表5')
    raw=raw.copy()
    for c in ['TotalCost','CPU','Feasible','Run','Seed']:
        raw[c]=pd.to_numeric(raw[c],errors='coerce')
    raw=raw[(raw['Feasible']==1)&raw['TotalCost'].notna()]
    groups=[]
    for (instance,variant),g in raw.groupby(['Instance','Variant'],sort=False):
        costs=g['TotalCost'];best_row=g.loc[costs.idxmin()]
        groups.append({'Instance':instance,'Variant':variant,'Runs':int(len(g)),
            'Best':float(costs.min()),'Avg':float(costs.mean()),'Worst':float(costs.max()),
            'Std':float(costs.std(ddof=1)) if len(g)>1 else 0.0,
            'CV(%)':float(costs.std(ddof=1)/costs.mean()*100) if len(g)>1 and costs.mean()!=0 else 0.0,
            'CPU':float(g['CPU'].mean()),'BestRun':int(best_row['Run']),
            'BestSeed':int(best_row['Seed']),'FeasibleRuns':int(g['Feasible'].sum())})
    summary=pd.DataFrame(groups)
    summary.to_csv(out/'最终消融实验_算例变体汇总.csv',index=False,encoding='utf-8-sig')
    # 长表：每算例×变体；宽表：论文表5格式
    metrics=['Best','Avg','CPU']
    wide=summary.pivot(index='Instance',columns='Variant',values=metrics)
    order=['NO_PD_DR','NO_PD_LS','FULL']
    wanted=pd.MultiIndex.from_product([metrics,order])
    wide=wide.reindex(columns=wanted)
    wide.columns=[f'{m}_{v}' for m,v in wide.columns]
    wide=wide.reset_index()
    # 只有完整包含用户指定15个仿真算例且每组运行完整时，才输出‘15个仿真算例平均’。
    sim=summary[summary['Instance'].isin(SIMULATION_INSTANCE_NAMES)]
    included_sim=set(sim['Instance'].unique())
    full_15=(set(SIMULATION_INSTANCE_NAMES).issubset(set(instances)) and
             included_sim.issuperset(set(SIMULATION_INSTANCE_NAMES)))
    mean_rows=[]
    if full_15:
        for variant in DEFAULT_VARIANT_CODES:
            g=sim[sim['Variant']==variant]
            complete=(g['Instance'].nunique()==15 and len(g)==15 and (g['Runs']==runs).all())
            mean_rows.append({'Instance':'15个仿真算例平均','Variant':variant,
                'InstanceCount':int(g['Instance'].nunique()),'MinRuns':int(g['Runs'].min()) if not g.empty else 0,
                'Complete':bool(complete),
                'Best':float(g['Best'].mean()) if complete else np.nan,
                'Avg':float(g['Avg'].mean()) if complete else np.nan,
                'CPU':float(g['CPU'].mean()) if complete else np.nan})
    means=pd.DataFrame(mean_rows)
    means.to_csv(out/'最终消融实验_15个仿真算例均值.csv',index=False,encoding='utf-8-sig')
    if not means.empty:
        means_w=means.pivot(index='Instance',columns='Variant',values=['Best','Avg','CPU'])
        means_w.columns=[f'{m}_{v}' for m,v in means_w.columns]
        means_w=means_w.reset_index()
        wide_all=pd.concat([wide,means_w],ignore_index=True,sort=False)
    else:
        wide_all=wide
    wide_all.to_csv(out/'表5_最终无重启消融实验_论文宽表.csv',index=False,encoding='utf-8-sig')
    real=summary[summary['Instance'].isin(REAL_INSTANCE_NAMES)]
    real.to_csv(out/'两个真实案例_单独结果.csv',index=False,encoding='utf-8-sig')
    # paired same-seed comparison relative to FULL
    pairs=[]
    for variant in ['NO_PD_DR','NO_PD_LS']:
        a=raw[raw.Variant=='FULL'][['Instance','Run','Seed','TotalCost']].rename(columns={'TotalCost':'FULL_Cost'})
        b=raw[raw.Variant==variant][['Instance','Run','Seed','TotalCost']].rename(columns={'TotalCost':'Variant_Cost'})
        p=b.merge(a,on=['Instance','Run','Seed'],how='inner')
        p['ComparedVariant']=variant
        p['Gap_vs_FULL']=p['Variant_Cost']-p['FULL_Cost']
        p['Gap(%)']=np.where(p['FULL_Cost'].abs()>1e-12,p['Gap_vs_FULL']/p['FULL_Cost']*100,np.nan)
        p['Outcome']=np.select([p['Gap_vs_FULL']>1e-8,p['Gap_vs_FULL']<-1e-8],['移除模块后变差','移除模块后变好'],default='持平')
        pairs.append(p)
    pair_df=pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()
    pair_df.to_csv(out/'最终消融实验_同随机种子配对.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(out/'最终消融实验_总结果.xlsx',engine='openpyxl') as w:
        raw.to_excel(w,sheet_name='原始运行记录',index=False)
        summary.to_excel(w,sheet_name='算例变体汇总',index=False)
        wide_all.to_excel(w,sheet_name='表5论文宽表',index=False)
        means.to_excel(w,sheet_name='15例均值',index=False)
        real.to_excel(w,sheet_name='真实案例',index=False)
        pair_df.to_excel(w,sheet_name='同种子配对',index=False)
    # 检查实验完整性
    checks=[]
    for instance in instances:
        for variant in DEFAULT_VARIANT_CODES:
            g=raw[(raw.Instance==instance)&(raw.Variant==variant)]
            checks.append({'Instance':instance,'Variant':variant,'Runs':int(g.Run.nunique()),
                           'FeasibleRuns':int(g.Feasible.sum()) if not g.empty else 0,
                           'Complete':bool(g.Run.nunique()==runs and g.Feasible.sum()==runs)})
    pd.DataFrame(checks).to_csv(out/'实验完整性检查.csv',index=False,encoding='utf-8-sig')
    return summary,wide_all,means,pair_df


def _resolve_scope(scope, custom_instances=None):
    if custom_instances:
        return list(custom_instances)
    if scope=='simulation': return list(SIMULATION_INSTANCE_NAMES)
    if scope=='real': return list(REAL_INSTANCE_NAMES)
    if scope=='all': return list(SIMULATION_INSTANCE_NAMES)+list(REAL_INSTANCE_NAMES)
    raise ValueError('scope必须为simulation、real或all')


def run_final_ablation():
    parser=argparse.ArgumentParser(description='IALNS最终版无重启消融实验：FULL/NO_PD_DR/NO_PD_LS')
    parser.add_argument('--base-dir',default=DEFAULT_BASE_DIR,help='算例Excel根目录，默认E:\\森林火灾巡检研究\\Code\\instances-2（递归查找各算例子文件夹）')
    parser.add_argument('--output-dir',default='IALNS_Taguchi最优参数_无重启消融实验结果')
    parser.add_argument('--scope',choices=['simulation','real','all'],default='simulation',help='批量求解15个仿真算例、2个真实案例或全部17个')
    parser.add_argument('--instance',action='append',help='指定算例名称，可重复传入；优先于scope')
    parser.add_argument('--runs',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=None,help='可选覆盖JSON迭代次数；正式实验建议不指定，默认500')
    parser.add_argument('--seed-base',type=int,default=1000)
    parser.add_argument('--radius',type=float,default=10.0)
    parser.add_argument('--truck-range',type=float,default=300.0,help='卡车最大续航km，建议与田口调优时保持一致（论文通常为250）')
    parser.add_argument('--drone-range',type=float,default=50.0)
    parser.add_argument('--parameter-file',default=None,help='最新Taguchi_最终最优参数.json完整路径')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--quick',action='store_true',help='快速自检：2次运行、40轮迭代')
    parser.add_argument('--skip-plots',action='store_true',help='仅输出统计表，不生成方案图及迭代曲线')
    parser.add_argument('--show-best',action='store_true')
    parser.add_argument('--verbose-iterations',action='store_true')
    args=parser.parse_args()
    # JSON参数是唯一的算法参数来源；默认附带本次上传L27获胜参数。
    embedded_path=Path(__file__).resolve().parent/'Taguchi_最终最优参数.json'
    param_path=Path(args.parameter_file).expanduser().resolve() if args.parameter_file else embedded_path
    if args.parameter_file and not param_path.is_file():
        parser.error(f'找不到指定的Taguchi参数JSON: {param_path}')
    taguchi=load_taguchi_parameters(param_path) if param_path.exists() else DEFAULT_IALNS_PARAMETERS
    if not param_path.exists():
        print('[提示] 同目录未找到Taguchi_最终最优参数.json；使用代码中已冻结的本次L27获胜参数')
    args.iterations=taguchi.max_iterations if args.iterations is None else args.iterations
    if args.quick:args.runs=2;args.iterations=40
    if args.runs<1 or args.iterations<1 or args.truck_range<=0 or args.drone_range<=0:
        parser.error('运行次数、迭代次数和续航参数必须为正数')
    instances=_resolve_scope(args.scope,args.instance)
    if not instances: parser.error('没有选择算例')
    base=Path(args.base_dir).resolve()
    global TOWER_RADIUS, TRUCK_MAX_RANGE, DRONE_MAX_RANGE
    TOWER_RADIUS=float(args.radius)
    TRUCK_MAX_RANGE=float(args.truck_range)
    DRONE_MAX_RANGE=float(args.drone_range)
    # 正式批量运行前预检查所有Excel，避免算到中途才发现某个算例文件缺失。
    missing=[]
    for case_name in instances:
        try:
            source_file(case_name,base)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        parser.error('以下算例Excel未找到，实验尚未开始：\n'+'\n'.join(missing))
    out=Path(args.output_dir)
    if not out.is_absolute(): out=Path(__file__).resolve().parent/out
    out.mkdir(parents=True,exist_ok=True)
    # 版本配置防止把旧版带重启结果混入最终无重启实验
    input_hashes={name:hashlib.sha256(source_file(name,base).read_bytes()).hexdigest() for name in instances}
    config={'version':ENGINE_VERSION,'taguchi_version':TAGUCHI_VERSION,'taguchi_source':TAGUCHI_SOURCE,
            'taguchi_parameters':asdict(taguchi), 'parameter_file':str(param_path) if param_path.exists() else 'frozen-embedded',
            'instances':instances, 'input_sha256':input_hashes,
            'runs':args.runs,'iterations':args.iterations,'seed_base':args.seed_base,
            'radius':args.radius,'truck_range':args.truck_range,'drone_range':args.drone_range,
            'variants':DEFAULT_VARIANT_CODES}
    cfg=out/'最终实验配置.json';raw_path=out/'最终消融实验_原始运行记录.csv'
    if raw_path.exists() and not args.resume:
        parser.error('输出目录已有结果。若继续同一配置请加--resume；否则请更换--output-dir。')
    if args.resume and raw_path.exists():
        if not cfg.exists() or json.loads(cfg.read_text(encoding='utf-8'))!=config:
            parser.error('已有结果与当前实验配置不一致，禁止混合，请使用新输出目录。')
        records=pd.read_csv(raw_path,encoding='utf-8-sig').to_dict('records')
    else: records=[]
    cfg.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'Taguchi_最终冻结参数_消融实验.json').write_text(
         json.dumps({'source':str(param_path) if param_path.exists() else 'embedded',
                     'algorithm':'IALNS_FULL_NO_RESTART','parameters':asdict(taguchi)},ensure_ascii=False,indent=2),encoding='utf-8')
    done={(str(r['Instance']),str(r['Variant']),int(r['Run']),int(r['Seed'])) for r in records}
    variants={c:ABLATION_VARIANTS[c] for c in DEFAULT_VARIANT_CODES}
    setup_chinese_fonts()
    total=len(instances)*len(DEFAULT_VARIANT_CODES)*args.runs
    job=0
    print('='*100)
    print('IALNS最终版无重启消融实验')
    print('最终算法：不包含任何停滞重启、精英扰动重启或温度恢复操作')
    print(f'算例范围={args.scope} | 算例数={len(instances)} | 变体={DEFAULT_VARIANT_CODES}')
    print(f'计划运行={total} | 每组运行={args.runs} | 每次迭代={args.iterations} | 覆盖半径={TOWER_RADIUS:g} km | 卡车续航={TRUCK_MAX_RANGE:g}km')
    print(f'Taguchi参数: {asdict(taguchi)}')
    print('='*100)
    for idx,name in enumerate(instances,1):
        data=load_instance(name,str(base))
        case_dir=out/safe_name(name);case_dir.mkdir(parents=True,exist_ok=True)
        save_distance_matrices(data,str(case_dir))
        print(f'\n[{idx}/{len(instances)}] {name}: 监测点={len(data.n)}, 候选塔={len(data.towers)}',flush=True)
        histories_by_variant={}
        for code in DEFAULT_VARIANT_CODES:
            variant=variants[code]
            vdir=case_dir/code;vdir.mkdir(parents=True,exist_ok=True)
            combo_records=[r for r in records if r['Instance']==name and r['Variant']==code]
            run_histories=[];best_sol=None;best_meta=None;best_cost=float('inf');best_run=None
            for run_id in range(1,args.runs+1):
                job+=1;seed=args.seed_base+run_id
                if (name,code,run_id,seed) in done:
                    # 续跑时从磁盘读取该次迭代历史；若缺失则重新复现该seed
                    hpath=vdir/f'迭代历史_Run{run_id:02d}.csv'
                    if hpath.exists():
                        hist=pd.read_csv(hpath).to_dict('records');run_histories.append(hist)
                        rr=[r for r in combo_records if int(r['Run'])==run_id]
                        if rr and float(rr[0]['TotalCost'])<best_cost:
                            best_cost=float(rr[0]['TotalCost']);best_run=run_id
                        continue
                solver=IALNS(data,variant=variant,max_iterations=args.iterations,parameters=taguchi)
                sol,meta=solver.solve(seed=seed,verbose=args.verbose_iterations)
                evaluate_solution(sol,data)
                if not sol.feasible:
                    print(f'警告：{name}/{code}/Run{run_id}未得到可行解',flush=True)
                rec=make_run_record(float(args.radius),name,variant,run_id,seed,sol,meta,data)
                rec['RestartEvents']=0
                records.append(rec);combo_records.append(rec);done.add((name,code,run_id,seed))
                pd.DataFrame(records).to_csv(raw_path,index=False,encoding='utf-8-sig')
                hist=meta['iteration_history'];run_histories.append(hist)
                pd.DataFrame(hist).to_csv(vdir/f'迭代历史_Run{run_id:02d}.csv',index=False,encoding='utf-8-sig')
                if not args.skip_plots:
                    _plot_run_curve(hist,vdir/f'迭代曲线_Run{run_id:02d}',f'{name} | {code} | Run {run_id} | 当前解与历史最优解')
                print(f'  [{job}/{total}] {code} Run={run_id:02d}/{args.runs} Seed={seed} '
                      f'Best={sol.cost:,.2f} CPU={meta["elapsed_time"]:.2f}s 重启=0',flush=True)
                if sol.feasible and sol.cost<best_cost:
                    best_cost=sol.cost;best_sol=sol.copy();best_meta=meta;best_run=run_id
            # 以该变体全部已完成运行记录确定真正Best；若内存中的解不是Best则按BestSeed复现。
            valid=[r for r in combo_records if int(r.get('Feasible',0))==1]
            if valid:
                bestrow=min(valid,key=lambda x:float(x['TotalCost']))
                best_run=int(bestrow['Run'])
                if best_sol is None or best_meta is None or abs(best_sol.cost-float(bestrow['TotalCost']))>1e-7:
                    solver=IALNS(data,variant=variant,max_iterations=args.iterations,parameters=taguchi)
                    best_sol,best_meta=solver.solve(seed=int(bestrow['Seed']))
                    evaluate_solution(best_sol,data)
            histories_by_variant[code]=run_histories
            if run_histories and not args.skip_plots:
                _plot_mean_best_curve(run_histories,vdir/'迭代曲线_多次运行均值',f'{name} | {code} | {args.runs}次运行历史最优成本均值')
            if best_sol is not None:
                save_solution_tables(best_sol,data,str(vdir))
                save_iteration_history(best_meta,str(vdir))
                save_operator_stats(best_meta,str(vdir))
                if not args.skip_plots:
                    visualize_solution(best_sol,data,str(vdir/'最优巡检方案.png'),title=f'{name} - {code}（无重启，Taguchi参数）')
                    _plot_run_curve(best_meta['iteration_history'],vdir/'迭代曲线_最佳运行',f'{name} | {code} | 最佳运行Run{best_run}')
                save_detailed_report(best_sol,data,best_meta,{
                    'best_cost':best_cost,
                    'avg_cost':float(np.mean([float(x['TotalCost']) for x in combo_records if int(x.get('Feasible',0))==1])),
                    'worst_cost':float(np.max([float(x['TotalCost']) for x in combo_records if int(x.get('Feasible',0))==1])),
                    'avg_cpu':float(np.mean([float(x['CPU']) for x in combo_records if int(x.get('Feasible',0))==1])),
                    'elapsed_time':best_meta['elapsed_time']},str(vdir/'最佳运行详细报告.txt'))
        if not args.skip_plots:
            _plot_three_variant_curve(histories_by_variant,variants,case_dir/'三种消融变体_迭代收敛曲线',f'{name} | 三种无重启消融变体收敛曲线')
        # 每完成一个算例，立即生成可用的中间统计，避免批量运行中断丢失成果
        temp=pd.DataFrame(records)
        temp.to_csv(raw_path,index=False,encoding='utf-8-sig')
        _make_final_table(temp,out,instances,args.runs)
    raw=pd.DataFrame(records)
    raw.to_csv(raw_path,index=False,encoding='utf-8-sig')
    summary,wide,means,pairs=_make_final_table(raw,out,instances,args.runs)
    print('\n最终消融实验完成。结果目录：',out)
    print('\n15个仿真算例平均结果：')
    print(means.to_string(index=False))


if __name__=='__main__':
    run_final_ablation()
