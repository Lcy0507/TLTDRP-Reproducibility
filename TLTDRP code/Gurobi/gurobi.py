

import os
import re
import math
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.lines as mlines

import gurobipy as gp
from gurobipy import GRB


# ============================================================
# 0) 全局配置
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_INSTANCE_DIR = r"E:\森林火灾巡检研究\Code\instances2"

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

# 年巡逻轮次
F = 360

# 固定成本（年度一次性）
cfW = 150000.0
cfT = 42000.0
cfD = 6000.0

# 运营成本（每轮）
coW = 1.0
coT = 0.5
coD = 0.22

# 能力参数
R = 10.0               # 瞭望塔覆盖半径 (km)
delta = 2              # 单塔无人机最大存放量
beta_max_T = 300.0     # 卡车单次续航 (km)
beta_max_D = 50.0      # 无人机往返续航 (km)
sigma = 1.3            # 卡车绕行系数

# 求解参数
TIME_LIMIT = 3600
MIP_GAP = 1e-6
OUTPUT_FLAG = 1
WRITE_LP = False       # 若想导出每个算例的LP模型文件，可改为 True
USE_GUROBI_LOGFILE = False  # Windows下中文路径常导致 Gurobi 报错 Unable to write file


# ============================================================
# 0) 工具函数
# ============================================================
def safe_name(name: str) -> str:
    """将名称转为安全文件夹名。"""
    return re.sub(r'[\\/:*?"<>|]+', '_', name).strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_ascii_name(name: str) -> str:
    """仅保留 ASCII 安全字符，供 Gurobi 的 write/log 等原生文件写出使用。"""
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._')
    return s or 'instance'


def get_ascii_export_dir() -> str:
    """为 Gurobi 原生写文件提供纯英文目录，避免中文路径报错。"""
    base = os.path.join(os.environ.get('SYSTEMDRIVE', 'C:'), 'gurobi_tmp_exports')
    os.makedirs(base, exist_ok=True)
    return base


def dist(coords: Dict[int, Tuple[float, float]], i: int, j: int) -> float:
    xi, yi = coords[i]
    xj, yj = coords[j]
    return math.hypot(xi - xj, yi - yj)


def truck_dist(coords: Dict[int, Tuple[float, float]], i: int, j: int) -> float:
    return sigma * dist(coords, i, j)


def status_name_from_code(status_code: int) -> str:
    status_codes = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.NUMERIC: "NUMERIC",
    }
    return status_codes.get(status_code, f"UNKNOWN_STATUS({status_code})")


def find_excel_for_instance(base_dir: str, instance_name: str) -> str:
    """
    兼容以下几种存放方式：
    0) E:/.../instances/实例名.xlsx
    2) E:/.../instances/实例名/实例名.xlsx
    3) E:/.../instances/实例名/*.xlsx （目录内仅一个或优先匹配同名）
    """
    direct_xlsx = os.path.join(base_dir, f"{instance_name}.xlsx")
    if os.path.isfile(direct_xlsx):
        return direct_xlsx

    instance_dir = os.path.join(base_dir, instance_name)
    if os.path.isdir(instance_dir):
        exact_inside = os.path.join(instance_dir, f"{instance_name}.xlsx")
        if os.path.isfile(exact_inside):
            return exact_inside

        xlsx_files = [
            os.path.join(instance_dir, f)
            for f in os.listdir(instance_dir)
            if f.lower().endswith(".xlsx") and not f.startswith("~$")
        ]
        if len(xlsx_files) == 1:
            return xlsx_files[0]
        if len(xlsx_files) > 1:
            # 优先找包含实例名的
            preferred = [p for p in xlsx_files if instance_name in os.path.basename(p)]
            if preferred:
                return preferred[0]
            return sorted(xlsx_files)[0]

    raise FileNotFoundError(
        f"未找到算例数据文件：{instance_name}\n"
        f"已尝试路径：\n"
        f"0) {direct_xlsx}\n"
        f"2) {os.path.join(base_dir, instance_name, instance_name + '.xlsx')}\n"
        f"3) {os.path.join(base_dir, instance_name)} 目录下的 .xlsx 文件"
    )


def calculate_potential_trucks(
    N1_list: List[int],
    depot_start_id: int,
    depot_end_id: int,
    coords: Dict[int, Tuple[float, float]],
    max_range: float,
) -> int:
    """仅用卡车覆盖全部 N1 节点时的潜在卡车数量估计。"""
    if not N1_list:
        return 0

    unvisited = set(N1_list)
    truck_count = 0

    while unvisited:
        truck_count += 1
        current_id = depot_start_id
        remaining_range = max_range

        while unvisited:
            nearest_id = None
            nearest_dist = float('inf')
            for node_id in unvisited:
                d = truck_dist(coords, current_id, node_id)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_id = node_id

            if nearest_id is None:
                break

            dist_to_node = truck_dist(coords, current_id, nearest_id)
            dist_back_to_depot = truck_dist(coords, nearest_id, depot_end_id)

            if remaining_range >= dist_to_node + dist_back_to_depot:
                remaining_range -= dist_to_node
                unvisited.remove(nearest_id)
                current_id = nearest_id
            else:
                break

    return truck_count


def X(v: Any) -> float:
    return 0.0 if v is None else v.X


# ============================================================
# 2) 数据准备
# ============================================================
def load_instance_data(excel_path: str) -> Dict[str, Any]:
    df = pd.read_excel(excel_path)

    required_cols = {"node", "x", "y", "Classification"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Excel 缺少必要列：{sorted(missing_cols)}")

    N1 = df.loc[df["Classification"] == "沿线道路监测节点", "node"].astype(int).tolist()
    N2 = df.loc[df["Classification"] == "非沿线道路监测节点", "node"].astype(int).tolist()
    N = N1 + N2

    L = df.loc[df["Classification"] == "候选瞭望塔", "node"].astype(int).tolist()

    DEPOTS = df.loc[df["Classification"] == "消防站", "node"].astype(int).tolist()
    if len(DEPOTS) < 2:
        raise ValueError("消防站节点至少需要2个（起点与终点，坐标可相同）。")

    depot_start = int(DEPOTS[0])
    depot_end = int(DEPOTS[1])

    G = [depot_start] + N1 + [depot_end]
    V = N + L

    coords = {int(r["node"]): (float(r["x"]), float(r["y"])) for _, r in df.iterrows()}

    alpha_max_W = math.ceil(len(L) / 2) if len(L) > 0 else 0
    U = list(range(1, alpha_max_W * 2 + 1))

    potential_trucks = calculate_potential_trucks(N1, depot_start, depot_end, coords, beta_max_T)
    alpha_max_T = math.ceil(potential_trucks / 2) if potential_trucks > 0 else 0
    K = list(range(1, potential_trucks + 1))

    y_cov = {(i, l): 1 if dist(coords, i, l) <= R + 1e-9 else 0 for i in N for l in L}
    dT = {(i, j): truck_dist(coords, i, j) for i in G for j in G if i != j}
    dD = {(i, j): dist(coords, i, j) for i in V for j in V if i != j}

    M_T = len(N1) + 2
    M_D = len(N) + 2

    return {
        "df": df,
        "N1": N1,
        "N2": N2,
        "N": N,
        "L": L,
        "DEPOTS": DEPOTS,
        "depot_start": depot_start,
        "depot_end": depot_end,
        "G": G,
        "V": V,
        "coords": coords,
        "alpha_max_W": alpha_max_W,
        "U": U,
        "potential_trucks": potential_trucks,
        "alpha_max_T": alpha_max_T,
        "K": K,
        "y_cov": y_cov,
        "dT": dT,
        "dD": dD,
        "M_T": M_T,
        "M_D": M_D,
    }


# ============================================================
# 3) 建模与求解
# ============================================================
def build_and_solve_model(instance_name: str, data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    N1 = data["N1"]
    N2 = data["N2"]
    N = data["N"]
    L = data["L"]
    G = data["G"]
    V = data["V"]
    U = data["U"]
    K = data["K"]
    y_cov = data["y_cov"]
    dT = data["dT"]
    dD = data["dD"]
    M_T = data["M_T"]
    M_D = data["M_D"]
    alpha_max_W = data["alpha_max_W"]
    alpha_max_T = data["alpha_max_T"]
    depot_start = data["depot_start"]
    depot_end = data["depot_end"]

    model_name = f"TLTDRP_{safe_name(instance_name)}"
    m = gp.Model(model_name)

    # ----------------------------
    # 3.0 决策变量
    # ----------------------------
    xW = m.addVars(L, vtype=GRB.BINARY, name="xW")
    xT = m.addVars(K, vtype=GRB.BINARY, name="xT")
    xD = m.addVars(U, vtype=GRB.BINARY, name="xD")

    zW = m.addVars(N, L, vtype=GRB.BINARY, name="zW")
    zT = m.addVars([(i, j, k) for i in G for j in G if i != j for k in K], vtype=GRB.BINARY, name="zT")
    zD = m.addVars([(i, j, u) for i in V for j in V if i != j for u in U], vtype=GRB.BINARY, name="zD")

    sT = m.addVars(G, K, vtype=GRB.CONTINUOUS, lb=0.0, name="sT")
    sD = m.addVars(V, U, vtype=GRB.CONTINUOUS, lb=0.0, name="sD")

    # ----------------------------
    # 3.2 目标函数（保持原代码不变）
    # ----------------------------
    fixed_cost = gp.quicksum(cfW * xW[l] for l in L) \
                 + gp.quicksum(cfT * xT[k] for k in K) \
                 + gp.quicksum(cfD * xD[u] for u in U)

    op_cost_per_round = gp.quicksum(coW * zW[i, l] for i in N for l in L) \
                        + gp.quicksum(coT * dT[i, j] * zT[i, j, k] for (i, j, k) in zT.keys()) \
                        + gp.quicksum(coD * dD[i, j] * zD[i, j, u] for (i, j, u) in zD.keys())

    m.setObjective(fixed_cost + F * op_cost_per_round, GRB.MINIMIZE)

    # ============================================================
    # 4) 约束（保持原代码逻辑）
    # ============================================================

    # 式（2）：沿线节点由塔/车/机三者之一巡检
    for i in N1:
        truck_in = gp.quicksum(zT[j, i, k] for k in K for j in G if j != i and (j, i, k) in zT)
        drone_in = gp.quicksum(zD[j, i, u] for u in U for j in V if j != i and (j, i, u) in zD)
        m.addConstr(gp.quicksum(zW[i, l] for l in L) + truck_in + drone_in == 1, name=f"eq2_cover_N1_{i}")

    # 式（3）：非沿线节点由塔/机二者之一巡检
    for i in N2:
        drone_in = gp.quicksum(zD[j, i, u] for u in U for j in V if j != i and (j, i, u) in zD)
        m.addConstr(gp.quicksum(zW[i, l] for l in L) + drone_in == 1, name=f"eq3_cover_N2_{i}")

    # 式（4）：塔巡检 => 在覆盖范围内
    for i in N:
        for l in L:
            m.addConstr(y_cov[i, l] >= zW[i, l], name=f"eq4_cov_{i}_{l}")

    # 式（5）：塔巡检 => 塔建设
    for i in N:
        for l in L:
            m.addConstr(zW[i, l] <= xW[l], name=f"eq5_build_{i}_{l}")

    # 式（6）：建塔数量上限
    m.addConstr(gp.quicksum(xW[l] for l in L) <= alpha_max_W, name="eq6_tower_limit")

    # 式（7）：未使用卡车无弧
    for (i, j, k) in zT.keys():
        m.addConstr(zT[i, j, k] <= xT[k], name=f"eq7_truck_use_{i}_{j}_{k}")

    # 式（8）：非沿线节点不能被卡车访问
    for i in N2:
        if i in G:
            m.addConstr(
                gp.quicksum(zT[j, i, k] for k in K for j in G if j != i and (j, i, k) in zT) == 0,
                name=f"eq8_no_truck_N2_{i}"
            )

    # 新增：卡车禁止巡检已建瞭望塔覆盖范围内的沿线监测节点
    for i in N1:
        for k in K:
            for l in L:
                m.addConstr(
                    gp.quicksum(zT[j, i, k] for j in G if j != i and (j, i, k) in zT) <= 1 - y_cov[i, l] * xW[l],
                    name=f"no_truck_enter_towercov_i{i}_k{k}_l{l}"
                )
                m.addConstr(
                    gp.quicksum(zT[i, j, k] for j in G if j != i and (i, j, k) in zT) <= 1 - y_cov[i, l] * xW[l],
                    name=f"no_truck_leave_towercov_i{i}_k{k}_l{l}"
                )

    # 式（9）：卡车必须从起点出发
    for k in K:
        m.addConstr(
            gp.quicksum(zT[depot_start, j, k] for j in N1 if (depot_start, j, k) in zT) == xT[k],
            name=f"eq9_depart_{k}"
        )

    # 式（20）：卡车必须返回终点
    for k in K:
        m.addConstr(
            gp.quicksum(zT[i, depot_end, k] for i in N1 if (i, depot_end, k) in zT) == xT[k],
            name=f"eq10_return_{k}"
        )

    # 禁止起点直接回终点
    for k in K:
        if (depot_start, depot_end, k) in zT:
            m.addConstr(zT[depot_start, depot_end, k] == 0, name=f"eq11_no_direct_{k}")

    # 卡车流平衡
    for k in K:
        for j in N1:
            in_flow = gp.quicksum(zT[i, j, k] for i in G if i != j and (i, j, k) in zT)
            out_flow = gp.quicksum(zT[j, g, k] for g in G if g != j and (j, g, k) in zT)
            m.addConstr(in_flow == out_flow, name=f"eq12_flow_{j}_{k}")

    # 卡车续航
    for k in K:
        m.addConstr(
            gp.quicksum(dT[i, j] * zT[i, j, k] for i in G for j in G if i != j and (i, j, k) in zT)
            <= beta_max_T * xT[k],
            name=f"eq13_range_{k}"
        )

    # sT[0,k] = 0
    for k in K:
        m.addConstr(sT[depot_start, k] == 0, name=f"eq14_s0_{k}")

    # 卡车 MTZ
    for k in K:
        for i in G:
            for j in G:
                if i != j and i != depot_end and j != depot_start and (i, j, k) in zT:
                    m.addConstr(
                        sT[j, k] >= sT[i, k] + 1 - M_T * (1 - zT[i, j, k]) - M_T * (1 - xT[k]),
                        name=f"eq15_mtz_{i}_{j}_{k}"
                    )

    for k in K:
        for i in G:
            m.addConstr(sT[i, k] >= 0, name=f"eq16_lb_{i}_{k}")
            m.addConstr(sT[i, k] <= len(N1) + 1, name=f"eq16_ub_{i}_{k}")

    # 卡车数量上限
    m.addConstr(gp.quicksum(xT[k] for k in K) <= alpha_max_T, name="eq17_truck_limit")

    # 未使用无人机无弧
    for (i, j, u) in zD.keys():
        m.addConstr(zD[i, j, u] <= xD[u], name=f"eq18_drone_use_{i}_{j}_{u}")

    # 每架无人机只从塔起飞一次
    for u in U:
        m.addConstr(
            gp.quicksum(zD[l, j, u] for l in L for j in N if l != j and (l, j, u) in zD) == xD[u],
            name=f"eq19_takeoff_{u}"
        )

    # 每架无人机只返回塔一次
    for u in U:
        m.addConstr(
            gp.quicksum(zD[i, l, u] for l in L for i in N if i != l and (i, l, u) in zD) == xD[u],
            name=f"eq20_land_{u}"
        )

    # 同塔起降
    for u in U:
        for l in L:
            m.addConstr(
                gp.quicksum(zD[l, j, u] for j in N if l != j and (l, j, u) in zD)
                == gp.quicksum(zD[i, l, u] for i in N if i != l and (i, l, u) in zD),
                name=f"eq21_same_tower_{l}_{u}"
            )

    # 单塔无人机容量
    for l in L:
        m.addConstr(
            gp.quicksum(zD[l, j, u] for u in U for j in N if l != j and (l, j, u) in zD) <= delta * xW[l],
            name=f"eq22_capacity_{l}"
        )

    # 起飞塔必须建设
    for u in U:
        for l in L:
            for j in N:
                if (l, j, u) in zD:
                    m.addConstr(zD[l, j, u] <= xW[l], name=f"eq23_takeoff_need_build_{l}_{j}_{u}")

    # 降落塔必须建设
    for u in U:
        for l in L:
            for i in N:
                if (i, l, u) in zD:
                    m.addConstr(zD[i, l, u] <= xW[l], name=f"eq24_land_need_build_{i}_{l}_{u}")

    # 禁止塔到塔飞行
    for u in U:
        for i in L:
            for j in L:
                if i != j and (i, j, u) in zD:
                    m.addConstr(zD[i, j, u] == 0, name=f"eq25_no_L2L_{i}_{j}_{u}")

    # 无人机流平衡
    for u in U:
        for j in N:
            in_flow = gp.quicksum(zD[i, j, u] for i in V if i != j and (i, j, u) in zD)
            out_flow = gp.quicksum(zD[j, h, u] for h in V if h != j and (j, h, u) in zD)
            m.addConstr(in_flow == out_flow, name=f"eq26_flow_{j}_{u}")

    # 无人机续航
    for u in U:
        m.addConstr(
            gp.quicksum(dD[i, j] * zD[i, j, u] for i in V for j in V if i != j and (i, j, u) in zD)
            <= beta_max_D * xD[u],
            name=f"eq27_range_{u}"
        )

    # 原代码保持
    for u in U:
        for l in L:
            m.addConstr(
                sD[l, u] <= M_D * (1 - gp.quicksum(zD[l, j, u] for j in N if l != j and (l, j, u) in zD)),
                name=f"eq28_anchor_{l}_{u}"
            )

    # 无人机 MTZ
    for u in U:
        for i in V:
            for j in N:
                if i != j and (i, j, u) in zD:
                    m.addConstr(
                        sD[j, u] >= sD[i, u] + 1 - M_D * (1 - zD[i, j, u]) - M_D * (1 - xD[u]),
                        name=f"eq29_mtz_{i}_{j}_{u}"
                    )

    for u in U:
        for i in V:
            m.addConstr(sD[i, u] >= 0, name=f"eq30_lb_{i}_{u}")
            m.addConstr(sD[i, u] <= (len(N) + 1) * xD[u], name=f"eq30_ub_{i}_{u}")

    # ============================================================
    # 5) 求解
    # ============================================================
    m.Params.OutputFlag = OUTPUT_FLAG
    m.Params.TimeLimit = TIME_LIMIT
    m.Params.MIPGap = MIP_GAP

    # 注意：Gurobi 在 Windows 下对包含中文的 LogFile / write 路径兼容性较差，
    # 很容易直接报错：Unable to write file。
    # 因此默认不设置 LogFile；若确实需要，可改到纯英文路径。
    if USE_GUROBI_LOGFILE:
        try:
            ascii_log_dir = get_ascii_export_dir()
            log_path = os.path.join(ascii_log_dir, f"{safe_ascii_name(instance_name)}.log")
            m.Params.LogFile = log_path
        except Exception as e:
            print(f"[警告] 无法创建 Gurobi LogFile，已跳过。原因: {e}")

    if WRITE_LP:
        try:
            ascii_export_dir = get_ascii_export_dir()
            lp_path = os.path.join(ascii_export_dir, f"{safe_ascii_name(instance_name)}.lp")
            m.write(lp_path)
            print(f"[提示] LP 文件已写出到纯英文目录: {lp_path}")
        except Exception as e:
            print(f"[警告] 无法写出 LP 文件，已跳过。原因: {e}")

    m.optimize()

    return {
        "model": m,
        "xW": xW,
        "xT": xT,
        "xD": xD,
        "zW": zW,
        "zT": zT,
        "zD": zD,
        "sT": sT,
        "sD": sD,
    }


# ============================================================
# 4) 结果提取
# ============================================================
def extract_truck_routes(data: Dict[str, Any], vars_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    G = data["G"]
    K = data["K"]
    dT = data["dT"]
    depot_start = data["depot_start"]
    depot_end = data["depot_end"]
    xT = vars_dict["xT"]
    zT = vars_dict["zT"]

    routes = []
    for k in K:
        if X(xT[k]) <= 0.5:
            continue

        route = [depot_start]
        current = depot_start
        route_miles = 0.0
        step_guard = 0

        while current != depot_end and step_guard <= len(G) + 5:
            step_guard += 1
            next_node = None
            for j in G:
                if j != current and (current, j, k) in zT and X(zT[current, j, k]) > 0.5:
                    next_node = j
                    route_miles += dT[current, j]
                    break

            if next_node is None:
                break

            route.append(next_node)
            current = next_node

        routes.append({
            "truck_id": k,
            "route": route,
            "route_text": " -> ".join(map(str, route)),
            "route_node_count": len(route),
            "route_miles": route_miles,
            "is_exceeded": int(route_miles > beta_max_T + 1e-9),
        })

    return routes


def extract_drone_routes(data: Dict[str, Any], vars_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    V = data["V"]
    L = data["L"]
    N = data["N"]
    U = data["U"]
    xD = vars_dict["xD"]
    zD = vars_dict["zD"]
    dD = data["dD"]

    routes = []
    for u in U:
        if X(xD[u]) <= 0.5:
            continue

        start_tower = None
        for l in L:
            found = False
            for j in N:
                if (l, j, u) in zD and X(zD[l, j, u]) > 0.5:
                    start_tower = l
                    found = True
                    break
            if found:
                break

        if start_tower is None:
            routes.append({
                "drone_id": u,
                "start_tower": None,
                "route": [],
                "route_text": "",
                "inspected_node_count": 0,
                "route_miles": 0.0,
                "is_exceeded": 0,
            })
            continue

        route = [start_tower]
        current = start_tower
        route_miles = 0.0
        step_guard = 0

        while step_guard <= len(V) + 5:
            step_guard += 1
            next_node = None
            for j in V:
                if j != current and (current, j, u) in zD and X(zD[current, j, u]) > 0.5:
                    next_node = j
                    route_miles += dD[current, j]
                    break

            if next_node is None:
                break

            route.append(next_node)
            current = next_node
            if current in L:
                break

        routes.append({
            "drone_id": u,
            "start_tower": start_tower,
            "route": route,
            "route_text": " -> ".join(map(str, route)),
            "inspected_node_count": len([n for n in route if n in N]),
            "route_miles": route_miles,
            "is_exceeded": int(route_miles > beta_max_D + 1e-9),
        })

    return routes


def extract_tower_assignments(data: Dict[str, Any], vars_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    N = data["N"]
    L = data["L"]
    coords = data["coords"]
    xW = vars_dict["xW"]
    zW = vars_dict["zW"]

    records = []
    for l in L:
        if X(xW[l]) <= 0.5:
            continue
        assigned_nodes = [i for i in N if X(zW[i, l]) > 0.5]
        for i in assigned_nodes:
            records.append({
                "tower_id": l,
                "tower_x": coords[l][0],
                "tower_y": coords[l][1],
                "node_id": i,
                "node_x": coords[i][0],
                "node_y": coords[i][1],
            })
    return records


def extract_node_assignment_summary(data: Dict[str, Any], vars_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    N = data["N"]
    N1 = set(data["N1"])
    L = data["L"]
    G = data["G"]
    V = data["V"]
    K = data["K"]
    U = data["U"]
    zW = vars_dict["zW"]
    zT = vars_dict["zT"]
    zD = vars_dict["zD"]

    rows = []
    for i in N:
        method = "未识别"
        resource_id = None

        # 塔巡检
        matched = False
        for l in L:
            if X(zW[i, l]) > 0.5:
                method = "瞭望塔"
                resource_id = l
                matched = True
                break
        if not matched and i in N1:
            for k in K:
                for j in G:
                    if j != i and (j, i, k) in zT and X(zT[j, i, k]) > 0.5:
                        method = "卡车"
                        resource_id = k
                        matched = True
                        break
                if matched:
                    break
        if not matched:
            for u in U:
                for j in V:
                    if j != i and (j, i, u) in zD and X(zD[j, i, u]) > 0.5:
                        method = "无人机"
                        resource_id = u
                        matched = True
                        break
                if matched:
                    break

        rows.append({
            "node_id": i,
            "node_type": "沿线道路监测节点" if i in N1 else "非沿线道路监测节点",
            "inspection_method": method,
            "resource_id": resource_id,
        })
    return rows


def generate_distance_matrix_csv(data: Dict[str, Any], output_dir: str) -> str:
    N = data["N"]
    L = data["L"]
    DEPOTS = data["DEPOTS"]
    coords = data["coords"]

    all_nodes = sorted(set(N + L + DEPOTS))
    distance_data = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                distance_data.append({
                    "From_Node": i,
                    "To_Node": j,
                    "From_X": coords[i][0],
                    "From_Y": coords[i][1],
                    "To_X": coords[j][0],
                    "To_Y": coords[j][1],
                    "Euclidean_Distance_km": dist(coords, i, j),
                    "Truck_Road_Distance_km": truck_dist(coords, i, j),
                })

    out_path = os.path.join(output_dir, "all_nodes_distance_matrix.csv")
    pd.DataFrame(distance_data).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ============================================================
# 5) 单算例结果文件输出
# ============================================================
def save_visualization(
    instance_name: str,
    data: Dict[str, Any],
    vars_dict: Dict[str, Any],
    output_dir: str,
) -> str:
    coords = data["coords"]
    depot_start = data["depot_start"]
    depot_end = data["depot_end"]
    N1 = data["N1"]
    N2 = data["N2"]
    N = data["N"]
    L = data["L"]

    xW = vars_dict["xW"]
    zW = vars_dict["zW"]
    zT = vars_dict["zT"]
    zD = vars_dict["zD"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_aspect('equal')

    all_x = [coords[n][0] for n in coords]
    all_y = [coords[n][1] for n in coords]
    x_min, x_max = min(all_x) - 5, max(all_x) + 5
    y_min, y_max = min(all_y) - 5, max(all_y) + 5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 起点消防站
    x_dep, y_dep = coords[depot_start]
    ax.scatter(x_dep, y_dep, marker='s', s=300, c='black', edgecolors='black', linewidths=2, zorder=10)
    ax.annotate(f'{depot_start}', (x_dep, y_dep), textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')

    # 终点消防站
    x_dep2, y_dep2 = coords[depot_end]
    ax.scatter(x_dep2, y_dep2, marker='D', s=300, c='black', edgecolors='black', linewidths=2, zorder=10)
    ax.annotate(f'{depot_end}', (x_dep2, y_dep2), textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')

    # 沿线道路监测节点
    for i in N1:
        x, y = coords[i]
        ax.scatter(x, y, marker='o', s=200, c='orange', edgecolors='darkorange', linewidths=2, zorder=10)
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(8, 5), fontsize=10, color='darkorange', fontweight='bold')

    # 非道路监测节点
    for i in N2:
        x, y = coords[i]
        ax.scatter(x, y, marker='o', s=200, c='skyblue', edgecolors='blue', linewidths=2, zorder=10)
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(8, 5), fontsize=10, color='blue', fontweight='bold')

    # 未建设的瞭望塔
    for l in L:
        if X(xW[l]) <= 0.5:
            x, y = coords[l]
            ax.scatter(x, y, marker='^', s=250, c='lightgray', edgecolors='gray', linewidths=2, zorder=10)
            ax.annotate(str(l), (x, y), textcoords="offset points", xytext=(8, 5), fontsize=9, color='gray')

    # 已建设的瞭望塔及覆盖范围
    for l in L:
        if X(xW[l]) > 0.5:
            x, y = coords[l]
            ax.scatter(x, y, marker='^', s=300, c='red', edgecolors='darkred', linewidths=2, zorder=10)
            ax.annotate(str(l), (x, y), textcoords="offset points", xytext=(8, 5), fontsize=10, color='darkred', fontweight='bold')
            circle = Circle((x, y), R, fill=False, edgecolor='hotpink', linestyle='--', linewidth=2, alpha=0.7, zorder=5)
            ax.add_patch(circle)
            fill_circle = Circle((x, y), R, fill=True, facecolor='lightpink', alpha=0.2, zorder=1)
            ax.add_patch(fill_circle)

    # 卡车路径
    for (i, j, k) in zT.keys():
        if X(zT[i, j, k]) > 0.5:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=20,
                                    color='purple', linewidth=2, zorder=8)
            ax.add_patch(arrow)

    # 无人机路径
    for (i, j, u) in zD.keys():
        if X(zD[i, j, u]) > 0.5:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15,
                                    color='green', linewidth=1.5, linestyle='--', zorder=7)
            ax.add_patch(arrow)

    # 瞭望塔巡检连线
    for i in N:
        for l in L:
            if X(zW[i, l]) > 0.5:
                x1, y1 = coords[l]
                x2, y2 = coords[i]
                ax.plot([x1, x2], [y1, y2], 'r--', linewidth=1.5, alpha=0.7, zorder=6)

    legend_elements = [
        mlines.Line2D([], [], marker='s', color='w', markerfacecolor='black', markersize=12, label='起点消防站', markeredgecolor='black'),
        mlines.Line2D([], [], marker='D', color='w', markerfacecolor='black', markersize=12, label='终点消防站', markeredgecolor='black'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='orange', markersize=12, label='道路沿线监测节点', markeredgecolor='darkorange'),
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor='skyblue', markersize=12, label='非道路监测节点', markeredgecolor='blue'),
        mlines.Line2D([], [], marker='^', color='w', markerfacecolor='red', markersize=12, label='已建设瞭望塔', markeredgecolor='darkred'),
        mlines.Line2D([], [], marker='^', color='w', markerfacecolor='lightgray', markersize=12, label='未建设候选塔', markeredgecolor='gray'),
        mlines.Line2D([], [], color='purple', linewidth=2, marker='>', markersize=8, label='卡车路径'),
        mlines.Line2D([], [], color='green', linewidth=1.5, linestyle='--', marker='>', markersize=8, label='无人机路径'),
        mlines.Line2D([], [], color='hotpink', linewidth=2, linestyle='--', marker='o', markersize=0, label='瞭望塔覆盖范围')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xlabel('X 坐标 (km)', fontsize=12)
    ax.set_ylabel('Y 坐标 (km)', fontsize=12)
    ax.set_title(f'{instance_name} - 森林火灾巡检瞭望塔覆盖+卡车/无人机路径', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'forest_fire_inspection_with_routes.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_instance_report(
    instance_name: str,
    excel_path: str,
    data: Dict[str, Any],
    vars_dict: Dict[str, Any],
    output_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    m = vars_dict["model"]
    status_name = status_name_from_code(m.Status)

    N = data["N"]
    N1 = data["N1"]
    N2 = data["N2"]
    L = data["L"]
    U = data["U"]
    K = data["K"]
    G = data["G"]
    V = data["V"]
    coords = data["coords"]
    alpha_max_W = data["alpha_max_W"]
    alpha_max_T = data["alpha_max_T"]
    depot_start = data["depot_start"]
    depot_end = data["depot_end"]
    dT = data["dT"]
    dD = data["dD"]

    xW = vars_dict["xW"]
    xT = vars_dict["xT"]
    xD = vars_dict["xD"]
    zW = vars_dict["zW"]
    zT = vars_dict["zT"]
    zD = vars_dict["zD"]

    has_solution = m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and m.SolCount > 0

    obj = m.ObjVal if has_solution else None
    best_bound = m.ObjBound if hasattr(m, "ObjBound") else None
    gap_pct = (m.MIPGap * 100.0) if hasattr(m, "MIPGap") else None

    sel_towers = [l for l in L if X(xW[l]) > 0.5] if has_solution else []
    sel_trucks = [k for k in K if X(xT[k]) > 0.5] if has_solution else []
    sel_drones = [u for u in U if X(xD[u]) > 0.5] if has_solution else []

    inspected_by_tower = [i for i in N for l in L if X(zW[i, l]) > 0.5] if has_solution else []
    inspected_by_truck = [
        i for i in N1 for k in K for j in G
        if j != i and (j, i, k) in zT and X(zT[j, i, k]) > 0.5
    ] if has_solution else []
    inspected_by_drone = [
        i for i in N for u in U for j in V
        if j != i and (j, i, u) in zD and X(zD[j, i, u]) > 0.5
    ] if has_solution else []

    all_inspected = set(inspected_by_tower + inspected_by_truck + inspected_by_drone)
    inspection_rate = (len(all_inspected) / len(N) * 100.0) if len(N) > 0 and has_solution else 0.0

    tower_fixed_cost = sum(cfW * X(xW[l]) for l in L) if has_solution else 0.0
    truck_fixed_cost = sum(cfT * X(xT[k]) for k in K) if has_solution else 0.0
    drone_fixed_cost = sum(cfD * X(xD[u]) for u in U) if has_solution else 0.0
    total_fixed = tower_fixed_cost + truck_fixed_cost + drone_fixed_cost

    tower_op_per = sum(coW * X(zW[i, l]) for i in N for l in L) if has_solution else 0.0
    truck_miles_per = sum(dT[i, j] * X(zT[i, j, k]) for (i, j, k) in zT.keys()) if has_solution else 0.0
    truck_op_per = coT * truck_miles_per
    drone_miles_per = sum(dD[i, j] * X(zD[i, j, u]) for (i, j, u) in zD.keys()) if has_solution else 0.0
    drone_op_per = coD * drone_miles_per
    total_op_per = tower_op_per + truck_op_per + drone_op_per
    annual_op_cost = total_op_per * F
    annual_total_cost = total_fixed + annual_op_cost

    truck_routes = extract_truck_routes(data, vars_dict) if has_solution else []
    drone_routes = extract_drone_routes(data, vars_dict) if has_solution else []

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append(f"      {instance_name} - 森林火灾巡检 TLTDRP 模型求解详细结果报告")
    output_lines.append("=" * 70)
    output_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"数据文件: {excel_path}")
    output_lines.append("")

    output_lines.append("【① 模型求解状态】")
    output_lines.append(f"    求解状态码: {m.Status}")
    output_lines.append(f"    状态说明: {status_name}")
    output_lines.append(f"    可行解数量: {m.SolCount}")
    output_lines.append("")

    output_lines.append("【② 终止原因及Gap值】")
    if m.Status == GRB.OPTIMAL:
        output_lines.append("    终止原因: 找到最优解")
    elif m.Status == GRB.TIME_LIMIT:
        output_lines.append("    终止原因: 达到时间限制")
    else:
        output_lines.append(f"    终止原因: {status_name}")
    output_lines.append(f"    当前Gap值: {gap_pct:.4f}%" if gap_pct is not None else "    当前Gap值: 无")
    output_lines.append(f"    最优目标值界限 (Best Bound): {best_bound:.6f}" if best_bound is not None else "    最优目标值界限 (Best Bound): 无")
    output_lines.append(f"    实际求解时间: {m.Runtime:.2f} 秒")
    output_lines.append("")

    output_lines.append("【③ 节点统计与巡检率】")
    output_lines.append(f"    待巡检监测节点总数: {len(N)} 个")
    output_lines.append(f"    实际被巡检节点数: {len(all_inspected)} 个")
    output_lines.append(f"    巡检率: {inspection_rate:.2f}%")
    output_lines.append(f"    沿线道路监测节点数: {len(N1)}")
    output_lines.append(f"    非沿线道路监测节点数: {len(N2)}")
    output_lines.append("")

    output_lines.append("【④ 成本与里程】")
    output_lines.append(f"    瞭望塔建设固定成本: {tower_fixed_cost:.2f}")
    output_lines.append(f"    卡车购置固定成本: {truck_fixed_cost:.2f}")
    output_lines.append(f"    无人机购置固定成本: {drone_fixed_cost:.2f}")
    output_lines.append(f"    固定成本合计: {total_fixed:.2f}")
    output_lines.append(f"    单次巡逻瞭望塔运营成本: {tower_op_per:.2f}")
    output_lines.append(f"    单次巡逻卡车运营成本: {truck_op_per:.2f}")
    output_lines.append(f"    单次巡逻无人机运营成本: {drone_op_per:.2f}")
    output_lines.append(f"    单次巡逻运营成本合计: {total_op_per:.2f}")
    output_lines.append(f"    全年运营成本合计: {annual_op_cost:.2f}")
    output_lines.append(f"    年度总成本: {annual_total_cost:.2f}")
    output_lines.append(f"    卡车单次巡逻总里程: {truck_miles_per:.2f} km")
    output_lines.append(f"    无人机单次巡逻总里程: {drone_miles_per:.2f} km")
    output_lines.append("")

    output_lines.append("【⑤ 设备使用详情】")
    output_lines.append(f"    建设瞭望塔数量: {len(sel_towers)} / 上限 {alpha_max_W}")
    output_lines.append(f"    使用卡车数量: {len(sel_trucks)} / 上限 {alpha_max_T}")
    output_lines.append(f"    使用无人机数量: {len(sel_drones)} / 上限 {len(U)}")
    output_lines.append(f"    建设瞭望塔编号: {sel_towers}")
    output_lines.append(f"    使用卡车编号: {sel_trucks}")
    output_lines.append(f"    使用无人机编号: {sel_drones}")
    output_lines.append("")

    if truck_routes:
        output_lines.append("    【卡车路径】")
        for row in truck_routes:
            flag = "【超限】" if row["is_exceeded"] else "【正常】"
            output_lines.append(
                f"      卡车 {row['truck_id']}: {row['route_text']} | 里程 {row['route_miles']:.2f} km {flag}"
            )
        output_lines.append("")

    if drone_routes:
        output_lines.append("    【无人机路径】")
        for row in drone_routes:
            flag = "【超限】" if row["is_exceeded"] else "【正常】"
            output_lines.append(
                f"      无人机 {row['drone_id']}: {row['route_text']} | 里程 {row['route_miles']:.2f} km {flag}"
            )
        output_lines.append("")

    output_lines.append("=" * 70)
    output_lines.append("                    报告结束")
    output_lines.append("=" * 70)

    report_path = os.path.join(output_dir, "solution_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    summary_row = {
        "instance_name": instance_name,
        "status_code": m.Status,
        "status_name": status_name,
        "sol_count": m.SolCount,
        "objective_value": obj,
        "best_bound": best_bound,
        "gap_percent": gap_pct,
        "runtime_seconds": m.Runtime,
        "node_count": len(N),
        "n1_count": len(N1),
        "n2_count": len(N2),
        "candidate_tower_count": len(L),
        "alpha_max_W": alpha_max_W,
        "potential_truck_count": data["potential_trucks"],
        "alpha_max_T": alpha_max_T,
        "max_drone_count": len(U),
        "used_tower_count": len(sel_towers),
        "used_truck_count": len(sel_trucks),
        "used_drone_count": len(sel_drones),
        "inspection_rate_percent": inspection_rate,
        "tower_fixed_cost": tower_fixed_cost,
        "truck_fixed_cost": truck_fixed_cost,
        "drone_fixed_cost": drone_fixed_cost,
        "total_fixed_cost": total_fixed,
        "tower_operating_cost_per_round": tower_op_per,
        "truck_miles_per_round": truck_miles_per,
        "truck_operating_cost_per_round": truck_op_per,
        "drone_miles_per_round": drone_miles_per,
        "drone_operating_cost_per_round": drone_op_per,
        "total_operating_cost_per_round": total_op_per,
        "annual_operating_cost": annual_op_cost,
        "annual_total_cost": annual_total_cost,
        "excel_path": excel_path,
        "result_dir": output_dir,
        "report_path": report_path,
    }
    return report_path, summary_row


def save_instance_csvs(data: Dict[str, Any], vars_dict: Dict[str, Any], output_dir: str) -> None:
    tower_assignments = extract_tower_assignments(data, vars_dict)
    truck_routes = extract_truck_routes(data, vars_dict)
    drone_routes = extract_drone_routes(data, vars_dict)
    node_summary = extract_node_assignment_summary(data, vars_dict)

    pd.DataFrame(tower_assignments).to_csv(os.path.join(output_dir, "tower_assignments.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(truck_routes).to_csv(os.path.join(output_dir, "truck_routes.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(drone_routes).to_csv(os.path.join(output_dir, "drone_routes.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(node_summary).to_csv(os.path.join(output_dir, "node_assignment_summary.csv"), index=False, encoding="utf-8-sig")

    # 保存弧变量明细
    zT = vars_dict["zT"]
    zD = vars_dict["zD"]
    dT = data["dT"]
    dD = data["dD"]

    truck_arc_rows = []
    for (i, j, k) in zT.keys():
        if X(zT[i, j, k]) > 0.5:
            truck_arc_rows.append({
                "truck_id": k,
                "from_node": i,
                "to_node": j,
                "distance_km": dT[i, j],
            })

    drone_arc_rows = []
    for (i, j, u) in zD.keys():
        if X(zD[i, j, u]) > 0.5:
            drone_arc_rows.append({
                "drone_id": u,
                "from_node": i,
                "to_node": j,
                "distance_km": dD[i, j],
            })

    pd.DataFrame(truck_arc_rows).to_csv(os.path.join(output_dir, "truck_arcs.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(drone_arc_rows).to_csv(os.path.join(output_dir, "drone_arcs.csv"), index=False, encoding="utf-8-sig")


# ============================================================
# 6) 批量汇总图表
# ============================================================
def save_summary_charts(summary_df: pd.DataFrame, output_dir: str) -> List[str]:
    chart_paths = []
    if summary_df.empty:
        return chart_paths

    df_plot = summary_df.copy()
    df_plot["display_name"] = df_plot["instance_name"]

    # 0) 年度总成本
    fig = plt.figure(figsize=(18, 8))
    plt.bar(df_plot["display_name"], df_plot["annual_total_cost"].fillna(0.0))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("年度总成本")
    plt.title("各算例年度总成本汇总")
    plt.tight_layout()
    path1 = os.path.join(output_dir, "summary_annual_total_cost.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append(path1)

    # 2) 求解时间
    fig = plt.figure(figsize=(18, 8))
    plt.bar(df_plot["display_name"], df_plot["runtime_seconds"].fillna(0.0))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("求解时间 (秒)")
    plt.title("各算例求解时间汇总")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "summary_runtime_seconds.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append(path2)

    # 3) 设备使用数量
    x = list(range(len(df_plot)))
    width = 0.25
    fig = plt.figure(figsize=(18, 8))
    plt.bar([p - width for p in x], df_plot["used_tower_count"].fillna(0.0), width=width, label="瞭望塔")
    plt.bar(x, df_plot["used_truck_count"].fillna(0.0), width=width, label="卡车")
    plt.bar([p + width for p in x], df_plot["used_drone_count"].fillna(0.0), width=width, label="无人机")
    plt.xticks(x, df_plot["display_name"], rotation=45, ha="right")
    plt.ylabel("使用数量")
    plt.title("各算例设备使用数量汇总")
    plt.legend()
    plt.tight_layout()
    path3 = os.path.join(output_dir, "summary_resource_usage.png")
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append(path3)

    # 4) Gap
    fig = plt.figure(figsize=(18, 8))
    plt.bar(df_plot["display_name"], df_plot["gap_percent"].fillna(0.0))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Gap (%)")
    plt.title("各算例最终 Gap 汇总")
    plt.tight_layout()
    path4 = os.path.join(output_dir, "summary_gap_percent.png")
    plt.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append(path4)

    return chart_paths


# ============================================================
# 7) 单个算例求解主流程
# ============================================================
def solve_one_instance(instance_name: str, base_instance_dir: str, output_root: str) -> Dict[str, Any]:
    instance_output_dir = os.path.join(output_root, f"结果_{safe_name(instance_name)}")
    ensure_dir(instance_output_dir)

    excel_path = find_excel_for_instance(base_instance_dir, instance_name)

    info_txt_path = os.path.join(instance_output_dir, "run_info.txt")
    with open(info_txt_path, "w", encoding="utf-8") as f:
        f.write(f"instance_name={instance_name}\n")
        f.write(f"excel_path={excel_path}\n")
        f.write(f"start_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        data = load_instance_data(excel_path)
        vars_dict = build_and_solve_model(instance_name, data, instance_output_dir)
        m = vars_dict["model"]

        if m.Status == GRB.INFEASIBLE:
            m.computeIIS()
            try:
                ascii_export_dir = get_ascii_export_dir()
                iis_path = os.path.join(ascii_export_dir, f"{safe_ascii_name(instance_name)}_infeasible_constraints.ilp")
                m.write(iis_path)
                print(f"[提示] IIS 文件已写出到纯英文目录: {iis_path}")
            except Exception as e:
                print(f"[警告] IIS 文件写出失败，已跳过。原因: {e}")

        distance_path = generate_distance_matrix_csv(data, instance_output_dir)
        report_path, summary_row = save_instance_report(instance_name, excel_path, data, vars_dict, instance_output_dir)

        if m.SolCount > 0 and m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            save_instance_csvs(data, vars_dict, instance_output_dir)
            fig_path = save_visualization(instance_name, data, vars_dict, instance_output_dir)
        else:
            fig_path = None

        # 单算例摘要表
        pd.DataFrame([summary_row]).to_csv(os.path.join(instance_output_dir, "instance_summary.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame([summary_row]).to_excel(os.path.join(instance_output_dir, "instance_summary.xlsx"), index=False)

        summary_row["distance_matrix_path"] = distance_path
        summary_row["figure_path"] = fig_path
        summary_row["error_message"] = None
        summary_row["traceback"] = None
        return summary_row

    except Exception as e:
        err_trace = traceback.format_exc()
        error_report_path = os.path.join(instance_output_dir, "error_report.txt")
        with open(error_report_path, "w", encoding="utf-8") as f:
            f.write(f"算例: {instance_name}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(str(e) + "\n\n")
            f.write(err_trace)

        return {
            "instance_name": instance_name,
            "status_code": None,
            "status_name": "ERROR",
            "sol_count": 0,
            "objective_value": None,
            "best_bound": None,
            "gap_percent": None,
            "runtime_seconds": None,
            "node_count": None,
            "n1_count": None,
            "n2_count": None,
            "candidate_tower_count": None,
            "alpha_max_W": None,
            "potential_truck_count": None,
            "alpha_max_T": None,
            "max_drone_count": None,
            "used_tower_count": None,
            "used_truck_count": None,
            "used_drone_count": None,
            "inspection_rate_percent": None,
            "tower_fixed_cost": None,
            "truck_fixed_cost": None,
            "drone_fixed_cost": None,
            "total_fixed_cost": None,
            "tower_operating_cost_per_round": None,
            "truck_miles_per_round": None,
            "truck_operating_cost_per_round": None,
            "drone_miles_per_round": None,
            "drone_operating_cost_per_round": None,
            "total_operating_cost_per_round": None,
            "annual_operating_cost": None,
            "annual_total_cost": None,
            "excel_path": None,
            "result_dir": instance_output_dir,
            "report_path": error_report_path,
            "distance_matrix_path": None,
            "figure_path": None,
            "error_message": str(e),
            "traceback": err_trace,
        }


# ============================================================
# 8) 批量求解主程序
# ============================================================
def main() -> None:
    print("=" * 90)
    print("TLTDRP 批量 Gurobi 精确求解开始")
    print("=" * 90)
    print(f"代码所在目录: {SCRIPT_DIR}")
    print(f"数据根目录: {BASE_INSTANCE_DIR}")
    print(f"计划求解算例数量: {len(INSTANCE_NAMES)}")
    print("")

    # 所有结果都放在与代码同目录下
    output_root = SCRIPT_DIR
    summary_rows = []

    batch_start = datetime.now()

    for idx, instance_name in enumerate(INSTANCE_NAMES, start=1):
        print("-" * 90)
        print(f"[{idx}/{len(INSTANCE_NAMES)}] 正在求解: {instance_name}")
        print("-" * 90)
        row = solve_one_instance(instance_name, BASE_INSTANCE_DIR, output_root)
        summary_rows.append(row)
        print(f"完成状态: {row.get('status_name')} | 结果目录: {row.get('result_dir')}")
        if row.get("error_message"):
            print(f"错误信息: {row['error_message']}")
        print("")

    summary_df = pd.DataFrame(summary_rows)

    # 按给定顺序排序
    order_map = {name: i for i, name in enumerate(INSTANCE_NAMES)}
    summary_df["_order"] = summary_df["instance_name"].map(order_map)
    summary_df = summary_df.sort_values(by="_order", na_position="last").drop(columns=["_order"])

    # 总汇总表：直接放在与代码同目录下
    summary_csv_path = os.path.join(output_root, "所有算例汇总表.csv")
    summary_xlsx_path = os.path.join(output_root, "所有算例汇总表.xlsx")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_excel(summary_xlsx_path, index=False)

    # 汇总图表
    chart_paths = save_summary_charts(summary_df, output_root)

    # 批量总报告
    batch_end = datetime.now()
    batch_report_path = os.path.join(output_root, "批量求解总报告.txt")
    with open(batch_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("TLTDRP 批量 Gurobi 精确求解总报告\n")
        f.write("=" * 90 + "\n")
        f.write(f"开始时间: {batch_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {batch_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {(batch_end - batch_start).total_seconds():.2f} 秒\n")
        f.write(f"数据根目录: {BASE_INSTANCE_DIR}\n")
        f.write(f"输出目录: {output_root}\n\n")

        f.write("算例结果概览：\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"- {row['instance_name']} | 状态={row['status_name']} | "
                f"目标值={row['objective_value']} | 用时={row['runtime_seconds']} 秒 | "
                f"结果目录={row['result_dir']}\n"
            )
        f.write("\n")
        f.write(f"汇总表 CSV: {summary_csv_path}\n")
        f.write(f"汇总表 Excel: {summary_xlsx_path}\n")
        for p in chart_paths:
            f.write(f"汇总图表: {p}\n")

    print("=" * 90)
    print("批量求解完成")
    print("=" * 90)
    print(f"总汇总表 CSV:  {summary_csv_path}")
    print(f"总汇总表 Excel:{summary_xlsx_path}")
    print(f"批量总报告:    {batch_report_path}")
    if chart_paths:
        print("汇总图表:")
        for p in chart_paths:
            print(f"  - {p}")
    print("\n每个算例的结果目录形式为：")
    print("  结果_小规模算例1_6个节点")
    print("  结果_小规模算例2_8个节点")
    print("  ...")


if __name__ == "__main__":
    main()
