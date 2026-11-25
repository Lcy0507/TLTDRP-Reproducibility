# -*- coding: utf-8 -*-
"""新林68节点：稳健性实验A（卡车禁巡放松）和B（无人机绕行）。
使用随附的原版独立无重启IALNS内核；不更换求解算法。
"""
import argparse, copy, hashlib, json, math, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
import importlib.util

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT/'IALNS_原版独立算法.py'
DEFAULT_EXCEL=ROOT/'新林案例_68个节点.xlsx'
PARAM=ROOT/'Taguchi_最终最优参数.json'

def load_engine():
    spec=importlib.util.spec_from_file_location('tlt_original_ialns',ENGINE)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def apply_scenario(mod, original, mode, relaxed, eta):
    """通过隔离的实验内核精确变更约束，不触碰原始算法源文件。"""
    data=copy.deepcopy(original)
    data.aerial_dist={key:val*eta for key,val in original.aerial_dist.items()}
    if mode=='WT': data.drone_limit=0
    if mode=='WD': data.truck_limit=0
    # 在实验A放松后，仅改变卡车禁巡判定：候选动作及最终可行性核查同时更新。
    # 原版函数使用全局名称 built_cover_nodes：替换为返回空集，仅针对卡车限制有效；
    # 该函数仅在卡车约束和卡车插入动作中调用，不参与瞭望塔自身的覆盖判断。
    mod.built_cover_nodes = (lambda sol,dat:set()) if relaxed else mod._original_built_cover_nodes
    return data

def set_font():
    candidates=['Microsoft YaHei','SimHei','Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Micro Hei','PingFang SC','Arial Unicode MS']
    plt.rcParams['font.sans-serif']=candidates+['DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['pdf.fonttype']=42
    plt.rcParams['ps.fonttype']=42

def savefig(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','eps']:
        try: fig.savefig(path.with_suffix('.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
        except Exception as e: print('导出失败',path,ext,str(e))
    plt.close(fig)

class CircleHandler(HandlerPatch):
    def create_artists(self,legend,orig,xdescent,ydescent,width,height,fontsize,trans):
        return [Circle((xdescent+width/2,ydescent+height/2),min(width,height)*.43,
                       facecolor='#ffe2ea',edgecolor='#db6598',linestyle=(0,(2,2,2,2,7,2)),
                       linewidth=1.7,transform=trans)]

def plot_plan(mod,sol,data,path,title):
    fig,ax=plt.subplots(figsize=(13,9));fig.subplots_adjust(right=.71)
    dash=(0,(2,2,2,2,7,2))
    for l in sorted(sol.built_towers):
        ax.add_patch(Circle(data.coords[l],mod.TOWER_RADIUS,facecolor='#ffe2ea',edgecolor='#db6598',
                            linestyle=dash,lw=1.5,alpha=.40,zorder=0))
    for l,nodes in sol.tower_visits.items():
        for n in nodes:
            a,b=data.coords[l],data.coords[n]
            ax.plot([a[0],b[0]],[a[1],b[1]],color='#d95a8c',ls=':',lw=1.25,zorder=2)
    for routes,col,ls in [(sol.truck_routes,'#773399','-'),(sol.drone_routes,'#168b62','-.')]:
        for route in routes.values():
            for i,j in zip(route[:-1],route[1:]):
                a,b=data.coords[i],data.coords[j]
                if math.dist(a,b)>1e-9:
                    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=col,lw=1.25,linestyle=ls),zorder=3)
    for nodes,marker,color,size in [
        (data.n1,'o','#ef9727',52),(data.n2,(3,1,0),'#2776b7',95),
        (sorted(set(data.towers)-sol.built_towers),'*','#888888',120),
        (sorted(sol.built_towers),'h','#d73032',130)]:
        if nodes:
            ax.scatter([data.coords[i][0] for i in nodes],[data.coords[i][1] for i in nodes],
                       marker=marker,c=color,s=size,edgecolors='black',linewidths=.3,zorder=6)
    depots=mod.unique_preserve([data.depot_start,data.depot_end])
    for d in depots:
        ax.scatter(*data.coords[d],marker='s',c='black',s=105,zorder=8)
        ax.annotate('消防站',data.coords[d],xytext=(8,-12),textcoords='offset points',fontsize=9)
    for i,n in enumerate(data.n):
        ax.annotate(str(i+1),data.coords[n],xytext=(6,3),textcoords='offset points',fontsize=7,zorder=9)
    for i,l in enumerate(data.towers):
        ax.annotate('T'+str(i+1),data.coords[l],xytext=(6,-7),textcoords='offset points',fontsize=7,zorder=9)
    legend=[Line2D([],[],marker='o',color='none',markerfacecolor='#ef9727',markeredgecolor='black',markersize=9,label='道路沿线监测节点'),
            Line2D([],[],marker=(3,1,0),color='none',markerfacecolor='#2776b7',markeredgecolor='black',markersize=12,label='非道路监测节点'),
            Line2D([],[],marker='h',color='none',markerfacecolor='#d73032',markeredgecolor='black',markersize=12,label='已建设瞭望塔'),
            Line2D([],[],marker='*',color='none',markerfacecolor='#888888',markeredgecolor='black',markersize=12,label='未建设瞭望塔'),
            Circle((0,0),1,facecolor='#ffe2ea',edgecolor='#db6598',linestyle=dash,label='瞭望塔覆盖范围'),
            Line2D([],[],color='#d95a8c',ls=':',lw=2,label='瞭望塔巡检路线'),
            Line2D([],[],marker='s',color='none',markerfacecolor='black',markersize=9,label='消防站'),
            Line2D([],[],color='#773399',ls='-',lw=2,label='卡车巡检路线'),
            Line2D([],[],color='#168b62',ls='-.',lw=2,label='无人机巡检路线')]
    ax.legend(handles=legend,loc='upper left',bbox_to_anchor=(1.03,1),fontsize=12,
              handler_map={Circle:CircleHandler()},frameon=True)
    xs=[p[0] for p in data.coords.values()];ys=[p[1] for p in data.coords.values()]
    pad=mod.TOWER_RADIUS+3
    ax.set_xlim(min(xs)-pad,max(xs)+pad);ax.set_ylim(min(ys)-pad,max(ys)+pad)
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    ax.set_xlabel('X 坐标 (km)',fontsize=12);ax.set_ylabel('Y 坐标 (km)',fontsize=12)
    ax.set_title(title,fontsize=13)
    savefig(fig,path)

def plot_A(rows,out):
    sub=rows[rows['experiment']=='A']
    # 以三元模式的原假设/放松后作Best Avg精确标值对比
    x=sub[(sub['mode']=='WTD') & (sub['feasible'])]
    if x.empty:return
    fig,ax=plt.subplots(figsize=(10,5.5))
    labels=['原假设：卡车禁巡塔覆盖道路节点','放松后：卡车允许巡检塔覆盖道路节点']
    for i,(flag,label) in enumerate(zip([False,True],labels)):
        a=x[x['relaxed']==flag]
        if a.empty:continue
        best=a.cost.min()/10000;avg=a.cost.mean()/10000
        for j,(name,value,hatch) in enumerate([('Best',best,'///'),('Avg',avg,'xxx')]):
            xpos=i*3+j
            ax.bar(xpos,value,width=.75,color=['#dddddd','#999999'][j],edgecolor='black',hatch=hatch)
            ax.text(xpos,value+max(1,value*.006),f'{value:.4f}',ha='center',fontsize=11)
    ax.set_xticks([.5,3.5],labels);ax.set_ylabel('年度总成本（万美元）')
    ax.set_title('实验A：三元模式原假设与放松后的Best和Avg对比')
    ax.legend(handles=[plt.Rectangle((0,0),1,1,fc='#dddddd',ec='black',hatch='///',label='Best'),
                       plt.Rectangle((0,0),1,1,fc='#999999',ec='black',hatch='xxx',label='Avg')])
    ax.set_ylim(bottom=0);fig.tight_layout();savefig(fig,out/'实验A_原假设放松前后_Best_Avg对比.png')

def plot_A_extra(rows,out):
    sub=rows[(rows.experiment=='A') & (rows['mode']=='WTD') & (rows.feasible)]
    if sub.empty:return
    labels=['原假设：卡车禁巡塔覆盖道路节点','放松后：允许卡车巡检塔覆盖道路节点']
    cols=['tower_fixed','truck_fixed','drone_fixed','tower_operating','truck_operating','drone_operating']
    names=['塔固定成本','卡车固定成本','无人机固定成本','塔运营成本','卡车运营成本','无人机运营成本']
    groups=[sub[sub.relaxed==flag] for flag in [False,True]]
    fig,ax=plt.subplots(figsize=(11,6));bottom=np.zeros(2)
    for j,(col,name) in enumerate(zip(cols,names)):
        values=np.array([g[col].mean()/10000 if len(g) else 0 for g in groups]);
        ax.bar([0,1],values,bottom=bottom,label=name,color=str(.90-j*.11),edgecolor='black',hatch=['','//','xx','..','\\','++'][j])
        bottom+=values
    for i,val in enumerate(bottom):ax.text(i,val+1,f'{val:.4f}',ha='center',fontsize=11)
    ax.set_xticks([0,1],labels);ax.set_ylabel('平均年度成本（万美元）')
    ax.set_title('实验A：原假设与放松后固定成本和运营成本构成')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=11)
    savefig(fig,out/'实验A_固定成本与运营成本构成.png')
    fig,ax=plt.subplots(figsize=(10,5.5));xs=np.arange(2)
    bottom=np.zeros(2)
    for j,(col,name) in enumerate(zip(['towers','trucks','drones'],['已建设瞭望塔','使用卡车','使用无人机'])):
        values=np.array([g[col].mean() if len(g) else 0 for g in groups]);ax.bar(xs,values,bottom=bottom,
            label=name,color=str(.85-j*.25),edgecolor='black',hatch=['//','xx','..'][j]);bottom+=values
    ax.set_xticks(xs,labels);ax.set_ylabel('平均资源使用数量')
    ax.set_title('实验A：原假设与放松后资源配置变化');ax.legend()
    savefig(fig,out/'实验A_资源配置变化.png')

def plot_B(summary,out):
    sub=summary[summary.experiment=='B']
    if sub.empty:return
    fig,ax=plt.subplots(figsize=(10,6))
    for mode,style,marker in [('WTD','-','o'),('WD','--','s')]:
        q=sub[(sub['mode']==mode)&(sub['feasible_runs']>0)].sort_values('eta')
        if q.empty:continue
        for metric,ls in [('Best','-'),('Avg',':')]:
            y=q[metric]/10000
            ax.plot(q.eta,y,linestyle=ls if mode=='WTD' else '--',marker=marker,
                    label=f'{"三元" if mode=="WTD" else "瞭望塔-无人机二元"} {metric}')
            for x,v in zip(q.eta,y):ax.annotate(f'{v:.2f}',(x,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xlabel('无人机飞行绕行系数 η');ax.set_ylabel('年度总成本（万美元）')
    ax.set_title('实验B：无人机绕行系数变化下的Best和Avg')
    ax.legend(loc='best');ax.grid(alpha=.2);fig.tight_layout();savefig(fig,out/'实验B_无人机绕行_Best_Avg.png')

def summarize(rows):
    res=[]
    for keys,g in rows.groupby(['experiment','mode','relaxed','eta'],dropna=False):
        f=g[g.feasible];r=dict(zip(['experiment','mode','relaxed','eta'],keys))
        r.update(total_runs=len(g),feasible_runs=len(f),feasibility_rate=len(f)/len(g),
                 Best=float(f.cost.min()) if len(f) else np.nan,Avg=float(f.cost.mean()) if len(f) else np.nan,
                 mean_coverage=float(g.coverage.mean()),mean_seconds=float(g.seconds.mean()),
                 mean_towers=float(f.towers.mean()) if len(f) else np.nan,
                 mean_trucks=float(f.trucks.mean()) if len(f) else np.nan,
                 mean_drones=float(f.drones.mean()) if len(f) else np.nan)
        res.append(r)
    return pd.DataFrame(res)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--excel',default=str(DEFAULT_EXCEL));ap.add_argument('--output-dir',default=str(ROOT/'稳健性实验A_B_结果'))
    ap.add_argument('--runs',type=int,default=10);ap.add_argument('--iterations',type=int,default=500)
    ap.add_argument('--seed-base',type=int,default=1000)
    ap.add_argument('--experiment',choices=['A','B','all'],default='all')
    ap.add_argument('--quick',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--show-plots',action='store_true',help='运行结束后打开生成图片（支持图形桌面时）')
    args=ap.parse_args()
    if args.quick:args.runs,args.iterations=1,5
    if args.runs<1 or args.iterations<1:ap.error('runs和iterations必须为正数')
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    mod=load_engine();mod._original_built_cover_nodes=mod.built_cover_nodes
    mod.TRUCK_MAX_RANGE=300.;mod.DRONE_MAX_RANGE=50.;mod.TRUCK_DETOUR_FACTOR=1.3
    mod.TOWER_RADIUS=10.;mod.TOWER_DRONE_CAPACITY=2
    mod.setup_chinese_fonts();set_font()
    excel=Path(args.excel).resolve();assert excel.is_file(),f'Excel不存在：{excel}'
    data=mod.load_any_instance(excel,'新林案例_68个节点')
    assert len(data.n)==68 and len(data.n1)==41 and len(data.n2)==27, '输入不是论文中的新林68节点结构'
    params=mod.load_taguchi_parameters(str(PARAM)) if PARAM.is_file() else mod.DEFAULT_IALNS_PARAMETERS
    scenarios=[]
    if args.experiment in ('A','all'):
        scenarios += [('A',m,r,1.) for m in ['WTD','WT'] for r in [False,True]]
    if args.experiment in ('B','all'):
        scenarios += [('B',m,False,e) for m in ['WTD','WD'] for e in [1.,1.05,1.10,1.20,1.30]]
    config=dict(excel_sha256=hashlib.sha256(excel.read_bytes()).hexdigest(),
                engine_sha256=hashlib.sha256(ENGINE.read_bytes()).hexdigest(),
                parameters=vars(params) if hasattr(params,'__dict__') else str(params),
                iterations=args.iterations,runs=args.runs,seed_base=args.seed_base,
                scenarios=scenarios,truck_range=300,drone_range=50,truck_detour=1.3,tower_radius=10)
    configpath=out/'实验配置.json'
    if configpath.exists() and json.loads(configpath.read_text(encoding='utf-8'))!=json.loads(json.dumps(config,ensure_ascii=False)):
        raise RuntimeError('现有结果配置不同，请使用新的output-dir，防止混合不同参数结果')
    configpath.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for exp,mode,relaxed,eta in scenarios:
        sid=f'{exp}_{mode}_{"放松后" if relaxed else "原假设"}_eta{eta:.2f}'
        folder=out/sid;folder.mkdir(exist_ok=True)
        d=apply_scenario(mod,data,mode,relaxed,eta)
        bestsol=None;bestcost=math.inf
        for run in range(1,args.runs+1):
            seed=args.seed_base+run;cache=folder/f'run_{run:02d}.pkl'
            if cache.is_file() and args.resume:
                with cache.open('rb') as f: result=pickle.load(f)
                sol=result['solution'];elapsed=result['seconds'];history=result['history']
            else:
                algo=mod.IALNS(d,parameters=params,max_iterations=args.iterations)
                sol,meta=algo.solve(seed=seed)
                elapsed=meta['elapsed_time'];history=meta['iteration_history']
                with cache.open('wb') as f:pickle.dump(dict(solution=sol,seconds=elapsed,history=history),f)
            violations=mod.constraint_violations(sol,d)
            feasible=(len(violations)==0 and len(sol.node_assignment)==68)
            cb=mod.cost_breakdown(sol,d)
            cover=len(set(sol.node_assignment).intersection(d.n))/len(d.n)
            row=dict(experiment=exp,mode=mode,relaxed=relaxed,eta=eta,scenario=sid,run=run,seed=seed,
                     feasible=feasible,coverage=cover,cost=cb.annual_total if feasible else np.nan,
                     seconds=elapsed,towers=len(sol.built_towers),trucks=len(sol.truck_routes),
                     drones=len(sol.drone_routes),violations=' | '.join(violations),
                     tower_fixed=cb.fixed_tower,truck_fixed=cb.fixed_truck,drone_fixed=cb.fixed_drone,
                     tower_operating=cb.patrol_tower*mod.PATROLS_PER_YEAR,
                     truck_operating=cb.patrol_truck*mod.PATROLS_PER_YEAR,
                     drone_operating=cb.patrol_drone*mod.PATROLS_PER_YEAR)
            records.append(row)
            pd.DataFrame(records).to_csv(out/'所有独立运行结果.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(history).to_csv(folder/f'迭代历史_Run{run:02d}.csv',index=False,encoding='utf-8-sig')
            if feasible and cb.annual_total<bestcost:
                bestsol,bestcost=sol,cb.annual_total
            print(sid,run,'可行' if feasible else '未获得全覆盖可行解',round(cb.annual_total/10000,4),flush=True)
        if bestsol is not None:
            plot_plan(mod,bestsol,d,folder/'最佳巡检方案.png',sid+' | 最佳可行巡检方案')
            with (folder/'最佳方案.pkl').open('wb') as f:pickle.dump(bestsol,f)
            pd.DataFrame([{'node':n,'service':v[0],'resource':v[1]} for n,v in bestsol.node_assignment.items()]).to_csv(
                folder/'最佳方案节点服务方式.csv',index=False,encoding='utf-8-sig')
    raw=pd.DataFrame(records);summary=summarize(raw)
    summary.to_csv(out/'稳健性汇总.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(out/'稳健性全部结果.xlsx',engine='openpyxl') as writer:
        raw.to_excel(writer,sheet_name='独立运行',index=False)
        summary.to_excel(writer,sheet_name='汇总',index=False)
    plot_A(raw,out);plot_A_extra(raw,out);plot_B(summary,out)
    if args.show_plots:
        import subprocess
        p=out/'实验A_原假设放松前后_Best_Avg对比.png'
        if p.exists():
            try:os.startfile(str(p))
            except Exception:pass
    print('完成，结果目录：',out)

if __name__=='__main__':main()
