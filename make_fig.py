import json, math, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

A=json.load(open("/tmp/mt_before.json")); B=json.load(open("/tmp/mt_after.json"))
assert A["tree"]!=B["tree"], "before/after came from the same tree"
CELLS=len(A["cells"]); assert CELLS==54
acc_a=sum(1 for v in A["cells"].values() if v["verdict"]=="accepted")
rai_a=sum(1 for v in A["cells"].values() if v["verdict"]=="raised")
acc_b=sum(1 for v in B["cells"].values() if v["verdict"]=="accepted")
rai_b=sum(1 for v in B["cells"].values() if v["verdict"]=="raised")
assert (acc_a,rai_a,acc_b,rai_b)==(44,6,0,0), (acc_a,rai_a,acc_b,rai_b)
assert all(A["controls"][k]==B["controls"][k] for k in A["controls"])

VALUES=["0","-4","True","nan","2.7","inf","'4'","None","[2]"]
FIELDS=["num_gpus","num_nodes"]
PROVS=["lerobot_local","groot","cosmos3"]
SHORT={"lerobot_local":"lerobot","groot":"GR00T","cosmos3":"Cosmos"}
COL={"accepted":"#c62828","raised":"#8e24aa","refused (domain)":"#2e7d32","refused (support)":"#1565c0"}
LBL={"accepted":"accepted","raised":"raised out\nof validate()","refused (domain)":"refused","refused (support)":"refused\n(multi-node)"}

placed=[]
def put(ax,x,y,s,**kw):
    placed.append((ax,y)); return ax.text(x,y,s,**kw)

fig=plt.figure(figsize=(15.6,9.4)); fig.patch.set_facecolor("white")
gs=fig.add_gridspec(3,2,height_ratios=[0.40,1.55,0.62],hspace=0.34,wspace=0.10,
                    left=0.055,right=0.975,top=0.945,bottom=0.045)

# ---- header: what the value does downstream ----
axh=fig.add_subplot(gs[0,:]); axh.axis("off"); axh.set_xlim(0,1); axh.set_ylim(0,1)
put(axh,0.0,0.80,"What each unusable process count does downstream (tree-independent)",
    fontsize=12.5,fontweight="bold")
xs=np.linspace(0.045,0.955,len(VALUES))
put(axh,0.0,0.50,"value",fontsize=9.6,fontweight="bold",ha="left")
put(axh,0.0,0.30,"$v>1$ selector",fontsize=9.6,ha="left")
put(axh,0.0,0.10,"LaunchConfig",fontsize=9.6,ha="left")
for x,v in zip(xs,VALUES):
    f=A["facts"][v]
    put(axh,x,0.50,v,fontsize=10.2,fontweight="bold",ha="center",family="monospace")
    g=f["gt1"]
    txt=("multi-proc" if g is True else "single-proc" if g is False else "TypeError")
    c=("#c62828" if g is True else "#e65100" if g is False else "#8e24aa")
    put(axh,x,0.30,txt,fontsize=9.0,ha="center",color=c)
    put(axh,x,0.10,f["launchconfig"],fontsize=9.0,ha="center",
        color=("#c62828" if f["launchconfig"]=="accepts" else "#555"))

# ---- the two verdict matrices ----
for col,(data,title) in enumerate([(A,"main  —  no domain on either field"),
                                   (B,"this change  —  one shared positive-count domain")]):
    ax=fig.add_subplot(gs[1,col]); ax.set_xlim(0,len(VALUES)); ax.set_ylim(0,6)
    ax.set_xticks(np.arange(len(VALUES))+0.5); ax.set_xticklabels(VALUES,fontsize=9.6,family="monospace")
    rows=[(p,f) for f in FIELDS for p in PROVS]
    ax.set_yticks(np.arange(len(rows))+0.5)
    ax.set_yticklabels([f"{SHORT[p]}  ·  {f}" for p,f in rows],fontsize=9.4)
    ax.invert_yaxis()
    for r,(p,f) in enumerate(rows):
        for c,v in enumerate(VALUES):
            k=data["cells"][f"{p}|{f}|{v}"]["verdict"]
            ax.add_patch(Rectangle((c,r),1,1,facecolor=COL[k],edgecolor="white",lw=1.6,alpha=0.93))
            ax.text(c+0.5,r+0.5,LBL[k],ha="center",va="center",fontsize=6.2,color="white",
                    fontweight="bold",linespacing=0.95)
    ax.set_title(title,fontsize=11.6,fontweight="bold",pad=9)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

# ---- footer: measured ledger ----
axf=fig.add_subplot(gs[2,:]); axf.axis("off"); axf.set_xlim(0,1); axf.set_ylim(0,1)
put(axf,0.0,0.90,f"Measured over {CELLS} (value × field × backend) cells — 9 unusable values × 2 fields × 3 launching backends",
    fontsize=11.2,fontweight="bold")
left=[
 ("accepted, run proceeded",              f"{acc_a} of {CELLS}",  "0 of 54", True),
 ("raised out of validate()",             f"{rai_a} of {CELLS}",  "0 of 54", True),
 ("refused before the run starts",        f"{CELLS-acc_a-rai_a} of {CELLS}", f"{CELLS} of {CELLS}", False),
]
y=0.66
for label,before,after,bad in left:
    put(axf,0.015,y,label,fontsize=9.9)
    put(axf,0.315,y,before,fontsize=9.9,family="monospace",
        color=("#c62828" if bad else "#2e7d32"),fontweight="bold")
    put(axf,0.425,y,"→",fontsize=9.9,color="#666")
    put(axf,0.465,y,after,fontsize=9.9,family="monospace",color="#2e7d32",fontweight="bold")
    y-=0.20
notes=[
 "num_nodes=8 is still refused as an unsupported topology on lerobot/GR00T — the guarded comparison still fires for a usable count.",
 "num_gpus 1 / 2 / 8 and num_nodes 1: verdicts byte-identical on both trees (12 control cells).",
 "LaunchConfig accepts 0, -4, True, nan, 2.7 and inf as nproc_per_node, so the preflight is the only place a caller can be told.",
]
y=0.66
for n in notes:
    put(axf,0.60,y,"• "+n,fontsize=8.5,color="#333",wrap=True)
    y-=0.20
put(axf,0.60,y+0.02,"",fontsize=8.5)

for ax,yy in placed:
    lo,hi=ax.get_ylim(); lo,hi=min(lo,hi),max(lo,hi)
    assert lo-0.06<=yy<=hi+0.08, f"text at y={yy} outside {ax.get_ylim()}"

out=pathlib.Path("/tmp/topology_domain.png")
fig.savefig(out,dpi=140,bbox_inches="tight",pad_inches=0.32,facecolor="white")
im=np.array(plt.imread(out)[:,:,:3]*255).astype(int)
for name,band in (("top",im[:8]),("bottom",im[-8:]),("left",im[:,:8]),("right",im[:,-8:])):
    n=int((np.abs(band-255).sum(2)>26).sum())
    assert n==0, f"{name} border has {n} non-white px"
print("OK",out,im.shape, f"accepted {acc_a}->{acc_b}, raised {rai_a}->{rai_b}")
