import sys, json, os, numpy as np, mujoco
from PIL import Image
root=os.path.expanduser('~/.strands_robots/assets')
reg=json.load(open('strands_robots/registry/robots.json'))['robots']
names=sys.argv[1:] or [n for n,s in reg.items() if s.get('asset')]
ok=[];bad=[]
for n in names:
    a=reg[n]['asset']; path=os.path.join(root,a['dir'],a['scene_xml'])
    try:
        m=mujoco.MjModel.from_xml_path(path); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
        r=mujoco.Renderer(m,480,640); cam=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m,cam)
        cam.azimuth=135; cam.elevation=-20; cam.distance*=0.8
        opt=mujoco.MjvOption(); opt.geomgroup[3]=0
        r.update_scene(d,cam,opt); img=r.render()
        Image.fromarray(img).save(f'docs/assets/img/robots/{n}.png'); r.close(); ok.append(n)
    except Exception as e: bad.append((n,str(e)[:90]))
print('ok',len(ok)); [print('BAD',b) for b in bad]
