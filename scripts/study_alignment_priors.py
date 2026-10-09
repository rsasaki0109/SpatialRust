import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'crates/spatialrust-py/examples'))
from align_point_clouds import align_files
ROOT=Path(__file__).resolve().parents[1]/'target/pose-prior-study'
ROOT.mkdir(parents=True, exist_ok=True)

def rotation(degrees):
    a=np.deg2rad(degrees); c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def matrix(r,t):
    m=np.eye(4);m[:3,:3]=r;m[:3,3]=t;return m

def write(path, xyz):
    header=f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA ascii\n'
    path.write_text(header+'\n'.join(' '.join(format(float(x),'.9g') for x in p) for p in xyz)+'\n')

def error(estimated, truth):
    u,_,v=np.linalg.svd(estimated[:3,:3]); r=u@v
    if np.linalg.det(r)<0: u[:,-1]*=-1; r=u@v
    angle=np.rad2deg(np.arccos(np.clip((np.trace(r@truth[:3,:3].T)-1)/2,-1,1)))
    return {'rotation_error_degrees':float(angle),'translation_error_metres':float(np.linalg.norm(estimated[:3,3]-truth[:3,3]))}
rows=[]
for geometry in ['ring','random']:
  for seed in ([0] if geometry=='ring' else range(5)):
    if geometry=='ring':
      a=np.arange(180)*2*np.pi/180;target=np.column_stack([np.cos(a),np.sin(a),np.zeros(180)]);r=rotation(20);t=np.zeros(3)
    else:
      target=np.random.default_rng(seed).uniform(-1,1,(180,3));r=rotation(60);t=np.array([5,-2,.5])
    source=target@r.T+t;truth=matrix(r.T,-r.T@t)
    sp=ROOT/f'{geometry}-{seed}-source.pcd';tp=ROOT/f'{geometry}-{seed}-target.pcd';write(sp,source);write(tp,target)
    for prior_name,prior in [('none',None),('correct',truth),('perturbed',matrix(rotation(3),np.array([.03,-.02,.01]))@truth)]:
      row={'geometry':geometry,'seed':seed,'prior':prior_name,'leaf_metres':.025,'gate_metres':.15,'iterations_per_stage':100}
      try:
        _,d=align_files(sp,tp,leaf=.025,max_distance=.15,iterations=100,initial_transform=prior)
        row.update(error(np.array(d['transform_source_to_target']),truth));row.update(converged=d['converged'],forward_fraction=d['aligned_support']['query_fraction'],reverse_fraction=d['aligned_reverse_support']['query_fraction'],rmse_metres=d['aligned_support']['gated_rmse_metres']);row['diagnostics']=d
      except Exception as e:row.update(error_type=type(e).__name__,error=str(e))
      rows.append(row)
(ROOT/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
for row in rows:print({k:v for k,v in row.items() if k!='diagnostics'})
