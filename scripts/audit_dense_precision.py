"""Source-defined one-step expectations, separated from actual frozen CPU APIs."""
import argparse
from importlib import import_module,metadata
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN_LF_SHA256='c52afd90aab29d05fb6627e0e94301a3bdea6b7c128038a1d156650870902924'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def load_plan(path=ROOT/'benchmarks/dense-precision-audit-plan.json'):
    path=Path(path)
    if path.stat().st_size>65536:raise ValueError('precision plan bound exceeded')
    raw=path.read_bytes();lf=raw.replace(b'\r\n',b'\n')
    if b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')) or hashlib.sha256(lf).hexdigest()!=PLAN_LF_SHA256:
        raise ValueError('require the exact committed precision plan')
    return json.loads(lf)

def camera(config):
    w,h=config['width'],config['height']
    return [0.9375*w,1.2083333333333333*h,(w-1)/2.,(h-1)/2.]

def euler_rotation(w):
    x,y,z=map(float,w);sx,cx=math.sin(x),math.cos(x);sy,cy=math.sin(y),math.cos(y);sz,cz=math.sin(z),math.cos(z)
    rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],float)
    ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],float)
    rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],float)
    return rz@ry@rx

def vector_rotation(w):
    w=np.asarray(w,dtype=np.float64);angle=float(np.linalg.norm(w))
    if angle==0:return np.eye(3)
    q=w*(math.sin(angle/2)/angle);x,y,z=q;c=math.cos(angle/2)
    return np.array([[1-2*(y*y+z*z),2*(x*y-c*z),2*(x*z+c*y)],
        [2*(x*y+c*z),1-2*(x*x+z*z),2*(y*z-c*x)],
        [2*(x*z-c*y),2*(y*z+c*x),1-2*(x*x+y*y)]])

def proper_pose(value,tolerance=1e-10):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(4,4) or not np.all(np.isfinite(value)) or np.max(np.abs(value[3]-[0,0,0,1]))>tolerance:
        raise ValueError('finite proper 4x4 pose required')
    r=value[:3,:3]
    if abs(float(np.linalg.det(r))-1)>tolerance or np.max(np.abs(r.T@r-np.eye(3)))>tolerance:
        raise ValueError('proper rotation required')
    return value

def pose_from_rt(rotation,translation):
    rotation=np.asarray(rotation,dtype=np.float64);translation=np.asarray(translation,dtype=np.float64)
    if rotation.shape!=(3,3) or translation.shape!=(3,):raise ValueError('native rotation 3x3 and translation 3 required')
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=translation
    return proper_pose(pose)

def retract(initial,delta,kind):
    delta=np.asarray(delta,dtype=float)
    if delta.shape!=(6,) or not np.all(np.isfinite(delta)) or kind not in ('vector','euler'):
        raise ValueError('finite six-vector and declared retraction required')
    step=np.eye(4);step[:3,:3]=vector_rotation(delta[:3]) if kind=='vector' else euler_rotation(delta[:3]);step[:3,3]=delta[3:]
    return proper_pose(step@proper_pose(initial))

def fixture(config):
    y,x=np.mgrid[:config['height'],:config['width']];fx,fy,cx,cy=camera(config)
    def texture(a,b):return np.floor(np.clip(127+35*np.sin(.20*a+.10*b)+40*np.cos(.17*b-.20*a)+20*np.sin(.07*a*b),0,255)+.5).astype(np.uint8)
    depth=2+.05*np.sin(.20*x)+.04*np.cos(.25*y)+.02*np.sin(.15*(x+y))
    initial=np.eye(4);initial[:3,:3]=euler_rotation([0,0,config['initial_z_rotation']]);initial[:3,3]=config['initial_translation']
    return dict(name=config['name'],source_gray=texture(x,y).tolist(),target_gray=texture(x-config['shift'][0],y-config['shift'][1]).tolist(),
        source_depth=depth.astype(np.float32).tolist(),target_depth=(depth+config['depth_offset']+config['depth_ramp'][0]*(x-cx)+config['depth_ramp'][1]*(y-cy)).astype(np.float32).tolist(),initial=initial.tolist())

def make_inputs(plan):return dict(schema='spatialrust.dense-precision-inputs.v1',plan_lf_sha256=PLAN_LF_SHA256,cases=[fixture(c) for c in plan['cases']])

def arrays(payload,config):
    if type(payload) is not dict or set(payload)!={'name','source_gray','target_gray','source_depth','target_depth','initial'} or payload['name']!=config['name']:
        raise ValueError('exact synthetic case schema required')
    values=[]
    for key in ('source_gray','source_depth','target_gray','target_depth'):
        value=payload[key];shape=(config['height'],config['width'])
        if type(value) is not list or len(value)!=shape[0] or any(type(row) is not list or len(row)!=shape[1] for row in value):raise ValueError('fixed synthetic raster shape required')
        gray=key.endswith('gray')
        if any(type(v) is not int if gray else type(v) not in (int,float) for row in value for v in row):raise ValueError('exact gray/depth scalar types required')
        a=np.asarray(value,dtype=np.float64)
        if not np.all(np.isfinite(a)) or np.any(a<0) or (gray and np.any(a>255)) or (not gray and (np.any(a<.1) or np.any(a>5))):raise ValueError('finite bounded synthetic input required')
        values.append(np.array(value,dtype=np.uint8 if gray else np.float32,copy=True,order='C'))
    proper_pose(payload['initial'])
    return tuple(values)

def _kernel64(a,kx,ky):
    padded=np.pad(np.asarray(a,dtype=np.float64),1,mode='edge');out=np.zeros(a.shape,np.float64);h,w=a.shape
    for dy,wy in enumerate(ky):
        for dx,wx in enumerate(kx):out+=float(wx*wy)*padded[dy:dy+h,dx:dx+w]
    return out

def _separable32(a,kx,ky):
    value=np.asarray(a,dtype=np.float32)
    for axis,kernel in ((1,kx),(0,ky)):
        padding=((0,0),(1,1)) if axis==1 else ((1,1),(0,0));p=np.pad(value,padding,mode='edge');total=np.zeros(value.shape,np.float64);h,w=value.shape
        for i,k in enumerate(kernel):
            part=p[:,i:i+w] if axis==1 else p[i:i+h,:]
            total+=(part*np.float32(k)).astype(np.float64)
        value=total.astype(np.float32)
    return value

def preprocess(data,precision):
    if precision not in ('native','reference'):raise ValueError('declared arithmetic model required')
    sg,sd,tg,td=data;smooth=[1.,2.,1.]
    if precision=='native':
        result=[_kernel64(a.astype(float)/255. if a.dtype==np.uint8 else a.astype(float),smooth,smooth)/16. for a in data]
    else:
        result=[_separable32(a.astype(np.float32)/np.float32(255) if a.dtype==np.uint8 else a,[.25,.5,.25],[.25,.5,.25]) for a in data]
    return result

def point_grid(z,intrinsics,quantized=False):
    fx,fy,cx,cy=intrinsics;y,x=np.mgrid[:z.shape[0],:z.shape[1]];z=np.asarray(z,dtype=float)
    if quantized:
        xx=((x-cx)*z*(1./fx)).astype(np.float32).astype(float);yy=((y-cy)*z*(1./fy)).astype(np.float32).astype(float)
    else:xx=(x-cx)*z/fx;yy=(y-cy)*z/fy
    return np.stack([xx,yy,z],axis=-1).reshape(-1,3)

def normal_model(payload,config,plan,precision):
    data=arrays(payload,config);sg,sd,tg,td=preprocess(data,precision);intrinsics=camera(config);initial=proper_pose(payload['initial']);h,w=sd.shape
    matching=point_grid(sd,intrinsics)@initial[:3,:3].T+initial[:3,3]
    fx,fy,cx,cy=intrinsics;u=np.trunc(fx*matching[:,0]/matching[:,2]+cx+.5).astype(int);v=np.trunc(fy*matching[:,1]/matching[:,2]+cy+.5).astype(int)
    y,x=np.mgrid[:h,:w]
    if not np.array_equal(u,x.ravel()) or not np.array_equal(v,y.ravel()) or np.max(np.abs(td.ravel()-matching[:,2]))>plan['controls']['max_depth_difference']:
        raise ValueError('fixed synthetic same-pixel initial correspondence assumption failed')
    gains=[.5/float(np.mean(sg,dtype=np.float64)),.5/float(np.mean(tg,dtype=np.float64))]
    sg=sg.astype(float)*gains[0];tg=tg.astype(float)*gains[1]
    if precision=='reference':sg=sg.astype(np.float32);tg=tg.astype(np.float32)
    if precision=='native':grad=[_kernel64(a,kx,ky)*.125 for a in (tg,td) for kx,ky in (([-1.,0.,1.],[1.,2.,1.]),([1.,2.,1.],[-1.,0.,1.]))]
    else:grad=[_separable32(a,kx,ky).astype(float)*.125 for a in (tg,td) for kx,ky in (([-1.,0.,1.],[1.,2.,1.]),([1.,2.,1.],[-1.,0.,1.]))]
    p=point_grid(sd,intrinsics,precision=='reference')@initial[:3,:3].T+initial[:3,3];px,py,pz=p.T
    def jacobian(gx,gy,depth):
        a=gx.ravel()*fx/pz;b=gy.ravel()*fy/pz;c=-(a*px+b*py)/pz
        return np.stack([-pz*b+py*c-(py if depth else 0),pz*a-px*c+(px if depth else 0),-py*a+px*b,a,b,c-(1 if depth else 0)],axis=1)
    jp=jacobian(grad[0],grad[1],False);jd=jacobian(grad[2],grad[3],True)
    rp=(tg-sg).astype(float).ravel();rd=td.astype(float).ravel()-pz;weight=plan['controls']['depth_weight']
    if precision=='reference':
        jp=jp*math.sqrt(1-weight);jd=jd*math.sqrt(weight);rp=rp*math.sqrt(1-weight);rd=rd*math.sqrt(weight)
        matrix=jp.T@jp+jd.T@jd;rhs=-jp.T@rp-jd.T@rd
    else:matrix=(1-weight)*(jp.T@jp)+weight*(jd.T@jd);rhs=-(1-weight)*(jp.T@rp)-weight*(jd.T@rd)
    condition=float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition>plan['tolerances']['maximum_normal_condition']:raise ValueError('synthetic normal condition bound exceeded')
    delta=np.linalg.solve(matrix,rhs);residual=float(np.max(np.abs(matrix@delta-rhs))/max(1.,float(np.max(np.abs(rhs)))))
    if residual>plan['tolerances']['normal_relative_residual']:raise ValueError('independent normal solve residual exceeded')
    return dict(matrix=matrix.tolist(),rhs=rhs.tolist(),delta=delta.tolist(),condition=condition,relative_solve_residual=residual,modeled_correspondences=h*w,gains=gains,
        predictions={kind:retract(initial,delta,kind).tolist() for kind in ('vector','euler')})

def array_record(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    return dict(shape=list(a.shape),dtype='float32',sha256=hashlib.sha256(a.tobytes()).hexdigest())

def reference_filter_observations(cpu,data):
    return [array_record(np.asarray(cpu.geometry.Image(np.asarray(a,dtype=np.float32)/np.float32(255) if a.dtype==np.uint8 else a).filter(cpu.geometry.ImageFilterType.Gaussian3)).copy()) for a in data]

def verify_sources(plan,source_manifest,pose_manifest):
    manifest=json.loads(Path(source_manifest).read_text(encoding='utf-8'));pose=json.loads(Path(pose_manifest).read_text(encoding='utf-8'))
    rows=manifest['files']+[pose];wanted={r['upstream_path']:r for r in plan['reference']['sources']}
    if manifest['commit']!=plan['reference']['commit'] or pose['commit']!=plan['reference']['commit'] or len(rows)!=len(wanted) or {r['upstream_path'] for r in rows}!=set(wanted):raise ValueError('exact pinned source set required')
    for row in rows:
        target=wanted[row['upstream_path']]
        if row['bytes']!=target['bytes'] or row['sha256']!=target['sha256'] or Path(row['path']).stat().st_size!=target['bytes'] or sha(row['path'])!=target['sha256']:raise ValueError('pinned reference source changed')
    for row in plan['native']['source_files']:
        if hashlib.sha256((ROOT/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=row['lf_sha256']:raise ValueError('native source changed')

def observe(plan,payloads,role):
    if role=='native':
        backend=import_module('spatialrust');module=Path(import_module('spatialrust.spatialrust').__file__);expected=plan['native']['sha256']
    elif role=='reference':
        backend=import_module('open3d.cpu.pybind');module=Path(backend.__file__);expected=plan['reference']['cpu_module_sha256']
        if metadata.version('open3d')!=plan['reference']['version']:raise ValueError('reference version changed')
    else:raise ValueError('declared observer required')
    if sha(module)!=expected:raise ValueError('frozen numerical module required')
    observations=[]
    for payload,config in zip(payloads,plan['cases']):
        item=dict(name=config['name']);observations.append(item)
        try:
            data=arrays(payload,config);initial=proper_pose(payload['initial']);fx,fy,cx,cy=camera(config)
            if role=='native':
                rotation,translation,report=backend.estimate_dense_rgbd_odometry(*data,fx,fy,cx,cy,initial=initial,fitting_model='rounded-sobel',**plan['controls'])
                transform=pose_from_rt(rotation,translation)
                if report['iterations']!=1 or report['fitting_model']!='rounded-sobel':raise ValueError('native one-step receipt invalid')
                item['native_receipt']=report
            else:
                def rgbd(gray,depth):
                    value=backend.geometry.RGBDImage();value.color=backend.geometry.Image(gray.astype(np.float32)/np.float32(255));value.depth=backend.geometry.Image(depth);return value
                source=rgbd(data[0],data[1]);target=rgbd(data[2],data[3]);intrinsic=backend.camera.PinholeCameraIntrinsic(config['width'],config['height'],fx,fy,cx,cy)
                options=backend.pipelines.odometry.OdometryOption();options.iteration_number_per_pyramid_level=backend.utility.IntVector([1]);options.depth_min=plan['controls']['min_depth'];options.depth_max=plan['controls']['max_depth'];options.depth_diff_max=plan['controls']['max_depth_difference']
                success,transform,information=backend.pipelines.odometry.compute_rgbd_odometry(source,target,intrinsic,initial,backend.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),options)
                if not success:raise ValueError('external one-step failed')
                item['public_gaussian_filters']=reference_filter_observations(backend,data)
                item['operator_observation_scope']='Fresh source-constructed public Image operations; hidden odometry state is not exposed.'
            item.update(status='success',transform=proper_pose(transform).tolist())
        except Exception as error:item.update(status='error',error_type=type(error).__name__,error=str(error))
    if sha(module)!=expected:raise ValueError('numerical module changed during probe')
    return dict(schema='spatialrust.dense-precision-observation.v1',role=role,module=dict(path=str(module),bytes=module.stat().st_size,sha256=expected),numpy_version=np.__version__,backend='CPU',runtime_jacobian_observation=None,cases=observations)

def audit(plan,inputs,native,reference):
    names=[r['name'] for r in plan['cases']]
    for value in (inputs,native,reference):
        if [r['name'] for r in value['cases']]!=names:raise ValueError('all fixed synthetic cases in original order required')
    if native['role']!='native' or reference['role']!='reference' or native['module']['sha256']!=plan['native']['sha256'] or reference['module']['sha256']!=plan['reference']['cpu_module_sha256']:raise ValueError('exact observer roles/modules required')
    cases=[]
    for config,payload,a,b in zip(plan['cases'],inputs['cases'],native['cases'],reference['cases']):
        row=dict(name=config['name']);cases.append(row)
        try:
            if a['status']!='success' or b['status']!='success':raise ValueError('retain failed runtime observation')
            models={kind:normal_model(payload,config,plan,kind) for kind in ('native','reference')}
            errors=dict(native=float(np.max(np.abs(proper_pose(a['transform'])-models['native']['predictions']['vector']))),reference=float(np.max(np.abs(proper_pose(b['transform'])-models['reference']['predictions']['euler']))))
            filters=[array_record(v) for v in preprocess(arrays(payload,config),'reference')]
            if filters!=b['public_gaussian_filters']:raise ValueError('public reference Gaussian filter differs from source model')
            row.update(models=models,prediction_max_differences=errors,public_gaussian_filters_match=True,actual_pose_max_difference=float(np.max(np.abs(np.asarray(a['transform'])-np.asarray(b['transform'])))),modeled_retraction_max_differences={kind:float(np.max(np.abs(np.asarray(v['predictions']['vector'])-np.asarray(v['predictions']['euler'])))) for kind,v in models.items()})
            row['accepted']=bool(errors['native']<=plan['tolerances']['native_pose_max_difference'] and errors['reference']<=plan['tolerances']['reference_pose_max_difference'])
        except Exception as error:row.update(accepted=False,error_type=type(error).__name__,error=str(error))
    return dict(schema='spatialrust.dense-precision-audit.v1',cases=cases,all_cases_accepted=all(c['accepted'] for c in cases),runtime_jacobian_observation=None,scope='Source-modeled equations/retractions and actual public synthetic one-step poses; no causal attribution to real-data accuracy.',operational_80_percent_proven=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--role',choices=['native','reference'],required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True);parser.add_argument('--source-manifest',type=Path,required=True);parser.add_argument('--pose-source-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();plan=load_plan()
    verify_sources(plan,args.source_manifest,args.pose_source_manifest)
    if not args.inputs.is_absolute() or args.inputs.stat().st_size>4*1024**2 or sha(args.inputs)!=args.inputs_sha256:raise ValueError('bounded frozen inputs required')
    inputs=json.loads(args.inputs.read_text(encoding='utf-8'))
    if set(inputs)!={'schema','plan_lf_sha256','cases'} or inputs['schema']!='spatialrust.dense-precision-inputs.v1' or inputs['plan_lf_sha256']!=PLAN_LF_SHA256 or [c['name'] for c in inputs['cases']]!=[c['name'] for c in plan['cases']]:raise ValueError('exact complete input set required')
    output=args.output
    if not output.is_absolute() or output.exists() or output.parent.resolve()!=Path(plan['resources']['result_root']).resolve() or shutil.disk_usage(output.parent).free<plan['resources']['minimum_free_bytes']:raise ValueError('fresh owned output and space floor required')
    result=observe(plan,inputs['cases'],args.role);result.update(inputs_sha256=args.inputs_sha256,plan_lf_sha256=PLAN_LF_SHA256,probe_sha256=sha(__file__),thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')})
    if sha(args.inputs)!=args.inputs_sha256:raise ValueError('input changed during probe')
    verify_sources(plan,args.source_manifest,args.pose_source_manifest)
    output.write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode('utf-8'));print(json.dumps(dict(role=args.role,cases=[dict(name=c['name'],status=c['status']) for c in result['cases']])))

if __name__=='__main__':main()
