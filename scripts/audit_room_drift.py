"""Complete frozen room metadata audit; never imports or calls an estimator."""
import argparse
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path
import stat

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dense_rgbd_protocol import CONTROLS
from evaluate_pose_reference import rigid
from rounded_sobel_dense_rgbd_protocol import POLICY as ROUNDED_POLICY
from reference_precision_dense_rgbd_protocol import POLICY as PRECISION_POLICY
from timestamped_trajectory_reference import decimal_timestamp_ns

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'benchmarks/room-drift-audit-plan.json'
PLAN_LF_SHA256 = '8e751ebbdd1662f82f7d571f97c46fd6a174137c231cd3dae8b1e525ff1dd34a'
POLICIES = {'rounded-sobel':ROUNDED_POLICY, 'reference-precision':PRECISION_POLICY}
COVARIATES = ('correspondence_fraction','photometric_rmse','depth_rmse_m',
              'support_fraction','support_rmse_m')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def load_plan(path=PLAN):
    raw=Path(path).read_bytes();lf=raw.replace(b'\r\n',b'\n')
    if len(raw)>65536 or b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')) or hashlib.sha256(lf).hexdigest()!=PLAN_LF_SHA256:
        raise ValueError('exact bounded prospective room protocol with uniform LF/CRLF required')
    return json.loads(lf)


def read_bound(record, maximum):
    if type(record) is not dict or set(record)!={'path','bytes','sha256'}:
        raise ValueError('exact path/bytes/SHA-256 input record required')
    path=Path(record['path'])
    if not path.is_absolute() or type(record['bytes']) is not int or not 0<record['bytes']<=maximum:
        raise ValueError('absolute bounded frozen input required')
    for p in [path]+list(path.parents):
        info=p.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400):
            raise ValueError('frozen inputs cannot traverse filesystem links')
    if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('regular frozen file required')
    raw=path.read_bytes()
    if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
        raise ValueError('frozen input identity changed')
    return raw


def finite(value, name, *, lower=None, upper=None):
    if type(value) not in (int,float) or not math.isfinite(value) or (lower is not None and value<lower) or (upper is not None and value>upper):
        raise ValueError('invalid finite scalar: '+name)
    return float(value)


def integer(value, name, lower, upper):
    if type(value) is not int or not lower<=value<=upper:
        raise ValueError('invalid integer: '+name)
    return value


def pose_error(actual, expected):
    actual,expected=rigid(actual),rigid(expected)
    vector=actual[:3,3]-expected[:3,3]
    rotation=Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3])
    return dict(translation_m=finite(math.hypot(*vector),'translation error',lower=0),
                translation_vector_m=vector.tolist(),
                rotation_degrees=float(np.degrees(rotation.magnitude())),
                rotation_vector_degrees=np.degrees(rotation.as_rotvec()).tolist())


def motion(pose):
    return pose_error(pose,np.eye(4))


def moments(values):
    values=[finite(v,'moment value',lower=0) for v in values]
    if not values:return dict(count=0,sse=0.,rmse=None,quantiles=None)
    try:sse=math.fsum(v*v for v in values)
    except OverflowError:raise ValueError('squared-error sum overflows') from None
    if not math.isfinite(sse):raise ValueError('squared-error sum overflows')
    ordered=sorted(values)
    def quantile(q):
        pos=q*(len(ordered)-1);lo=int(pos);hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(pos-lo)*(ordered[hi]-ordered[lo])
    return dict(count=len(values),sse=sse,rmse=math.sqrt(sse/len(values)),
                quantiles={str(q):quantile(q) for q in (0.,.25,.5,.75,.95,1.)})


def pearson(x,y):
    if len(x)!=len(y):raise ValueError('paired correlation observations required')
    n=len(x)
    if n<2:return dict(count=n,value=None,undefined_reason='fewer-than-two-observations')
    x=[finite(v,'correlation x') for v in x];y=[finite(v,'correlation y') for v in y]
    sx=max(map(abs,x));sy=max(map(abs,y))
    if sx==0 or sy==0:return dict(count=n,value=None,undefined_reason='constant-covariate-or-error')
    a=[v/sx for v in x];b=[v/sy for v in y];ma=math.fsum(a)/n;mb=math.fsum(b)/n
    a=[v-ma for v in a];b=[v-mb for v in b];na=math.hypot(*a);nb=math.hypot(*b)
    if na==0 or nb==0:return dict(count=n,value=None,undefined_reason='constant-covariate-or-error')
    coefficient=math.fsum((u/na)*(v/nb) for u,v in zip(a,b))
    return dict(count=n,value=max(-1.,min(1.,coefficient)),undefined_reason=None)


def references(raw,times,max_gap):
    stamps=[];xyz=[];quaternions=[]
    for line in raw.decode('ascii').splitlines():
        fields=line.split('#',1)[0].split()
        if not fields:continue
        if len(fields)!=8:raise ValueError('reference requires exact TUM rows')
        stamp=decimal_timestamp_ns(fields[0]);values=np.asarray([float(v) for v in fields[1:]],float)
        if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1.)>1e-3 or (stamps and stamp<=stamps[-1]):
            raise ValueError('reference poses must be finite unit-quaternion/time ordered')
        stamps.append(stamp);xyz.append(values[:3]);quaternions.append(values[3:])
    if not stamps:raise ValueError('nonempty reference required')
    xyz=np.asarray(xyz);quaternions=np.asarray(quaternions);result=[]
    for time in times:
        j=bisect_left(stamps,time);pose=None
        if j<len(stamps) and stamps[j]==time:
            pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(quaternions[j]).as_matrix();pose[:3,3]=xyz[j]
        elif 0<j<len(stamps) and max(time-stamps[j-1],stamps[j]-time)<=max_gap:
            interval=(stamps[j]-stamps[j-1])/1e9;offset=(time-stamps[j-1])/1e9
            pose=np.eye(4);pose[:3,:3]=Slerp([0.,interval],Rotation.from_quat(quaternions[j-1:j+1]))([offset]).as_matrix()[0]
            alpha=(time-stamps[j-1])/(stamps[j]-stamps[j-1]);pose[:3,3]=xyz[j-1]+alpha*(xyz[j]-xyz[j-1])
        result.append(rigid(pose) if pose is not None else None)
    return result


def pair_indices(times,delta,gap):
    pairs=[]
    for i,t in enumerate(times):
        j=bisect_left(times,t+delta);candidates=[k for k in (j-1,j) if i<k<len(times)]
        if candidates:
            target=min(candidates,key=lambda k:(abs(times[k]-t-delta),k))
            if abs(times[target]-t-delta)<=gap:pairs.append((i,target))
    return pairs


def _close(actual,expected,tolerance,name):
    if type(actual) is bool or type(expected) is bool or not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected)>tolerance:
        raise ValueError('frozen value differs: '+name)


def _compare_error(value,stored,c):
    if type(stored) is not dict or set(stored)!={'translation_m','rotation_degrees'}:
        raise ValueError('exact original translation/rotation error record required')
    for key,tol in [('translation_m',c['translation_comparison_tolerance_m']),('rotation_degrees',c['rotation_comparison_tolerance_degrees'])]:
        _close(value[key],finite(stored[key],key,lower=0),tol,key)


def _raw_receipt(report,check,label,c):
    if report.get('fitting_model')!=label or report.get('fitting_preprocessing')!='raw' or report.get('final_support_only') is not True or report.get('fitting_policy')!=POLICIES[label] or report.get('dense_settings')!=CONTROLS:
        raise ValueError('complete original fitting/final policy required')
    integer(report.get('iterations'),'iterations',c['iterations'],c['iterations'])
    valid=integer(report.get('valid_source_pixels'),'valid pixels',1,c['image_pixels'])
    count=integer(report.get('correspondences'),'correspondences',c['min_correspondences'],valid)
    fraction=finite(report.get('correspondence_fraction'),'correspondence fraction',lower=c['min_correspondence_fraction'],upper=1.)
    _close(fraction,count/valid,1e-12,'raw fraction denominator')
    integer(report.get('support_count'),'support count',3,c['image_pixels'])
    values={key:finite(report.get(key),key,lower=0,upper=1. if key.endswith('fraction') else None) for key in COVARIATES}
    if values['support_fraction']<c['min_support_fraction']:raise ValueError('original full-cloud support required')
    stats=check.get('statistics',{})
    for key in ('valid_source_pixels','correspondences'):
        if type(stats.get(key)) is not int or stats[key]!=report[key]:raise ValueError('original independent raw count differs')
    for key in ('correspondence_fraction','photometric_rmse','depth_rmse_m'):
        _close(values[key],finite(stats.get(key),key,lower=0),1e-12 if key.endswith('fraction') else 1e-10,'independent raw '+key)
    return dict(valid_source_pixels=valid,correspondences=count,**values)


def analyze_case(label,frozen,scored,raw_audit,score_audit,reference,c):
    if label not in POLICIES or frozen.get('schema')!='spatialrust.timestamped-trajectory.v1' or frozen.get('length_unit')!='m' or frozen.get('sensor_frame')!='tum_rgb_optical' or frozen.get('reference_used_for_generation') is not False:
        raise ValueError('original reference-free metre optical trajectory required')
    if scored.get('schema')!='spatialrust.timestamped-accuracy.v1' or raw_audit.get('schema')!='spatialrust.dense-rgbd-audit.v1' or score_audit.get('schema')!='spatialrust.tum-independent-audit.v1':
        raise ValueError('original score and independent audit schemas required')
    if scored.get('max_reference_gap_ns')!=c['max_reference_gap_ns'] or scored.get('rpe_requested_delta_ns')!=c['rpe_delta_ns']:
        raise ValueError('original timestamp controls required')
    for key in ('source_sha256','plan_sha256'):
        if scored.get(key)!=frozen.get(key):raise ValueError('original trajectory/score lineage differs')
    if raw_audit.get('reference_motion_used') is not False or raw_audit.get('manifest_sha256')!=frozen.get('manifest_sha256'):
        raise ValueError('original reference-free raw audit binding required')
    poses=frozen.get('poses',[]);n=c['planned_frames']
    if len(poses)!=n or len(scored.get('rows',[]))!=n or raw_audit.get('planned_frames')!=n:
        raise ValueError('complete planned timeline required')
    times=[];actual=[];lost=False
    for row in poses:
        t=integer(row.get('timestamp_ns'),'timestamp',0,2**63-1)
        if times and t<=times[-1]:raise ValueError('strictly ordered integer timestamps required')
        times.append(t)
        if row.get('status')=='success':
            if lost:raise ValueError('a disconnected reset cannot pass')
            actual.append(rigid(row['world_from_sensor']))
        elif row.get('status')=='error' and type(row.get('error')) is str and row['error']:
            actual.append(None);lost=True
        else:raise ValueError('every planned status/error must be retained')
    truth=references(reference,times,c['max_reference_gap_ns'])
    matched=[i for i in range(n) if actual[i] is not None and truth[i] is not None]
    anchor=matched[0] if matched else None;gauge=truth[anchor]@np.linalg.inv(actual[anchor]) if anchor is not None else None
    if scored.get('first_pose_alignment_anchor_index')!=anchor or (gauge is None)!=(scored.get('first_pose_alignment') is None):
        raise ValueError('original first-joint gauge/anchor required')
    if gauge is not None and not np.allclose(gauge,rigid(scored['first_pose_alignment']),rtol=0,atol=c['pose_composition_tolerance']):
        raise ValueError('original first-pose gauge differs')
    if actual[0] is not None and not np.allclose(actual[0],np.eye(4),rtol=0,atol=c['seed_tolerance']):
        raise ValueError('sole identity initial world anchor required')
    checks=raw_audit.get('pair_checks',[]);expected_checks=[i for i in range(1,n) if actual[i] is not None]
    if [row.get('index') for row in checks]!=expected_checks or raw_audit.get('successful_pairs')!=len(checks) or raw_audit.get('causal_initialization_pairs')!=len(checks) or raw_audit.get('failed_frame_indices')!=[i for i,a in enumerate(actual) if a is None]:
        raise ValueError('complete causal/raw audit coverage required')
    checks={row['index']:row for row in checks};rows=[];previous_relative=None
    for i,(a,t,stored) in enumerate(zip(actual,truth,scored['rows'])):
        if stored.get('timestamp_ns')!=times[i] or stored.get('generation_status')!=poses[i]['status'] or stored.get('reference_matched') is not (t is not None):
            raise ValueError('all original score coverage/statuses required')
        row=dict(index=i,timestamp_ns=times[i],elapsed_s=(times[i]-times[0])/1e9,status=poses[i]['status'],reference_matched=t is not None,aligned_error=None,unadjusted_error=None,adjacent=None,raw=None)
        if a is not None and t is not None:
            row['aligned_error']=pose_error(gauge@a,t);row['unadjusted_error']=pose_error(a,t)
            _compare_error(row['unadjusted_error'],stored.get('raw_error'),c);_compare_error(row['aligned_error'],stored.get('first_pose_aligned_error'),c)
        elif 'raw_error' in stored or 'first_pose_aligned_error' in stored:raise ValueError('missing reference cannot acquire an error')
        if i and a is not None:
            report=poses[i]['registration'];relative=rigid(report['current_from_previous']);inverse=rigid(report['previous_from_current']);initial=rigid(report['initial_current_from_previous'])
            expected_seed=np.eye(4) if previous_relative is None else previous_relative
            seed_index=report.get('initial_previous_pair_index')
            if report.get('initial_seed_policy')!='previous-relative-increment' or seed_index!=(i-1 if i>1 else None) or (i>1 and type(seed_index) is not int) or not np.allclose(initial,expected_seed,rtol=0,atol=c['seed_tolerance']):
                raise ValueError('original causal seed required')
            if not np.allclose(inverse,np.linalg.inv(relative),rtol=0,atol=c['seed_tolerance']) or not np.allclose(actual[i-1]@inverse,a,rtol=0,atol=c['pose_composition_tolerance']):
                raise ValueError('original inverse/world direction differs')
            row['raw']=_raw_receipt(report,checks[i],label,c);previous_relative=relative
            if truth[i-1] is not None and t is not None:
                ae=np.linalg.inv(actual[i-1])@a;gt=np.linalg.inv(truth[i-1])@t
                row['adjacent']=dict(source_index=i-1,target_index=i,error=pose_error(ae,gt),actual_motion=motion(ae),reference_motion=motion(gt))
        rows.append(row)
    pairs=pair_indices(times,c['rpe_delta_ns'],c['max_reference_gap_ns'])
    if [(r.get('source_index'),r.get('target_index')) for r in scored.get('rpe_pairs',[])]!=pairs:
        raise ValueError('complete original RPE pair denominator/ties required')
    rpe=[]
    for (i,j),stored in zip(pairs,scored['rpe_pairs']):
        evaluated=all(v[k] is not None for v in (actual,truth) for k in (i,j))
        if stored.get('evaluated') is not evaluated or stored.get('delta_ns')!=times[j]-times[i]:raise ValueError('RPE coverage/duration differs')
        row=dict(source_index=i,target_index=j,elapsed_s=(times[i]-times[0])/1e9,delta_ns=times[j]-times[i],evaluated=evaluated,error=None,window_mean_raw=None)
        if evaluated:
            ae=np.linalg.inv(actual[i])@actual[j];gt=np.linalg.inv(truth[i])@truth[j]
            row.update(error=pose_error(ae,gt),actual_motion=motion(ae),reference_motion=motion(gt));_compare_error(row['error'],stored.get('error'),c)
            parts=[r['raw'] for r in rows[i+1:j+1]]
            if any(p is None for p in parts):raise ValueError('a complete RPE window needs every original registration')
            row['window_mean_raw']={key:math.fsum(p[key] for p in parts)/len(parts) for key in COVARIATES}
        elif 'error' in stored:raise ValueError('unevaluated RPE pair cannot contain an error')
        rpe.append(row)
    counts=dict(planned_poses=n,generated_poses=sum(a is not None for a in actual),evaluated_poses=len(matched),unmatched_reference_poses=sum(t is None for t in truth),rpe_planned_pairs=len(rpe),rpe_evaluated_pairs=sum(p['evaluated'] for p in rpe))
    if any(type(scored.get(k)) is not int or scored[k]!=v or score_audit.get('counts',{}).get(k)!=v for k,v in counts.items()):raise ValueError('original full score counts differ')
    return dict(label=label,counts=counts,anchor_index=anchor,original_first_pose_gauge=gauge.tolist() if gauge is not None else None,rows=rows,rpe_pairs=rpe,**_summarize(rows,rpe,scored,score_audit,counts,c)),actual,truth


def _errors(items,field):
    values=[item[field] for item in items if item[field] is not None]
    return {key:moments([v[key] for v in values]) for key in ('translation_m','rotation_degrees')}


def _summarize(rows,rpe,scored,score_audit,counts,c):
    aligned=_errors(rows,'aligned_error');raw=_errors(rows,'unadjusted_error');relative=_errors(rpe,'error')
    adjacent=_errors([row['adjacent'] for row in rows if row['adjacent'] is not None],'error')
    for name,value in [('ate_raw_rmse_m',raw['translation_m']['rmse']),('ate_first_pose_aligned_rmse_m',aligned['translation_m']['rmse']),('rpe_translation_rmse_m',relative['translation_m']['rmse'])]:
        for other in [scored.get(name),score_audit.get('aggregates',{}).get(name)]:
            if (value is None)!=(other is None):raise ValueError('original aggregate coverage differs')
            if value is not None:_close(value,finite(other,name,lower=0),c['translation_comparison_tolerance_m'],name)
    n=len(rows);bins=[]
    for b in range(c['chronological_bins']):
        frame=[r for r in rows if c['chronological_bins']*r['index']//n==b]
        pairs=[p for p in rpe if c['chronological_bins']*p['source_index']//n==b]
        steps=[r['adjacent'] for r in frame if r['adjacent'] is not None]
        metrics=dict(aligned=_errors(frame,'aligned_error'),adjacent=_errors(steps,'error'),rpe=_errors(pairs,'error'))
        covariates={key:moments([r['raw'][key] for r in frame if r['raw'] is not None]) for key in COVARIATES}
        bins.append(dict(bin=b,indices=[r['index'] for r in frame],planned_frames=len(frame),reference_matched=sum(r['reference_matched'] for r in frame),generated_frames=sum(r['status']=='success' for r in frame),adjacent_evaluated=len(steps),rpe_planned=len(pairs),rpe_evaluated=sum(p['evaluated'] for p in pairs),metrics=metrics,raw_covariates=covariates))
    for name,total in [('aligned',aligned),('adjacent',adjacent),('rpe',relative)]:
        for key in ('translation_m','rotation_degrees'):
            count=sum(b['metrics'][name][key]['count'] for b in bins)
            sse=math.fsum(b['metrics'][name][key]['sse'] for b in bins)
            if count!=total[key]['count']:raise ValueError('binned denominator does not reconstruct complete timeline')
            _close(sse,total[key]['sse'],1e-12*max(1.,total[key]['sse']),'binned squared errors')
            for b in bins:
                b['metrics'][name][key]['whole_sse_fraction']=b['metrics'][name][key]['sse']/total[key]['sse'] if total[key]['sse'] else 0.
    associations={}
    for name,events,cov in [('adjacent',[r for r in rows if r['adjacent'] is not None],'raw'),('one_second',[p for p in rpe if p['evaluated']],'window_mean_raw')]:
        associations[name]={}
        for metric in ('translation_m','rotation_degrees'):
            y=[(r['adjacent']['error'] if name=='adjacent' else r['error'])[metric] for r in events]
            associations[name][metric]={key:pearson([r[cov][key] for r in events],y) for key in COVARIATES}
    return dict(metrics=dict(aligned=aligned,unadjusted=raw,adjacent=adjacent,rpe=relative),chronological_bins=bins,associations=associations,
        original_acceptance=bool(counts['generated_poses']==counts['planned_poses'] and counts['evaluated_poses']/counts['planned_poses']>=.95 and aligned['translation_m']['rmse'] is not None and aligned['translation_m']['rmse']<=.1 and relative['translation_m']['rmse'] is not None and relative['translation_m']['rmse']<=.05))


def compare_cases(cases,actual,truth):
    labels=list(cases)
    if len(labels)!=2:raise ValueError('both fixed candidates required')
    a,b=labels
    if [r['timestamp_ns'] for r in cases[a]['rows']]!=[r['timestamp_ns'] for r in cases[b]['rows']]:
        raise ValueError('candidate timelines differ')
    ga=np.asarray(cases[a]['original_first_pose_gauge']);gb=np.asarray(cases[b]['original_first_pose_gauge']);rows=[]
    for i,(pa,pb,ta,tb) in enumerate(zip(actual[a],actual[b],truth[a],truth[b])):
        if (ta is None)!=(tb is None):raise ValueError('candidate reference coverage differs')
        if ta is not None and not np.allclose(ta,tb,rtol=0,atol=1e-12):raise ValueError('candidate reference poses differ')
        common=all(v is not None for v in (pa,pb,ta,tb))
        rows.append(dict(index=i,timestamp_ns=cases[a]['rows'][i]['timestamp_ns'],common_reference_matched=common,error=pose_error(gb@pb,ga@pa) if common else None))
    return dict(first_candidate=a,second_candidate=b,alignment='Each original first-pose gauge only; no additional fit',rows=rows,metrics=_errors(rows,'error'))


def json_value(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON object key')
            result[key]=value
        return result
    def number(token):
        result=float(token)
        if not math.isfinite(result):raise ValueError('nonfinite JSON number')
        return result
    def constant(token):raise ValueError('nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_float=number,parse_constant=constant)


def render(result,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    with plt.rc_context({'path.simplify':False,'agg.path.chunksize':0,'svg.hashsalt':'spatialrust-room-drift-v1'}):
        figure,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        counts={}
        for label,case in result['cases'].items():
            rows=case['rows'];pairs=case['rpe_pairs'];x=[r['elapsed_s'] for r in rows]
            def curve(field,key):return [r[field][key] if r[field] is not None else math.nan for r in rows]
            axes[0,0].plot(x,curve('aligned_error','translation_m'),label=label)
            axes[0,1].plot(x,curve('aligned_error','rotation_degrees'),label=label)
            axes[1,0].plot(x,[r['adjacent']['error']['translation_m'] if r['adjacent'] is not None else math.nan for r in rows],label=label)
            axes[1,1].plot([p['elapsed_s'] for p in pairs],[p['error']['translation_m'] if p['error'] is not None else math.nan for p in pairs],label=label)
            counts[label]=dict(all_frame_samples=len(rows),all_planned_rpe_pairs=len(pairs))
        for axis,title,ylabel in zip(axes.flat,['Original first-pose gauge: position','Original first-pose gauge: orientation','Adjacent step error','Original 1s relative error'],['Translation error (m)','Rotation error (degrees)','Translation error (m)','Translation error (m)']):
            axis.set(title=title,xlabel='Time from first planned frame (s)',ylabel=ylabel);axis.grid(alpha=.25);axis.legend()
        figure.suptitle('Complete observed room trajectories; original global acceptance remains separate')
        figure.savefig(directory/'drift.svg',metadata={'Date':None})
        figure.savefig(directory/'preview.png',dpi=140)
        plt.close(figure)
    return dict(matplotlib=matplotlib.__version__,plotted_counts=counts,path_simplification=False,missing_reference_samples='NaN gaps retained')


def run(output,plan):
    output=Path(output);root=Path(plan['resources']['result_root'])
    if not output.is_absolute() or output.resolve()!=root.resolve()/'audit' or output.exists() or not root.is_dir():
        raise ValueError('fresh exact owned audit output required')
    cases={};actual={};truth={};bindings={};lineage=[]
    for item in plan['cases']:
        label=item['label'];raws={k:read_bound(v,plan['resources']['input_file_max_bytes']) for k,v in item['files'].items()}
        read_bound(item['native_extension'],plan['resources']['input_file_max_bytes'])
        values={k:json_value(v) for k,v in raws.items() if k!='reference_file'}
        frozen=values['estimates'];score=values['evaluation_file'];audit=values['projective_audit_file'];independent=values['score_audit_file']
        if frozen.get('method')!=item['estimator']:raise ValueError('fixed candidate estimator required')
        digest=item['files']['estimates']['sha256'];ref=item['files']['reference_file']['sha256']
        if score.get('estimates_sha256')!=digest or audit.get('estimates_sha256')!=digest or independent.get('estimates_sha256')!=digest or independent.get('evaluation_sha256')!=item['files']['evaluation_file']['sha256'] or score.get('reference_sha256')!=ref or independent.get('reference_sha256')!=ref:
            raise ValueError('exact original independent input bindings required')
        cases[label],actual[label],truth[label]=analyze_case(label,frozen,score,audit,independent,raws['reference_file'],plan['controls'])
        lineage.append((frozen['source_sha256'],frozen['calibration_sha256'],ref));bindings[label]=dict(files=item['files'],native_extension=item['native_extension'])
    if len(set(lineage))!=1:raise ValueError('same original recording/calibration/reference required')
    result=dict(schema='spatialrust.complete-room-drift-audit.v1',plan_lf_sha256=PLAN_LF_SHA256,cases=cases,comparison=compare_cases(cases,actual,truth),bindings=bindings,scope=plan['scope'],association_scope=plan['associations'],new_estimator_calls=0,sensor_image_reads=0,archive_reads=0,sensor_payload_copies=0,runtime_jacobian_observation=None,runtime_normal_system_observation=None,observed_development_only=True,operational_80_percent_proven=False)
    for item in plan['cases']:
        for record in item['files'].values():read_bound(record,plan['resources']['input_file_max_bytes'])
    output.mkdir()
    result['plot']=render(result,output)
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();plan=load_plan()
    result=run(args.output_dir,plan)
    print(json.dumps(dict(cases={name:dict(counts=c['counts'],metrics=c['metrics'],accepted=c['original_acceptance']) for name,c in result['cases'].items()},operational_80_percent_proven=False)))


if __name__=='__main__':main()
