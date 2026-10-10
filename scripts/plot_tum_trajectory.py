"""Standalone post-fit TUM trajectories and error curves from hash-bound cases."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from timestamped_trajectory_reference import read_tum


def read_bound(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('plot input differs from frozen hash: '+str(path))
    return raw


def plot(manifest_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest_raw = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_raw)
    reference_raw = read_bound(manifest['reference'], manifest['reference_sha256'])
    reference = read_tum(reference_raw)
    if not 1 <= len(manifest['cases']) <= 8 or Path(output).exists():
        raise ValueError('require 1..8 cases and a fresh output directory')
    cases = []
    for item in manifest['cases']:
        est_raw = read_bound(item['estimates'], item['estimates_sha256'])
        score_raw = read_bound(item['accuracy'], item['accuracy_sha256'])
        est, score = json.loads(est_raw), json.loads(score_raw)
        if score['estimates_sha256'] != item['estimates_sha256'] or score['reference_sha256'] != manifest['reference_sha256']:
            raise ValueError('plot case does not bind the supplied reference/estimates')
        if len(est['poses']) != len(score['rows']):
            raise ValueError('plot cannot drop planned poses')
        if cases and (est['source_sha256'], est['calibration_sha256']) != (cases[0]['est']['source_sha256'], cases[0]['est']['calibration_sha256']):
            raise ValueError('comparison requires the same source/calibration')
        cases.append(dict(item=item, est=est, score=score, est_raw=est_raw, score_raw=score_raw))
    origin = min(c['est']['poses'][0]['timestamp_ns'] for c in cases)
    last = max(c['est']['poses'][-1]['timestamp_ns'] for c in cases)
    ref_xyz = np.array([r['world_from_sensor'][:3,3] for r in reference if origin <= r['timestamp_ns'] <= last])
    fig = plt.figure(figsize=(13,9), layout='constrained')
    spatial = fig.add_subplot(2,2,1,projection='3d')
    planar = fig.add_subplot(2,2,2)
    ate = fig.add_subplot(2,2,3)
    rpe = fig.add_subplot(2,2,4)
    spatial.plot(*ref_xyz.T, color='black', linewidth=2, label='Publisher reference')
    planar.plot(ref_xyz[:,0],ref_xyz[:,1],color='black',linewidth=2,label='Publisher reference')
    all_xyz = [ref_xyz]
    for c in cases:
        est, score, label = c['est'], c['score'], c['item']['label']
        alignment = np.array(score['first_pose_alignment'],float) if score['first_pose_alignment'] is not None else None
        xyz, distance, times = [], [], []
        for p,s in zip(est['poses'],score['rows']):
            if p['timestamp_ns'] != s['timestamp_ns']:
                raise ValueError('plot timelines differ')
            times.append((p['timestamp_ns']-origin)/1e9)
            xyz.append((alignment @ np.array(p['world_from_sensor']))[:3,3] if alignment is not None and p['status']=='success' else [np.nan]*3)
            distance.append(s.get('first_pose_aligned_error',{}).get('translation_m',np.nan))
        xyz = np.array(xyz)
        spatial.plot(*xyz.T,label=label)
        planar.plot(xyz[:,0],xyz[:,1],label=label)
        all_xyz.append(xyz[np.isfinite(xyz).all(axis=1)])
        ate.plot(times,distance,label='{} ({}/{})'.format(label,score['evaluated_poses'],score['planned_poses']))
        rx,ry=[],[]
        for pair in score['rpe_pairs']:
            rx.append(times[pair['source_index']])
            ry.append(pair.get('error',{}).get('translation_m',np.nan))
        rpe.plot(rx,ry,label='{} ({}/{})'.format(label,score['rpe_evaluated_pairs'],score['rpe_planned_pairs']))
    combined = np.vstack([a for a in all_xyz if len(a)])
    lower,upper=combined.min(axis=0),combined.max(axis=0)
    midpoint=(lower+upper)/2
    radius=max(float(np.max(upper-lower))/2,.01)*1.05
    for axis,value in zip(('x','y','z'),midpoint):
        getattr(spatial,'set_'+axis+'lim')(value-radius,value+radius)
    spatial.set_box_aspect((1,1,1))
    spatial.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title='Common world frame; first-pose SE(3) gauge')
    planar.set(xlabel='World X (m)',ylabel='World Y (m)',title='XY projection')
    planar.set_aspect('equal',adjustable='datalim')
    ate.set(xlabel='Seconds from first depth frame',ylabel='Translation error (m)',title='Pointwise ATE; missing reference remains a gap')
    rpe.set(xlabel='Source-frame time (s)',ylabel='Translation error (m)',title='One-second relative pose error')
    for axis in (spatial,planar,ate,rpe):
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    fig.suptitle('TUM freiburg1_xyz — frozen estimates, no scale or best-fit trajectory alignment',fontsize=12)
    # Verify plotted bytes before producing artifacts.
    read_bound(manifest['reference'],manifest['reference_sha256'])
    for c in cases:
        read_bound(c['item']['estimates'],c['item']['estimates_sha256'])
        read_bound(c['item']['accuracy'],c['item']['accuracy_sha256'])
    if Path(manifest_path).read_bytes()!=manifest_raw:
        raise ValueError('plot manifest changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/'trajectory.png',dpi=180)
    fig.savefig(output/'trajectory.svg')
    plt.close(fig)
    def metric(value):return 'unevaluated' if value is None else '{:.6f}'.format(value)
    rows=[]
    for c in cases:
        s=c['score'];label=html.escape(c['item']['label'])
        rows.append('<tr><td>{}</td><td>{}/{}/{}</td><td>{}</td><td>{}/{}</td><td>{}</td></tr>'.format(
            label,s['planned_poses'],s['generated_poses'],s['evaluated_poses'],metric(s['ate_first_pose_aligned_rmse_m']),
            s['rpe_evaluated_pairs'],s['rpe_planned_pairs'],metric(s['rpe_translation_rmse_m'])))
    svg=(output/'trajectory.svg').read_text(encoding='utf-8');svg=svg[svg.index('<svg'):]
    document=('<!doctype html><html lang="en"><meta charset="utf-8"><title>TUM trajectory evidence</title>'
              '<style>body{font:15px system-ui;margin:2rem}td,th{padding:.5rem;text-align:left}svg{width:100%;height:auto}</style>'
              '<h1>TUM freiburg1_xyz: frozen trajectory evidence</h1>'
              '<p>Same source, calibration and geometric budgets. Native and Open3D stopping semantics differ; '
              'this same-sequence diagnostic does not support a speed ranking or a new blind holdout claim.</p>'
              '<table><tr><th>Method</th><th>Planned / generated / evaluated</th><th>First-pose ATE RMSE (m)</th>'
              '<th>Evaluated / planned RPE pairs</th><th>RPE RMSE (m)</th></tr>'+''.join(rows)+'</table>'+svg+
              '<p>All planned failures and unmatched poses remain in the denominators. Geometric support and '
              'successful processing do not establish the correct trajectory. Publisher reference: '
              'https://cvg.cit.tum.de/data/datasets/rgbd-dataset</p></html>')
    (output/'report.html').write_text(document,encoding='utf-8')
    receipt=dict(schema='spatialrust.tum-plot.v1',manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                 plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
                 artifacts={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('trajectory.png','trajectory.svg','report.html')})
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    plot(args.manifest,args.output_dir)
    print(args.output_dir/'report.html')


if __name__=='__main__':
    main()
