"""Optional OpenCV PnP comparison verifies the paired experiment contract."""
import sys
from pathlib import Path

import pytest

pytest.importorskip('cv2')
pytest.importorskip('spatialrust')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import compare_pnp_failures as study


def test_paired_inputs_and_outlier_complementarity():
    rows=study.run(seeds=1,budget=300)
    assert len(rows)==48
    pairs={}
    for row in rows:
        pairs.setdefault((row['geometry'],row['condition'],row['seed']),[]).append(row)
    for group in pairs.values():
        assert len(group)==4
        assert len({r['object_sha256'] for r in group})==len({r['image_sha256'] for r in group})==1
        assert all(r['correspondences']==100 for r in group)
    contaminated={r['method']:r for r in pairs[('volume','outliers',0)]}
    assert contaminated['spatialrust_ransac']['recovered']
    assert not contaminated['spatialrust_plain']['recovered']
    assert contaminated['opencv_ransac']['recovered']
    assert not contaminated['opencv_plain']['recovered']
    assert all(r['recovered'] for r in pairs[('volume','clean',0)])
    assert 'minimal sample sizes differ' in study.__doc__
    assert 'cannot determine all rotation' in study.render(rows)


def test_existing_output_rejected_before_study(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['compare','--output-dir',str(tmp_path)])
    monkeypatch.setattr(study,'run',lambda *args:pytest.fail('study before output validation'))
    with pytest.raises(SystemExit):study.main()
