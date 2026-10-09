"""Check installed runtime bytes against a wheel, then exercise conversion/ICP/PCD.

Run with the fresh virtualenv's Python after installing the wheel. This fails
for an editable or different installed native module, and records artifact hashes.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
import xml.etree.ElementTree as ET
import zipfile


def verify(wheel):
    import numpy as np
    import spatialrust as sr
    distribution=importlib.metadata.distribution('spatialrust')
    direct=distribution.read_text('direct_url.json')
    if direct and json.loads(direct).get('dir_info',{}).get('editable'):
        raise ValueError('editable install does not validate a wheel')
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist()
        required=['spatialrust/__init__.py','spatialrust/__init__.pyi','spatialrust/py.typed']
        natives=[name for name in names if name.startswith('spatialrust/') and name.endswith(('.so','.pyd'))]
        if len(natives)!=1 or any(names.count(name)!=1 for name in required+natives):
            raise ValueError('wheel must contain one native module and unique package/type files')
        hashes={}
        for name in required+natives:
            installed=Path(distribution.locate_file(name)).resolve()
            expected=archive.read(name)
            if installed.read_bytes()!=expected:
                raise ValueError(f'installed bytes differ from wheel: {name}')
            hashes[name]=hashlib.sha256(expected).hexdigest()
        imported=Path(importlib.import_module('spatialrust.spatialrust').__file__).resolve()
        if imported!=Path(distribution.locate_file(natives[0])).resolve():
            raise ValueError('imported native module is outside installed wheel')
        if Path(sr.__file__).resolve()!=Path(distribution.locate_file(required[0])).resolve():
            raise ValueError('imported Python package is outside installed wheel')
    xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    source=sr.PointCloud.from_xyz(xyz+[np.float32(.01),np.float32(-.02),np.float32(.005)])
    target=sr.PointCloud.from_xyz(xyz)
    result=sr.register_icp(source,target,.2,10)
    aligned=sr.apply_transform(source,result.transform())
    np.testing.assert_allclose(aligned.xyz(),xyz,atol=1e-6)
    count,fraction,rmse=sr.DistanceSupportIndex(target).support(aligned,.001)
    if count!=4 or fraction!=1 or rmse is None or rmse>1e-6:
        raise ValueError('installed wheel support regression')
    with tempfile.TemporaryDirectory(prefix='spatialrust-wheel-') as temporary:
        path=Path(temporary)/'aligned.pcd'
        sr.write(str(path),aligned)
        np.testing.assert_array_equal(sr.read(str(path)).xyz(),aligned.xyz())
    return dict(schema='spatialrust.installed-wheel-validation.v1',wheel_name=Path(wheel).name,
        wheel_sha256=hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
        package_file_sha256=hashes,version=distribution.version,python=platform.python_version(),
        platform=platform.platform(),numpy=np.__version__,native_file=str(imported),
        noneditable_installed_bytes_match=True,icp_and_support_passed=True,exact_xyz_roundtrip=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--junit',type=Path)
    parser.add_argument('--expected-skips',type=int,default=3)
    args=parser.parse_args()
    if args.receipt.exists():
        parser.error('receipt already exists')
    result=verify(args.wheel)
    if args.junit:
        tree=ET.parse(args.junit).getroot()
        cases=list(tree.iter('testcase'))
        skipped=sum(case.find('skipped') is not None for case in cases)
        failed=sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
        if not cases or failed or skipped!=args.expected_skips:
            raise ValueError('wheel tests failed, absent, or unexpectedly skipped')
        result['tests']=dict(passed=len(cases)-skipped,skipped=skipped,failed=failed,
                            junit_sha256=hashlib.sha256(args.junit.read_bytes()).hexdigest())
    with args.receipt.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
    print(f"Verified installed wheel {result['wheel_name']}; native bytes and runtime checks match")


if __name__=='__main__':
    main()
