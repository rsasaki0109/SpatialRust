"""Persistent bounded protocol for the optional native PCL ICP comparator."""
from pathlib import Path
import selectors
import subprocess
import tempfile

import numpy as np


class PclIcpProcess:
    def __init__(self,path):
        self.path=Path(path).resolve(strict=True)
        self.diagnostics=tempfile.TemporaryFile()
        self.process=subprocess.Popen([str(self.path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.diagnostics,text=True,bufsize=1)
        self.selector=selectors.DefaultSelector()
        self.selector.register(self.process.stdout,selectors.EVENT_READ)
        self.version=None

    def align(self,source,target,gate,budget):
        for xyz in (source,target):
            if not isinstance(xyz,np.ndarray) or xyz.dtype!=np.float32 or xyz.ndim!=2 or xyz.shape[1]!=3 or not 3 <= len(xyz) <= 1000000 or not np.isfinite(xyz).all():
                raise ValueError('PCL comparison requires finite float32 Nx3 arrays with 3..1000000 points')
        if not np.isfinite(gate) or gate <= 0 or type(budget) is not int or not 1 <= budget <= 1000:
            raise ValueError('PCL gate/budget invalid')
        payload=f'{len(source)} {len(target)} {gate:.17g} {budget}\n'
        payload+='\n'.join(' '.join(format(float(v),'.9g') for v in row) for xyz in (source,target) for row in xyz)+'\n'
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError('PCL comparator terminated; check runtime library paths') from error
        if not self.selector.select(timeout=30):
            self.process.kill()
            raise RuntimeError('PCL comparator exceeded 30-second response budget')
        values=self.process.stdout.readline().split()
        if len(values)!=19:
            self.diagnostics.seek(0)
            detail=self.diagnostics.read(4096).decode(errors='replace')
            raise RuntimeError('PCL comparator returned malformed response: '+detail)
        version=values[0]
        if self.version is not None and self.version!=version:
            raise RuntimeError('PCL version changed during comparison')
        self.version=version
        iterations=int(values[1]);seconds=float(values[2])
        pose=np.asarray(values[3:],dtype=np.float64).reshape(4,4)
        if not 0 <= iterations <= budget or not np.isfinite(seconds) or seconds < 0 or not np.isfinite(pose).all():
            raise RuntimeError('PCL comparator returned invalid measurements')
        return pose,iterations,seconds

    def close(self):
        self.selector.close()
        self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait(timeout=2)
        self.process.stdout.close()
        self.diagnostics.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
