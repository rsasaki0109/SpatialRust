"""Range admission preserves a version-bound prefix without overlapping payloads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
import acquire_heldout_rgbd_ranges as ranges
from heldout_dense_rgbd_protocol import selection


class Response(io.BytesIO):
    def __init__(self,body,source,start,end,total,change=None):
        super().__init__(body);self.status=206;self.url=source['url'];self.headers={'ETag':source['etag'],'Content-Range':'bytes {}-{}/{}'.format(start,end,total),'Content-Length':str(end-start+1)}
        if change=='status':self.status=200
        elif change=='etag':self.headers['ETag']='other'
        elif change=='start':self.headers['Content-Range']='bytes {}-{}/{}'.format(start+1,end,total)
        elif change=='total':self.headers['Content-Range']='bytes {}-{}/{}'.format(start,end,total+1)
        elif change=='encoding':self.headers['Content-Encoding']='gzip'
        elif change=='length':self.headers['Content-Length']=str(end-start)


class Opener:
    def __init__(self,body,source,change=None):self.body=body;self.source=source;self.change=change;self.requests=[]
    def open(self,request,timeout):
        a,b=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups());self.requests.append((a,b))
        body=self.body[a:b+1]
        if self.change=='truncated':body=body[:-1]
        elif self.change=='overflow':body+=b'x'
        return Response(body,self.source,a,b,len(self.body),self.change)


def case(tmp_path,monkeypatch,prefix=0):
    root=tmp_path/'inputs';root.mkdir();body=b'\x1f\x8b'+bytes(range(40))
    source=copy.deepcopy(selection()['sources'][0]);source['content_length']=len(body)
    limits=copy.deepcopy(selection()['resources']);limits['archives_max_bytes']=1024
    monkeypatch.setattr(ranges.shutil,'disk_usage',lambda _:SimpleNamespace(free=32*1024**3))
    if prefix:
        directory=root/source['sequence'];directory.mkdir();(directory/(source['sequence']+'.tgz.part')).write_bytes(body[:prefix])
        (directory/'acquisition.json').write_text(json.dumps(dict(source=source,status='incomplete')),encoding='utf-8')
    return root,source,limits,body


@pytest.mark.parametrize('prefix',[0,7,41])
def test_nonoverlapping_ranges_keep_original_prefix_and_exact_file(tmp_path,monkeypatch,prefix):
    root,source,limits,body=case(tmp_path,monkeypatch,prefix);opener=Opener(body,source)
    result=ranges.acquire_ranges(source,root,limits,opener=opener,segment_bytes=8)
    archive=Path(result['archive']['path']);assert archive.read_bytes()==body
    assert result['archive']['sha256']==hashlib.sha256(body).hexdigest()
    assert result['preserved_original_prefix']['sha256']==hashlib.sha256(body[:prefix]).hexdigest()
    assert result['downloaded_payload_bytes']==len(body)-prefix and result['additional_payload_copies']==0
    assert opener.requests[0][0]==prefix
    assert all(a[1]+1==b[0] for a,b in zip(opener.requests,opener.requests[1:]))
    with pytest.raises(FileExistsError):ranges.acquire_ranges(source,root,limits,opener=opener,segment_bytes=8)


@pytest.mark.parametrize('change',['status','etag','start','total','encoding','length','truncated','overflow'])
def test_wrong_range_identity_or_body_cannot_replace_prefix(tmp_path,monkeypatch,change):
    root,source,limits,body=case(tmp_path,monkeypatch,7);opener=Opener(body,source,change)
    with pytest.raises(ValueError):ranges.acquire_ranges(source,root,limits,opener=opener,segment_bytes=8)
    directory=root/source['sequence'];partial=directory/(source['sequence']+'.tgz.part')
    assert partial.read_bytes().startswith(body[:7]) and not (directory/(source['sequence']+'.tgz')).exists()
    assert json.loads((directory/'acquisition-ranges.json').read_text(encoding='utf-8'))['status']=='failed'


def test_low_space_refuses_before_range_request(tmp_path,monkeypatch):
    root,source,limits,body=case(tmp_path,monkeypatch,7);opener=Opener(body,source)
    monkeypatch.setattr(ranges.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='20GiB'):ranges.acquire_ranges(source,root,limits,opener=opener)
    assert opener.requests==[]
