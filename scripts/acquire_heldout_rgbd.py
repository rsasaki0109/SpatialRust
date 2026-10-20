"""Acquire only fixed opaque public archives; never inspect images or reference poses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.request
from datetime import datetime,timezone

from heldout_dense_rgbd_protocol import ROOT, selection
from regular_file_inventory import inventory


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def regular_tree(path):
    path=Path(path)
    nodes=[path]+list(path.parents)
    for node in nodes:
        info=node.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0):
            raise ValueError('source root and ancestors must not be links or reparse points')


def preflight(root,minimum):
    root=Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('existing absolute source root required')
    regular_tree(root)
    resolved=root.resolve()
    if resolved==ROOT or ROOT in resolved.parents:
        raise ValueError('sensor archives must stay outside the repository')
    if shutil.disk_usage(resolved).free<minimum:
        raise ValueError('20GiB source free-space floor required before admission')
    return resolved


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('fixed source endpoint must not redirect')


def acquire(source,root,resources,*,opener=None):
    root=preflight(root,resources['minimum_free_bytes'])
    name=source['sequence']
    if name not in ('rgbd_dataset_freiburg1_plant','rgbd_dataset_freiburg1_room'):
        raise ValueError('fixed source recording required')
    expected_url='https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/'+name+'.tgz'
    if source['url']!=expected_url or source['effective_url']!=expected_url or type(source['content_length']) is not int or not 0<source['content_length']<=resources['archives_max_bytes']:
        raise ValueError('fixed HTTPS source URL and bounded exact length required')
    destination=root/name
    if destination.exists():raise FileExistsError('preserve existing source acquisition and failures')
    destination.mkdir()
    partial=destination/(name+'.tgz.part');archive=destination/(name+'.tgz')
    journal=destination/'acquisition.json'
    started=time.monotonic();count=0;digest=hashlib.sha256();first=b''
    record=dict(schema='spatialrust.heldout-acquisition.v1',sequence=name,source=source,
        started_at_utc=datetime.now(timezone.utc).isoformat(),free_before_bytes=shutil.disk_usage(root).free,
        image_or_reference_interpretation=False,status='incomplete',samples=[])
    def save():
        journal.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    try:
        # Before opening a remote stream, account for existing source payloads.
        previous=inventory(root,filename_pattern='*.tgz*').logical_bytes
        if previous+source['content_length']>resources['archives_max_bytes']:
            raise ValueError('combined source archive budget exceeded')
        if opener is None:opener=urllib.request.build_opener(NoRedirect())
        request=urllib.request.Request(expected_url,headers={'Accept-Encoding':'identity','If-Match':source['etag']})
        save()
        with opener.open(request,timeout=30) as response:
            if response.status!=200 or response.url!=expected_url or int(response.headers.get('Content-Length','-1'))!=source['content_length'] or response.headers.get('ETag')!=source['etag'] or response.headers.get('Content-Encoding','identity')!='identity':
                raise ValueError('source status, endpoint, length, ETag or encoding changed')
            with partial.open('xb') as output:
                while True:
                    if shutil.disk_usage(root).free<resources['minimum_free_bytes'] or time.monotonic()-started>resources['source_command_max_seconds']:
                        raise ValueError('source free-space or duration bound breached')
                    block=response.read(1024**2)
                    if not block:break
                    if count+len(block)>source['content_length'] or previous+count+len(block)>resources['archives_max_bytes']:
                        raise ValueError('source length or combined archive budget exceeded')
                    first=(first+block)[:2] if len(first)<2 else first
                    output.write(block);digest.update(block);count+=len(block)
                    if count%(32*1024**2)==0:
                        record['samples'].append(dict(bytes=count,free_bytes=shutil.disk_usage(root).free))
                        save();print('{} bytes {}/{}'.format(name,count,source['content_length']),flush=True)
        if count!=source['content_length'] or first!=b'\x1f\x8b' or sha(partial)!=digest.hexdigest():
            raise ValueError('source truncated, not gzip, or disk hash differs from stream')
        regular_tree(destination)
        assert partial.parent.resolve()==archive.parent.resolve()==destination.resolve() and not archive.exists()
        partial.rename(archive)
        record.update(status='complete',archive=dict(path=str(archive),bytes=count,sha256=digest.hexdigest()),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),free_after_bytes=shutil.disk_usage(root).free,
            downloaded_payload_bytes=count,additional_payload_copies=0)
        save();return record
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),
            downloaded_payload_bytes=count,finished_at_utc=datetime.now(timezone.utc).isoformat(),
            preserved_partial=dict(path=str(partial),bytes=partial.stat().st_size,sha256=sha(partial)) if partial.exists() else None)
        save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['plant','room'],required=True)
    parser.add_argument('--input-root',type=Path,required=True)
    args=parser.parse_args();spec=selection()
    source=next(row for row in spec['sources'] if row['sequence']=='rgbd_dataset_freiburg1_'+args.scene)
    result=acquire(source,args.input_root,spec['resources'])
    print(json.dumps(dict(sequence=result['sequence'],archive=result['archive'],status=result['status'])))


if __name__=='__main__':main()
