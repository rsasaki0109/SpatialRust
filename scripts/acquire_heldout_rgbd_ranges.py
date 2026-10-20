"""Verified non-overlapping HTTP ranges for the same fixed opaque source bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
from datetime import datetime,timezone

from acquire_heldout_rgbd import NoRedirect,preflight,regular_tree,sha
from heldout_dense_rgbd_protocol import selection
from regular_file_inventory import inventory

SEGMENT_BYTES=128*1024**2


def prefix_sha(path,count):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while count:
            block=stream.read(min(count,1024**2))
            if not block:raise ValueError('prefix is truncated')
            digest.update(block);count-=len(block)
    return digest.hexdigest()


def acquire_ranges(source,root,resources,*,opener=None,segment_bytes=SEGMENT_BYTES):
    root=preflight(root,resources['minimum_free_bytes'])
    name=source['sequence'];expected='https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/'+name+'.tgz'
    if name not in ('rgbd_dataset_freiburg1_plant','rgbd_dataset_freiburg1_room') or source['url']!=expected or source['effective_url']!=expected or type(source['content_length']) is not int or not 0<source['content_length']<=resources['archives_max_bytes'] or not 1<=segment_bytes<=SEGMENT_BYTES:
        raise ValueError('fixed source identity and bounded range size required')
    destination=root/name;partial=destination/(name+'.tgz.part');archive=destination/(name+'.tgz');journal=destination/'acquisition-ranges.json'
    if archive.exists() or journal.exists():raise FileExistsError('preserve previous completed source/range attempts')
    if destination.exists():
        previous=json.loads((destination/'acquisition.json').read_text(encoding='utf-8'))
        if previous['source']!=source or previous['status']=='complete' or not partial.is_file():
            raise ValueError('resume requires the original incomplete identity and preserved prefix')
        regular_tree(partial)
    else:destination.mkdir()
    initial=partial.stat().st_size if partial.exists() else 0
    if initial>source['content_length']:raise ValueError('source prefix exceeds the fixed archive length')
    prefix=dict(bytes=initial,sha256=prefix_sha(partial,initial) if initial else hashlib.sha256(b'').hexdigest())
    other=inventory(root,filename_pattern='*.tgz*').logical_bytes-initial
    if other+source['content_length']>resources['archives_max_bytes']:raise ValueError('combined source archive budget exceeded')
    record=dict(schema='spatialrust.heldout-range-acquisition.v1',source=source,sequence=name,
        started_at_utc=datetime.now(timezone.utc).isoformat(),status='incomplete',segments=[],
        preserved_original_prefix=prefix,image_or_reference_interpretation=False,additional_payload_copies=0)
    def save():journal.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    count=initial;started=time.monotonic();save()
    if opener is None:opener=urllib.request.build_opener(NoRedirect())
    try:
        while count<source['content_length']:
            begin=count;end=min(begin+segment_bytes,source['content_length'])-1
            request=urllib.request.Request(expected,headers={'Accept-Encoding':'identity','If-Match':source['etag'],'Range':'bytes={}-{}'.format(begin,end)})
            item=dict(start=begin,end=end,bytes=0);record['segments'].append(item);save()
            with opener.open(request,timeout=30) as response:
                if response.status!=206 or response.url!=expected or response.headers.get('ETag')!=source['etag'] or response.headers.get('Content-Range')!='bytes {}-{}/{}'.format(begin,end,source['content_length']) or int(response.headers.get('Content-Length','-1'))!=end-begin+1 or response.headers.get('Content-Encoding','identity')!='identity':
                    raise ValueError('fixed source range/status/length/ETag/encoding mismatch')
                with partial.open('ab' if partial.exists() else 'xb') as output:
                    while True:
                        if shutil.disk_usage(root).free<resources['minimum_free_bytes'] or time.monotonic()-started>resources['source_command_max_seconds']:
                            raise ValueError('range acquisition free-space or duration bound breached')
                        block=response.read(1024**2)
                        if not block:break
                        if count+len(block)>end+1:raise ValueError('range body exceeds declared end')
                        output.write(block);count+=len(block);item['bytes']+=len(block)
            if count!=end+1:raise ValueError('range body truncated')
            item['complete']=True;save();print('{} verified bytes {}/{}'.format(name,count,source['content_length']),flush=True)
        if partial.stat().st_size!=source['content_length'] or prefix_sha(partial,initial)!=prefix['sha256']:
            raise ValueError('completed file length or preserved prefix changed')
        with partial.open('rb') as stream:
            if stream.read(2)!=b'\x1f\x8b':raise ValueError('completed source is not gzip')
        digest=sha(partial);regular_tree(partial)
        if partial.parent.resolve()!=archive.parent.resolve() or archive.exists():raise ValueError('unexpected source rename target')
        partial.rename(archive)
        if prefix_sha(archive,initial)!=prefix['sha256']:raise ValueError('preserved prefix differs after source freeze')
        record.update(status='complete',archive=dict(path=str(archive),bytes=count,sha256=digest),
            downloaded_payload_bytes=count-initial,preserved_prefix_in_completed_archive=True,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),free_after_bytes=shutil.disk_usage(root).free)
        save();return record
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),
            downloaded_payload_bytes=count-initial,preserved_partial=dict(path=str(partial),bytes=partial.stat().st_size,sha256=sha(partial)) if partial.exists() else None,
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['plant','room'],required=True)
    parser.add_argument('--input-root',type=Path,required=True)
    args=parser.parse_args();spec=selection();source=next(r for r in spec['sources'] if r['sequence']=='rgbd_dataset_freiburg1_'+args.scene)
    result=acquire_ranges(source,args.input_root,spec['resources'])
    print(json.dumps(dict(sequence=result['sequence'],archive=result['archive'],status=result['status'])))


if __name__=='__main__':main()
