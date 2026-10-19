"""Only the named unavailable external comparator may add a wheel-test skip."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

SOURCE=Path(__file__).resolve().parents[1]/'tools/verify_installed_wheel.py'
SPEC=importlib.util.spec_from_file_location('installed_wheel_junit_check',SOURCE)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def report(tmp_path, *, external=0, damage=None):
    root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite')
    for i in range(3):
        node=ET.SubElement(suite,'testcase',classname='existing.optional',name='optional_'+str(i))
        ET.SubElement(node,'skipped',message='existing optional model')
    for _ in range(external):
        node=ET.SubElement(suite,'testcase',classname='tests.test_tum_open3d_rgbd',
            name='test_installed_legacy_cpu_images_own_float_buffers_without_estimation')
        ET.SubElement(node,'skipped',message="could not import 'open3d.cpu.pybind': No module named 'open3d'")
    if damage=='name':node.set('name','different_test')
    if damage=='module':node.set('classname','other_test_module')
    if damage=='reason':node.find('skipped').set('message','real correctness regression')
    if damage=='failure':ET.SubElement(ET.SubElement(suite,'testcase',name='regression'),'failure')
    if damage=='extra_skip':ET.SubElement(ET.SubElement(suite,'testcase',name='regression'),'skipped')
    if damage=='empty':root=ET.Element('testsuites')
    path=tmp_path/'report.xml';ET.ElementTree(root).write(path,encoding='utf-8')
    return path


@pytest.mark.parametrize('external',[0,1])
def test_baseline_skip_count_and_one_identified_external_skip(tmp_path,external):
    result=MODULE.verify_junit(report(tmp_path,external=external))
    assert result['failed']==0 and result['skipped']==3+external
    assert result['optional_external_comparator_skipped']==external


@pytest.mark.parametrize('damage',['name','module','reason','failure','extra_skip','empty'])
def test_other_skips_failures_and_absent_coverage_reject(tmp_path,damage):
    with pytest.raises(ValueError):MODULE.verify_junit(report(tmp_path,external=1,damage=damage))


def test_duplicate_external_skip_cannot_mask_missing_coverage(tmp_path):
    with pytest.raises(ValueError):MODULE.verify_junit(report(tmp_path,external=2))
