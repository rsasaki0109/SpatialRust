"""Expose pytest failures in Actions annotations without downloading job logs."""
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def escape(value):
    return value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


def annotations(path):
    root = ET.parse(path).getroot()
    for case in root.iter('testcase'):
        for failure in list(case):
            if failure.tag not in ('failure', 'error'):
                continue
            name = '.'.join(filter(None, (case.get('classname'), case.get('name'))))
            detail = failure.text or failure.get('message', '')
            yield '::error::' + escape(name + '\n' + detail)


def main():
    # Actions consumes UTF-8 commands even when Windows redirects cp1252 stdout.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('junit', type=Path)
    args = parser.parse_args()
    if not args.junit.exists():
        print('::warning::JUnit report is absent; failure happened before test reporting.')
        return
    for annotation in annotations(args.junit):
        print(annotation)


if __name__ == '__main__':
    main()

