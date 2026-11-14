"""Verify archived original files; --require-large checks restored Release data."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify(require_large=False):
 manifest=json.loads((ROOT/'archive_manifest.json').read_text());checked=skipped=0;errors=[]
 for r in manifest['entries']:
  p=ROOT/r['path']
  if r['placement']=='release' and not require_large and not p.exists():skipped+=1;continue
  if not p.exists():errors.append('missing: '+r['path']);continue
  if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:errors.append('hash/size: '+r['path'])
  checked+=1
 inventory=ROOT/'git_files_manifest.json'
 if inventory.exists():
  for r in json.loads(inventory.read_text())['files']:
   p=ROOT/r['path']
   if not p.exists() or sha(p)!=r['sha256']:errors.append('Git inventory: '+r['path'])
 report=dict(status='PASS' if not errors else 'FAIL',checked_original_files=checked,not_restored_release_files=skipped,errors=errors)
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--require-large',action='store_true');a=p.parse_args();r=verify(a.require_large);raise SystemExit(0 if r['status']=='PASS' else 1)
