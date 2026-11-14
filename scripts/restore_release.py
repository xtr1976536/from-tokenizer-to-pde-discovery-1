"""Check Release parts, stream lossless extraction, then check formal files."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parts',type=Path,required=True);parser.add_argument('--destination',type=Path,default=ROOT);a=parser.parse_args();dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
 for archive in json.loads((ROOT/'release_assets.json').read_text())['archives']:
  compressed=hashlib.sha256();n=0
  for part in archive['parts']:
   p=a.parts/part['name'];assert p.stat().st_size==part['bytes'] and sha(p)==part['sha256'],part['name']
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):compressed.update(b);n+=len(b)
  assert compressed.hexdigest()==archive['sha256'] and n==archive['bytes'],archive['archive']
  paths=[str(a.parts/p['name']) for p in archive['parts']];cat=subprocess.Popen(['cat',*paths],stdout=subprocess.PIPE);z=subprocess.Popen(['zstd','-dc'],stdin=cat.stdout,stdout=subprocess.PIPE);cat.stdout.close()
  with tarfile.open(fileobj=z.stdout,mode='r|') as stream:
   for member in stream:
    path=dest/member.name;assert path.resolve().is_relative_to(dest),member.name
    assert not member.issym() and not member.islnk() and not member.isdev(),member.name
    stream.extract(member,dest,filter='data')
  assert z.wait()==0 and cat.wait()==0
  print('RESTORED',archive['archive'],flush=True)
 subprocess.run(['python',str(ROOT/'scripts/verify_public_archive.py'),'--require-large'],check=True)
if __name__=='__main__':main()
