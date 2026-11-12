"""Verify that a resumed download is range checked and hash checked without network."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/download_model.py'
spec=importlib.util.spec_from_file_location('pilot_download',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('wrong_range',[False,True])
def test_resume_range_and_hash(tmp_path,monkeypatch,wrong_range):
    names=['config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json','tokenizer.json']
    data={name:('test-content-'+name).encode() for name in names}
    files=[dict(Type='blob',Path=n,Size=len(data[n]),Sha256=hashlib.sha256(data[n]).hexdigest()) for n in names]
    (tmp_path/'model.safetensors.part').write_bytes(data['model.safetensors'][:3])
    calls=[]
    class Response:
        def __init__(self,content=b'',code=200,headers=None):
            self.content=content;self.status_code=code;self.headers=headers or {}
        def raise_for_status(self):pass
        def json(self):return {'Code':200,'Data':{'Files':files}}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,*args):yield self.content
    def get(url,**kwargs):
        if '/api/v1/' in url:return Response()
        name=url.rsplit('/',1)[-1];headers=kwargs['headers'];calls.append((name,headers))
        if 'Range' in headers:
            offset=2 if wrong_range else 3
            return Response(data[name][3:],206,{'Content-Range':f'bytes {offset}-{len(data[name])-1}/{len(data[name])}'})
        return Response(data[name])
    monkeypatch.setattr(module.requests,'get',get)
    monkeypatch.setattr(sys,'argv',['download_model.py','--output',str(tmp_path)])
    if wrong_range:
        with pytest.raises(ValueError,match='Incorrect resume range'):module.main()
        assert not (tmp_path/'model.safetensors').exists()
    else:
        module.main()
        assert (tmp_path/'model.safetensors').read_bytes()==data['model.safetensors']
        assert (tmp_path/'download_receipt.json').exists()
        assert ('model.safetensors',{'Range':'bytes=3-','Accept-Encoding':'identity'}) in calls
