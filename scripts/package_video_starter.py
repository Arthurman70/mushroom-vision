"""Package only the verified public starter frames and their provenance."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parents[1]
collection=root/'data/video-discovery-v1'
manifest=json.loads((collection/'manifest.json').read_text(encoding='utf-8'))
target=root/'dist/mycoscout-video-starter-collection.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ['manifest.json','contact-sheet.jpg','COLLECTION_NOTES.md']:
        archive.write(collection/name,'video-discovery-v1/'+name)
    for frame in manifest['frames']:
        source=(collection/frame['path']).resolve()
        if not source.is_relative_to(collection.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest()!=frame['sha256']:
            raise ValueError('Frame failed provenance validation')
        archive.write(source,'video-discovery-v1/'+frame['path'])
    archive.write(root/'docs/VIDEO_GRADING.md','VIDEO_GRADING.md')
with zipfile.ZipFile(target) as archive: assert archive.testzip() is None
report={'file':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'frames':len(manifest['frames']),'videos':len(manifest['sources']),'humanGradesIncluded':False,'licenseNotice':'See video-discovery-v1/COLLECTION_NOTES.md and source licenses in manifest.json'}
(root/'dist/video-starter-artifact.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
