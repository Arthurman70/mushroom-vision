"""Build a local, script-free comparison with source attribution and all boxes."""
import base64
import html
import io
import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/black-trumpet-discovery-v2'
RUNS = {'generic': 'v2-discovery-generic', 'aware': 'v2-discovery-black-aware',
        'tiled': 'v2-discovery-black-aware-tiled', 'generic-tiled': 'v2-discovery-generic-tiled'}


def read_predictions(folder):
    predictions = {}
    for path in folder.glob('*.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('schema') == 'mycoscout.unlocated-candidates.v1':
            predictions[value['image']['sha256']] = value['candidates']
    return predictions


def rectangles(boxes, css):
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box['bboxPixelsXYXY']
        result.append(f'<rect class="{css}" x="{x1:.2f}" y="{y1:.2f}" width="{x2-x1:.2f}" height="{y2-y1:.2f}"/>')
    return ''.join(result)


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    runs = {key: read_predictions(ROOT / 'experiments/results' / directory) for key, directory in RUNS.items()}
    css = '''body{background:#101d18;color:#edf0dc;font:15px system-ui;margin:0;padding:25px}main{max-width:1250px;margin:auto}a{color:#b9d4a7}p{line-height:1.65}.controls{position:sticky;top:0;z-index:2;background:#192b23ee;padding:15px;border:1px solid #52644b;display:flex;gap:20px;flex-wrap:wrap}input{accent-color:#efb65b}h1{font:36px Georgia}h2{font:20px Georgia}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin-top:25px}.photo-card{background:#192b23;border:1px solid #35483c;padding:16px;border-radius:10px}.frame{position:relative}.frame img{display:block;width:100%;height:auto}.frame svg{position:absolute;inset:0;width:100%;height:100%}rect{fill:none;stroke-width:3;vector-effect:non-scaling-stroke}.label-box{stroke:#72ff8c}.generic-box{stroke:#ffb956}.aware-box{stroke:#ff6eaf}.tiled-box{stroke:#68ccff}.generic-tiled-box{stroke:#b2a0ff}.fine{font-size:11px;color:#afbcab}.counts{font-size:12px;line-height:1.8}body:has(#labels:not(:checked)) .label-box,body:has(#generic:not(:checked)) .generic-box,body:has(#aware:not(:checked)) .aware-box,body:has(#tiled:not(:checked)) .tiled-box,body:has(#generic-tiled:not(:checked)) .generic-tiled-box{display:none}@media(max-width:700px){body{padding:15px}.grid{grid-template-columns:1fr}}'''
    controls = [('labels', 'Reviewed shapes · green', True), ('generic', 'Generic · amber', True),
                ('aware', 'Black-trumpet prompt · pink', False), ('tiled', 'Black prompt + tiles · blue', False),
                ('generic-tiled', 'Generic + tiles · purple', False)]
    parts = [f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Black trumpet detection review</title><style>{css}</style><main>',
             '<p><a href="/lab.html">← Research fieldbook</a></p><h1>Black trumpets: inspect the evidence.</h1>',
             '<p>12 selected development photos, 30 approximate fruiting-body boxes. One assistant reviewed the boxes before seeing model predictions. These are observation-labeled black-trumpet photos; species identity is not independently verified. There are no mushroom-free controls or measured camera distances in this audit.</p>',
             '<p class="fine">The boxed region can include an entire body while a model draws only its cap, affecting overlap scores. Frozen labels are preserved, including ambiguity; they have not been adjusted to match results. Tile experiments cover only four context photos. A missing tiled overlay means that photo was not tested with tiles.</p>',
             '<div class="controls">']
    for key, label, checked in controls:
        parts.append(f'<label><input id="{key}" type="checkbox" {"checked" if checked else ""}>{html.escape(label)}</label>')
    parts.append('</div><div class="grid">')
    for item in manifest['images']:
        image = ImageOps.exif_transpose(Image.open(DATA / item['path'])).convert('RGB')
        image.thumbnail((850, 850))
        buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        counts = [f"Reviewed: {len(item['annotations'])}"]
        boxes = rectangles(item['annotations'], 'label-box')
        for key, results in runs.items():
            if item['sha256'] in results:
                boxes += rectangles(results[item['sha256']], key + '-box')
                counts.append(f"{key}: {len(results[item['sha256']])}")
        source = item['photoUrl']
        if not source.startswith('https://www.inaturalist.org/photos/'):
            raise ValueError('Unexpected source URL')
        attribution = item['photoLicense']['attribution']
        license_url = item['photoLicense']['url']
        if not license_url.startswith('https://creativecommons.org/'):
            raise ValueError('Unexpected license URL')
        parts.append(f'<article class="photo-card"><h2>{html.escape(item["id"])}</h2><div class="frame"><img alt="Observation-labeled black-trumpet photo with optional detection overlays" src="data:image/jpeg;base64,{encoded}"><svg viewBox="0 0 {item["width"]} {item["height"]}">{boxes}</svg></div><p class="counts">{html.escape(" · ".join(counts))}</p><p class="fine">{html.escape(attribution)} · <a href="{html.escape(license_url, quote=True)}" rel="noreferrer">{html.escape(item["photoLicense"]["code"])}</a>. Resized review copy; overlays added.<br><a href="{html.escape(source, quote=True)}" rel="noreferrer">Original photo and attribution</a></p></article>')
    parts.append('</div></main></html>')
    output = DATA / 'photo-review.html'
    output.write_text(''.join(parts), encoding='utf-8')
    print(json.dumps({'output': str(output), 'photos': len(manifest['images']), 'scriptFree': True}))


if __name__ == '__main__':
    main()
