"""Rebuild the review vocabulary without changing the Michigan curriculum scope.

Names and aliases from the cited curriculum are source names, not asserted current
synonyms. Optional acquisition taxonomy records resolve new exact-name genera.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    ('unknown', 'Unknown / another fungus'), ('morels', 'Morels'),
    ('chanterelles_trumpets', 'Chanterelles and trumpets'), ('boletes', 'Boletes'),
    ('gilled', 'Other gilled mushrooms'), ('puffballs', 'Puffballs and earthballs'),
    ('tooth_fungi', 'Tooth fungi'), ('polypores', 'Shelves, brackets and conks'),
    ('crust_parasitized', 'Crusts and parasitized forms'), ('truffles', 'Truffles'),
    ('ascomycetes', 'Cups, saddles and morel lookalikes'),
    ('jelly_fungi', 'Jelly fungi'), ('coral_fungi', 'Coral and club fungi'),
    ('stinkhorns', 'Stinkhorns'),
]
GENUS_GROUPS = {
    **dict.fromkeys('Armillaria Desarmillaria Pholiota Galerina Omphalotus Coprinus Coprinopsis Entoloma Lactarius Lactifluus Russula Pleurotus Pleurocybella Hypsizygus Lentinus Lentinellus Crepidotus Stropharia Amanita Cortinarius Inocybe Hypholoma Gymnopilus Psilocybe Agaricus Mycena Lepiota Chlorophyllum'.split(), 'gilled'),
    **dict.fromkeys('Morchella'.split(), 'morels'),
    **dict.fromkeys('Gyromitra Verpa Helvella'.split(), 'ascomycetes'),
    **dict.fromkeys('Cantharellus Craterellus'.split(), 'chanterelles_trumpets'),
    **dict.fromkeys('Hericium Hydnum Dentinum Sarcodon Hydnellum'.split(), 'tooth_fungi'),
    **dict.fromkeys('Ganoderma Grifola Meripilus Inonotus Laetiporus Cerioporus Polyporus Bondarzewia Trametes'.split(), 'polypores'),
    **dict.fromkeys('Calvatia Scleroderma'.split(), 'puffballs'),
    'Stereum': 'crust_parasitized', 'Hypomyces': 'crust_parasitized', 'Tuber': 'truffles',
    'Auricularia': 'jelly_fungi', 'Tremella': 'jelly_fungi', 'Ramaria': 'coral_fungi', 'Phallus': 'stinkhorns',
}


def stable_id(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def main():
    original = json.loads((ROOT / 'config/michigan-mushrooms-v1.json').read_text(encoding='utf-8'))
    taxa = {}
    sources = {item['id']: item for item in original['sources']}
    for entry in original['entries']:
        for item in entry.get('taxa', []) + entry.get('relatedTaxa', []):
            name = item['name']
            # Do not silently normalize descriptive forms or a source spelling.
            if not re.fullmatch(r'[A-Z][a-z]+(?: [a-z]+)?', name):
                continue
            key = stable_id(name)
            genus = name.split()[0]
            row = taxa.setdefault(key, {
                'id': key, 'scientificName': name, 'rank': 'species' if ' ' in name else 'genus',
                'broadGroupId': GENUS_GROUPS.get(genus, entry['groupId']),
                'sourceIds': [], 'sourceNames': [], 'aliases': [], 'inatTaxonId': None,
                'nameResolution': 'source_name_unresolved',
                'modelSupport': 'catalog_only', 'speciesAccuracyMeasured': False,
                'salesEligibility': 'not_assessed', 'requiresHumanReview': True,
            })
            row['sourceIds'] = sorted(set(row['sourceIds'] + entry.get('sourceIds', [])))
            row['sourceNames'].append({'name': name, 'status': item.get('status'), 'entryId': entry['id']})
            for alias in item.get('aliases', []):
                if alias not in row['aliases']:
                    row['aliases'].append(alias)

    # Exact API resolutions supplied by the separate, rights-aware downloader.
    acquisition = ROOT / 'config/research-taxa-sources-v1.json'
    if acquisition.exists():
        raw = json.loads(acquisition.read_text(encoding='utf-8'))
        resolutions = raw.get('queries', raw.get('taxa', raw.get('classes', []))) if isinstance(raw, dict) else raw
        if isinstance(resolutions, dict):
            resolutions = list(resolutions.values())
        for value in resolutions:
            resolved = value.get('taxon', value)
            name = resolved.get('name', resolved.get('scientificName'))
            taxon_id = resolved.get('id', resolved.get('taxonId', resolved.get('taxon_id')))
            if not isinstance(name, str) or not isinstance(taxon_id, int) or not re.fullmatch(r'[A-Z][a-z]+(?: [a-z]+)?', name):
                continue
            key = stable_id(name)
            sid = 'inat-' + str(taxon_id)
            sources[sid] = {'id': sid, 'title': 'iNaturalist taxon record: ' + name, 'url': 'https://www.inaturalist.org/taxa/' + str(taxon_id)}
            row = taxa.setdefault(key, {'id': key, 'scientificName': name, 'rank': resolved.get('rank', 'species' if ' ' in name else 'genus'),
                'broadGroupId': GENUS_GROUPS.get(name.split()[0], 'unknown'), 'sourceIds': [], 'sourceNames': [], 'aliases': [],
                'modelSupport': 'catalog_only', 'speciesAccuracyMeasured': False, 'salesEligibility': 'not_assessed', 'requiresHumanReview': True})
            row.update(inatTaxonId=taxon_id, nameResolution='inat_exact_name_resolved')
            row['sourceIds'] = sorted(set(row['sourceIds'] + [sid]))

    catalog = {'schema': 'mushroom-vision.research-taxa.v1', 'version': '1.0.0', 'updatedAt': '2026-10-29',
        'title': 'Mushroom Vision research identification vocabulary',
        'notice': 'These are accepted research annotations, not trained species predictions. The existing classifier predicts broad categories only. Names inherited from study guides may be historical or unresolved. This vocabulary does not expand Michigan certification or sales eligibility.',
        'groups': [{'id': key, 'label': label} for key, label in GROUPS],
        'taxa': sorted(taxa.values(), key=lambda item: item['scientificName']),
        'sources': [sources[key] for key in sorted({sid for row in taxa.values() for sid in row['sourceIds']})],
        'model': {'speciesModelTrained': False, 'speciesAccuracyMeasured': False, 'catalogAdditionTrainsModel': False}}
    (ROOT / 'config/research-taxa-v1.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    helper = (ROOT / 'scripts/research-taxonomy-runtime.js').read_text(encoding='utf-8')
    module = '// Generated by scripts/build-research-taxonomy.py; edit source JSON generation and runtime.\nexport const RESEARCH_CATALOG = ' + json.dumps(catalog, ensure_ascii=False, separators=(',', ':')) + ';\n' + helper
    (ROOT / 'community/public/research-taxonomy.js').write_text(module, encoding='utf-8')
    (ROOT / 'public/research-taxonomy.js').write_text(module, encoding='utf-8')
    (ROOT / 'public/research-taxon-editor.js').write_bytes((ROOT / 'community/public/research-taxon-editor.js').read_bytes())
    print(json.dumps({'taxa': len(taxa), 'species': sum(t['rank'] == 'species' for t in taxa.values()), 'sources': len(catalog['sources'])}))


if __name__ == '__main__':
    main()
