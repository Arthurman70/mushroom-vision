# Michigan named-target coverage in the frozen V3 collection

The collection has **3,250 photos**, weakly labeled from public observations. **33 of the 35 specifically named curriculum targets** have at least one photo whose observation taxon uses that exact name. **32** have an exact-name photo in training. These counts do not establish species identification accuracy; the trained classes remain broader mushroom groups.

The original catalog and dataset manifest remain unchanged. This audit checks label presence, recorded taxonomy ancestry and partition integrity; it does not inspect every image or reassess the curriculum sources.

Dataset SHA256: `fea75afafabbb21f80075286769b84aafd5e556feb28abf60813de5075e69bcf`.
Catalog SHA256: `3a2e8966fe3aeaa00a1ddb9dc148e9f54259dc6e8eb78af1342adeaa617c0692`.

## Named targets

Exact counts below use the recorded observation taxon name. Descendant counts are separate and do not duplicate exact-name photos. Source aliases are recorded in the JSON audit; historical names are not treated as current equivalence.

| Curriculum target | Exact-name photos | Train | Validation | Test | Additional descendant photos |
|---|---:|---:|---:|---:|---:|
| Armillaria gallica | 15 | 5 | 4 | 6 | 0 |
| Armillaria mellea | 30 | 19 | 6 | 5 | 0 |
| Armillaria tabescens | 0 | 0 | 0 | 0 | 0 |
| Calvatia gigantea | 100 | 50 | 29 | 21 | 0 |
| Craterellus fallax | 264 | 178 | 41 | 45 | 0 |
| Coprinus comatus | 114 | 69 | 27 | 18 | 0 |
| Entoloma abortivum | 101 | 68 | 13 | 20 | 0 |
| Ganoderma applanatum | 37 | 24 | 7 | 6 | 0 |
| Ganoderma sessile | 16 | 11 | 2 | 3 | 0 |
| Ganoderma curtisii | 14 | 10 | 0 | 4 | 0 |
| Ganoderma tsugae | 10 | 6 | 2 | 2 | 0 |
| Grifola frondosa | 102 | 62 | 20 | 20 | 0 |
| Hericium erinaceus | 22 | 15 | 0 | 7 | 0 |
| Hericium coralloides | 56 | 23 | 24 | 9 | 0 |
| Hericium americanum | 22 | 15 | 1 | 6 | 0 |
| Hydnum repandum | 60 | 31 | 19 | 10 | 0 |
| Hydnum rufescens | 11 | 6 | 2 | 3 | 0 |
| Hypomyces lactifluorum | 100 | 56 | 15 | 29 | 0 |
| Inonotus obliquus | 98 | 55 | 25 | 18 | 2 |
| Laetiporus sulphureus | 85 | 64 | 10 | 11 | 0 |
| Laetiporus cincinnatus | 4 | 2 | 2 | 0 | 0 |
| Morchella americana | 53 | 36 | 9 | 8 | 0 |
| Morchella angusticeps | 5 | 2 | 2 | 1 | 0 |
| Morchella diminutiva | 12 | 9 | 1 | 2 | 0 |
| Morchella exuberans | 0 | 0 | 0 | 0 | 0 |
| Morchella importuna | 1 | 1 | 0 | 0 | 0 |
| Morchella punctipes | 16 | 7 | 5 | 4 | 0 |
| Pleurotus ostreatus | 16 | 7 | 4 | 5 | 0 |
| Pleurotus populinus | 1 | 1 | 0 | 0 | 0 |
| Pleurotus pulmonarius | 15 | 12 | 2 | 1 | 0 |
| Cerioporus squamosus | 107 | 69 | 15 | 23 | 0 |
| Polyporus umbellatus | 55 | 33 | 12 | 10 | 0 |
| Stropharia rugosoannulata | 94 | 57 | 14 | 23 | 6 |
| Trametes versicolor | 120 | 72 | 24 | 24 | 0 |
| Tuber canaliculatum | 1 | 0 | 0 | 1 | 0 |

## Gaps that matter

- **Armillaria tabescens:** no exact-name example. The honey class includes Desarmillaria tabescens and D. caespitosa, but the legacy curriculum name requires explicit current taxonomic reconciliation. They are not counted as automatic species equivalents.
- **Morchella exuberans:** no matching exact-name, source-synonym or descendant example.
- **Tuber canaliculatum:** one test photo and none in training or validation. The classifier must leave this category unsupported.
- **Morchella importuna** and **Pleurotus populinus:** one exact-name photo each. **Laetiporus cincinnatus:** four. Such presence is not reliable species coverage.
- **Cantharellus** and **Hydnum:** open genus scopes. Recorded descendant photos are shown separately; broad class coverage does not prove each Michigan species is covered.
- Aborted Entoloma form, chaga growth form, mushroom condition and related specimen-level restrictions have no verified image labels here.
- Boletes remain a separate research class outside the 20 documented study entries; counts do not resolve Michigan sale scope.

## Broad research classes

| Class | Photos | Train | Validation | Test | Distinct public observers |
|---|---:|---:|---:|---:|---:|
| Honey mushrooms | 118 | 72 | 18 | 28 | 91 |
| Giant puffball | 100 | 50 | 29 | 21 | 77 |
| Chanterelles | 130 | 70 | 23 | 37 | 98 |
| Black trumpets | 490 | 300 | 88 | 102 | 286 |
| Shaggy mane | 114 | 69 | 27 | 18 | 100 |
| Shrimp of the woods | 101 | 68 | 13 | 20 | 72 |
| Reishi and artist conks | 110 | 71 | 13 | 26 | 90 |
| Hen of the woods | 102 | 62 | 20 | 20 | 84 |
| Lion mane and branching tooth fungi | 113 | 61 | 27 | 25 | 92 |
| Hedgehog mushrooms | 111 | 64 | 22 | 25 | 88 |
| Lobster mushroom | 100 | 56 | 15 | 29 | 77 |
| Chaga | 100 | 56 | 25 | 19 | 65 |
| Chicken of the woods | 118 | 80 | 20 | 18 | 97 |
| True morels | 130 | 84 | 22 | 24 | 105 |
| Oyster mushrooms | 114 | 67 | 21 | 26 | 99 |
| Dryad saddle | 107 | 69 | 15 | 23 | 94 |
| Umbrella polypore | 55 | 33 | 12 | 10 | 48 |
| Wine cap | 100 | 61 | 14 | 25 | 78 |
| Turkey tail | 120 | 72 | 24 | 24 | 101 |
| Michigan truffle | 1 | 0 | 0 | 1 | 1 |
| Boletes research coverage | 161 | 68 | 51 | 42 | 105 |
| Other fungi comparison | 655 | 373 | 123 | 159 | 345 |

The complete aggregate audit is in `config/michigan-dataset-coverage-v3.json`; the identical public UI asset is `public/michigan-coverage.json`. Neither contains observation IDs, uploader names, coordinates, photo paths or photo URLs. Named curriculum sources are linked from the existing catalog and retained in the aggregate audit.

MAMI source index: [Mushroom Study Guides](https://midwestmycology.org/mushroom-study-guides/). See `docs/MICHIGAN_SCOPE.md` for source-specific scope, exclusions, certification and inspection rules. Dataset labels and model output provide no legal clearance or edibility decision.
