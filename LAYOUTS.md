# Gallery layouts

Press `L` to cycle through `composition → field → fixed grid`. Composition is the default. Home and search remember their layouts separately.

| Layout | Desktop | Mobile | Behavior |
|---|---:|---:|---|
| Composition | Home/text: 12; image search: 8 + query | Home: 9; text: 8; image search: 3 + query | Search results paginate |
| Field | All loaded results | All loaded results | Ranked Canvas neighborhoods; drag to pan and scroll or use controls to zoom |
| Fixed grid | 6 × 4: 24 tiles | 2 × 4: 8 tiles | Search results paginate |

Home sampling and both search types currently request up to 48 results. Field is structured for larger result sets through viewport-aware drawing and a bounded thumbnail cache.
