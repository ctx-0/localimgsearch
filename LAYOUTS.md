# Gallery layouts

Press `L` to cycle through `composition → field → fixed grid → river → contact sheet`. Composition is the default. Home and search remember their layouts separately.

| Layout | Desktop | Mobile | Behavior |
|---|---:|---:|---|
| Composition | Home/text: 12; image search: 8 + query | Home: 9; text: 8; image search: 3 + query | Search results paginate |
| Field | All loaded results | All loaded results | Ranked Canvas neighborhoods; drag to pan and scroll or use controls to zoom |
| Fixed grid | 6 × 4: 24 tiles | 2 × 4: 8 tiles | Search results paginate |
| River | All loaded results | All loaded results | Natural aspect ratios, justified rows, continuous scroll |
| Contact sheet | 8 columns | 4 columns; 3 on small screens | Cropped 4:3 tiles, continuous scroll |

Home sampling and both search types currently request up to 48 results. Field is structured for larger result sets through viewport-aware drawing and a bounded thumbnail cache. In scroll layouts, the image-search query appears in a separate strip and does not reduce the result count.
