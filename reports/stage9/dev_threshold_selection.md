# Stage 9 — Dev threshold / window selection

## Window scheme (8k sha256-sorted Stage-6?dev subset)

| Scheme | F1@0.5 | F1@0.1 | miss | coverage |
|--|--:|--:|--:|--:|
| A non-overlap | 0.6268 | 0.4334 | 0.0000 | 1.0000 |
| B 50% overlap | 0.6706 | 0.4591 | 0.0000 | 1.0000 |

**Selected:** `B` — higher F1@0.5 by >0.002

## Threshold (grid on `window_select_8k_dev_subset`)

- Official height: `0.1`
- **Locked (dev):** `0.7`
- Dev F1@0.5 @ locked: `0.6760`
- Dev F1@0.5 @ official: `0.6706`

Confirm must not re-select.
