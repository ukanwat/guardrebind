# Search versus no-search clear rates (round-2 adaptive runs)

| run | n | cleared by candidate 1 (no feedback) | cleared within the 20 random candidates | cleared within budget |
|---|---|---|---|---|
| base_search_null | 50 | 14/50 (28%, Wilson [17%, 42%]) | 28/50 | 35/50 |
| t_seed0_search_null | 50 | 3/50 (6%, Wilson [2%, 16%]) | 16/50 | 26/50 |
| base_search | 50 | 11/50 (22%, Wilson [13%, 35%]) | 50/50 | 50/50 |
| t_seed0_search | 50 | 2/50 (4%, Wilson [1%, 13%]) | 18/50 | 47/50 |
| bt_seed0_search | 50 | 4/50 (8%, Wilson [3%, 19%]) | 15/50 | 34/50 |

- base: 50 search wins; plain-text-control wins 35, of which 35 are also search wins; search-win positions {'append': 3, 'middle': 42, 'prepend': 5}
- t_seed0: 47 search wins; plain-text-control wins 26, of which 26 are also search wins; search-win positions {'append': 12, 'middle': 1, 'prepend': 34}
