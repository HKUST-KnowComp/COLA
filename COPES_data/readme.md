All commonsense causal data is in COPES.json.
The data is recorded in the format of json records with three fileds:
1. story: the story of 5 sentences sampled from RocStories.
2. cause_idx: the events that have a causal relation with the last event (indices start from 0).
3. res_idx: the result, just the last event.

The split_idx.json provide the indices for validation and testing data, indexing from 0.
