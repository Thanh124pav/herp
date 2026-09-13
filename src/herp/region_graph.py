"""Observed directed chain transitions, including the ordinary-reset root."""
from collections import Counter

class RegionGraph:
    def __init__(self):
        self.edge_counts = Counter()

    def observe_path(self, region_ids):
        for a,b in zip(region_ids,region_ids[1:]):
            if a != b:
                self.edge_counts[a,b] += 1

    def children(self, region_id):
        return sorted(b for (a,b), n in self.edge_counts.items() if a==region_id and n)

    def child_probabilities(self, region_id):
        counts = {b:self.edge_counts[region_id,b] for b in self.children(region_id)}
        total = sum(counts.values())
        return {b:n/total for b,n in counts.items()}
