import json
from pathlib import Path

# Merge chunk into semantic
chunk = json.loads(Path("graphify-out/.graphify_chunk_01.json").read_text())
semantic = {
    "nodes": chunk.get("nodes", []),
    "edges": chunk.get("edges", []),
    "hyperedges": chunk.get("hyperedges", []),
}
Path("graphify-out/.graphify_semantic.json").write_text(json.dumps(semantic, indent=2))

# Merge AST + semantic
ast = json.loads(Path("graphify-out/.graphify_ast.json").read_text())
all_nodes = ast.get("nodes", []) + semantic["nodes"]
all_edges = ast.get("edges", []) + semantic["edges"]
all_hyperedges = semantic["hyperedges"]

seen_ids = set()
deduped_nodes = []
for n in all_nodes:
    if n["id"] not in seen_ids:
        seen_ids.add(n["id"])
        deduped_nodes.append(n)

merged = {
    "nodes": deduped_nodes,
    "edges": all_edges,
    "hyperedges": all_hyperedges,
}
Path("graphify-out/.graphify_merged.json").write_text(json.dumps(merged, indent=2))
print(f"Merged: {len(deduped_nodes)} nodes, {len(all_edges)} edges")

# Build graph using graphify
from graphify.build import build_from_json
G = build_from_json(merged)
print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Cluster communities
from graphify.cluster import cluster
raw_communities = cluster(G)
# Convert to node-to-community mapping if generate() expects dict[int, list[str]]
# The signature was: communities: 'dict[int, list[str]]'
communities = raw_communities
print(f"Communities: {len(communities)}")

# Generate outputs
from graphify.report import generate
report = generate(
    G=G,
    communities=communities,
    cohesion_scores={},
    community_labels={},
    god_node_list=[],
    surprise_list=[],
    detection_result={"total_files": 1, "total_words": 1000},
    token_cost={"input": 0, "output": 0},
    root='root'
)
Path("graphify-out/GRAPH_REPORT.md").write_text(report)
print("GRAPH_REPORT.md written")

# Visualize (no visualize module found, skipping or placeholder)
print("Visualize module not found in graphify, skipping HTML generation.")

# Save graph JSON
graph_out = {
    "nodes": deduped_nodes,
    "edges": all_edges,
    "hyperedges": all_hyperedges,
    "communities": communities,
}
Path("graphify-out/graph.json").write_text(json.dumps(graph_out, indent=2))
print("graph.json written")
