import json
import networkx as nx
from graphify.export import to_html
from pathlib import Path

# Load graph data
graph_path = Path("graphify-out/graph.json")
data = json.loads(graph_path.read_text())

# Create NetworkX graph
G = nx.Graph()
for n in data["nodes"]:
    G.add_node(n["id"], **n)
for e in data.get("edges", []):
    # Mapping 'source' and 'target' as they appear in the JSON
    G.add_edge(e["source"], e["target"], **e)
if "hyperedges" in data:
    G.graph["hyperedges"] = data["hyperedges"]

# Load communities
communities = data.get("communities", {})
# Convert keys from string to int if necessary
communities = {int(k): v for k, v in communities.items()}

# Generate HTML
to_html(G, communities, "graphify-out/graph.html")
print("Successfully generated graphify-out/graph.html")
