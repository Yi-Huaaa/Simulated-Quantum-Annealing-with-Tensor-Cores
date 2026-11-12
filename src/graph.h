// Weighted undirected graphs for MAX-CUT, read from G-set formatted files.
#pragma once

#include <string>
#include <vector>

struct Edge {
    int u;     // 0-based endpoint
    int v;     // 0-based endpoint
    float w;   // edge weight
};

struct Graph {
    int num_nodes = 0;
    std::vector<Edge> edges;
};

// Reads a graph in G-set format (https://web.stanford.edu/~yyye/yyye/Gset/):
//
//   <num_nodes> <num_edges>
//   <u> <v> <weight>        (one line per edge, 1-based node indices)
//
// Throws std::runtime_error on malformed input.
Graph load_gset(const std::string& path);

// Total weight of the edges whose endpoints lie on different sides of the
// partition. spins[i] must be +1 or -1 for every node i.
double cut_value(const Graph& graph, const std::vector<signed char>& spins);
