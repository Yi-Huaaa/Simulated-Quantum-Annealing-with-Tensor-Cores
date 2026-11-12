#include "graph.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

Graph load_gset(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open graph file '" + path + "'");
    }

    long long num_nodes = 0, num_edges = 0;
    if (!(in >> num_nodes >> num_edges) || num_nodes <= 0 || num_edges < 0) {
        throw std::runtime_error("'" + path + "': invalid header, expected '<num_nodes> <num_edges>'");
    }

    Graph graph;
    graph.num_nodes = static_cast<int>(num_nodes);
    graph.edges.reserve(static_cast<size_t>(num_edges));
    for (long long e = 0; e < num_edges; ++e) {
        long long u = 0, v = 0;
        double w = 0.0;
        if (!(in >> u >> v >> w)) {
            std::ostringstream msg;
            msg << "'" << path << "': expected " << num_edges << " edges, could only read " << e;
            throw std::runtime_error(msg.str());
        }
        if (u < 1 || u > num_nodes || v < 1 || v > num_nodes || u == v) {
            std::ostringstream msg;
            msg << "'" << path << "': invalid edge (" << u << ", " << v << ") on edge line " << e + 1;
            throw std::runtime_error(msg.str());
        }
        graph.edges.push_back({static_cast<int>(u - 1), static_cast<int>(v - 1), static_cast<float>(w)});
    }
    return graph;
}

double cut_value(const Graph& graph, const std::vector<signed char>& spins) {
    double cut = 0.0;
    for (const Edge& e : graph.edges) {
        if (spins[e.u] != spins[e.v]) {
            cut += e.w;
        }
    }
    return cut;
}
