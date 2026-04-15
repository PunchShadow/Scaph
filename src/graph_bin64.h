#ifndef __GRAPH_BIN64_H__
#define __GRAPH_BIN64_H__

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "page.h"

#ifdef APPSSSP
#ifndef WEIGHT
#define WEIGHT
#endif
#endif

inline bool scaph_bin64_has_suffix(const std::string& s, const std::string& suffix)
{
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool scaph_is_bin64_format(const std::string& path)
{
    return scaph_bin64_has_suffix(path, ".bcsr64") ||
           scaph_bin64_has_suffix(path, ".bwcsr64");
}

inline bool scaph_is_bin64_weighted(const std::string& path)
{
    return scaph_bin64_has_suffix(path, ".bwcsr64");
}

template<typename vertex_t,typename index_t,typename value_t>
bool fetch_graph_bcsr64(const char *graphfile,
                        vertex_t &vert_count, index_t &edge_count,
                        index_t *&csr_idx, vertex_t *&csr_ngh,
                        vertex_t *&csr_wgh, vertex_t *&csr_deg,
                        vertex_t **&h_page, Page<vertex_t> *&h_page_list,
                        int &list_size, vertex_t &pagesize)
{
    const bool weighted = scaph_is_bin64_weighted(std::string(graphfile));

    FILE *file = std::fopen(graphfile, "rb");
    if (!file) {
        std::cerr << "Failed to open binary graph file: " << graphfile << std::endl;
        std::exit(-1);
    }

    uint64_t u64_num_nodes = 0;
    uint64_t u64_num_edges = 0;
    if (std::fread(&u64_num_nodes, sizeof(uint64_t), 1, file) != 1 ||
        std::fread(&u64_num_edges, sizeof(uint64_t), 1, file) != 1) {
        std::cerr << "Failed to read header from " << graphfile << std::endl;
        std::fclose(file);
        std::exit(-1);
    }

    if (u64_num_nodes > static_cast<uint64_t>(std::numeric_limits<vertex_t>::max())) {
        std::cerr << "Graph has " << u64_num_nodes
                  << " vertices, exceeding Scaph's vertex_t range." << std::endl;
        std::fclose(file);
        std::exit(-1);
    }
    if (u64_num_edges > static_cast<uint64_t>(std::numeric_limits<index_t>::max())) {
        std::cerr << "Graph has " << u64_num_edges
                  << " edges, exceeding Scaph's index_t range." << std::endl;
        std::fclose(file);
        std::exit(-1);
    }

    vert_count = static_cast<vertex_t>(u64_num_nodes);
    edge_count = static_cast<index_t>(u64_num_edges);

    csr_idx = new index_t[vert_count + 1];
    {
        std::vector<uint64_t> tmp_pointer(vert_count);
        if (vert_count > 0 &&
            std::fread(tmp_pointer.data(), sizeof(uint64_t), vert_count, file) != vert_count) {
            std::cerr << "Failed to read node pointer array from " << graphfile << std::endl;
            std::fclose(file);
            std::exit(-1);
        }
        for (vertex_t i = 0; i < vert_count; ++i) {
            csr_idx[i] = static_cast<index_t>(tmp_pointer[i]);
        }
        csr_idx[vert_count] = edge_count;
    }

    csr_deg = new vertex_t[vert_count];
    for (vertex_t i = 0; i < vert_count; ++i) {
        csr_deg[i] = static_cast<vertex_t>(csr_idx[i + 1] - csr_idx[i]);
    }

    const vertex_t node_words =
        static_cast<vertex_t>(sizeof(Node<index_t, vertex_t>) / sizeof(vertex_t));
    const vertex_t edge_words =
        weighted
            ? static_cast<vertex_t>(sizeof(Edge<vertex_t, value_t>) / sizeof(vertex_t))
            : static_cast<vertex_t>(1);

    auto single_cost = [&](vertex_t v) -> uint64_t {
        uint64_t deg = static_cast<uint64_t>(csr_deg[v]);
        return deg * edge_words + node_words;
    };

    uint64_t max_single = 0;
    for (vertex_t v = 0; v < vert_count; ++v) {
        uint64_t c = single_cost(v);
        if (c > max_single) max_single = c;
    }

    const uint64_t orig_pagesize = static_cast<uint64_t>(pagesize);
    uint64_t needed_pagesize = orig_pagesize;
    if (max_single + 1 >= needed_pagesize) {
        uint64_t new_size = max_single + 8;
        if (new_size > static_cast<uint64_t>(std::numeric_limits<vertex_t>::max())) {
            std::cerr << "Required page size " << new_size
                      << " exceeds vertex_t range." << std::endl;
            std::fclose(file);
            std::exit(-1);
        }
        needed_pagesize = new_size;
        std::cerr << "[bcsr64] auto-growing pagesize from " << orig_pagesize
                  << " to " << needed_pagesize
                  << " vertex_t slots to fit max-degree vertex." << std::endl;
    }
    pagesize = static_cast<vertex_t>(needed_pagesize);

    auto page_cost = [&](vertex_t l, vertex_t r) -> uint64_t {
        uint64_t edges = static_cast<uint64_t>(csr_idx[r + 1] - csr_idx[l]);
        uint64_t nodes = static_cast<uint64_t>(r - l + 1);
        return edges * edge_words + nodes * node_words;
    };

    const uint64_t limit = static_cast<uint64_t>(pagesize);
    std::vector<Page<vertex_t>> page_list;
    vertex_t left = 0;
    while (left < vert_count) {
        vertex_t hi_limit = vert_count - 1;
        uint64_t probe64 = static_cast<uint64_t>(left) + pagesize;
        vertex_t hi = probe64 > static_cast<uint64_t>(hi_limit)
                          ? hi_limit
                          : static_cast<vertex_t>(probe64);

        if (page_cost(left, left) >= limit) {
            std::cerr << "Vertex " << left << " has degree " << csr_deg[left]
                      << " which does not fit in a single page of "
                      << pagesize << " vertex_t slots." << std::endl;
            std::fclose(file);
            std::exit(-1);
        }

        vertex_t lo = left;
        vertex_t best = left;
        if (page_cost(left, hi) < limit) {
            best = hi;
        } else {
            while (lo <= hi) {
                vertex_t mid = lo + (hi - lo) / 2;
                if (page_cost(left, mid) < limit) {
                    best = mid;
                    lo = mid + 1;
                } else {
                    if (mid == 0) break;
                    hi = mid - 1;
                }
            }
        }

        Page<vertex_t> cur;
        cur.left = left;
        cur.right = best;
        cur.nodenum = static_cast<unsigned int>(best - left + 1);
        cur.edgenum = static_cast<unsigned int>(csr_idx[best + 1] - csr_idx[left]);
        page_list.push_back(cur);

        left = best + 1;
    }

    list_size = static_cast<int>(page_list.size());
    h_page_list = new Page<vertex_t>[list_size];
    std::memcpy(h_page_list, page_list.data(), list_size * sizeof(Page<vertex_t>));

#ifdef SCAPH_KEEP_CSR_FOR_CHECK
    const bool full_materialize = true;
#else
    const bool full_materialize = false;
#endif

    if (full_materialize) {
        csr_ngh = new vertex_t[edge_count];
        csr_wgh = weighted ? new vertex_t[edge_count] : nullptr;

        const size_t chunk_edges = size_t(1) << 20;
        if (weighted) {
            struct OutEdgeWeighted64 { uint64_t end; uint64_t w8; };
            std::vector<OutEdgeWeighted64> buf(chunk_edges);
            index_t remaining = edge_count;
            index_t pos = 0;
            while (remaining > 0) {
                size_t n = remaining > chunk_edges
                              ? chunk_edges
                              : static_cast<size_t>(remaining);
                if (std::fread(buf.data(), sizeof(OutEdgeWeighted64), n, file) != n) {
                    std::cerr << "Failed to read weighted edges from " << graphfile << std::endl;
                    std::fclose(file);
                    std::exit(-1);
                }
                for (size_t i = 0; i < n; ++i) {
                    csr_ngh[pos + i] = static_cast<vertex_t>(buf[i].end);
                    csr_wgh[pos + i] = static_cast<vertex_t>(buf[i].w8);
                }
                pos += n;
                remaining -= n;
            }
        } else {
            std::vector<uint64_t> buf(chunk_edges);
            index_t remaining = edge_count;
            index_t pos = 0;
            while (remaining > 0) {
                size_t n = remaining > chunk_edges
                              ? chunk_edges
                              : static_cast<size_t>(remaining);
                if (std::fread(buf.data(), sizeof(uint64_t), n, file) != n) {
                    std::cerr << "Failed to read edges from " << graphfile << std::endl;
                    std::fclose(file);
                    std::exit(-1);
                }
                for (size_t i = 0; i < n; ++i) {
                    csr_ngh[pos + i] = static_cast<vertex_t>(buf[i]);
                }
                pos += n;
                remaining -= n;
            }
        }
    } else {
        csr_ngh = nullptr;
        csr_wgh = nullptr;
    }

    const long long edges_file_offset =
        static_cast<long long>(16) + static_cast<long long>(u64_num_nodes) *
                                        static_cast<long long>(sizeof(uint64_t));
    const size_t edge_record_bytes = weighted ? (2 * sizeof(uint64_t)) : sizeof(uint64_t);

    h_page = new vertex_t*[list_size];
    std::vector<uint64_t> edge_buf;
    std::vector<std::pair<uint64_t,uint64_t>> wedge_buf;

    for (int pid = 0; pid < list_size; ++pid) {
        const vertex_t l = h_page_list[pid].left;
        const vertex_t r = h_page_list[pid].right;
        const unsigned int nodenum = h_page_list[pid].nodenum;
        const unsigned int edgenum = h_page_list[pid].edgenum;

        cudaError_t ce = cudaMallocHost(
            reinterpret_cast<void**>(&h_page[pid]),
            static_cast<size_t>(pagesize + pagesize / 2) * sizeof(vertex_t));
        if (ce != cudaSuccess) {
            std::cerr << "cudaMallocHost failed for page " << pid << ": "
                      << cudaGetErrorString(ce) << std::endl;
            std::fclose(file);
            std::exit(-1);
        }

        Node<index_t, vertex_t> *lnode = new Node<index_t, vertex_t>[nodenum];
        for (vertex_t id = l; id <= r; ++id) {
            unsigned int off = static_cast<unsigned int>(id - l);
            lnode[off].vtx = id;
            lnode[off].idx = csr_idx[id] - csr_idx[l];
            lnode[off].len = csr_idx[id + 1] - csr_idx[id];
        }

        const size_t node_bytes = static_cast<size_t>(nodenum) *
                                  sizeof(Node<index_t, vertex_t>);
        std::memcpy(h_page[pid], lnode, node_bytes);

        vertex_t *edge_dst = h_page[pid] + (node_bytes / sizeof(vertex_t));
        const uint64_t base = static_cast<uint64_t>(csr_idx[l]);

        if (weighted) {
            Edge<vertex_t, value_t> *ledge =
                edgenum > 0 ? new Edge<vertex_t, value_t>[edgenum] : nullptr;

            if (full_materialize) {
#ifdef WEIGHT
                for (unsigned int eid = 0; eid < edgenum; ++eid) {
                    ledge[eid].ngr = csr_ngh[base + eid];
                    ledge[eid].wgh = static_cast<value_t>(csr_wgh[base + eid]);
                }
#else
                for (unsigned int eid = 0; eid < edgenum; ++eid) {
                    ledge[eid].ngr = csr_ngh[base + eid];
                }
#endif
            } else if (edgenum > 0) {
                long long off = edges_file_offset +
                                static_cast<long long>(base) *
                                    static_cast<long long>(edge_record_bytes);
                if (std::fseek(file, off, SEEK_SET) != 0) {
                    std::cerr << "fseek failed for page " << pid << std::endl;
                    std::fclose(file);
                    std::exit(-1);
                }
                if (wedge_buf.size() < edgenum) wedge_buf.resize(edgenum);
                if (std::fread(wedge_buf.data(), edge_record_bytes, edgenum, file) != edgenum) {
                    std::cerr << "fread weighted edges failed for page " << pid << std::endl;
                    std::fclose(file);
                    std::exit(-1);
                }
                for (unsigned int eid = 0; eid < edgenum; ++eid) {
                    ledge[eid].ngr = static_cast<vertex_t>(wedge_buf[eid].first);
#ifdef WEIGHT
                    ledge[eid].wgh = static_cast<value_t>(wedge_buf[eid].second);
#endif
                }
            }
            std::memcpy(edge_dst, ledge,
                        static_cast<size_t>(edgenum) *
                            sizeof(Edge<vertex_t, value_t>));
            delete[] ledge;
        } else {
            if (full_materialize) {
                std::memcpy(edge_dst, csr_ngh + base,
                            static_cast<size_t>(edgenum) * sizeof(vertex_t));
            } else if (edgenum > 0) {
                long long off = edges_file_offset +
                                static_cast<long long>(base) *
                                    static_cast<long long>(edge_record_bytes);
                if (std::fseek(file, off, SEEK_SET) != 0) {
                    std::cerr << "fseek failed for page " << pid << std::endl;
                    std::fclose(file);
                    std::exit(-1);
                }
                if (edge_buf.size() < edgenum) edge_buf.resize(edgenum);
                if (std::fread(edge_buf.data(), sizeof(uint64_t), edgenum, file) != edgenum) {
                    std::cerr << "fread edges failed for page " << pid << std::endl;
                    std::fclose(file);
                    std::exit(-1);
                }
                for (unsigned int eid = 0; eid < edgenum; ++eid) {
                    edge_dst[eid] = static_cast<vertex_t>(edge_buf[eid]);
                }
            }
        }
        delete[] lnode;
    }

    std::fclose(file);
    return true;
}

#endif
