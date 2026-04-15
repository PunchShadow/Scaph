#ifndef __PR_KERNEL_CUH__
#define __PR_KERNEL_CUH__

#include "page.h"
#include "gpu_tools.cuh"
#include "cub/cub/util_ptx.cuh"

// ---------------------------------------------------------------------------
// PageRank is atypical for Scaph: it needs {delta, residual, value, label}
// per vertex, and uses atomicAdd instead of writeMin. The framework's
// generic gpu_push_csr is therefore *not* used for PR (gpu_push_batch
// dispatches to gpu_push_csr_pr below when APPPR is defined).
//
// The dummy cond / update templates still exist only so that the PR
// translation unit can parse gpu_push_csr's template body (phase-1 lookup).
// They are never instantiated and their semantics are irrelevant.
// ---------------------------------------------------------------------------

template<typename value_t>
__device__ __forceinline__ bool cond(value_t /*src_v*/, value_t /*dst_v*/)
{
    return false;
}

template<typename value_t>
__device__ __forceinline__ bool update(value_t /*src_v*/, value_t & /*dst_v*/)
{
    return false;
}

template<typename value_t>
__device__ __forceinline__ bool active(value_t new_value, value_t old_value)
{
    return !(new_value == old_value);
}

// ---------------------------------------------------------------------------
// PageRank initialization kernel — mirrors EMOGI's `initialize`.
//   value[v]    = 1 - alpha
//   delta[v]    = (deg > 0) ? (1 - alpha) * alpha / deg : 0
//   residual[v] = 0
//   label[v]    = true
// Note: delta lives in Scaph's `d_value` slot (value_t float) so that
// gpu_push_csr_pr can pull the scatter source from the same buffer that
// Scaph's compression pipeline already reads for ordinary push-style apps.
// ---------------------------------------------------------------------------
template<typename vertex_t, typename value_t>
__global__ void pr_init_kernel(bool *label, value_t *delta, value_t *residual,
                               value_t *pr_value, vertex_t *degree,
                               vertex_t vert_count, value_t alpha)
{
    vertex_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const vertex_t stride = gridDim.x * blockDim.x;
    const value_t one_minus_alpha = (value_t)1.0 - alpha;
    while (tid < vert_count) {
        pr_value[tid] = one_minus_alpha;
        const vertex_t deg = degree[tid];
        delta[tid]    = (deg > 0) ? (one_minus_alpha * alpha / (value_t)deg)
                                  : (value_t)0;
        residual[tid] = (value_t)0;
        label[tid]    = true;
        tid += stride;
    }
}

// ---------------------------------------------------------------------------
// PageRank update kernel — mirrors EMOGI's `update`.
//   if residual[v] > tolerance:
//       value[v] += residual[v]
//       delta[v]  = (deg > 0) ? residual[v] * alpha / deg : 0
//       residual[v] = 0
//       label[v]   = true        (re-activates v for next push round)
//       *changed   = 1
// ---------------------------------------------------------------------------
template<typename vertex_t, typename value_t>
__global__ void pr_update_kernel(bool *label, value_t *delta, value_t *residual,
                                 value_t *pr_value, vertex_t *degree,
                                 vertex_t vert_count, value_t tolerance,
                                 value_t alpha, int *changed)
{
    vertex_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const vertex_t stride = gridDim.x * blockDim.x;
    while (tid < vert_count) {
        const value_t r = residual[tid];
        if (r > tolerance) {
            pr_value[tid] += r;
            const vertex_t deg = degree[tid];
            delta[tid]    = (deg > 0) ? (r * alpha / (value_t)deg) : (value_t)0;
            residual[tid] = (value_t)0;
            label[tid]    = true;
            *changed      = 1;
        }
        tid += stride;
    }
}

// ---------------------------------------------------------------------------
// PageRank push kernel — structurally identical to gpu_push_csr but does
// atomicAdd(residual[ngh], delta[node]) instead of writeMin. It respects
// Scaph's paged CSR layout (Node[] vlist | Edge[] nlist inside each page).
// The `delta` argument is aliased to Scaph's d_value buffer.
// ---------------------------------------------------------------------------
template<typename vertex_t, typename index_t, typename value_t>
__global__ void gpu_push_csr_pr(value_t *delta, value_t *residual, bool *stat,
                                vertex_t *page, vertex_t nodenum,
                                int /*pageid*/, Page<vertex_t> * /*pglist*/)
{
    const vertex_t TBSIZE    = blockDim.x;
    const vertex_t work_size = nodenum / gridDim.x;
    const vertex_t bound     = nodenum - gridDim.x * (nodenum / gridDim.x);
    const vertex_t bnode     = blockIdx.x < bound
                                   ? blockIdx.x * (work_size + 1)
                                   : bound * (work_size + 1) +
                                         (blockIdx.x - bound) * work_size;
    const vertex_t enode     = blockIdx.x < bound ? bnode + (work_size + 1)
                                                  : bnode + work_size;
    const vertex_t venode    = bnode + (enode - bnode + gridDim.x) /
                                            gridDim.x * gridDim.x;
    const vertex_t WPSIZE    = 32;

    __shared__ vertex_t tbowner;
    __shared__ vertex_t tbpos;
    __shared__ vertex_t tblen;
    __shared__ value_t  tbdata;
    __shared__ value_t  twvalue[8];

    Node<index_t, vertex_t> *vlist = (Node<index_t, vertex_t> *)(page);
    Edge<vertex_t, value_t> *nlist = (Edge<vertex_t, value_t> *)(
        page + nodenum * sizeof(Node<index_t, vertex_t>) / sizeof(vertex_t));

    int tid = threadIdx.x + bnode;
    while (tid < venode) {
        unsigned int len  = 0;
        unsigned int pos  = 0;
        unsigned int node = 0;
        if (tid < enode) {
            node = vlist[tid].vtx;
            pos  = vlist[tid].idx;
            len  = vlist[tid].len;
            if (len == 0) {
                stat[node] = false;
            }
            len = stat[node] ? len : 0;
        }
        if (threadIdx.x == 0) {
            tbowner = TBSIZE + 1;
        }
        __syncthreads();

        // CTA-level processing: one node's entire edge list scanned by the block.
        while (true) {
            if (len >= TBSIZE) tbowner = threadIdx.x;
            __syncthreads();
            if (tbowner == TBSIZE + 1) break;
            if (tbowner == threadIdx.x) {
                tbpos       = pos;
                tblen       = len;
                stat[node]  = false;
                tbdata      = delta[node];
                pos = 0;
                len = 0;
            }
            __syncthreads();
            unsigned int tpos = tbpos;
            unsigned int tlen = tblen;
            value_t scatter   = tbdata;
            if (threadIdx.x == 0) tbowner = TBSIZE + 1;
            for (unsigned int ii = threadIdx.x; ii < tlen; ii += TBSIZE) {
                Edge<vertex_t, value_t> ng = nlist[tpos + ii];
                atomicAdd(&residual[ng.ngr], scatter);
            }
            __syncthreads();
        }

        // Warp-level processing: one node's edge list scanned by a warp.
        unsigned int warp_id = threadIdx.x / WPSIZE;
        unsigned int lane_id = cub::LaneId();
        while (__any_sync(0xffffffff, len >= WPSIZE)) {
            int mask   = __ballot_sync(0xffffffff, len >= WPSIZE ? 1 : 0);
            int leader = __ffs(mask) - 1;
            unsigned int tpos = __shfl_sync(0xffffffff, pos, leader);
            unsigned int tlen = __shfl_sync(0xffffffff, len, leader);
            if (leader == lane_id) {
                stat[node]       = false;
                twvalue[warp_id] = delta[node];
                len = 0;
                pos = 0;
            }
            __sync_warp(1);
            value_t scatter = twvalue[warp_id];
            for (unsigned int ii = lane_id; ii < tlen; ii += WPSIZE) {
                Edge<vertex_t, value_t> ng = nlist[tpos + ii];
                atomicAdd(&residual[ng.ngr], scatter);
            }
        }

        // Thread-level processing: short edge lists handled inline.
        if (len > 0) {
            stat[node]      = false;
            value_t scatter = delta[node];
            for (unsigned int ii = 0; ii < len; ii++) {
                Edge<vertex_t, value_t> ng = nlist[pos + ii];
                atomicAdd(&residual[ng.ngr], scatter);
            }
        }
        tid += blockDim.x;
        __syncthreads();
    }
}

#endif
