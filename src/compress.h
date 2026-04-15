/*************************************************************************
	> File Name: compress.h
	> Author:
	> Mail:
	> Created Time: 2019年01月09日 星期三 13时25分21秒
 ************************************************************************/

#ifndef _COMPRESS_H
#define _COMPRESS_H

#include <string.h>
#include "pipe.h"
#include "page.h"
#include "tools.h"
#include "parallel.h"
#include "pagehandle.h"

template <typename vertex_t, typename index_t, typename value_t>
inline int compress_stat(vertex_t *Page_src, vertex_t *Page_dst, bool *nodestat,
                         vertex_t left, unsigned int srcnodenum, unsigned int dstnodenum,
                         double &endtime, int &tid)
{
	typedef Node<index_t, vertex_t> NodeT;
	typedef Edge<vertex_t, value_t> EdgeT;
	const unsigned int node_words =
		static_cast<unsigned int>(sizeof(NodeT) / sizeof(vertex_t));

	EdgeT *edgesrc = reinterpret_cast<EdgeT *>(Page_src + srcnodenum * node_words);
	EdgeT *edgedst = reinterpret_cast<EdgeT *>(Page_dst + dstnodenum * node_words);
	NodeT *nodesrc = reinterpret_cast<NodeT *>(Page_src);
	NodeT *nodedst = reinterpret_cast<NodeT *>(Page_dst);

	index_t offset = 0;
	int counter = 0;
	for (unsigned int i = 0; i < srcnodenum; ++i) {
		vertex_t node = static_cast<vertex_t>(i) + left;
		if (nodestat[node]) {
			nodedst[counter].vtx = node;
			nodedst[counter].idx = offset;
			nodedst[counter].len = nodesrc[i].len;
			memcpy(edgedst + offset, edgesrc + nodesrc[i].idx,
			       static_cast<size_t>(nodesrc[i].len) * sizeof(EdgeT));
			offset += nodesrc[i].len;
			counter++;
		}
	}
	endtime = wtime();
	return counter;
}

template <typename vertex_t, typename index_t, typename value_t>
inline void parallel_compress_stat_async(vertex_t **Pages, bool *nodestat,
                                          vertex_t *pagenodes, vertex_t *pagedatas,
                                          Page<vertex_t> *pagelist, VPstat *vpstats,
                                          bool *cached, int listsize,
                                          vertex_t pagesize, vertex_t chunksize,
                                          vertex_t thres, double *endtimes, int *tids,
                                          MFinFout<int> &wqueue)
{
	omp_set_num_threads(20);
	parallel_for(int i = 0; i < listsize; i++) {
		if (!cached[i] && pagedatas[i] > 0 && pagedatas[i] <= thres) {
			int counter = compress_stat<vertex_t, index_t, value_t>(
				Pages[i], Pages[i] + pagesize, nodestat, pagelist[i].left,
				pagelist[i].nodenum, pagenodes[i], endtimes[i], tids[i]);
			assert((unsigned int)counter == pagenodes[i]);
			vpstats[i].nodenum = pagenodes[i];
			vpstats[i].datanum = pagedatas[i];
			vpstats[i].chunknum = (pagedatas[i] + chunksize - 1) / chunksize;
			vpstats[i].shared = true;
			wqueue.Write(i);
		}
	}
}

#endif
