#include "app.cuh"
#include <iostream>
#include <string.h>
#include <vector>

extern int FLAGS_src;
extern bool FLAGS_check;
extern int FLAGS_runs;
extern double Total_throughput;
extern double Total_kernel_time;

using namespace std;

template<typename vertex_t,typename value_t>
__global__ void sssp_init(value_t *value, bool *stat,vertex_t vert_count)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = gridDim.x*blockDim.x;
    const value_t MAXPATH = (value_t)(~(value_t)0) >> 1;
    while(tid < vert_count){
          stat[tid]  = false;
          value[tid] = MAXPATH;
          tid += stride;
    }
}

template<typename vertex_t,typename value_t>
__global__ void setsrc(value_t *value, bool *stat,vertex_t src)
{
    value[src] = 0;
    stat[src] = true;
}

template<typename vertex_t,typename index_t,typename value_t>
class sssp:public App<vertex_t,index_t,value_t>
{
public:
    sssp(int argc,char **argv):App<vertex_t,index_t,value_t>(argc,argv){};
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::d_value;
    using App<vertex_t,index_t,value_t>::d_stat;
    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::csr_ngh;
    using App<vertex_t,index_t,value_t>::csr_wgh;
    using App<vertex_t,index_t,value_t>::num_blks;
    using App<vertex_t,index_t,value_t>::num_thds;
    using App<vertex_t,index_t,value_t>::copystream;
    virtual void gpu_init();
    virtual bool check();
};

template<typename vertex_t,typename index_t,typename value_t>
void sssp<vertex_t,index_t,value_t>::gpu_init()
{
    sssp_init<vertex_t,value_t><<<num_blks,num_thds,0,copystream>>>(d_value,d_stat,vert_count);
    setsrc<vertex_t,value_t><<<1,1,0,copystream>>>(d_value,d_stat,FLAGS_src);
    cudaStreamSynchronize(copystream);
}

template<typename vertex_t,typename index_t,typename value_t>
bool sssp<vertex_t,index_t,value_t>::check()
{
    const value_t MAXPATH = (value_t)(~(value_t)0) >> 1;
    value_t *vertex_value = new value_t[vert_count];

    #pragma omp parallel for schedule(static)
    for(vertex_t id = 0; id < vert_count; id++){
        vertex_value[id] = MAXPATH;
    }

    vertex_value[FLAGS_src] = 0;
    vector<vertex_t> frontier;
    frontier.reserve(1024);
    frontier.push_back(FLAGS_src);

    int nthreads = omp_get_max_threads();
    vector<vector<vertex_t>> local_next(nthreads);
    vector<char> in_next(vert_count, 0);

    double t0 = wtime();
    while(!frontier.empty()){
        for(int t = 0; t < nthreads; t++) local_next[t].clear();
        size_t fsize = frontier.size();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            vector<vertex_t> &my_next = local_next[tid];
            #pragma omp for schedule(dynamic, 1024) nowait
            for(size_t i = 0; i < fsize; i++){
                vertex_t u = frontier[i];
                value_t du = vertex_value[u];
                if(du == MAXPATH) continue;
                index_t vbg = csr_idx[u];
                index_t ved = csr_idx[u+1];
                for(index_t eid = vbg; eid < ved; eid++){
                    vertex_t v = csr_ngh[eid];
                    value_t w = (value_t)csr_wgh[eid];
                    value_t nd = du + w;
                    value_t od = vertex_value[v];
                    while(od > nd){
                        if(__sync_bool_compare_and_swap(&vertex_value[v], od, nd)){
                            char prev = __atomic_exchange_n(&in_next[v], (char)1, __ATOMIC_ACQ_REL);
                            if(prev == 0) my_next.push_back(v);
                            break;
                        }
                        od = vertex_value[v];
                    }
                }
            }
        }

        frontier.clear();
        for(int t = 0; t < nthreads; t++){
            for(size_t k = 0; k < local_next[t].size(); k++){
                in_next[local_next[t][k]] = 0;
            }
            frontier.insert(frontier.end(), local_next[t].begin(), local_next[t].end());
        }
    }
    double t1 = wtime();

    long long errors = 0;
    #pragma omp parallel for reduction(+:errors) schedule(static)
    for(vertex_t vid = 0; vid < vert_count; vid++){
        if(h_value[vid] != vertex_value[vid]){
            errors++;
        }
    }
    if(errors > 0){
        int printed = 0;
        for(vertex_t vid = 0; vid < vert_count && printed < 10; vid++){
            if(h_value[vid] != vertex_value[vid]){
                cout<<"vid: "<<vid<<" gpu: "<<h_value[vid]<<" cpu: "<<vertex_value[vid]<<endl;
                printed++;
            }
        }
    }
    delete[] vertex_value;
    cout<<"cpu parallel sssp time : "<<(t1-t0)<<" seconds"<<endl;
    cout<<"total errors           : "<<errors<<endl;
    return errors == 0;
}

int main(int argc,char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#if defined(LARGESIZE)
    typedef unsigned long int vertex_t;
    typedef unsigned long int index_t;
    typedef unsigned long int value_t;
#elif defined(MIDIUMSIZE)
    typedef unsigned int      vertex_t;
    typedef unsigned long int index_t;
    typedef unsigned int      value_t;
#else
    typedef unsigned int      vertex_t;
    typedef unsigned int      index_t;
    typedef unsigned int      value_t;
#endif

    printf("|=============================Execute===============================|\n");
    sssp<vertex_t,index_t,value_t> mysssp(argc,argv);
    mysssp.load_graph();
    double time1,time2;
    int runs = FLAGS_runs > 0 ? FLAGS_runs : 1;
    time1 = wtime();
    for(int i = 0; i < runs; i++) {
        mysssp.run();
    }
    time2 = wtime();
    double avg_t = (time2 - time1) / runs;
    printf("|      Avgrage(%2d Rounds) SSSP wall     : %.8f seconds        |\n", runs, avg_t);
    printf("|      Avgrage(%2d Rounds) SSSP kernel   : %.8f seconds        |\n", runs, Total_kernel_time/runs);
    printf("|===================================================================|\n");
    if(FLAGS_check){
        cudaDeviceSynchronize();
#ifdef SCAPH_KEEP_CSR_FOR_CHECK
        if(mysssp.check()){
            cout<<"check passed!"<<endl;
        }
#else
        cout<<"--check requires building with -DSCAPH_KEEP_CSR_FOR_CHECK"<<endl;
#endif
    }
    return 0;
}
