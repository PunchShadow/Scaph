#include "app.cuh"
#include "tools.h"
#include <iostream>
#include <string.h>
#include <vector>

extern bool FLAGS_check;
extern int FLAGS_runs;
extern double Total_throughput;
extern double Total_kernel_time;

using namespace std;

template<typename value_t,typename vertex_t>
__global__ void init(value_t *value, bool *stat,vertex_t vert_count)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = gridDim.x*blockDim.x;
    while(tid < vert_count){
        stat[tid]  = true;
        value[tid] = tid;
        tid += stride;
    }
}

template<typename vertex_t,typename index_t,typename value_t>
class wcc:public App<vertex_t,index_t,value_t>
{
public:
    wcc(int argc,char **argv):App<vertex_t,index_t,value_t>(argc,argv){};
    using App<vertex_t,index_t,value_t>::h_stat;
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::d_stat;
    using App<vertex_t,index_t,value_t>::d_value;
    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::copystream;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::csr_ngh;
    virtual void cpu_init();
    virtual void gpu_init();
    virtual bool check();
};

template<typename vertex_t,typename index_t,typename value_t>
void wcc<vertex_t,index_t,value_t>::cpu_init()
{
}

template<typename vertex_t,typename index_t,typename value_t>
void wcc<vertex_t,index_t,value_t>::gpu_init()
{
    init <value_t,vertex_t><<<256,256,0,copystream>>> (d_value,d_stat,vert_count);
    if (cudaSuccess != cudaGetLastError()) {
        std::cout<<"kernel lauch failed"<<std::endl;
        exit(0);
    }
    cudaStreamSynchronize(copystream);
}

template<typename vertex_t,typename index_t,typename value_t>
bool wcc<vertex_t,index_t,value_t>::check()
{
    int vnodes = vert_count;
    value_t *vertex_value = new value_t[vnodes];

    for(vertex_t id = 0; id < vert_count; id++){
        vertex_value[id] = id;
    }

    int active = vert_count;
    
    while(active > 0){
        active = 0;
        for(int id = 0; id < vert_count; id++){
            vertex_t vid = id;
            index_t vbg = csr_idx[vid];
            index_t ved = csr_idx[vid+1];
            for(index_t nid = vbg; nid < ved; nid++){
                vertex_t ng = csr_ngh[nid];
                if(vertex_value[vid] < vertex_value[ng]){
                  vertex_value[ng] = vertex_value[vid];
                  active++;
                }
            }
        }
    }
    int error = 0;
    for(vertex_t vid = 0; vid < vert_count; vid++){
        if(h_value[vid] != vertex_value[vid]){
           if(error == 0){
              cout<<"vid: "<<vid<<" gpu: "<<h_value[vid]<<" cpu: "<<vertex_value[vid]<<endl;
           }
           error++;
        }
    }
    cout<<"total error count: "<<error<<endl;
    delete vertex_value;
    return true;
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
    wcc<vertex_t,index_t,value_t> mywcc(argc,argv);
    mywcc.load_graph();
    double time1,time2;
    int runs = FLAGS_runs > 0 ? FLAGS_runs : 1;
    time1 = wtime();
    for(int i = 0; i < runs; i++) {
        mywcc.run();
    }
    time2 = wtime();
    double avg_t = (time2 - time1) / runs;
    printf("|      Avgrage(%2d Rounds) WCC wall      : %.8f seconds        |\n", runs, avg_t);
    printf("|      Avgrage(%2d Rounds) WCC kernel    : %.8f seconds        |\n", runs, Total_kernel_time/runs);
    printf("|===================================================================|\n");
    if(FLAGS_check){
#ifdef SCAPH_KEEP_CSR_FOR_CHECK
        if(mywcc.check()){
            cout<<"check passed!"<<endl;
        }
#else
        cout<<"--check requires building with -DSCAPH_KEEP_CSR_FOR_CHECK"<<endl;
#endif
    }
    return 0;
}
