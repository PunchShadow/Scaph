#include <iostream>
#include <string.h>
#include <vector>
// #include <gflags/gflags.h>
#include "app.cuh"
#include "tools.h"

using namespace std;

extern int FLAGS_src;
extern bool FLAGS_check;
extern int FLAGS_threshold;
extern int FLAGS_runs;
extern double Total_throughput;
extern double Total_kernel_time;

// DECLARE_int32(src);
// DECLARE_bool(check);

template<typename T>
__host__ __device__ inline T bfs_infty()
{
    return (T)(~(T)0) >> 1;
}

template<typename vertex_t,typename value_t>
__global__ void bfs_init(value_t *value, bool *stat,vertex_t vert_count)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = gridDim.x*blockDim.x;
    const value_t MAXDEPTH = bfs_infty<value_t>();
    while(tid < vert_count){
          stat[tid]  = false;
          value[tid] = MAXDEPTH;
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
class bfs:public App<vertex_t,index_t,value_t>
{
public:
    bfs(int argc,char **argv):App<vertex_t,index_t,value_t>(argc,argv){};
    using App<vertex_t,index_t,value_t>::h_stat;
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::d_stat;
    using App<vertex_t,index_t,value_t>::d_value;
    using App<vertex_t,index_t,value_t>::num_blks;
    using App<vertex_t,index_t,value_t>::num_thds;
    using App<vertex_t,index_t,value_t>::copystream;
    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::csr_ngh;
    virtual bool check();
    virtual void gpu_init();
};

template<typename vertex_t,typename index_t,typename value_t>
void bfs<vertex_t,index_t,value_t>::gpu_init()
{
    bfs_init<vertex_t,value_t><<<num_blks,num_thds,0,copystream>>>(d_value,d_stat,vert_count);
    setsrc<vertex_t,value_t><<<1,1,0,copystream>>>(d_value,d_stat,FLAGS_src);
    // setsrc<vertex_t,value_t><<<1,1,0,copystream>>>(d_value,d_stat,src);
    cudaStreamSynchronize(copystream);
}

template<typename vertex_t,typename index_t,typename value_t>
bool bfs<vertex_t,index_t,value_t>::check()
{
    const value_t MAXDEPTH = bfs_infty<value_t>();
    value_t *vertex_value = new value_t[vert_count];

    #pragma omp parallel for schedule(static)
    for(vertex_t id = 0; id < vert_count; id++){
        vertex_value[id] = MAXDEPTH;
    }

    vector<vertex_t> frontier;
    frontier.reserve(1024);
    frontier.push_back(FLAGS_src);
    vertex_value[FLAGS_src] = 0;
    value_t level = 0;

    int nthreads = omp_get_max_threads();
    vector<vector<vertex_t>> local_next(nthreads);

    double t0 = wtime();
    while(!frontier.empty()){
        for(int t = 0; t < nthreads; t++) local_next[t].clear();
        size_t fsize = frontier.size();
        value_t next_level = level + 1;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            vector<vertex_t> &my_next = local_next[tid];
            #pragma omp for schedule(dynamic, 1024) nowait
            for(size_t i = 0; i < fsize; i++){
                vertex_t u = frontier[i];
                index_t vbg = csr_idx[u];
                index_t ved = csr_idx[u+1];
                for(index_t eid = vbg; eid < ved; eid++){
                    vertex_t v = csr_ngh[eid];
                    if(vertex_value[v] != MAXDEPTH) continue;
                    value_t expected = MAXDEPTH;
                    if(__sync_bool_compare_and_swap(&vertex_value[v], expected, next_level)){
                        my_next.push_back(v);
                    }
                }
            }
        }

        frontier.clear();
        for(int t = 0; t < nthreads; t++){
            frontier.insert(frontier.end(), local_next[t].begin(), local_next[t].end());
        }
        level = next_level;
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
    cout<<"cpu parallel bfs time  : "<<(t1-t0)<<" seconds"<<endl;
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
    
    
       
    FLAGS_threshold = 3;
    bfs<vertex_t,index_t,value_t> mybfs(argc,argv);
    mybfs.load_graph(); // 50% is original
    double time1,time2,avg_time12;


    int i;
    int runs = FLAGS_runs > 0 ? FLAGS_runs : 1;
    time1 = wtime();
	for(i=0;i<runs;i++)
    {
        mybfs.run();
    }
    time2 = wtime();
    avg_time12=(time2-time1)/runs;
    printf("|========================Results(alpha=37.50%%)======================|\n");
    printf("|      Avgrage(%2d Rounds) BFS wall      : %.8f seconds       |\n", runs, avg_time12);
    printf("|      Avgrage(%2d Rounds) BFS kernel    : %.8f seconds       |\n", runs, Total_kernel_time/runs);
    printf("|      Avgrage(%2d Rounds) Throughput    : %.8f GTEPS          |\n", runs, Total_throughput/runs);
    
    
/*
    printf("|========================Results(alpha=37.50%%)======================|\n");
    FLAGS_threshold = 3;
    bfs<vertex_t,index_t,value_t> mybfs1(argc,argv);
    mybfs1.load_graph();
	mybfs1.run();

    printf("|========================Results(alpha=25.00%)======================|\n");
    FLAGS_threshold = 2; 
    bfs<vertex_t,index_t,value_t> mybfs2(argc,argv);
    mybfs2.load_graph();
	mybfs2.run();    

  
    printf("|========================Results(alpha=37.50%)======================|\n");
    FLAGS_threshold = 3; 
    bfs<vertex_t,index_t,value_t> mybfs3(argc,argv);
    mybfs3.load_graph();   
	mybfs3.run();  

 	
    printf("|========================Results(alpha=50.00%)======================|\n");
    FLAGS_threshold = 4; 
    bfs<vertex_t,index_t,value_t> mybfs4(argc,argv);
    mybfs4.load_graph();
    mybfs4.run();  

  
    printf("|========================Results(alpha=62.50%)======================|\n");
    FLAGS_threshold = 5; 
    bfs<vertex_t,index_t,value_t> mybfs5(argc,argv);
    mybfs5.load_graph();
	mybfs5.run(); 
    

     printf("|========================Results(alpha=75.00%)======================|\n");
    FLAGS_threshold = 6; 
    bfs<vertex_t,index_t,value_t> mybfs6(argc,argv);
    mybfs6.load_graph();
	mybfs6.run(); 



    printf("|========================Results(alpha=87.50%)======================|\n");
    FLAGS_threshold = 7; 
    bfs<vertex_t,index_t,value_t> mybfs7(argc,argv);
    mybfs7.load_graph();
	mybfs7.run(); 


    printf("|========================Results(alpha=100.00%)=====================|\n");
    FLAGS_threshold = 8; 
    bfs<vertex_t,index_t,value_t> mybfs8(argc,argv);
    mybfs8.load_graph();
	mybfs8.run(); 
  */

    printf("|===================================================================|\n");

    if(FLAGS_check){
        cudaDeviceSynchronize();
#ifdef SCAPH_KEEP_CSR_FOR_CHECK
        if(mybfs.check()){
            cout<<"check passed!"<<endl;
        }
#else
        cout<<"--check requires building with -DSCAPH_KEEP_CSR_FOR_CHECK"<<endl;
#endif
    }
    return 0;
}
