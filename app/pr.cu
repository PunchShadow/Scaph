#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

#include "app.cuh"
#include "pr_kernel.cuh"
#include "tools.h"

extern int FLAGS_runs;
extern bool FLAGS_check;
extern int FLAGS_threshold;
extern int FLAGS_num_blks;
extern int FLAGS_num_thds;
extern double Total_throughput;
extern double Total_kernel_time;

// ---------------------------------------------------------------------------
// PageRank parameters — aligned with EMOGI/pagerank.cu defaults:
//   alpha     = 0.85
//   tolerance = 0.01
//   max_iter  = 5000
// ---------------------------------------------------------------------------
double FLAGS_alpha     = 0.85;
double FLAGS_tolerance = 0.01;
int    FLAGS_max_iter  = 5000;

using namespace std;

static void pr_parse_args(int argc, char **argv)
{
    auto starts_with = [](const std::string &s, const std::string &p) {
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](const std::string &key) -> std::string {
            if (a == key && i + 1 < argc) { return std::string(argv[++i]); }
            if (starts_with(a, key + "=")) return a.substr(key.size() + 1);
            return std::string();
        };
        std::string v;
        if (!(v = take("--alpha")).empty() || !(v = take("-alpha")).empty()) {
            FLAGS_alpha = std::atof(v.c_str());
        } else if (!(v = take("--tolerance")).empty() || !(v = take("-tolerance")).empty() ||
                   !(v = take("--tol")).empty() || !(v = take("-tol")).empty()) {
            FLAGS_tolerance = std::atof(v.c_str());
        } else if (!(v = take("--max_iter")).empty() || !(v = take("-max_iter")).empty() ||
                   !(v = take("--iter")).empty() || !(v = take("-iter")).empty()) {
            FLAGS_max_iter = std::atoi(v.c_str());
        }
    }
}

template<typename vertex_t, typename index_t, typename value_t>
class pagerank : public App<vertex_t, index_t, value_t> {
public:
    pagerank(int argc, char **argv) : App<vertex_t, index_t, value_t>(argc, argv) {}

    using App<vertex_t,index_t,value_t>::vert_count;
    using App<vertex_t,index_t,value_t>::edge_count;
    using App<vertex_t,index_t,value_t>::d_value;       // reused as delta
    using App<vertex_t,index_t,value_t>::h_value;
    using App<vertex_t,index_t,value_t>::d_stat;
    using App<vertex_t,index_t,value_t>::h_stat;
    using App<vertex_t,index_t,value_t>::d_degree;
    using App<vertex_t,index_t,value_t>::d_residual;
    using App<vertex_t,index_t,value_t>::h_residual;
    using App<vertex_t,index_t,value_t>::d_pr_value;
    using App<vertex_t,index_t,value_t>::h_pr_value;
    using App<vertex_t,index_t,value_t>::d_changed;
    using App<vertex_t,index_t,value_t>::num_blks;
    using App<vertex_t,index_t,value_t>::num_thds;
    using App<vertex_t,index_t,value_t>::copystream;
    using App<vertex_t,index_t,value_t>::csr_idx;
    using App<vertex_t,index_t,value_t>::csr_ngh;
    using App<vertex_t,index_t,value_t>::h_total_datas;

    virtual void gpu_init() override;
    virtual void run() override;
    virtual bool check() override;
};

template<typename vertex_t, typename index_t, typename value_t>
void pagerank<vertex_t,index_t,value_t>::gpu_init()
{
    pr_init_kernel<vertex_t, value_t>
        <<<num_blks, num_thds, 0, copystream>>>(
            d_stat, d_value, d_residual, d_pr_value, d_degree,
            (vertex_t)vert_count, (value_t)FLAGS_alpha);
    cudaStreamSynchronize(copystream);
}

template<typename vertex_t, typename index_t, typename value_t>
void pagerank<vertex_t,index_t,value_t>::run()
{
    this->gpu_init();
    this->iter_init();

    int iter = 0;
    bool changed = true;
    double total_times = 0;

    while (changed && iter < FLAGS_max_iter) {
        double t1 = wtime();

        // Push phase: scatter delta -> residual through Scaph's page pipeline.
        if (h_total_datas > 0) {
            this->gpu_push_batch();
        }

        // Update phase: residual -> pr_value, rebuild delta, re-activate.
        int zero = 0;
        H_ERR(cudaMemcpyAsync(d_changed, &zero, sizeof(int),
                              cudaMemcpyHostToDevice, copystream));
        pr_update_kernel<vertex_t, value_t>
            <<<num_blks, num_thds, 0, copystream>>>(
                d_stat, d_value, d_residual, d_pr_value, d_degree,
                (vertex_t)vert_count, (value_t)FLAGS_tolerance,
                (value_t)FLAGS_alpha, d_changed);
        int changed_h = 0;
        H_ERR(cudaMemcpyAsync(&changed_h, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost, copystream));
        cudaStreamSynchronize(copystream);
        changed = (changed_h != 0);

        // Rebuild expectq / h_total_datas based on the new label[] state.
        this->iter_init();

        double t2 = wtime();
        total_times += (t2 - t1);
        iter++;
    }

    double throughputi =
        (double)edge_count / (total_times > 1e-12 ? total_times : 1e-12) / 1e9;
    printf("|                         PR   costs    : %.8f seconds        |\n",
           total_times);
    printf("|                         Throughput    : %.8f GTEPS          |\n",
           throughputi);
    printf("|                         iterations    : %-26d|\n", iter);
    if (!changed) {
        printf("|                         converged     : yes                       |\n");
    } else {
        printf("|                         converged     : no (hit max_iter)         |\n");
    }
    Total_throughput  += throughputi;
    Total_kernel_time += total_times;

    if (FLAGS_check) {
        H_ERR(cudaMemcpyAsync(h_pr_value, d_pr_value,
                              (size_t)vert_count * sizeof(value_t),
                              cudaMemcpyDeviceToHost, copystream));
        cudaStreamSynchronize(copystream);
    }
}

template<typename vertex_t, typename index_t, typename value_t>
bool pagerank<vertex_t,index_t,value_t>::check()
{
    const value_t alpha = (value_t)FLAGS_alpha;
    const value_t tol   = (value_t)FLAGS_tolerance;

    value_t *ref_value    = new value_t[vert_count];
    value_t *ref_delta    = new value_t[vert_count];
    value_t *ref_residual = new value_t[vert_count];
    char    *ref_label    = new char[vert_count];

    const value_t one_minus_alpha = (value_t)1.0 - alpha;

    #pragma omp parallel for schedule(static)
    for (vertex_t i = 0; i < (vertex_t)vert_count; i++) {
        ref_value[i] = one_minus_alpha;
        index_t deg = csr_idx[i+1] - csr_idx[i];
        ref_delta[i] = (deg > 0) ? (one_minus_alpha * alpha / (value_t)deg)
                                 : (value_t)0;
        ref_residual[i] = (value_t)0;
        ref_label[i]    = 1;
    }

    double t0 = wtime();
    int iter = 0;
    bool changed = true;
    while (changed && iter < FLAGS_max_iter) {
        // Push phase: parallel over active vertices, atomic residual add.
        #pragma omp parallel for schedule(dynamic, 1024)
        for (vertex_t u = 0; u < (vertex_t)vert_count; u++) {
            if (ref_label[u]) {
                value_t d = ref_delta[u];
                index_t beg = csr_idx[u];
                index_t end = csr_idx[u+1];
                for (index_t j = beg; j < end; j++) {
                    vertex_t v = csr_ngh[j];
                    #pragma omp atomic
                    ref_residual[v] += d;
                }
                ref_label[u] = 0;
            }
        }
        // Update phase: residual -> value, rebuild delta, set label.
        int any_changed = 0;
        #pragma omp parallel for schedule(static) reduction(|:any_changed)
        for (vertex_t u = 0; u < (vertex_t)vert_count; u++) {
            if (ref_residual[u] > tol) {
                ref_value[u] += ref_residual[u];
                index_t deg = csr_idx[u+1] - csr_idx[u];
                ref_delta[u] = (deg > 0) ? (ref_residual[u] * alpha / (value_t)deg)
                                         : (value_t)0;
                ref_residual[u] = (value_t)0;
                ref_label[u]    = 1;
                any_changed     = 1;
            }
        }
        changed = (any_changed != 0);
        iter++;
    }
    double t1 = wtime();

    // Compare GPU vs CPU. Float non-determinism forces a relative tolerance.
    long long errors  = 0;
    double    max_abs = 0.0;
    double    max_rel = 0.0;
    const value_t abs_tol = (value_t)(10.0 * FLAGS_tolerance); // 10x convergence tol
    const value_t rel_tol = (value_t)0.05;                     // 5% relative

    int printed = 0;
    for (vertex_t i = 0; i < (vertex_t)vert_count; i++) {
        value_t g = h_pr_value[i];
        value_t c = ref_value[i];
        value_t ad = fabsf(g - c);
        value_t denom = fabsf(c) > (value_t)1e-6 ? fabsf(c) : (value_t)1.0;
        value_t rd = ad / denom;
        if ((double)ad > max_abs) max_abs = ad;
        if ((double)rd > max_rel) max_rel = rd;
        if (ad > abs_tol && rd > rel_tol) {
            if (printed < 10) {
                cout << "vid: " << i << " gpu: " << g << " cpu: " << c
                     << " abs: " << ad << " rel: " << rd << endl;
                printed++;
            }
            errors++;
        }
    }

    cout << "cpu parallel pr time  : " << (t1 - t0) << " seconds" << endl;
    cout << "cpu pr iterations     : " << iter << endl;
    cout << "max abs diff          : " << max_abs << endl;
    cout << "max rel diff          : " << max_rel << endl;
    cout << "total errors          : " << errors << endl;

    delete[] ref_value;
    delete[] ref_delta;
    delete[] ref_residual;
    delete[] ref_label;

    return errors == 0;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#if defined(LARGESIZE)
    typedef unsigned long int vertex_t;
    typedef unsigned long int index_t;
    typedef float             value_t;
#else
    typedef unsigned int      vertex_t;
    typedef unsigned int      index_t;
    typedef float             value_t;
#endif

    pr_parse_args(argc, argv);

    printf("|=============================Execute===============================|\n");
    // Full-copy threshold by default for PR — all vertices are active in the
    // first iteration, so there is no benefit to aggressive compression.
    FLAGS_threshold = 8;
    pagerank<vertex_t, index_t, value_t> mypr(argc, argv);
    mypr.load_graph();

    printf("|                       alpha           : %-26.4f|\n", FLAGS_alpha);
    printf("|                       tolerance       : %-26.6f|\n", FLAGS_tolerance);
    printf("|                       max_iter        : %-26d|\n", FLAGS_max_iter);

    int runs = FLAGS_runs > 0 ? FLAGS_runs : 1;
    double time1 = wtime();
    for (int i = 0; i < runs; i++) {
        mypr.run();
    }
    double time2 = wtime();
    double avg_t = (time2 - time1) / runs;

    printf("|========================Results====================================|\n");
    printf("|      Avgrage(%2d Rounds) PR   wall     : %.8f seconds        |\n",
           runs, avg_t);
    printf("|      Avgrage(%2d Rounds) PR   kernel   : %.8f seconds        |\n",
           runs, Total_kernel_time / runs);
    printf("|      Avgrage(%2d Rounds) Throughput    : %.8f GTEPS          |\n",
           runs, Total_throughput / runs);
    printf("|===================================================================|\n");

    if (FLAGS_check) {
        cudaDeviceSynchronize();
#ifdef SCAPH_KEEP_CSR_FOR_CHECK
        if (mypr.check()) {
            cout << "check passed!" << endl;
        }
#else
        cout << "--check requires building with -DSCAPH_KEEP_CSR_FOR_CHECK" << endl;
#endif
    }
    return 0;
}
