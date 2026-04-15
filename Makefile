CUCC    ?= nvcc
SM      ?= 86
CUFLAGS ?= -arch=sm_$(SM)
CUFLAGS += -Xcompiler -fopenmp -lpthread
CUFLAGS += -I src/ -I app/ -I ./
# -I ./gflags/include/ -lrt ./gflags/lib/libgflags.a

ifdef DEBUG
	CUFLAGS += -O0 -G -g
else
	CUFLAGS += -O3
endif

.PHONY: all
all: bfs bfs64 wcc wcc64 sssp sssp64 pr pr64

bfs: app/bfs.cu app/bfs_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/bfs.cu -DAPPBFS -o bfs $(CUFLAGS)

bfs64: app/bfs.cu app/bfs_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/bfs.cu -DAPPBFS -DLARGESIZE -o bfs64 $(CUFLAGS)

bfs_check: app/bfs.cu app/bfs_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/bfs.cu -DAPPBFS -DSCAPH_KEEP_CSR_FOR_CHECK -o bfs_check $(CUFLAGS)

bfs_check64: app/bfs.cu app/bfs_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/bfs.cu -DAPPBFS -DLARGESIZE -DSCAPH_KEEP_CSR_FOR_CHECK -o bfs_check64 $(CUFLAGS)

wcc: app/wcc.cu app/wcc_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/wcc.cu -DAPPWCC -o wcc $(CUFLAGS)

wcc64: app/wcc.cu app/wcc_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/wcc.cu -DAPPWCC -DLARGESIZE -o wcc64 $(CUFLAGS)

wcc_check: app/wcc.cu app/wcc_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/wcc.cu -DAPPWCC -DSCAPH_KEEP_CSR_FOR_CHECK -o wcc_check $(CUFLAGS)

sssp: app/sssp.cu app/sssp_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/sssp.cu -DAPPSSSP -o sssp $(CUFLAGS)

sssp64: app/sssp.cu app/sssp_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/sssp.cu -DAPPSSSP -DLARGESIZE -o sssp64 $(CUFLAGS)

sssp_check: app/sssp.cu app/sssp_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/sssp.cu -DAPPSSSP -DSCAPH_KEEP_CSR_FOR_CHECK -o sssp_check $(CUFLAGS)

sssp_check64: app/sssp.cu app/sssp_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/sssp.cu -DAPPSSSP -DLARGESIZE -DSCAPH_KEEP_CSR_FOR_CHECK -o sssp_check64 $(CUFLAGS)

pr: app/pr.cu app/pr_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/pr.cu -DAPPPR -o pr $(CUFLAGS)

pr64: app/pr.cu app/pr_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/pr.cu -DAPPPR -DLARGESIZE -o pr64 $(CUFLAGS)

pr_check: app/pr.cu app/pr_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/pr.cu -DAPPPR -DSCAPH_KEEP_CSR_FOR_CHECK -o pr_check $(CUFLAGS)

pr_check64: app/pr.cu app/pr_kernel.cuh $(wildcard src/*) Makefile
	$(CUCC) app/pr.cu -DAPPPR -DLARGESIZE -DSCAPH_KEEP_CSR_FOR_CHECK -o pr_check64 $(CUFLAGS)

.PHONY: clean
clean:
	rm -f bfs bfs64 bfs_check bfs_check64 wcc wcc64 wcc_check sssp sssp64 sssp_check sssp_check64 pr pr64 pr_check pr_check64
