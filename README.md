# SQA-fully

## Makefile usage

```
cd SQA-fully/  # should work in this directory
make           # compile all binaries
make speed     # test speed for different N,M
```
## run.sh
* three files inside
* parameters:
  * N = 1024 ~ 32768
  * M = 4 - 512
  * M2 = 15 - 1024
* run.sh -> get exe files
* com.sh -> get running time
* cut.sh -> grep out "avg time, gemm, judge" times
