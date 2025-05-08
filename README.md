# PID-Join \[SIGMOD '23\]

This repository contains the source code for [PID-Join \[SIGMOD '23\]](https://doi.org/10.1145/3589258), a fast processing-in-DIMM join algorithm designed and optimized for UPMEM DIMMs.
Please cite the following paper if you utilize PID-Join in your research.

```bibtex
@article{lim2023pidjoin,
  author  = {Chaemin Lim and Suhyun Lee and Jinwoo Choi and Jounghoo Lee and Seongyeon Park and Hanjun Kim and Jinho Lee and Youngsok Kim},
  title   = {{Design and Analysis of a Processing-in-DIMM Join Algorithm: A Case Study with UPMEM DIMMs}},
  journal = {Proceedings of the ACM on Management of Data (PACMMOD)},
  volume  = {1},
  number  = {2},
  year    = {2023},
}
```

## System Configuration

- Intel Xeon Gold 5222 CPU
- 1 DDR4 channel with 2 64-GB DDR4-2400 DIMMs
- 4 DDR4 channels, each with two UPMEM DIMMs
- Ubuntu 18.04 (x64)

## Prerequisites

- g++
- python3.6
- matplotlib
- numpy
- Pandas
- Scipy
- The driver for UPMEM SDK (version 2021.3, available from the [UPMEM website](https://sdk.upmem.com/).)

## Directories
- src/ # The source codes for libpidjoin.so
- test/ # The test srcs for PID-Join
- upmem-2021.3.0-Linux-x86_64/ # Contains the modified upmem sdk version 2021.3 for pid-join

## Download Upmem SDK
```
# firstly download upmem sdk then,
cp -r {your upmem sdk dir}/lib {your pid-join dir}/upmem-2021.3.0-Linux-x86_64/;
cp -r {your upmem sdk dir}/share {your pid-join dir}/upmem-2021.3.0-Linux-x86_64/;
```

## Environment Setup
```
cd {your pid-join dir};
source ./scripts/upmem_env.sh
```

## Build UPMEM SDK
```
cd {your pid-join dir};
cd upmem-2021.3.0-Linux-x86_64/src/backends/;
./load.sh
```

## Build PID-Join Library
```
cd {your pid-join dir};
make lib -j
```

## Build tests for PID-Join
```
cd {your pid-join dir};
make test -j
```
