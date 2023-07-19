#include "idpHandler.hpp"


#include <thread>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <thread>
#include <string.h>



namespace pidjoin
{
    void *IDPArgumentHandler::ConfigureMicrobenchmarkArgRank(
        int rank_id, IDPHandler*idp_handler, DPUKernelParams_t &arg_rank,
        int benchamark_type, int tasklet_num)
    {
        arg_rank.clear();
        microbenchmark_arg *arg = 
            (microbenchmark_arg *)malloc(sizeof(microbenchmark_arg) * NUM_DPU_RANK);
        
        arg[0].benchmark_type = benchamark_type;
        arg[0].tasklet_num = tasklet_num;
        arg_rank.push_back((char*)(arg + 0));

        for (int i = 1; i < NUM_DPU_RANK; i++)
        {
            arg[i] = arg[0];
            // Add arg
            arg_rank.push_back((char*)(arg + i));
        }

        return NULL;
    }
}
