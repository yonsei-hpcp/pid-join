#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/types.h>

#include <unistd.h>

#include "pidjoin.hpp"

using namespace pidjoin;


void TestPidJoin(int64_t number_of_tuples, int number_of_ranks)
{
    int64_t num_tuples = number_of_tuples;
    kv_pair_t* left_columns = (kv_pair_t*)malloc(sizeof(kv_pair_t) * num_tuples);
    kv_pair_t* right_columns = (kv_pair_t*)malloc(sizeof(kv_pair_t) * num_tuples);

    // setting up immediate values
    for (int64_t i = 0; i < num_tuples; i++)
    {
        left_columns[i].lvalue = i+1;
        left_columns[i].rvalue = i+3;
        right_columns[i].lvalue =  i+1;
        right_columns[i].rvalue = i+3;
    }

    JoinInstance join_instance(number_of_ranks); 

    join_instance.LoadColumn((void*)left_columns, num_tuples, "left");
    std::cout << "LoadColumn " << "left" << " Done." << std::endl;
    join_instance.LoadColumn((void*)right_columns, num_tuples, "right");
    std::cout << "LoadColumn " << "right" << " Done." << std::endl;
    
    ResultBuffers_t result_buffers = join_instance.ExecuteJoin("phj");

    // Join Outputs are stored in result_buffers

    int64_t joined_tuples = 0;
    int64_t zero_padding_counts = 0;
    
    int buffer_cnts = result_buffers.first.size();

    int64_t buffer_size_sum = 0;

    std::cout << "Start validations" << std::endl; 

    for (int b = 0; b < buffer_cnts; b++)
    {   
        int64_t elem_cnt = result_buffers.second[b] / sizeof(kv_pair_t);
        buffer_size_sum += result_buffers.second[b];
        
        for (int e = 0; e < elem_cnt; e++)
        {
            if (result_buffers.first[b][e].lvalue == result_buffers.first[b][e].rvalue)
            {
                if (result_buffers.first[b][e].lvalue == 0)
                {
                    zero_padding_counts++;
                }
                else
                {
                    joined_tuples++;
                }
            }
        }
    }
    
    printf("%ld Tuples are joined. %ld for zero padding. \n", 
        joined_tuples,
        zero_padding_counts);

    free (left_columns);
    free (right_columns);
    // free (result_ptr);
}


int main(int argc, char** argv)
{    
    printf("Join with 1 rank ...\n");
    
    TestPidJoin(50*1024*1024, 1);
    
    printf("Join with 16 ranks ...\n");
    
    TestPidJoin(32*1024*1024, 16);

    return 0;
}