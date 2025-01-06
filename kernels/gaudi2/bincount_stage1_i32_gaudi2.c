#pragma tpc_printf(enable)

void main(tensor input, tensor output, unsigned int partitions, unsigned int partition_size, unsigned int bins)
{
    const int one = 1;

    int5 start = get_index_space_offset();
    int5 end = start + get_index_space_size();
    int5 input_coord = { 0 };
    int5 target_coord = { 0 };

    int inputSize = get_dim_size(input, 0);

    //printf("partitions=%d\n", partitions);
    //printf("partition_size=%d\n", partition_size);
    //printf("bins=%d\n", bins);

    //printf("start: (%d, ", start[0]);
    //printf("%d, ", start[1]);
    //printf("%d, ", start[2]);
    //printf("%d, ", start[3]);
    //printf("%d)\n",  start[4]);
    //printf("end:   (%d, ", end[0]);
    //printf("%d, ", end[1]);
    //printf("%d, ", end[2]);
    //printf("%d, ", end[3]);
    //printf("%d)\n", end[4]);
    //printf("inputSize=%d\n", inputSize);

    for(int i = start[0]; i < end[0]; i++)
    {
        for(int j = 0; j < partition_size; j++)
        {
            input_coord[0] = i * partition_size + j;

            __global__ int* s_addr_inVal = (__global__ int*)gen_addr(input_coord, input);
            bool pred = s_i32_cmp_less(input_coord[0], inputSize);

            target_coord[0] = i;
            target_coord[1] = s_i32_ld_g(s_addr_inVal, 0, 0, pred, 0);

            __global__ int* s_addr_outVal = (__global__ int*)gen_addr(target_coord, output);
            bool s_cmp_res  = s_i32_cmp_less(target_coord[1], bins, 0, 0, pred, 0);

            int s_outVal = s_i32_ld_g(s_addr_outVal, 0, 0, s_cmp_res, 0);

            //printf("input_coord=(%d)\n", input_coord[0]);
            //printf("target_coord=(%d, ", target_coord[0]);
            //printf("%d)\n", target_coord[1]);
            //printf("s_outVal=%d\n", s_outVal);

            int s_outVal_st = s_i32_add(s_outVal, one, 0, 0, s_cmp_res, 0);
            s_i32_st_g(s_addr_outVal, s_outVal_st, 0, s_cmp_res, 0);
        }
    }

    cache_flush();
}
