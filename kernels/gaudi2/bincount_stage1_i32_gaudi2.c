//#pragma tpc_printf(enable)

__local__ int output_local[256];

void main(tensor input, tensor output, unsigned int partitions, unsigned int partition_size, unsigned int bins)
{
    const int one = 1;
    const int step  = 4;

    int5 start = get_index_space_offset();
    int5 end = start + get_index_space_size();

    int5 input_coord_0 = { 0 };
    int5 input_coord_1 = { 0 };
    int5 input_coord_2 = { 0 };
    int5 input_coord_3 = { 0 };

    int5 target_coord_0 = { 0 };
    int5 target_coord_1 = { 0 };
    int5 target_coord_2 = { 0 };
    int5 target_coord_3 = { 0 };

    int input_size = get_dim_size(input, 0);
    int input_bound = 0;

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
    //printf("input_size=%d\n", input_size);

    for(int i = start[0]; i < end[0]; i++)
    {
        for (int k = 0; k < bins; k++)
        {
            output_local[k] = 0;
        }

        input_bound = s_i32_min((i + 1) * partition_size, input_size);
        //printf("i=%d, ", i);
        //printf("input_bound=%d\n", input_bound);

        for(int j = 0; j < partition_size; j+=step)
        {

            input_coord_0[0] = i * partition_size + j + 0;
            input_coord_1[0] = i * partition_size + j + 1;
            input_coord_2[0] = i * partition_size + j + 2;
            input_coord_3[0] = i * partition_size + j + 3;

            __global__ int* s_addr_inVal_0 = (__global__ int*)gen_addr(input_coord_0, input);
            __global__ int* s_addr_inVal_1 = (__global__ int*)gen_addr(input_coord_1, input);
            __global__ int* s_addr_inVal_2 = (__global__ int*)gen_addr(input_coord_2, input);
            __global__ int* s_addr_inVal_3 = (__global__ int*)gen_addr(input_coord_3, input);
            bool pred_0 = s_i32_cmp_less(input_coord_0[0], input_bound);
            bool pred_1 = s_i32_cmp_less(input_coord_1[0], input_bound);
            bool pred_2 = s_i32_cmp_less(input_coord_2[0], input_bound);
            bool pred_3 = s_i32_cmp_less(input_coord_3[0], input_bound);

            target_coord_0[0] = i;
            target_coord_1[0] = i;
            target_coord_2[0] = i;
            target_coord_3[0] = i;
            target_coord_0[1] = s_i32_ld_g(s_addr_inVal_0, 0, 0, pred_0, 0);
            target_coord_1[1] = s_i32_ld_g(s_addr_inVal_1, 0, 0, pred_1, 0);
            target_coord_2[1] = s_i32_ld_g(s_addr_inVal_2, 0, 0, pred_2, 0);
            target_coord_3[1] = s_i32_ld_g(s_addr_inVal_3, 0, 0, pred_3, 0);

            //__global__ int* s_addr_outVal_0 = (__global__ int*)gen_addr(target_coord_0, output);
            //__global__ int* s_addr_outVal_1 = (__global__ int*)gen_addr(target_coord_1, output);
            //__global__ int* s_addr_outVal_2 = (__global__ int*)gen_addr(target_coord_2, output);
            //__global__ int* s_addr_outVal_3 = (__global__ int*)gen_addr(target_coord_3, output);
            bool s_cmp_res_0 = s_i32_cmp_less(target_coord_0[1], bins, 0, 0, pred_0, 0);
            bool s_cmp_res_1 = s_i32_cmp_less(target_coord_1[1], bins, 0, 0, pred_1, 0);
            bool s_cmp_res_2 = s_i32_cmp_less(target_coord_2[1], bins, 0, 0, pred_2, 0);
            bool s_cmp_res_3 = s_i32_cmp_less(target_coord_3[1], bins, 0, 0, pred_3, 0);

            //int s_outVal_0 = s_i32_ld_g(s_addr_outVal_0, 0, 0, s_cmp_res_0, 0);
            //int s_outVal_st_0 = s_i32_add(s_outVal_0, one, 0, 0, s_cmp_res_0, 0);
            //s_i32_st_g(s_addr_outVal_0, s_outVal_st_0, 0, s_cmp_res_0, 0);

            //int s_outVal_1 = s_i32_ld_g(s_addr_outVal_1, 0, 0, s_cmp_res_1, 0);
            //int s_outVal_st_1 = s_i32_add(s_outVal_1, one, 0, 0, s_cmp_res_1, 0);
            //s_i32_st_g(s_addr_outVal_1, s_outVal_st_1, 0, s_cmp_res_1, 0);

            //int s_outVal_2 = s_i32_ld_g(s_addr_outVal_2, 0, 0, s_cmp_res_2, 0);
            //int s_outVal_st_2 = s_i32_add(s_outVal_2, one, 0, 0, s_cmp_res_2, 0);
            //s_i32_st_g(s_addr_outVal_2, s_outVal_st_2, 0, s_cmp_res_2, 0);

            //int s_outVal_3 = s_i32_ld_g(s_addr_outVal_3, 0, 0, s_cmp_res_3, 0);
            //int s_outVal_st_3 = s_i32_add(s_outVal_3, one, 0, 0, s_cmp_res_3, 0);
            //s_i32_st_g(s_addr_outVal_3, s_outVal_st_3, 0, s_cmp_res_3, 0);

            int s_outVal_0 = s_i32_add(output_local[target_coord_0[1]], one, 0, 0, s_cmp_res_0, 0);
            s_i32_st_l((unsigned)&output_local[target_coord_0[1]], s_outVal_0, 0, s_cmp_res_0, 0);
            int s_outVal_1 = s_i32_add(output_local[target_coord_1[1]], one, 0, 0, s_cmp_res_1, 0);
            s_i32_st_l((unsigned)&output_local[target_coord_1[1]], s_outVal_1, 0, s_cmp_res_1, 0);
            int s_outVal_2 = s_i32_add(output_local[target_coord_2[1]], one, 0, 0, s_cmp_res_2, 0);
            s_i32_st_l((unsigned)&output_local[target_coord_2[1]], s_outVal_2, 0, s_cmp_res_2, 0);
            int s_outVal_3 = s_i32_add(output_local[target_coord_3[1]], one, 0, 0, s_cmp_res_3, 0);
            s_i32_st_l((unsigned)&output_local[target_coord_3[1]], s_outVal_3, 0, s_cmp_res_3, 0);

            //if (j > partition_size / 2)
            //{
            //    printf("input_coord_0=(%d)\n", input_coord_0[0]);
            //    printf("target_coord_0=(%d, ", target_coord_0[0]);
            //    printf("%d)\n", target_coord_0[1]);
            //    printf("s_cmp_res_0=%d, ", s_cmp_res_0 ? 1 : 0);
            //    printf("pred_0=%d\n", pred_0 ? 1 : 0);
            //    //printf("s_outVal_0=%d\n", s_outVal_0);
            //    printf("input_coord_1=(%d)\n", input_coord_1[0]);
            //    printf("target_coord_1=(%d, ", target_coord_1[0]);
            //    printf("%d)\n", target_coord_1[1]);
            //    printf("s_cmp_res_1=%d, ", s_cmp_res_1 ? 1 : 0);
            //    printf("pred_1=%d\n", pred_1 ? 1 : 0);
            //    //printf("s_outVal_1=%d\n", s_outVal_1);
            //    printf("input_coord_2=(%d)\n", input_coord_2[0]);
            //    printf("target_coord_2=(%d, ", target_coord_2[0]);
            //    printf("%d)\n", target_coord_2[1]);
            //    printf("s_cmp_res_2=%d, ", s_cmp_res_2 ? 1 : 0);
            //    printf("pred_2=%d\n", pred_2 ? 1 : 0);
            //    //printf("s_outVal_2=%d\n", s_outVal_2);
            //    printf("input_coord_3=(%d)\n", input_coord_3[0]);
            //    printf("target_coord_3=(%d, ", target_coord_3[0]);
            //    printf("%d)\n", target_coord_3[1]);
            //    printf("s_cmp_res_3=%d, ", s_cmp_res_3 ? 1 : 0);
            //    printf("pred_3=%d\n", pred_3 ? 1 : 0);
            //    //printf("s_outVal_3=%d\n", s_outVal_3);
            //}
        }

        for (int k = 0; k < bins; k++)
        {
            target_coord_0[1] = k;
            __global__ int* s_addr_outVal = (__global__ int*)gen_addr(target_coord_0, output);
            s_i32_st_g(s_addr_outVal, output_local[k]);
        }
    }

    cache_flush();
}
