#pragma tpc_printf(enable)

#define STEP    (64)

void main(tensor inputA, tensor inputB, tensor outputC)
{
    int5 start = get_index_space_offset();
    int5 end = start + get_index_space_size();
    int5 targetCoord = { 0 };
    //printf("%d\n", start[0]);
    //printf("%d\n", start[1]);
    //printf("%d\n", start[2]);
    //printf("%d\n", start[3]);
    //printf("%d\n", start[4]);
    //printf("%d\n", end[0]);
    //printf("%d\n", end[1]);
    //printf("%d\n", end[2]);
    //printf("%d\n", end[3]);
    //printf("%d\n", end[4]);
    //unsigned int dim0 = get_dim_size(inputA, 0);
    //unsigned int dim1 = get_dim_size(inputA, 1);
    //unsigned int dim2 = get_dim_size(inputA, 2);
    //unsigned int dim3 = get_dim_size(inputA, 3);
    //unsigned int dim4 = get_dim_size(inputA, 4);
    //printf("%d\n", dim0);
    //printf("%d\n", dim1);
    //printf("%d\n", dim2);
    //printf("%d\n", dim3);
    //printf("%d\n", dim4);

    for(int i = start[0]; i < end[0]; i++)
    {
        targetCoord[0] = i * STEP;
        for(int j = start[1]; j < end[1]; j++)
        {
            targetCoord[1] = j;

            float64 a = v_f32_ld_tnsr_b(targetCoord, inputA);
            //printf("%f\n", a[0]);
            float64 b = v_f32_ld_tnsr_b(targetCoord, inputB);
            float64 c = v_f32_add_b(a, b);
            v_f32_st_tnsr(targetCoord, outputC, c);
        }
    }
}
