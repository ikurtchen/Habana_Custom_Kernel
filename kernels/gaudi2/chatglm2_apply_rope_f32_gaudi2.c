#pragma tpc_printf(enable)

#define STEP            (64)
#define UNROLL_COUNT    (4)

void main(tensor input0, tensor input1, tensor output)
{
    // const int width   = 0;
    // const int height  = 1;
    // const int batch   = 2;
    // const int sequence = 3;

    int5 start = get_index_space_offset();
    int5 end = start + get_index_space_size();

    int5 input0_coords = { 0 };
    int5 input1_coords = { 0 };
    int5 output_coords = { 0 };

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
    //unsigned int dim0 = get_dim_size(input0, 0);
    //unsigned int dim1 = get_dim_size(input0, 1);
    //unsigned int dim2 = get_dim_size(input0, 2);
    //unsigned int dim3 = get_dim_size(input0, 3);
    //unsigned int dim4 = get_dim_size(input0, 4);
    //printf("%d\n", dim0);
    //printf("%d\n", dim1);
    //printf("%d\n", dim2);
    //printf("%d\n", dim3);
    //printf("%d\n", dim4);

    for(int i = start[0]; i < end[0]; i++)
    {
        input0_coords[0] = i * STEP;
        input1_coords[0] = i * STEP;
        output_coords[0] = i * STEP;
        for(int j = start[1]; j < end[1]; j++)
        {
            for(int k = 0; k < UNROLL_COUNT; k++)
            {
                input0_coords[1] = j * UNROLL_COUNT + k;
                input1_coords[1] = 0;
                output_coords[1] = j * UNROLL_COUNT + k;

                for(int o = 0; o < end[2]; o++)
                {
                    input0_coords[2] = o;
                    input1_coords[2] = o;
                    output_coords[2] = o;

                    for(int p = 0; p < end[3]; p++)
                    {
                        input0_coords[3] = p;
                        input1_coords[3] = p;
                        output_coords[3] = p;

                        float64 a = v_f32_ld_tnsr_b(input0_coords, input0);
                        //printf("%f\n", a[0]);
                        float64 b = v_f32_ld_tnsr_b(input1_coords, input1);
                        float64 c = v_f32_add_b(a, b);
                        v_f32_st_tnsr(output_coords, output, c);
                    }
                }
            }
        }
    }
}
