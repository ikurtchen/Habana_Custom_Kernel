#pragma tpc_printf(enable)

#define STEP    (64)

void main(tensor inputA, tensor inputB, tensor outputC)
{
    int5 start = get_index_space_offset();
    int5 end = start + get_index_space_size();
    int5 targetCoord = { 0 };
    // doesn't work
    //printf("Index space: (%d, %d, %d, %d, %d), (%d, %d, %d, %d, %d)\n", start[0], start[1], start[2], start[3], start[4], end[0], end[1], end[2], end[3], end[4]);
    // work
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
    for(int i = start[0]; i < end[0]; i++)
    {
        targetCoord[0] = i * STEP;
        float64 a = v_f32_ld_tnsr_b(targetCoord, inputA);
        //printf("%f\n", a[0]);
        float64 b = v_f32_ld_tnsr_b(targetCoord, inputB);
        float64 c = v_f32_add_b(a, b);
        v_f32_st_tnsr(targetCoord, outputC, c);
    }
}
