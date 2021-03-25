/**********************************************************************
Copyright (c) 2018 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

void main(
    tensor ifm,
    tensor ofm
)
{

    const int depth  = 0;
    const int width  = 1;
    const int height = 2;
    const int batch  = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // depth
    const int depthStep  = 64;
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd   = index_space_end[depth]   * depthStep;

    // width
    const int widthStep  = 1;
    const int widthStart = 0;
    const int widthEnd   = get_dim_size(ifm, 1);

    // height
    const int heightStep  = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd   = index_space_end[height]   * heightStep;

    // batch
    const int batchStep  = 1;
    const int batchStart = index_space_start[batch] * batchStep;
    const int batchEnd   = index_space_end[batch]   * batchStep;

    int5 ifmCoords = { depthStart, widthStart, heightStart, batchStart, 0 };

    float64 x;
    float64 y;
    float64 sum;

    float64 zero = v_f32_mov_s(0.f);

    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d;

        for (int b = batchStart; b < batchEnd; b += batchStep)
        {
            ifmCoords[batch] = b;

            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                ifmCoords[height] = h;

                sum = zero;

                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    ifmCoords[width] = w;

                    // load input pixel
                    x = v_f32_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);
                    // exp_f32(float64 input)
                    y = v_exp_cephes_f32(x);

                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool64(v_u32_cmp_geq_b(d + V_LANE_ID_32, (unsigned)depthEnd, 0, to_bool64((bool256){0}), 1, 0));
                    y = v_f32_mov_vb(zero, 0, y, to_bool64(pred), 0);

                    // Sum up the values in a vector
                    sum = sum + y;

                }


                sum = v_reciprocal_f32(sum);

                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    ifmCoords[width] = w;

                    x = v_f32_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);
                    y = v_exp_cephes_f32(x);

                    // Multiply exp(x) * 1/(sum_of_exponents)
                    x = y * sum;

                    v_f32_st_tnsr(ifmCoords, ofm, x, 0, 1, 0);
               }
            }
        }
    }

}