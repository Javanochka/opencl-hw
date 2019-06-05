#define SWAP(a,b) {__local double * tmp=a; a=b; b=tmp;}

__kernel void prefix_sum_group(__global double * input, __global double * output, __local double * a, __local double * b, uint size) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    if (gid < size) {
        a[lid] = b[lid] = input[gid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1) {
        if(lid > (s-1)) {
            b[lid] = a[lid] + a[lid-s];
        } else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }

    if (gid < size) {
        output[gid] = a[lid];
    }
}

__kernel void block_copy(__global double *input, __global double *output, uint input_size, uint output_size) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);

    uint ind = gid / block_size + 1;

    if (gid < input_size && ind < output_size && 1 + gid == ind * block_size) {
        output[ind] = input[gid];
    }
}

__kernel void block_add(__global double *partial_input, __global double *input, __global double *output, uint size) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);

    if (gid < size) {
        output[gid] = input[gid] + partial_input[gid / block_size];
    }
}