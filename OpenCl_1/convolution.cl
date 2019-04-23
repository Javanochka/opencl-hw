__kernel void convolution_step(__global double * a, __global double * b, __global double * c, int n, int m)
{
   int id = get_global_id(0);
   if (id < n * n) {
      int i = id / n;
      int j = id % n;
      int hm = (m - 1) / 2;
      c[i * n + j] = 0;
      for (int k = -hm; k <= hm; k++) {
         for (int l = -hm; l <= hm; l++) {
            if (i + k >= 0 && i + k < n && j + l >= 0 && j + l < n) {
               c[i * n + j] += a[(i + k) * n + j + l] * b[(k + hm) * m + l + hm];
            }
         }
      }
   }
}