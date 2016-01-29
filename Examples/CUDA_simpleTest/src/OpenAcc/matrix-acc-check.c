/* matrix-acc-check.c */
#define SIZE 1000
float a[SIZE][SIZE];
float b[SIZE][SIZE];
float c[SIZE][SIZE];
float seq[SIZE][SIZE];
 
int main()
{
  int i,j,k;
   
  // Initialize matrices.
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      a[i][j] = (float)i + j;
      b[i][j] = (float)i - j;
      c[i][j] = 0.0f;
    }
  }
   
  // Compute matrix multiplication.
#pragma acc kernels copyin(a,b) copy(c)
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      for (k = 0; k < SIZE; ++k) {
    c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
 
  // ****************
  // double-check the OpenACC result sequentially on the host
  // ****************
  // Initialize the seq matrix
  for(i = 0; i < SIZE; ++i) 
    for(j = 0; j < SIZE; ++j) 
      seq[i][j] = 0.f;
   
  // Perform the multiplication
  for (i = 0; i < SIZE; ++i) 
    for (j = 0; j < SIZE; ++j) 
      for (k = 0; k < SIZE; ++k) 
    seq[i][j] += a[i][k] * b[k][j];
   
  // check all the OpenACC matrices
  for (i = 0; i < SIZE; ++i)
    for (j = 0; j < SIZE; ++j)
      if(c[i][j] != seq[i][j]) {
    printf("Error %d %d\n", i,j);
    exit(1);
      }
  printf("OpenACC matrix multiplication test was successful!\n");
   
  return 0;
}