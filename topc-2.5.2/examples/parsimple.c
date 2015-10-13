/* This is the simple example from the manual meant to illustrate
 * the basics of the TOP-C model.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <topc.h>

#define MAX 3000000
#define INCR MAX/10 /* We assume INCR divides MAX exactly */

int array[MAX];
int idx;
int max_int;

TOPC_BUF GenerateTaskInput() {
  int input_task;
  if (idx >= MAX) return NOTASK;
  input_task = idx;
  idx = idx + INCR;
  return TOPC_MSG( &input_task, sizeof(input_task) );
}
TOPC_BUF DoTaskRandom( int *ignore ) {
  int rand_int[INCR];
  int i;
  for (i = 0; i < INCR; i++)
    rand_int[i] = rand();
  return TOPC_MSG( rand_int, INCR * sizeof(int) );
}
TOPC_ACTION CheckTaskRandom( int *input, int rand_vals[] ) {
  int curr_idx = *input;
  memcpy( array+curr_idx, rand_vals, INCR * sizeof(int) );
  return NO_ACTION;
}

TOPC_BUF GenerateTaskInMax() {
  int *input_task;
  if (idx >= MAX) return NOTASK;
  input_task = array + idx;
  idx = idx + INCR;
  return TOPC_MSG( input_task, INCR * sizeof(int) );
}
TOPC_BUF DoTaskMax( int subarray[] ) {
  int i;
  int max=0;
  for (i = 0; i < INCR; i++)
    if ( subarray[i] > max )
      max = subarray[i];
  return TOPC_MSG( &max, sizeof(max) );
}
TOPC_ACTION CheckTaskMax( int ignore[], int *output ) {
  int curr_max = *output;
  if ( curr_max > max_int )
    max_int = curr_max;
  return NO_ACTION;
}

int main( int argc, char **argv ) {
  printf("simple.c uses large static data.  This can fail in some config's\n");
  /* Change default to no trace; Can override with:  ./a.out --TOPC-trace=1 */
  TOPC_OPT_trace = 0;
  TOPC_init( &argc, &argv );
  idx = 0; /* Initialize idx, and randomize values of array[] */
  TOPC_master_slave( GenerateTaskInput, DoTaskRandom, CheckTaskRandom, NULL );
  if (TOPC_is_master())
    printf("Finished randomizing integers.\n");
  idx = 0; /* Re-initialize idx to 0, and find max. value in array[] */
  TOPC_master_slave( GenerateTaskInMax, DoTaskMax, CheckTaskMax, NULL );
  TOPC_finalize();
  printf("The maximum integer is:  %d\n", max_int);
  return 0;
}
