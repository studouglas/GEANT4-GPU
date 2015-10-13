/********************************************************************
 ********************************************************************
 ** USAGE:  topcc [ --seq | --mpi | --pthread ] THIS_FILE
 ** ALGORITHM:  8 queens, recursive solution (abort slaves after first sol.)
 ** shared data:  none
 ** task input:   column to place queen from first row
 ** task output:  positions of 8 queens
 ** update:  none
 ** NOTE:  Redefine NQUEENS to other than 8 for other size boards
 ********************************************************************
 ********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "topc.h"

#define NQUEENS 8
#define SIZEOF_SOL (NQUEENS*sizeof(int))
#define MAX_SOLS 10000

int check( int cols[], int new_row, int new_col ) {
  int row;
  for ( row = 0; row < new_row; row++ ) {
    if ( (new_col == cols[row])
         || (new_col-cols[row] == new_row-row)
         || (new_col-cols[row] == row-new_row) )
      return 0;
  }
  cols[new_row] = new_col;
  return 1;
}

void record_sols( int sols[][NQUEENS], int sol[NQUEENS], int *num_sols ) {
  memcpy( &sols[*num_sols][0], sol, SIZEOF_SOL);
  (*num_sols)++;
  assert(*num_sols <= MAX_SOLS);
}

/* FOR EFFICIENCY ONLY:  (avoid copying global_output, as done by TOPC_MSG())
 * TOPC_thread_private illustrates how to make TOPC_MSG_PTR() in do_task()
 *   compatible with shared memory model.
 * Most programs would use TOPC_MSG(), allocate task_out locally on stack,
 *   and TOPC_MSG() would copy it to the TOP-C address space.
 * Ideally, if C allowed it, we would just write:
 *    THREAD_PRIVATE struct band *global_output = NULL;
 * and the first call to do_task() would malloc the memory.
 */
typedef struct out { int num_sols; int sols[MAX_SOLS][NQUEENS]; }
        TOPC_thread_private_t;
#define global_output TOPC_thread_private
TOPC_thread_private_t global_output_debug() { /* needed for debugging in gdb */
  return global_output;
}

void trace_result(int cols[], struct out *output)
{ printf("  cols: %d %d %d %d %d\n",
         cols[0], cols[1], cols[2], cols[3], cols[4]),
  printf("  output->num_sols, global_output.num_sols: %d\n",
         output->num_sols, global_output.num_sols);
}

TOPC_BUF do_task( int *cols ) {
  int *sols, num_sols = 0;
  int col4, col5, col6, col7;

  sols = (int *)(global_output.sols);

  for ( col4 = 0; col4 < NQUEENS; col4++ ) {
    if ( ! check( cols, 4, col4 ) ) continue; /* try next column */
    for ( col5 = 0; col5 < NQUEENS; col5++ ) {
      if ( ! check( cols, 5, col5 ) ) continue; /* try next column */
      for ( col6 = 0; col6 < NQUEENS; col6++ ) {
        if ( ! check( cols, 6, col6 ) ) continue; /* try next column */
        for ( col7 = 0; col7 < NQUEENS; col7++ ) {
          if ( check( cols, 7, col7 ) )
            record_sols( (int (*)[NQUEENS])sols, cols, &num_sols );
  } } } }

  global_output.num_sols = num_sols;
  return TOPC_MSG_PTR( &global_output,
                       (char *)(global_output.sols[num_sols])
                         - (char *)&global_output );
}
TOPC_ACTION check_task_result( int *dummy, struct out *output ) {
  if (output->num_sols > 0) {
    memcpy( (global_output.sols[global_output.num_sols]),
            output->sols,
            (output->num_sols)*SIZEOF_SOL );
    global_output.num_sols += output->num_sols;
  }
  return NO_ACTION;
}

int main( int argc, char **argv ) {
  int cols[NQUEENS], (*sols)[NQUEENS];
  int col0, col1, col2, col3;

  global_output.num_sols = 0;

  TOPC_OPT_trace=0;
#ifdef __cplusplus
  TOPC_OPT_trace_result = (TOPC_trace_result_ptr)trace_result;
#else
  TOPC_OPT_trace_result = trace_result;
#endif
  TOPC_init( &argc, &argv );
  TOPC_raw_begin_master_slave( do_task, check_task_result, NULL );
  if (TOPC_is_master()) {
    for ( col0 = 0; col0 < NQUEENS; col0++ ) {
      check( cols, 0, col0 ); /* This sets cols[0]=col0; */
      for ( col1 = 0; col1 < NQUEENS; col1++ ) {
        if ( ! check( cols, 1, col1 ) ) continue; /* try next column */
        for ( col2 = 0; col2 < NQUEENS; col2++ ) {
          if ( ! check( cols, 2, col2 ) ) continue; /* try next column */
          for ( col3 = 0; col3 < NQUEENS; col3++ ) {
            if ( ! check( cols, 3, col3 ) ) continue; /* try next column */
            else TOPC_raw_submit_task_input( TOPC_MSG(cols, SIZEOF_SOL) );
  } } } } }
  TOPC_raw_end_master_slave();
  TOPC_finalize();

  if (TOPC_is_master()) {
    printf("number of solutions:  %d\n", global_output.num_sols);
    if (global_output.num_sols != 92) {
      printf("INTERNAL ERROR:  It should have been 82.\n");
      exit(1);
    }
    /* slave tasks can return in different order than originated */
    sols = (int (*)[NQUEENS])(global_output.sols);
    printf("middle solution (approx %d-th solution): %d %d %d %d %d %d %d %d\n",
           global_output.num_sols/2+1,
           sols[global_output.num_sols/2][0], sols[global_output.num_sols/2][1],
           sols[global_output.num_sols/2][2], sols[global_output.num_sols/2][3],
           sols[global_output.num_sols/2][4], sols[global_output.num_sols/2][5],
           sols[global_output.num_sols/2][6], sols[global_output.num_sols/2][7]
          );
    /*
    for (col0=0; col0<global_output.num_sols; col0++)
      printf("solution (%d-th solution):  %d %d %d %d %d %d %d %d\n",
             col0+1,
             sols[col0][0], sols[col0][1],
             sols[col0][2], sols[col0][3],
             sols[col0][4], sols[col0][5],
             sols[col0][6], sols[col0][7]
            );
    */

    if (NQUEENS == 8 && global_output.num_sols != 92) {
      printf("\nBad output.  Number solutions was %d instead of 92.\n",
	     global_output.num_sols);
      exit(1);
    }
  }
  return 0;
}
