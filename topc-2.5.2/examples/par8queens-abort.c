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

static int done = 0;

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

TOPC_BUF generate_task_input() {
  static int col0 = -1;
  col0++;
  if (col0 >= NQUEENS || done) return NOTASK;
  else return TOPC_MSG( &col0, sizeof(col0) );
}

/* FOR EFFICIENCY ONLY:  (avoid copying cols, as done by TOPC_MSG())
 * TOPC_thread_private illustrates how to make TOPC_MSG_PTR() in do_task()
 *   compatible with shared memory model.
 * Most programs would use TOPC_MSG(), allocate task_out locally on stack,
 *   and TOPC_MSG() would copy it to the TOP-C address space.
 * Ideally, if C allowed it, we would just write:
 *    THREAD_PRIVATE int *cols = NULL;
 * and the first call to do_task() would malloc the memory.
 */
typedef int *TOPC_thread_private_t;
#define cols TOPC_thread_private
TOPC_thread_private_t cols_debug() {  /* needed for debugging in gdb */
  return cols;
}

TOPC_BUF do_task( int *input ) {
  int col0 = *input, col1, col2, col3, col4, col5, col6, col7;

  if (cols == NULL)
#  ifdef __cplusplus
    /* initialize to all zero: */
    cols = (int *)calloc(NQUEENS, sizeof(*cols));
#  else
    cols = calloc(NQUEENS, sizeof(*cols)); /* initialize to all zero */
#  endif
  cols[0] = col0;

  for ( col1 = 0; col1 < NQUEENS; col1++ ) {
    if ( ! check( cols, 1, col1 ) ) continue; /* try next column */
    for ( col2 = 0; col2 < NQUEENS; col2++ ) {
      if ( ! check( cols, 2, col2 ) ) continue; /* try next column */
      for ( col3 = 0; col3 < NQUEENS; col3++ ) {
        if ( ! check( cols, 3, col3 ) ) continue; /* try next column */
  for ( col4 = 0; col4 < NQUEENS; col4++ ) {
    if ( ! check( cols, 4, col4 ) ) continue; /* try next column */
    for ( col5 = 0; col5 < NQUEENS; col5++ ) {
      if ( ! check( cols, 5, col5 ) ) continue; /* try next column */
      for ( col6 = 0; col6 < NQUEENS; col6++ ) {
        if (TOPC_is_abort_pending()) {
          fprintf(stderr, "Slave %d aborting.\n", TOPC_rank());
          fprintf(stderr, "Checked so far:  [%d, %d, %d, %d, %d, %d, %d, -]\n",
                  col0, col1, col2, col3, col4, col5, col6);
        } /* It works to call TOPC_is_abort_pending() several times in a task */
        if (TOPC_is_abort_pending())
          return TOPC_MSG(NULL,0);
        if ( ! check( cols, 6, col6 ) ) continue; /* try next column */
        for ( col7 = 0; col7 < NQUEENS; col7++ ) {
          if ( check( cols, 7, col7 ) )
            return TOPC_MSG_PTR(cols, SIZEOF_SOL);
  } } } } } } }
  return TOPC_MSG(NULL,0);
}

int global_output[NQUEENS];

TOPC_ACTION check_task_result( int *input, int *output ) {
  if (done)
    return NO_ACTION;
  if (output) {
    printf("Found solution for col0 = %d; aborting tasks.\n", *input);
    TOPC_abort_tasks();
  }
  memcpy( global_output, output, NQUEENS*sizeof(*output) );
  done = 1;
  /* If output == NULL (no solution), leave slave idle */
  return NO_ACTION;
}

void trace_input(int *col0) {
  printf("col0:  %d", *col0);
}
int main( int argc, char **argv ) {
  TOPC_OPT_trace=2;
#ifdef __cplusplus
  TOPC_OPT_trace_input = (TOPC_trace_input_ptr)trace_input;
#else
  TOPC_OPT_trace_input = trace_input;
#endif
  TOPC_init( &argc, &argv );
  TOPC_master_slave( generate_task_input, do_task, check_task_result, NULL );
  TOPC_finalize();

  if (TOPC_is_master()) {
    int *x;

    x = global_output;
    printf("first solution: %d %d %d %d %d %d %d %d\n",
	   x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
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
  }
  return 0;
}
