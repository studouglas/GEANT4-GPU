/********************************************************************
 ********************************************************************
 ** USAGE:  topcc [ --seq | --mpi | --pthread ] THIS_FILE
 ** ALGORITHM:  Quicksort
 ** shared data:  array_global
 ** task input:  start and end indices
 **   struct qs_task {int start; int end; unsigned int random_seed;} *indices;
 ** task output:  subarray:  array_global[indices->start .. indices->end - 1]
 ** update:  None, result saved on master
 ********************************************************************
 ********************************************************************/

// The strategy is to use TOPC_raw_begin/end_master_slave().
// Each slave precomputes quicksort() recursively down to PAR_LEVEL.
// The master then calls quicksort().  When it reaches PAR_LEVEL,
//   TOPC_raw_submit_task_input(_input_) is called within quicksort() by the
//   master, where _input_ is TOPC_MSG() containing indices:  {start,end}
// The slave then calls qs_do_task(), which calls:
//   quicksort(array,start,end,PAR_LEVEL);
// Another alternative would be for TOPC_MSG() to include the
//   entire subarray:  array[start..end]
//   which would avoid the need for the slave to precompute down to PAR_LEVEL,
//   but it would involve higher communication costs.

#ifndef PAR_LEVEL
#define PAR_LEVEL 3 /* Level at which master dispatches sorting to slave */
#endif

#ifndef ARRAY_LEN
#define ARRAY_LEN 100
#endif

#include "topc.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <assert.h>

struct qs_task {
  int start;
  int end;
};

/*==================== MODIFICATION OF SEQUENTIAL CODE =================*/

/* Either master will call this with level == 0 and it will sort through 
 *   the recursive level PAR_LEVEL, or else a slave will call it with 
 *   with level == PAR_LEVEL and sort the given subarray entirely.
 */
void quicksort( int array[], int start, int end, int level ) {
  int i = start, j = end, pivot = array[start];
  int tmp;
  static struct qs_task indices;

  level = level + 1;
  if ( start >= end ) return;

  if ( level == PAR_LEVEL ) {
    if ( TOPC_is_master() ) {
      indices.start = start;
      indices.end = end;
      TOPC_raw_submit_task_input( TOPC_MSG(&indices, sizeof(indices)) );
    }
    return;
  }

  /* interchange so nums <= pivot come first, and nums > pivot come later */
  while (1) {
    while( (array[i] <= pivot) && (i <= end) ) i++;
    while( array[j] > pivot ) j--; /* j>0 always, since pivot is an array val */
    if ( i < j ) {
      tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
    }
    else {
      i--;
      j++;
      array[start] = array[i];
      array[i] = pivot;
      break;
    }
  }

  assert( i == j-1 );
  quicksort( array, start, i - 1, level );  /* pivot==array[i] is in order */
  quicksort( array, j, end, level );
}

void print_array( char const *string, int array[], int len ) {
  int i;
  printf("%s", string);
  for ( i=0; i < len; i++ )
    printf("%d ", array[i]);
  printf("\n");
}

/*==================== DEFINITION OF TOP-C CALLBACKS =================*/

int *array_global = NULL;
TOPC_BUF qs_do_task( struct qs_task *indices ) {
  quicksort( array_global, indices->start, indices->end, PAR_LEVEL );
  return TOPC_MSG_PTR( array_global + indices->start,
                       (indices->end - indices->start + 1) * sizeof(int) );
}
TOPC_ACTION qs_check_task_result( struct qs_task *indices, int array[] ) {
  // In shared memory (--pthread or --seq), array modified in place.
  if ( array_global + indices->start != array )
    memcpy( array_global + indices->start, array,
            (indices->end - indices->start + 1) * sizeof(int) );
  return NO_ACTION;
}
void qs_trace_input(struct qs_task *indices) {
  printf("[%d, %d]", indices->start, indices->end );
}
void qs_trace_result(struct qs_task *indices, int array[]) {
  int len = indices->end - indices->start + 1;
  if (len>8) printf("[END OF ARRAY OMMITTED] ");
  print_array( "", array, (len>8 ? 8 : len) );
}

/*==================== MAIN ROUTINE WITH INITIALIZATION ===================*/

int main( int argc, char **argv ) { int array[ARRAY_LEN];
  int i;

  // Set up defaults.  Can be overridden, e.g.:  ./a.out --TOPC_trace=0
#ifdef __cplusplus
  TOPC_OPT_trace_input = (TOPC_trace_input_ptr)qs_trace_input;
  TOPC_OPT_trace_result = (TOPC_trace_result_ptr)qs_trace_result;
#else
  TOPC_OPT_trace_input = qs_trace_input;
  TOPC_OPT_trace_result = qs_trace_result;
#endif
  TOPC_OPT_trace = 2;  /* 2 is default */
  TOPC_init(&argc, &argv);

  // PROBLEM:  HOW TO ELEGANTLY DISTRIBUTE COMMON SEED?
  srandom( (unsigned int)time(NULL) );
  for ( i=0; i < ARRAY_LEN; i++ ) array[i] = ( random() & ((1<<10)-1) );
  array_global = array; /* pass array to qs_do_task & qs_check_task_result */

  /* Tell slaves to each do quicksort sequentially until PAR_LEVEL */
  if (!TOPC_is_master()) quicksort( array, 0, ARRAY_LEN - 1, 0 );

  /* The parallel stuff happens here */
  if (TOPC_is_master()) print_array("BEFORE:  ", array, ARRAY_LEN - 1);
  TOPC_raw_begin_master_slave( qs_do_task, qs_check_task_result,
                               NULL );  /* update_shared_memory not called */
  if (TOPC_is_master()) quicksort( array, 0, ARRAY_LEN - 1, 0 );
  TOPC_raw_end_master_slave();
  if (TOPC_is_master()) print_array("AFTER:  ", array, ARRAY_LEN - 1);

  TOPC_finalize();
  return 0;
}
