/********************************************************************
 ********************************************************************
 ** USAGE:  topcc [ --seq | --mpi | --pthread ] THIS_FILE
 ** ALGORITHM:  Quicksort
 ** shared data:  array_global
 ** task input:   start and end indices
 **   struct qs_task {int start; int end; unsigned int random_seed;} *indices;
 ** task output:  subarray:  array_global[indices->start .. indices->end - 1]
 ** update:       None, result saved on master
 ** FOR A LARGER RUN, COMPILE WITH:  -DARRAY_LEN=10000 -DPAR_LEVEL=12
 ********************************************************************
 ********************************************************************/

// The strategy is to use TOPC_master_slave();
// Each slave precomputes quicksort() recursively down to
//   the recursive level, PAR_LEVEL.
// The master also precomputes quicksort() recursively down to PAR_LEVEL,
//   but the master also records what index pair:  {start,end}
//   would have been invoked at the next call to quicksort()
// This index pair is saved in array_of_tasks, and used by
//   qs_generate_task_input() to generate tasks for the slaves.
// The master then calls TOPC_master_slave()
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
  unsigned int random_seed;
};
int *array_of_tasks = NULL;
int length_of_array_of_tasks = 0;
int length_of_array_of_tasks_filled = 0;
#define max(x,y) ( (x)<(y) ? (y) : (x) )

/*==================== MODIFICATION OF SEQUENTIAL CODE =================*/

/* Either master will call this with level == 0 and it will sort through
 *   the recursive level PAR_LEVEL, or else a slave will call it with
 *   with level == PAR_LEVEL and sort the given subarray entirely.
 */
void quicksort( int array[], int start, int end, int level ) {
  int i = start, j = end, pivot = array[start];
  int tmp;

  level = level + 1;
  if ( start >= end ) return;

  if ( level == PAR_LEVEL ) {
    if ( TOPC_is_master() ) {
      /* record [start,end] indices for qs_generate_task_input() */
      if ( length_of_array_of_tasks_filled >= length_of_array_of_tasks - 1 ) {
	length_of_array_of_tasks = max(100,2*length_of_array_of_tasks);
#      ifdef __cplusplus
        array_of_tasks = (int *)realloc( array_of_tasks,
                                  sizeof(int)*length_of_array_of_tasks );
#      else
        array_of_tasks = realloc( array_of_tasks,
                           sizeof(int)*length_of_array_of_tasks );
#      endif
      }
      array_of_tasks[ length_of_array_of_tasks_filled++ ] = start;
      array_of_tasks[ length_of_array_of_tasks_filled++ ] = end;
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
TOPC_BUF qs_generate_task_input() {
  static struct qs_task indices;
  static int random_numbers_initialized = 0;
  /* indices.random_seed == 0 means sort array[start..end];
     else initialize random array using seed */
  if ( ! random_numbers_initialized ) {
    while ( ! (indices.random_seed = (unsigned int)time(NULL)) ) ;
    random_numbers_initialized = 1;
  }
  else if (length_of_array_of_tasks_filled >= length_of_array_of_tasks)
    return NOTASK;
  else {
    indices.start = array_of_tasks[length_of_array_of_tasks_filled++];
    indices.end = array_of_tasks[length_of_array_of_tasks_filled++];
    indices.random_seed = 0; /* no random seed */
  }
  return TOPC_MSG( &indices, sizeof(indices) );
}
TOPC_BUF qs_do_task( struct qs_task *indices ) {
  /* if  indices.random_seed != 0,  UPDATE:  init random array */
  if ( indices->random_seed ) return TOPC_MSG(NULL,0);
  quicksort( array_global, indices->start, indices->end, PAR_LEVEL );
  return TOPC_MSG_PTR( array_global + indices->start,
                       (indices->end - indices->start + 1) * sizeof(int) );
}
TOPC_ACTION qs_check_task_result( struct qs_task *indices, int array[] ) {
  if ( indices->random_seed ) return UPDATE; /* init random array */
  // In shared memory (--pthread or --seq), array modified in place.
  if ( array_global + indices->start != array )
    memcpy( array_global + indices->start, array,
            (indices->end - indices->start + 1) * sizeof(int) );
  return NO_ACTION;
}
void qs_update_shared_data( struct qs_task *indices, int array[] ) {
  int i;
  srandom( indices->random_seed );
  for ( i=0; i < ARRAY_LEN; i++ ) array_global[i] = ( random() & ((1<<10)-1) );
}

void qs_trace_input(struct qs_task *indices) {
  if ( indices->random_seed )
    printf("initializing:  random seed:  %d", indices->random_seed );
  else
    printf("[%d, %d]", indices->start, indices->end );
}
void qs_trace_result(struct qs_task *indices, int array[]) {
  int len = indices->end - indices->start + 1;
  if ( indices->random_seed ) printf("Initializing ...");
  else {
    if (len>8) printf("[END OF ARRAY OMMITTED] ");
    print_array( "", array, (len>8 ? 8 : len) );
  }
}

/*==================== MAIN ROUTINE WITH INITIALIZATION ===================*/

int main( int argc, char **argv ) {
  int array[ARRAY_LEN];

  // Set up defaults.  Can be overridden, e.g.:  ./a.out --TOPC-trace=0
#ifdef __cplusplus
  TOPC_OPT_trace_input = (TOPC_trace_input_ptr)qs_trace_input;
  TOPC_OPT_trace_result = (TOPC_trace_result_ptr)qs_trace_result;
#else
  TOPC_OPT_trace_input = qs_trace_input;
  TOPC_OPT_trace_result = qs_trace_result;
#endif
  TOPC_OPT_trace = 2;  /* 2 is default */
  TOPC_init(&argc, &argv);

  /* pass array to qs_do_task, qs_check_task_result, qs_update_shared_data */
  array_global = array;
  /* random_numbers_initialized == 0:  initialize array, and return NOTASK */
  TOPC_master_slave( qs_generate_task_input, qs_do_task, qs_check_task_result,
                     qs_update_shared_data );

  /* Do quicksort sequentially until PAR_LEVEL; if master, record tasks */
  /* This will set array_of_tasks (array of [start..end] pairs for
     slaves to sort) */
  quicksort( array, 0, ARRAY_LEN - 1, 0 );

  /* The parallel stuff happens here */
  if (TOPC_is_master()) print_array("BEFORE:  ", array, ARRAY_LEN - 1);
  length_of_array_of_tasks = length_of_array_of_tasks_filled;
  length_of_array_of_tasks_filled = 0; /* count up again */
  /* random_numbers_initialized == 1 &&
   * length_of_array_of_tasks > length_of_array_of_tasks_filled:  start sorting
   */
  TOPC_master_slave( qs_generate_task_input, qs_do_task, qs_check_task_result,
                     NULL );
  if (TOPC_is_master()) print_array("AFTER:  ", array, ARRAY_LEN - 1);
  if (TOPC_is_master()) free(array_of_tasks);

  TOPC_finalize();
  return 0;
}
