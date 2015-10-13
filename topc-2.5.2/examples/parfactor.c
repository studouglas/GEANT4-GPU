/********************************************************************
 ********************************************************************
 ** USAGE:  topcc [ --seq | --mpi | --pthread ] THIS_FILE
 ** ALGORITHM:  Euclidean sieve for factorization
 ** shared data:  num_to_factor (divided by each factor found)
 ** task input:   input (test factor for num_to_factor)
 ** task output:  1 or 0 (true or false)
 ** update:  Divide num_to_factor by num
 ********************************************************************
 ********************************************************************/

#include <stdlib.h> // for atol() [ Note long is only 32 bits on many mach's]
#include <limits.h> // for LONG_MAX
#include <errno.h>
#include <string.h> // for strcmp
#include <strings.h> // for strcmp
#include "topc.h"

#ifndef TASK_INTERVAL
#define TASK_INTERVAL 1000
#endif

// Used if TOPC_OPT_trace == 2, or: ./a.out --TOPC-trace=2
void trace_input( long *input ) { printf("%ld", *input); } 
void trace_result( long *input, long *output ) {
  if (*output == 1) printf("TRUE");
  if (*output == 0) printf("FALSE");
}

long factors[1000], orig_num, max_factor=1, num_to_factor, next_num;
// TOP-C callback functions:
TOPC_BUF GenerateTaskInput(void);
TOPC_BUF DoTask( long *input );
TOPC_ACTION CheckTaskResult( long *input, long *output );
void UpdateSharedData( long *input, long *output );
long get_argument( int argc, char **argv );

int main(int argc, char *argv[])
{
  int i;

  // Set up defaults.  Can be overridden, e.g.:  ./a.out --TOPC-trace=0
#ifdef __cplusplus
  TOPC_OPT_trace_input = (TOPC_trace_input_ptr)trace_input;
  TOPC_OPT_trace_result = (TOPC_trace_result_ptr)trace_result;
#else
  TOPC_OPT_trace_input = trace_input;
  TOPC_OPT_trace_result = trace_result;
#endif
  TOPC_OPT_trace = 2;
  TOPC_init(&argc, &argv);
  // This occurs in MPICH 1.2.4:
  if (argc > 2 && strcmp("-p4pg", argv[argc-2])) {
    printf("MPI bug:  MPI changed value of argc;"
	   "  Don't use argc with this MPI\n");
    exit(1);
  }
  // Important:  Inspect command line only after TOPC_init()
  orig_num = num_to_factor = get_argument(argc, argv);

  factors[0]=0; // factors[0] is number of factors found so far
  next_num = 2 - TASK_INTERVAL; // last number factored
  TOPC_master_slave(GenerateTaskInput, DoTask, CheckTaskResult,
                    UpdateSharedData);

  if ( TOPC_is_master() ) {
    for (i = 1; i <= factors[ 0 ]; i++)
      { printf("%ld ", factors[i]); fflush(stdout); }
    printf("\n");
  }
  TOPC_finalize();
  // printf("%d: Exiting\n", TOPC_rank()); fflush(stdout);
  return 0;
}

TOPC_BUF GenerateTaskInput() {
  next_num = next_num + TASK_INTERVAL;
  if ( next_num > num_to_factor ) return NOTASK;
  else return TOPC_MSG(&next_num, sizeof(next_num));
}

TOPC_BUF DoTask( long *input )
{
  long limit;
  long num = *input;
  long task_out;

  limit = num + TASK_INTERVAL;
  for ( ; num < limit; num++ )
    if (num_to_factor % num == 0) {
      task_out = 1;
      return TOPC_MSG(&task_out, sizeof(long));
    }
  task_out = 0;
  return TOPC_MSG(&task_out, sizeof(long));
}

TOPC_ACTION CheckTaskResult( long *input, long *output )
{
  if ( *output == 0 ) return NO_ACTION;
  if ( TOPC_is_up_to_date() ) return UPDATE;
  // Suppose we factor 1006 and a slave reports 1006 before 2 is reported
  if ( *input < max_factor ) return UPDATE;
  else return REDO;
}

void UpdateSharedData( long *input, long *output )
{
  long limit, num = *input;
  int i;
  limit = num + TASK_INTERVAL;
  // Suppose we factor 1006 and a slave reports 1006 before 2 is reported
  for ( ; num < limit; num++ ) {
    if ( orig_num % num == 0 ) {
      if ( num < max_factor ) {
        max_factor = 1; // re-compute max_factor
        for( i = 1; i <= factors[0]; i++ ) {
          if (factors[i] > max_factor) max_factor = factors[i];
          while (factors[i] % num == 0 && factors[i] != num) {
            factors[i] = factors[i] / num;
            num_to_factor = num_to_factor * num;
          }
        }
      }
      while (num_to_factor % num == 0) {
        factors[ ++factors[0] ] = num;
	if (num > max_factor) max_factor = num;
        num_to_factor = num_to_factor / num;
        if (num_to_factor == 1) return;
      }
    }
  }
}

long get_argument(int argc, char **argv) {
  long num_to_factor = 123456789;
  if (argc == 1) { // If command line doesn't have num_to_factor
    if (TOPC_is_master())
      printf("WARNING: Missing number arg on command line. Using 123456789\n");
  } else {
    errno = 0;
    /* Replace strtol by atol(argv[1]) if your UNIX doesn't have strtol */
    num_to_factor = strtol(argv[1], NULL, 10);
    if (num_to_factor > 0 && errno == ERANGE) {
      if (TOPC_is_master())
        printf("Argument out of range.  Max. integer is:  %ld\n", LONG_MAX);
      exit(1);
    }
  }
  if (num_to_factor < 2) {
    fprintf(stderr, "Can't factor number less than 2\n"); exit(1);
  }
  return num_to_factor;
}
