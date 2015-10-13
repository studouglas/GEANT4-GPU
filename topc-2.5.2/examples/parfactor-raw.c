/********************************************************************
 ********************************************************************
 ** USAGE:  topcc [ --seq | --mpi | --pthread ] THIS_FILE
 ** ALGORITHM:  Euclidean sieve for factorization
 **   Demonstrates use of TOPC_raw_XXX
 **     for parallelizing legacy sequential code
 ** shared memory:  num_to_factor (divided by each factor found)
 ** task input:   input (test factor for num_to_factor)
 ** task output:  1 or 0 (true or false)
 ** update:  Divide num_to_factor by num
 ********************************************************************
 ********************************************************************/

#include <stdlib.h> // atoi()
#include <string.h> // strcmp()
#include <strings.h> // strcmp()
#include "topc.h"

#ifndef TASK_INTERVAL
#define TASK_INTERVAL 1000
#endif

#ifndef FALSE
enum {FALSE, TRUE};
#endif

// Used if TOPC_OPT_trace == 2, or: ./a.out --TOPC-trace=2
void trace_input( long *input ) { printf("%ld", *input); } 
void trace_result( long *input, long *output ) {
  if (*output == 1) printf("TRUE");
  if (*output == 0) printf("FALSE");
}

long factors[1000], max_factor=1, num_to_factor;
void MasSlaveFactor( long );

int main(int argc, char *argv[])
{
  int i;
  char *arg = argv[1];

#ifdef __cplusplus
  TOPC_OPT_trace_input = (TOPC_trace_input_ptr)trace_input;
  TOPC_OPT_trace_result = (TOPC_trace_result_ptr)trace_result;
#else
  TOPC_OPT_trace_input = trace_input;
  TOPC_OPT_trace_result = trace_result;
#endif
  TOPC_OPT_trace = 2;
  TOPC_init(&argc, &argv);
  if (argc > 2 && strcmp("-p4pg", argv[argc-2])) {
    // This occurs in MPICH 1.2.4
    printf("MPI bug:  MPI changed value of argc;"
           "  Don't use argc with this MPI\n");
    exit(1);
  }
  // Important:  Inspect command line only after TOPC_init()
  if (argc == 1 ) {
    if (TOPC_is_master())
      printf("WARNING: Missing number arg on command line.  Using 123456789\n");
    arg = "123456789";
  }

  MasSlaveFactor( strtol(arg, NULL, 10) );/* No error checking in this vers. */
  if ( TOPC_is_master() ) {
    for (i = 1; i <= factors[ 0 ]; i++) printf("%ld ", factors[i]);
    printf("\n"); }
  TOPC_finalize();
  // printf("%d: Exiting\n", TOPC_rank()); fflush(stdout);
  return 0;
}

TOPC_BUF DoTask( long *input )
{
  long limit;
  long num = *input;
  long task_out;

  // printf("DoTask(%d)\n", num); fflush(stdout);
  limit = num + TASK_INTERVAL;
  for ( ; num < limit; num++ )
    if (num_to_factor % num == 0) {
      // printf("%d: DoTask returning 1\n", TOPC_rank()); fflush(stdout);
      task_out = 1;
      return TOPC_MSG(&task_out, sizeof(long));
    }
  // printf("%d: DoTask returning 0\n", TOPC_rank()); fflush(stdout);
  task_out = 0;
  return TOPC_MSG(&task_out, sizeof(long));
}

TOPC_ACTION CheckTaskResult( long *input, long *output )
{
  // printf("CheckTaskResult()\n"); fflush(stdout);
  if ( *output == 0 ) return NO_ACTION;
  // NOTE: Suppose we factor 12, 2 is outstanding, and 4 is reported to factor
  if ( *input < max_factor ) return UPDATE;
  if ( TOPC_is_up_to_date() ) return UPDATE;
  return REDO;
}

void UpdateSharedMemory( long *input, long *output )
{
  long limit, num = *input;
  int i;
  // printf("%d: UpdateSharedMemory()\n", TOPC_rank()); fflush(stdout);
  limit = num + TASK_INTERVAL;
  // NOTE: Suppose we factor 12, 4 is reported as factor before 2 is reported
  for ( ; num < limit; num++ ) {
    if ( num_to_factor % num == 0 ) {
      if ( num < max_factor ) {
        max_factor = 1; // re-compute max_factor
        for( i = 1; i <= factors[0]; i++ ) {
          if (factors[i] > max_factor) max_factor = factors[i];
          while (factors[i] % num == 0) {
            factors[i] = factors[i] / num;
            num_to_factor = num_to_factor * num;
          }
        }
      }
      while (num_to_factor % num == 0) {
        factors[ ++factors[0] ] = num;
        num_to_factor = num_to_factor / num;
        if (num_to_factor == 1) return;
      }
    }
  }
}

void MasSlaveFactor( long num )
{ long last_num;

  factors[0]=0;
  num_to_factor = num;
  //printf("Entering master_slave()\n"); fflush(stdout);
  TOPC_raw_begin_master_slave(DoTask, CheckTaskResult,
                              UpdateSharedMemory);
  for ( last_num = 2; last_num <= num_to_factor; last_num += TASK_INTERVAL )
    TOPC_raw_submit_task_input(TOPC_MSG(&last_num, sizeof(last_num)));
  TOPC_raw_end_master_slave();
}
