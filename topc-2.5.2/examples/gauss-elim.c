/********************************************************************
 ********************************************************************
 ** USAGE:  topcc [ --seq | --mpi | --pthread ] THIS_FILE
 **         ./a.out                    [ using default input data, or]
 **         ./a.out dimension number_of_bands    [ using random data ]
 ** ALGORITHM:  Group adjacent rows of matrix into bands (sets of rows)
 **             TASK:  Do Gaussian elimination on input band and then
 **                    use other bands to further reduce input band.
 ** shared data:  rows of matrix grouped into bands:
 **               matrix, first_unred_band, first_unred_col[] (read/write)
 **               n (dimension), b (number of bands), band_size (read-only)
 ** task input:   band number
 ** task output:  struct band_info: first_unred_band, first_unred_col,
 **		  rows of band after all possible eliminations done
 ** update:       copy band to shared data of all slaves
 ** FOR A LARGER RUN, INVOKE BINARY AS:  ./a.out 120 30
 ********************************************************************
 ********************************************************************/

/* #define DEBUG */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> // strcmp()
#include <strings.h> // strcmp()
#include "topc.h"

#define MAX_DIM 1000
/* DIM is default, if no args given */
#define DIM 4
#define BANDS 2
#define BAND_SIZE ((DIM+BANDS-1)/BANDS) /* For b | n, this is just n/b */
int band_size = BAND_SIZE;
float matrix_vals[DIM*DIM] = { 20.0,  2.0,  3.0,  4.0,
                                5.0, 20.0,  7.0,  8.0,
                                9.0, 10.0, 20.0, 12.0,
                               13.0, 14.0, 15.0, 20.0
};
float *matrix = matrix_vals;
/* Columns 0 up to first_unred_col[b] of band b are 0 */
int *first_unred_col; /* BANDS elements, malloc in main() */
int first_unred_band = 0; /* Done when first_unred_band = b */
int *band_is_busy;     /* BANDS elements, malloc in main() */

#define max(x,y) ( (x) < (y) ? y : x )
struct band_info {
  int band_no;
  int first_unred_band;
  int first_unred_col;
  /* float band[BAND_SIZE * DIM]; */
  /* task_out will use dynamically allocated space to extend band:
   * malloc( sizeof(struct band_info)+BAND_SIZE * DIM*sizeof(float) )
   */
  float band[1];
};

void mytrace_input ( int *in_msg )
{ printf("band %d", *in_msg);
}
void mytrace_result ( int *ignore, struct band_info *out_msg )
{ printf("row vector: %5.1f %5.1f ...",
	 (out_msg->band)[0], out_msg->band[1] );
}

/* These will be set to DIM and BANDS inside main() */
int n, b;  /* n = matrix dimension; b = number of bands */

void reduce_row(float *matrix_rows[], int row_to_red, int row, int pivot)
{ float scalar = matrix_rows[row_to_red][pivot] / matrix_rows[row][pivot];
  float *row1 = matrix_rows[row_to_red];
  float *row2 = matrix_rows[row];
  int j;

#ifdef DEBUG
  printf("reducing row %d, using row %d and pivot %d\n", row_to_red,row,pivot);
#endif
  row1[pivot] = 0;
  for (j = pivot+1; j < n; j++)
    row1[j] = row1[j] - scalar * row2[j];
}

TOPC_BUF generate_task_input() {
  static int i;

  for (i = first_unred_band; i < b ; i++)
    if (first_unred_col[i] <= first_unred_col[max(first_unred_band-1,0)]
        && band_is_busy[i] == 0) {
      band_is_busy[i] = 1;
      return TOPC_MSG(&i, sizeof(i)); }
  return NOTASK; }

/* FOR EFFICIENCY ONLY:  (avoid copying task_out, as done by TOPC_MSG())
 * TOPC_thread_private illustrates how to make TOPC_MSG_PTR() in do_task()
 *   compatible with shared memory model.
 * Most programs would use TOPC_MSG(), allocate task_out locally on stack,
 *   and TOPC_MSG() would copy it to the TOP-C address space.
 * Ideally, if C allowed it, we would just write:
 *    THREAD_PRIVATE struct band_info *global_output = NULL;
 * and the first call to do_task() would malloc the memory.
 * Instead, we write this extra code for full generality, to run in the
 * shared memory model or any other memory model.
 */
typedef struct band_info *TOPC_thread_private_t;
#define task_out TOPC_thread_private
TOPC_thread_private_t task_out_debug() {  /* needed for debugging in gdb */
  return task_out; 
}

#define sizeof_task_out (sizeof(struct band_info) + n*band_size*sizeof(*matrix))

TOPC_BUF do_task(int *band_ptr) /* 0 <= band_no <= b - 1 */
{ int band_no = *band_ptr;
  int row_start = band_no * band_size;
  int row_end = (band_no + 1) * band_size;
  int row, i, j, k;
  int pivot = max(first_unred_col[max(first_unred_band-1,0)], 0);
  /* float *matrix_rows[DIM]; */
  float *matrix_rows[MAX_DIM];

  if ((void *)task_out == NULL)
#  ifdef __cplusplus
    task_out = (struct band_info *)malloc( sizeof_task_out );
#  else
    task_out = malloc( sizeof_task_out );
#  endif
  /* Create array of matrix rows */
  for (row = 0; row < n; row++)
    matrix_rows[row] = &matrix[row*n];
  /* Copy over the band to be modified */
  for (i = 0, k = 0; i < band_size; i++) /* copy band */
    for (j = 0; j < n; j++, k++)
      task_out->band[k] = matrix_rows[band_no*band_size+i][j];
  /* Set matrix_rows to use the band to be modified */
  for (i = 0; i < band_size; i++)
    matrix_rows[band_no*band_size+i] = &task_out->band[i*n];

  /* Will always be true:  generate_task_input() satisfied this condition */
  if (first_unred_col[band_no] <= first_unred_col[max(first_unred_band-1,0)]) {
    /* reduce the band, band_no, by all rows above that band */
    for (row = max(first_unred_col[band_no], 0);
         row < first_unred_band * band_size; row++) /* row is fully reduced */
      /* reduce band [row_start..row_end-1] by row */
      for (i = row_start; i < row_end; i++)
        reduce_row(matrix_rows, i, row, row); /* pivot == row */
    /* This is non-zero block at or below first unreduced, diagonal block;
       Diagonalize the band, band_no, in place */
    for (row = row_start; row < row_end; row++, pivot++)
      for (i = row + 1; i < row_end; i++)
        reduce_row(matrix_rows, i, row, pivot);
  }

  /* task_out->band is now correct;  Fill in other new information */
  task_out->band_no = band_no;
  task_out->first_unred_band =
    ( band_no == first_unred_band ? first_unred_band+1 : first_unred_band );
  task_out->first_unred_col = task_out->first_unred_band * band_size;
  return TOPC_MSG_PTR(task_out, sizeof_task_out);
}

TOPC_ACTION check_task_result(int *band_ptr, struct band_info *output_ptr)
{  return UPDATE;
}

void update_shared_data(int *band_ptr, struct band_info *output_ptr)
{ float *mat_ptr = output_ptr->band;
  int i, band_no = *band_ptr;

  if (output_ptr->first_unred_band > first_unred_band)
    first_unred_band = output_ptr->first_unred_band;
  first_unred_col[output_ptr->band_no]
    = output_ptr->first_unred_col;
  for (i = band_no*band_size*n; i < (band_no+1)*band_size*n; i++)
    matrix[i] = *(mat_ptr++);
  band_is_busy[band_no] = 0; /* (Only master needs this) */
#ifdef DEBUG
if ( TOPC_is_master() )
  { int i, j;
    printf("updating band %d\n", band_no);
    for (i = 0; i < n; i++, printf("\n"))
      for (j = 0; j < n; j++)
        printf("%5.1f ", matrix[i*n+j]);
  }
#endif
}

/* The next four callbacks set a common seed on all processes */
unsigned int seed;
TOPC_BUF gen_seed() {
  static int done = 0;
  if (!done) {
    seed = (unsigned int)time(NULL) >> 6; /* Use same seed for 64 seconds */
    done = 1;
    return TOPC_MSG( &seed, sizeof(seed) );
  } else {
    return NOTASK;
  }
}
TOPC_BUF do_seed( unsigned int *ignore ) {
  return TOPC_MSG( NULL, 0 );
}
TOPC_ACTION check_seed( unsigned int *ignore, void *ignore2 ) {
  return UPDATE;
}
void update_seed( unsigned int *new_seed, void *ignore ) {
  seed = *new_seed;
}

int main(int argc, char *argv[])
{ int i, j;
  TOPC_init(&argc, &argv);

  if (argc > 2 && strcmp("-p4pg", argv[argc-2])) {
    // This occurs in MPICH 1.2.4
    printf("MPI bug:  MPI changed value of argc;"
           "  Don't use argc with this MPI\n");
    exit(1);
  }
  if ( argc ==  1) {
    n = DIM;  /* set dimension */
    b = BANDS;   /* number of bands */
  } else {
    if ( argc != 3 ) {
      if (TOPC_is_master())
        printf("Usage:  ./a.out DIM NUM_BANDS  [ or no args to default ]\n");
      exit(1);
    }
    n = atoi(argv[1]);
    b = atoi(argv[2]);
    if (n > MAX_DIM) {
      printf("dim > %d; Re-compile with larger MAX_DIM.\n", MAX_DIM); exit(1);
    }
    matrix = (float *)malloc(n*n*sizeof(*matrix));
    TOPC_master_slave( gen_seed, do_seed, check_seed, update_seed );
    srandom(seed);
    for ( i = 0; i < n*n; i++ )
      matrix[i] = 10.0*rand()/(RAND_MAX+1.0);
  }
#ifdef __cplusplus
  first_unred_col = (int *)malloc(b*sizeof(*first_unred_col));
  band_is_busy = (int *)malloc(b*sizeof(*band_is_busy));
#else
  first_unred_col = malloc(b*sizeof(*first_unred_col));
  band_is_busy = malloc(b*sizeof(*band_is_busy));
#endif
  band_size = (n+b-1)/b; /* For b | n, this is just n/b */
  for (i = 0; i < b; i++) {
    /* Columns 0 until first_unred_col[i] of band i are 0 cols and
      next block is diagonalized; -1 means nothing diagonalized */
    first_unred_col[i] = -1;
    band_is_busy[i] = 0;
  }
  if ( TOPC_is_master() ) {
    printf("Random Matrix:\n");
    for (i = 0; i < n; i++, printf("\n"))
      for (j = 0; j < n; j++)
        printf("%5.1f ", matrix[i*n+j]);
  }
#ifdef __cplusplus
  TOPC_OPT_trace_input = (TOPC_trace_input_ptr)mytrace_input;
  TOPC_OPT_trace_result = (TOPC_trace_result_ptr)mytrace_result;
#else
  TOPC_OPT_trace_input = mytrace_input;
  TOPC_OPT_trace_result = mytrace_result;
#endif
  TOPC_OPT_trace = 2;  /* 2 is default; Set to 0 for no tracing */
  TOPC_master_slave(generate_task_input, do_task, check_task_result,
	       update_shared_data);
  if ( TOPC_is_master() ) {
    printf("Result Matrix:\n");
    for (i = 0; i < n; i++, printf("\n"))
      for (j = 0; j < n; j++)
        printf("%5.1f ", matrix[i*n+j]);
  }
  TOPC_finalize();
  return 0;
 }

#if 0
/* THIS VERSION CAN BE USED TO DRIVE A SEQUENTIAL VERSION */
/* In this case, the parallel version required a new, more
   complicated algorithm than the sequential version */
float matrix[DIM*DIM] = {20.0, 2.0, 3.0, 4.0, 5.0, 20.0, 7.0, 8.0,
                         9.0, 10.0, 20.0, 12.0, 13.0, 14.0, 15.0, 20.0};
int main()
{ int row, i, j;
  n = DIM; /* Used by reduce_row */
  void reduce_row(float *, int, int, int);
  for (row = 0; row < n; row++)
    for (i = row + 1; i < n; i++)
      reduce_row(matrix, i, row, row); /* row already reduced */
  for (i = 0; i < n; i++, printf("\n"))
    for (j = 0; j < n; j++)
      printf("%5.1f ", matrix[i*n+j]);
  return 0;
}
#endif
