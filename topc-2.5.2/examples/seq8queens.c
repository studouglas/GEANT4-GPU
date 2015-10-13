/* This is the model sequential code, on which the parallel code is based. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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

int main( int argc, char **argv ) {
  int cols[NQUEENS], sols[MAX_SOLS][NQUEENS], num_sols = 0;
  int col0, col1, col2, col3, col4, col5, col6, col7;
  for ( col0 = 0; col0 < NQUEENS; col0++ ) {
    check( cols, 0, col0 ); /* This sets cols[0]=col0; */
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
                if ( ! check( cols, 6, col6 ) ) continue; /* try next column */
                for ( col7 = 0; col7 < NQUEENS; col7++ ) {
                  if ( check( cols, 7, col7 ) )
                    record_sols( sols, cols, &num_sols );
  } } } } } } } }
  printf("number of solutions:  %d\n", num_sols);
  printf("middle solution (%d-th solution):  %d %d %d %d %d %d %d %d\n",
         num_sols/2+1,
         sols[num_sols/2][0], sols[num_sols/2][1],
         sols[num_sols/2][2], sols[num_sols/2][3],
         sols[num_sols/2][4], sols[num_sols/2][5],
         sols[num_sols/2][6], sols[num_sols/2][7]
        );
  /*
  for (col0=0; col0<num_sols; col0++)
    printf("solution (%d-th solution):  %d %d %d %d %d %d %d %d\n",
           col0+1,
           sols[col0][0], sols[col0][1],
           sols[col0][2], sols[col0][3],
           sols[col0][4], sols[col0][5],
           sols[col0][6], sols[col0][7]
          );
  */
  return 0;
}
