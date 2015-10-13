/* This is the model sequential code, on which the parallel code is based. */

#ifndef ARRAY_LEN
#define ARRAY_LEN 100
#endif

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include <assert.h>

void quicksort( int array[], int start, int end, int level ) {
  int i = start, j = end, pivot = array[start];
  int tmp;

  level = level + 1;
  if ( start >= end ) return;

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

void print_array( char *string, int array[], int len ) {
  int i;
  printf("%s", string);
  for ( i=0; i < len; i++ )
    printf("%d ", array[i]);
  printf("\n");
}

int main( int argc, char **argv ) {
  int array[ARRAY_LEN];
  int i;

  srandom( (unsigned int)time(NULL) );
  for ( i=0; i < ARRAY_LEN; i++ ) array[i] = ( random() & ((1<<10)-1) );

  print_array("BEFORE:  ", array, ARRAY_LEN);
  quicksort( array, 0, ARRAY_LEN - 1, 0 );
  print_array("AFTER:  ", array, ARRAY_LEN);
  return 0;
}
