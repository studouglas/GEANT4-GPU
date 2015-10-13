/*   TO COMPILE:  gcc -o ssh-procgroup ssh-procgroup.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_LEN 100

int main( int argc, char **argv )
{
  char *p4pg_file = "procgroup";
  FILE *fin;
  char *t0;
  int i = 0;
  int j, k;
  char *buf;
  char pg_array[1000][BUF_LEN];

  if ( (fin = fopen(p4pg_file, "r")) )
  {
    while( fgets( buf = pg_array[i], BUF_LEN, fin ) ) {
      if ( buf[0] == '#' ) continue; /* # means comment */
      if ( buf[0] == '\n' ) continue; /* skip null lines */

      t0 = strtok(buf, " ");
      if ( !strcmp(t0, "local") )
        continue;
      i++;
    }
  } else {
    fprintf(stderr, "Couldn't find procgroup file in current directory.\n");
    exit(1);
  }

  // Now use pg_array to create ssh commands and call system()
  for (j = 0; j < i; j++) {
    char command[256];
    sprintf( command, "ssh %s", pg_array[j] );
    for (k = 1; k < argc; k++)
      sprintf( command + strlen(command), " %s", argv[k] );
    if ( j % 5 != 0 )
      sprintf( command + strlen(command), " &" );
    printf("%s\n", command);
    system( command );
  }

  return 0;
}
