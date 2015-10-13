/* Usage:
 *         char *tmp_procgroup, ***argv;
 *         int *argc;
 *         TOPC_pre_extend_procgroup(procgroup_file, num_slaves, argc, argv);
 *         append to argv:   -p4pg tmp_procgroup
 *         MPI_init( argc, argv );
 *         TOPC_post_extend_procgroup(argc, argv);
 *
 * #define TEST // for standalone test of this file
 *
 * Extends procgroup file using relative pathnames:
 * "-" means pathname of current binary
 * relative pathname means relative to pathname of _current_ binary.
 *
 * NOTE:  It the current working directory and pathname of the current
 *        binary are different, then:
 *   The procgroup is taken relative to the current working directory.
 *   The binary of the slave process is found relative to the directory
 *     of the binary of the master process.
 *   This makes it easy to set up and use multiple architectures,
 *     as seen in the following procgroup file.
 * local 0
 * alphahost.university.edu 1 ../alpha/a.out
 * sparchost.university.edu 1 ../sparc/a.out
 * linuxhost.university.edu 1 ../linuxhost/a.out
 *   This procgroup, and the input files of the program can be placed
 *   in a working directory, along with a link to the desired master binary.
 * PRINCIPLES:
 *   Let SLAVE_PATH be the path of the slave as given in the procgroup file,
 *     and let MASTER_DIR be the directory of the master process as invoked on
 *     on the command line.
 *   If SLAVE_PATH is an absolute path, TOP-C tries to create a slave
 *     process from a binary at SLAVE_PATH.
 *   If SLAVE_PATH is relative and if MASTER_DIR is an absolute path,
 *     then TOP-C tries to create a slave process from MASTER_DIR/SLAVE_PATH
 *   If SLAVE_PATH and MASTER_DIR are both relative, TOP-C tries to create
 *     a slave at $PWD/MASTER_DIR/SLAVE_PATH
 *   If SLAVE_PATH is `-', then TOP-C tries to ceate a slave at
 *     MASTER_DIR (if it's absolute), or $PWD/MASTER_DIR (if MASTER_DIR is rel.)
 *   In all cases, if the procgroup line contains command line arguments,
 *     those command line arguments are passed to the slave application
 *     as its first arguments, and any arguments on the master command
 *     line are appended to the list of arguments.
 * EXAMPLE: If you are at /home/gene/proj and execute ../mydir/a.out 17,
 *   TOP-C will look in /home/gene/proj for a procgroup file,
 *   and if the procgroup file has:  ./subdir/binary -remote
 *   then TOP-C executes:  /home/gene/proj/../mydir/subdir/binary -remote 17
 * NOTE ALSO:
 * If <num_slaves> is less than number of slaves specified by procgroup,
 *   then use first <num_slaves> from procgroup.
 * If <num_slaves> is more than number of slaves specified by procgroup,
 *   then re-use hosts from procgroup in the same order.
 * (If procgroup=(local,hostA,hostB), and num_slaves=5,
 *   use (local,hostA,hostB,hostA,hostB,hostA).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "comm.h"    // comm.h also defines HAVE_PTHREAD, etc., by default

#define MAX_HOSTS 1001

static char *tmp_procgroup = NULL;

#define MAX_LEN 161

/* Expand relative pathnames, "./a.out", "../subdir/a.out", and "-" */
static int expand_path( char *path, char *arg0 ) {
  char buf[MAX_LEN], *suffix;
  int offset, changed=0;
  /* sscanf: whitespace=[ \t\n]*, %s=[^ \t\n]+ (largest matches) */
  sscanf( path, " %*s %*s %n", &offset);
  suffix = path + offset;
  if (strlen(suffix) >= MAX_LEN)
    ERROR("procgroup file:  a path is longer than %d", MAX_LEN);
  strncpy(buf, suffix, MAX_LEN);
  if (buf[strlen(buf)-1] == '\n')
    buf[strlen(buf)-1] = '\0';
  if ( buf[1] == '\t' ) /* Replace user tab by canonical whitespace */
    buf[1] = ' ';
  if ( (buf[0] == '.' && buf[1] == '/')
       || (buf[0] == '.' && buf[1] == '.' && buf[2] == '/')
       || (buf[0] == '-' && buf[1] == '\0')
       || (buf[0] == '-' && buf[1] == ' ') ) {
    changed = 1;
    /* ADD: $PWD if arg0 is a relative pathname */
    if ( arg0[0] != '/' ) {
      TOPC_OPT_getpwd( suffix, 100 );
      suffix[ strlen(suffix) + 1 ] = '\0';
      suffix[ strlen(suffix) ] = '/';
      suffix += strlen(suffix);
    }
    /* ADD: arg0 */
    sprintf( suffix, "%s\n", arg0 );
    suffix += strlen(suffix);
    /* ADD: add buf (if original path a relative pathname) */
    if ( buf[0] == '.' ) {
      /* then backtrack to directory or arg0, and add rel. path, buf */
      while ( *(--suffix) != '/' ) {}
      sprintf( suffix+1, "%s\n", buf );
    }
    /* ADD: add args of buf (if buf is "- ...") */
    if ( buf[0] == '-' && buf[1] == ' ' )
      sprintf( suffix-1, "%s\n", buf+1 );
  }
  return changed;
}

static void read_and_expand_procgroup
  (char *procgroup_file, int num_slaves, char *arg0,
   char *hosts[], int *num_hosts, int *is_changed) // last three: output vars
{
  FILE *procfile;
  int c;

  *num_hosts = *is_changed = 0;
  procfile = fopen(procgroup_file, "r");
  // In options.c, we already verified that procgroup_file exists.
  if ( procfile == NULL ) {
    perror("TOP-C:  read_and_expand_procgroup");
    ERROR("TOP-C:  TOPC_OPT_pre_extend_procgroup:"
          "  Couldn't read procgroup file:  %s\n", procgroup_file);
  }
  c = getc(procfile);
  while ( c != EOF
         && ((num_slaves == UNINITIALIZED) || (*num_hosts <= num_slaves+1)) ) {
    switch ( c ) {
    case EOF:
      break;
    case '\n':
      c = getc(procfile);
      break;
    case '#':
      while( (c=getc(procfile)) != '\n' && c != EOF ) {}
      break;
    case ' ':
      while( (c=getc(procfile)) == ' ' ) {}
      break;
    default:
      ungetc( c, procfile );
      hosts[(*num_hosts)++]=fgets( malloc(MAX_LEN+1), MAX_LEN+1, procfile );
      if ( strlen(hosts[(*num_hosts)-1]) >= MAX_LEN )
	ERROR("Line of procgroup file is more than %d characters\n",
	       MAX_LEN);
      if (*num_hosts > MAX_HOSTS)
	ERROR("*** Number of hosts in procgroup file(%d) exceeds max (%d)\n"
	      "    Increase procgroup.c:MAX_HOSTS\n", *num_hosts, MAX_HOSTS);
      if (expand_path( hosts[(*num_hosts)-1], arg0 ))
	*is_changed = 1;
      c = getc(procfile);
      break;
    } // end switch
  } // end while
  fclose(procfile);
}

static char *write_tmp_procgroup(char *hosts[], int num_hosts, int num_slaves)
{
  char *outname = malloc(80);
  FILE *outfile;
  char *str;
  int i, j, num;

  sprintf(outname, "%s/procgroup.%d",
	  ( (str = getenv("TMPDIR")) ? str : "/tmp"), getpid());
  outfile = fopen(outname, "w");
  if ( outfile == NULL ) {
    perror("TOP-C:  write_tmp_procgroup");
    ERROR("Couldn't create temporary file %s\n"
          "        Check filesystem and environment variable TMPDIR\n",
          outname);
  }
  for ( i=0, j=0; i<=num_slaves; i++, j++ ) {
    if ( j >= num_hosts ) j = 1;
    num = fprintf(outfile, "%s", hosts[j]);
    if (num != (int)strlen(hosts[j])) {
      perror("TOP-C:  write_tmp_procgroup");
      ERROR("Couldn't write to %s\n", outname);
    }
  }
  fclose(outfile);
  return outname;
}

static void modify_argc_and_argv(char *new_procgroup, int *argc, char ***argv)
{
  static char **new_argv;
  int i;

  // Reserve 3 extra slots for:  "-p4pg" <new_procgroup> NULL
  new_argv = malloc( (*argc + 3) * sizeof(char **) );
  for (i = 0; i <= *argc; i++)
    new_argv[i] = (*argv)[i];
  new_argv[*argc] = "-p4pg";
  new_argv[*argc + 1] = new_procgroup;
  new_argv[*argc + 2] = NULL;
  *argv = new_argv;
  *argc += 2;
}

/* In TOPC_OPT_pre_init */
void TOPC_OPT_pre_extend_procgroup
  (char *procgroup_file, int num_slaves, int *argc, char ***argv)
{
  char *hosts[MAX_HOSTS];
  int num_hosts;
  int is_changed;
  int i;

  if ( num_slaves != UNINITIALIZED
       && (num_slaves < 1 || num_slaves > 1000) )
    ERROR("TOP-C:  Number of slaves, %d, not in range [1..1000].\n",
	   num_slaves);

  // hosts, num_hosts and is_changed are output parameters
  read_and_expand_procgroup(procgroup_file, num_slaves, (*argv)[0],
			    hosts, &num_hosts, &is_changed);
  if (num_slaves == UNINITIALIZED)
    num_slaves = num_hosts - 1;
  if ( is_changed || (num_hosts != num_slaves+1) )
    procgroup_file = tmp_procgroup
                   = write_tmp_procgroup(hosts, num_hosts, num_slaves);
  for (i = 0; i < num_hosts; i++)
    free(hosts[i]);
  if (0 != strncmp(procgroup_file, "./procgroup", 12)
      && 0 != strncmp((*argv)[*argc-1], "-p4amslave", 11))
    modify_argc_and_argv(procgroup_file, argc, argv);
}

void TOPC_OPT_post_extend_procgroup(int *argc, char ***argv) {
  if( tmp_procgroup ) {
    unlink( tmp_procgroup );
    tmp_procgroup = NULL;
    // Maybe MPI_init() wasn't mpinu, and so didn't know about -p4pg flag.
    if ( *argc >= 3 && 0 == strncmp((*argv)[*argc-2], "-p4pg", 6) ) {
      (*argv)[*argc-2] = NULL;
      *argc -= 2;
    }
  }
}


#ifdef TEST
main(int myargc, char *myargv[]) {
  /*
  char *myargv[1] = {"/myhome/me/mya.out"};
  int myargc = 1;
  */
  char buf[100];

  TOPC_OPT_pre_extend_procgroup("procgroup", 9, &myargc, &myargv);
  printf("DONE:  %s\n", tmp_procgroup);
  system( (sprintf(buf, "cat %s", tmp_procgroup), buf) );
  TOPC_OPT_post_extend_procgroup();
  exit(0);
}
#endif
