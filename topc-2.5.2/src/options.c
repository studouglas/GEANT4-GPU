 /**********************************************************************
  * TOP-C (Task Oriented Parallel C)                                   *
  * Copyright (c) 2004 Gene Cooperman <gene@ccs.neu.edu>               *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU Lesser General Public         *
  * License as published by the Free Software Foundation; either       *
  * version 2.1 of the License, or (at your option) any later version. *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * Lesser General Public License for more details.                    *
  *                                                                    *
  * You should have received a copy of the GNU Lesser General Public   *
  * License along with this library (see file COPYING); if not, write  *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact Gene Cooperman          *
  * <gene@ccs.neu.edu>.                                                *
  **********************************************************************/

/*  ISSUE:  In options.c, clean up standalone test */

/***********************************************************************
 ***********************************************************************
 ** USAGE:  get_topc_options( int *argc, char ***argv );
 **   Command line options:  --TOPC-XXX= (variable:  TOPC_OPT_XXX)
 ***********************************************************************
 ***********************************************************************/

/*  gcc -DTEST options.c; ./a.out --TOPC-help   for testing */

#include <sys/types.h>  // for open()
#include <sys/stat.h>   // for open()
#include <fcntl.h>      // for open()
#include <stdio.h>      // for fopen()
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>
#include <time.h> // time()
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>
#include "topc.h"
#include "comm.h"    // comm.h also defines HAVE_PTHREAD, etc., by default
#ifdef HAVE_PTHREAD
#  ifdef HAVE_PTHREAD_SETCONCURRENCY
// Define __USE_UNIX98 this for sake of pthread_setconcurrency()                
// Consider no longer using old pthread_setconcurrency()                        
#    define __USE_UNIX98
#    include <pthread.h>
#  endif
#endif

#ifdef TEST
int COMM_is_shared_memory(void) { return 1; }
int COMM_node_count()
  {extern int TOPC_OPT_num_slaves; return TOPC_OPT_num_slaves + 1;};
#define UNINITIALIZED -1;
#endif

#define ARGS_MAX 100
static char *saved_args[ARGS_MAX]; /* from argv[] */
static int args_idx = 0; /* from argv[] */

/* Defaults; types should also be declared extern in topc.h and comm.h */
#define bool int
int TOPC_OPT_num_slaves = UNINITIALIZED;
int TOPC_OPT_aggregated_tasks = 1;
int TOPC_OPT_slave_wait = 0; /* seconds */
int TOPC_OPT_slave_timeout = 1800;
int TOPC_OPT_trace = 2;
bool TOPC_OPT_help = 0;     /* --TOPC-help (no arg) sets it to 1 */
/* Set to 2, to distinguish from --TOPC-help --TOPC-verbose, which
   will be detected by TOPC_OPT_verbose == 1 */
bool TOPC_OPT_verbose = 2;  /* --TOPC-verbose (no arg) sets it to 1 */
bool TOPC_OPT_stats = 0;  /* --TOPC-stats (no arg) sets it to 1 */
char *TOPC_OPT_procgroup = "./procgroup";
char *TOPC_OPT_topc_log = "-"; /* - = stdout, but should set to topc.log if is "" */
int TOPC_OPT_safety = SAFETY_NONE;

enum opt_type { BOOL, INT, STRING };

struct option {
  char *option;
  void *var_ptr;
  enum opt_type type;
  char *help_string;
};

/* COULD COMPUTE DEFAULT NUM_SLAVES FIRST */

static struct option cmd_opt[] = {
  { "--TOPC-help", &TOPC_OPT_help, BOOL, "display this message" },
  { "--TOPC-stats", &TOPC_OPT_stats, BOOL, "display stats before and after" },
  { "--TOPC-verbose", &TOPC_OPT_verbose, BOOL, "set verbose mode" },
  { "--TOPC-num-slaves", &TOPC_OPT_num_slaves, INT,
    "number of slaves (sys-defined default)" },
  { "--TOPC-aggregated-tasks", &TOPC_OPT_aggregated_tasks, INT,
    "number of tasks to aggregate" },
  { "--TOPC-slave-wait", &TOPC_OPT_slave_wait, INT,
    "secs before slave starts (use w/ gdb attach)"},
  { "--TOPC-slave-timeout", &TOPC_OPT_slave_timeout, INT,
    "dist mem: secs to die if no msgs, 0=never"},
  { "--TOPC-trace", &TOPC_OPT_trace, INT,
    "trace (0: notrace, 1: trace, 2: user trace fncs.)" },
  { "--TOPC-procgroup", &TOPC_OPT_procgroup, STRING, "procgroup file (--mpi)" },
  { "--TOPC-topc-log", &TOPC_OPT_topc_log, STRING,
                       "NOT IMPL:  log file for TOPC output (\"-\" = stdout)" },
  { "--TOPC-safety", &TOPC_OPT_safety, INT,
    "[0..20]: higher turns off optimizations,\n"
    "                           try with --TOPC-verbose" },
  { "", NULL, BOOL, "" }
};

#define COL2 27
#define COL3 55

static char *print_arg_type(enum opt_type type) {
  switch (type) {
  case BOOL: return "[=<0/1>]";
  case INT: return "=<int>";
  case STRING: return "=<string>";
  default: return "";
  }
}

static void print_help(char *command) {
  struct option *opt;
  int len1, len2;
  printf("\nTOP-C Version " VER_STR " (" UPDATE_STR ");  (%s memory model)\n",
         COMM_mem_model);
  printf("Usage:  %s [ [TOPC_OPTION | APPLICATION_OPTION] ...]\n", command);
  printf("  where TOPC_OPTION is one of:\n");
  for ( opt = cmd_opt; opt->option[0] != '\0'; opt++ ) {
    printf("%s%s%n", opt->option, print_arg_type(opt->type), &len1 );
    printf("%*s%s %n", COL2 - len1, " ", opt->help_string, &len2 );
    if ( len1 + len2 > COL3 ) printf("\n%*s", COL3, " ");
    else printf("%*s", COL3 - len1 - len2, " ");
    switch (opt->type) {
    case BOOL:
      printf("[default: false]\n");
      break;
    case INT:
      printf("[default: %d]\n", *(int *)(opt->var_ptr));
      break;
    case STRING:
      printf("[default: \"%s\"]\n", *(char **)(opt->var_ptr));
      break;
    default:
      ERROR("FATAL:  invalid option type:  %d\n", opt->type);
    }
  }
  printf("\nThe environment variable TOPC_OPTS and the init file ~/.topcrc\n");
  printf("  are also examined for options (format:  --TOPC-xxx ...).\n");
  printf("You can change the defaults in the application source code.\n");
  printf("  For example, to change the default to --TOPC-trace=0,\n");
  printf("  add `TOPC_OPT_trace = 0;' before `TOPC_init(&argc, &argv);'\n");
  if (TOPC_OPT_verbose == 1) {
    printf("\nEffect of --TOPC-safety=<int>:\n"
           "  safety: >=%d: all;"
	   "  >=%d: no TOPC_MSG_PTR;"
           "  >=%d: no TOP-C memory mgr (uses malloc/free);\n"
	   "  >=%d: no aggregation of tasks;\n"
           "  >=%d: no TOPC_abort_tasks; >=%d: no receive thread on slave;\n"
           "  >=%d: default atomic read/write for DoTask, UpdateSharedData;\n"
           "   =%d: only 2 slaves; =%d: only 1 slave\n"
           "  >=%d: no timeout (no SIGALRM, dangerous for runaways)\n",
           SAFETY_NONE, SAFETY_NO_MSG_PTR,
	   SAFETY_NO_MEM_MGR, SAFETY_NO_AGGREG,
           SAFETY_NO_ABORT_TASKS, SAFETY_NO_RCV_THREAD,
           SAFETY_DFLT_ATOMIC_READ_WRITE, SAFETY_TWO_SLAVES, SAFETY_ONE_SLAVE,
	   SAFETY_NO_TIMEOUT
          );
  } else
    printf("Try `--TOPC-help --TOPC-verbose' for more information.\n");
  // TOPC_MSG_PTR() requires memory manager to MEM_malloc_ptr()
  //   header to record pointer.
  assert( SAFETY_NO_MEM_MGR > SAFETY_NO_MSG_PTR );
}

static void set_option( struct option *opt, char *val ) {
  char *endptr; // Used to check validity of strtol() arg.
  switch (opt->type) {
  case BOOL:
    if ( val[0] == '\0' ) {
      val = NULL;
      break;
    } /* else parse val in next case */
  case INT:
  case STRING:
    if ( val[0] != '=' )
      ERROR( "FATAL:  %s=<val>:  option takes %s argument.\n",
	      opt->option, (opt->type==INT ? "an int" : "a string") );
    else
      val = val + 1; // val was "=<arg>", now val is "<arg>"
    break;
  }
  switch (opt->type) {
  case BOOL:
    if ( val == NULL || *val == '\0' || *val == '1' )
      *(int *)(opt->var_ptr) = 1;
    else if ( *val == '0' )
      *(int *)(opt->var_ptr) = 0;
    else
      WARNING("Boolean option %s must have value 0, 1, or no arg"
              " (default: false)\n", opt->option);
    break;
  case INT:
    *(int *)(opt->var_ptr) =
      strtol( val, &endptr, 10 );
    if (*endptr != '\0')
      ERROR("Invalid integer value for %s=%s\n", opt->option, val);
    break;
  case STRING:
    *(char **)(opt->var_ptr) = malloc( strlen( val ) );
    strcpy( *(char **)(opt->var_ptr), val );
    break;
  }
}

static void get_options(int *argc, char ***argv) {
  struct option *opt;
  int i, j, end_options = 0;
  char *ch;

  for ( i = 1; i < *argc && args_idx < ARGS_MAX; i++)
    saved_args[args_idx++] = (*argv)[i];

#ifdef TEST
printf("BEGIN:  *argc=%d\n", *argc);
#endif
  for ( j = 1; j < *argc; j++ ) {
#ifdef TEST
printf("(*argv)[%d]=%s\n", j, (*argv)[j]);
#endif
    if ( strncmp( "--", (*argv)[j], 3) == 0 )
      end_options = 1; /* Stop processing TOP-C options; Leave "--" for app */
    if ( ! end_options && strncmp( (*argv)[j], "--TOPC", 6 ) == 0 ) {
      /* Convert --TOPC_the_option=XX-YY  to:  --TOPC-the-option=XX-YY */
      for (ch = (*argv)[j]+6; *ch != '\0' && *ch != '=' && ch-(*argv)[j]<80;
                              ch++)
        if (*ch == '_')
          *ch = '-';
      /* Look for matching --TOPC option */
      for ( opt = cmd_opt; opt->option[0] != '\0'; opt++ ) {
        if ( strncmp( opt->option, (*argv)[j], strlen(opt->option) ) == 0 ) {
          set_option( opt, (*argv)[j] + strlen(opt->option) );
          break;
        }
      }
      if ( opt->option[0] == '\0' ) { // if "--TOPC" did not match anything
        printf("\n*** Invalid option:  %s\n\n", (*argv)[j]);
        print_help(**argv);
        exit(1);
      }
    }
  }
#ifdef TEST
printf("END:  *argc=%d\n", *argc);
#endif
}
static void get_options_from_string( char * opts_str ) {
  int argc = 1;
  char * argv[100];
  char **argv_ptr = argv;

  // Copy string since get_options saves argv[] strings in saved_args
  opts_str = strcpy( malloc( strlen(opts_str)+1 ), opts_str );
  argv[argc] = strtok( opts_str, " \t" );
  argv[0] = "UNKNOWN_BINARY";
  while ( argv[argc] != NULL )
    argv[++argc] = strtok( NULL, " \t" );
  get_options( &argc, &argv_ptr );
}

static void strip_options(int *argc, char ***argv) {
  int i, j, end_options = 0;
#ifdef TEST
printf("BEGIN:  *argc=%d\n", *argc);
#endif
  for ( i = 1, j = 1; j < *argc; j++ ) {
#ifdef TEST
printf("(*argv)[%d]=%s\n", j, (*argv)[j]);
#endif
    if ( strncmp( "--", (*argv)[j], 3) == 0 )
      end_options = 1;
    if ( end_options || (strncmp((*argv)[j], "--TOPC", 6) != 0) ) {
      (*argv)[i] = (*argv)[j];
      i++;
    }
  }
  (*argv)[i] = (*argv)[j]; // Copy final null pointer
  *argc = i;
#ifdef TEST
printf("END:  *argc=%d\n", *argc);
#endif
}

static time_t start_time;
// These are defined in procgroup.c
void TOPC_OPT_pre_extend_procgroup
  (char *procgroup_file, int num_slaves, int *argc, char ***argv);
void TOPC_OPT_post_extend_procgroup(int *argc, char ***argv);

// Note this answers true even for other MPI's.
// This minor bug doesn't seem to hurt us.
static int has_procgroup_file() {
  static int val = -1;
  if (val == -1) {
    val = ! COMM_is_shared_memory()
          && 0 != strncmp(COMM_mem_model, "sequential", 3);
  }
  return val;
}

static void print_config_info() {
  char buf[61];

  printf("\n");
  printf("TOP-C Version " VER_STR " (" UPDATE_STR ")\n");
  system("date");
  system("uptime");
  start_time = time(NULL);
  if ( 0 == gethostname(buf, 60) )
    printf("hostname:  %.60s\n", buf);
  else
    printf("hostname:  <gethostname() failed>\n");
  TOPC_OPT_getpwd(buf, 60);
  printf("current directory:  %.60s\n", buf);
  printf("safety level:  %d\n", TOPC_OPT_safety);
  printf("memory model:  %s\n", COMM_mem_model);
  if ( has_procgroup_file() )
    printf("procgroup file:  %s\n", TOPC_OPT_procgroup);
}

void TOPC_OPT_pre_init( int *argc, char ***argv ) {
  struct stat buf;
  FILE *topcrc;
  char *opts_str;

  // Process .toprc
  if ( opts_str = getenv( "HOME" ) ) {
#define MAX_PATH 10000
    char path[MAX_PATH];
    strncat( path, opts_str, MAX_PATH );
    strncat( path + strlen(path), ".topcrc", MAX_PATH - strlen(path) );
    topcrc = fopen( path, "r" );
    if ( topcrc != NULL && (opts_str = fgets( path, MAX_PATH, topcrc )) )
      get_options_from_string( path );
  }

  // Process environment variable, TOPC_OPTS
  if ( opts_str = getenv("TOPC_OPTS") )
    get_options_from_string( opts_str );

  // Process command line arguments from (argc, argv) from main
  get_options(argc, argv);

  if (TOPC_OPT_procgroup[0] == '~') {
    char *home = getenv("HOME");
    char *tmp = malloc(strlen(TOPC_OPT_procgroup)+strlen(home));
    sprintf(tmp, "%s%s", home, TOPC_OPT_procgroup+1);
    TOPC_OPT_procgroup = tmp;
  }
  if (TOPC_OPT_aggregated_tasks > 1 && COMM_is_shared_memory() ) {
    //Problem is do_task_wrapper, which uses static local variables
    //  for output buffer.
    WARNING("TOPC_OPT_aggregated_tasks > 1 not supported for shared memory.\n"
	    "  Resetting TOPC_OPT_aggregated_tasks from %d to %d.");
    TOPC_OPT_aggregated_tasks = 1;
  }

  if (TOPC_OPT_safety == SAFETY_TWO_SLAVES)
    TOPC_OPT_num_slaves = 2;
  if (TOPC_OPT_safety >= SAFETY_ONE_SLAVE)
    TOPC_OPT_num_slaves = 1;

  if (TOPC_OPT_help) {
    print_help( **argv ); // **argv == command name ($0 on command line)
    if( TOPC_OPT_stats )
      print_config_info();
    exit( 0 );
  }

#ifndef TEST
  /*If dist. mem, using p4/MPI convention (procgroup/-p4amslave), not a slave.*/
  if ( has_procgroup_file()
       && 0 != strncmp((*argv)[*argc-1], "-p4amslave", 11) ) {
    if ( 0 == stat( TOPC_OPT_procgroup, &buf ) )
      TOPC_OPT_pre_extend_procgroup(TOPC_OPT_procgroup, TOPC_OPT_num_slaves,
                                    argc, argv);
    else /* The  MPI may not be MPINU; MPINU can signal err. if necessary */
      WARNING("procgroup file is:  %s", TOPC_OPT_procgroup);
  }
#endif
}

void TOPC_OPT_post_init( int *argc, char ***argv ) {
  strip_options(argc, argv);
  if (! TOPC_is_master()) {
/*  Consider this, or "%.60s connecting ..." in pre_init: */
#   if 0
    if ( TOPC_OPT_stats && ! COMM_is_shared_memory() ) {
      char buf[60];
      if ( 0 == gethostname(buf, 60) )
        printf("%.60s connected.\n", buf);
      else
        printf("hostname:  <gethostname() failed>\n");
    }
#   endif
    return;
  }

#ifndef TEST
  // If COMM_init() didn't consume any "-p4pg ...", this will strip it.
  TOPC_OPT_post_extend_procgroup(argc, argv);
#endif
  /* These things are known best only after initialization. */
  if( TOPC_OPT_stats ) {
    int i;
    print_config_info();
    printf("number of slaves:  %d\n", COMM_node_count() - 1);
#ifdef HAVE_PTHREAD
#  ifdef HAVE_PTHREAD_SETCONCURRENCY
    if (COMM_is_shared_memory()) {
      /* threads:  slaves + 1  (no thread should need to context switch) */
      printf("number of threads (master+slaves+stubs):  %d\n",
  	     pthread_getconcurrency() );
    }
#  endif
#endif
    printf("CALLED AS: ");
    for (i = 0; saved_args[i] != NULL && i < ARGS_MAX; i++)
      printf(" %s", saved_args[i]);
    printf("\n\n");
  }
}

/* Returns info as string terminated by '\n' */
static char * getcpuinfo( char * buf, char * substr ) {
  substr = strstr( buf, substr );
  if ( substr != NULL )
    substr = strstr( substr, ":" );
  if ( substr == NULL )
    substr = "";
  return substr;
}
static void print_cpuinfo () {
  char buf[10001];
  char *tmp;
  char *model, *bogomips, *cache;
  int num;
  int fd = open("/proc/cpuinfo", O_RDONLY);
  if ( fd != -1 ) {
    num = read( fd, buf, 10000 );
    if ( num < 10 ) return;
    buf[num] = '\0';
    buf[10000] = '\0';
    model = getcpuinfo( buf, "model name" );
    cache = getcpuinfo( buf, "cache size" );
    bogomips = getcpuinfo( buf, "bogomips" );
    close( fd );
    tmp = strchr( buf, '\n' );
    while ( tmp != NULL ) {
      tmp[0] = '\0';  /* Now replace markers by end of string */
      tmp = strchr( tmp + 1, '\n' );
    }
    printf("CPU%s (cache%s; bogomips%s)\n", model, cache, bogomips);
  }
}
/* Copied from Linux include/asm-* /timex.h */
// i386 sparc mips powerpc alpha arm
#ifdef __GNUC__
# if defined(__ppc__)
#  define __powerpc__
# endif
# if defined(_PPC_PRECISE)
#  define __ppc64__
# endif
// In macros below, val is "long long"
# if defined(__i386__)
#  define CPU_TYPE "Pentium or Pentium-compatible"
// Macro more conventionally known as rdtscll:
#  define set_cycles(val) __asm__ __volatile__ ("rdtsc" : "=A" (val))
# elif defined(__alpha__)
#  define CPU_TYPE "Alpha"
// Only the low 32 bits are available as a continuously counting entity.
#  define set_cycles(val) { unsigned int ret; \
	__asm__ __volatile__ ("rpcc %0" : "=r"(ret)); val = ret; }
# elif defined(__ppc64__)
#  define CPU_TYPE "PowerPC, 64-bit ?"
#  define set_cycles(val) { unsigned long ret; \
        /* MULTIPLYING BY 10 TO GET CLOCK RATE.  IS THIS CORRECT? */ \
	/*  "?" ADDED TO CPU_TYPE UNTIL THIS IS RESOLVED */ \
        __asm__ __volatile__("mftb %0" : "=r" (ret) : ); val = 10 * ret; }
# elif defined(__powerpc__) || defined(__PPC__)
#  define CPU_TYPE "PowerPC, 32-bit"
#  define set_cycles(val) val = 0
# elif defined(__sparc64__)
#  define CPU_TYPE "Sparc, 64-bit"
// #  define set_cycles(val) { unsigned long ret; \
//         __asm__ __volatile__("rd %%tick, %0" : "=r" (ret)); val = ret; }
#  define set_cycles(val) val = 0
# elif defined(__sparc__)
#  define CPU_TYPE "Sparc, 32-bit"
#  define set_cycles(val) val = 0
# else
#  define CPU_TYPE "Undetermined CPU"
#  define set_cycles(val) val = 0
# endif
#endif
static void print_cpu_speed() {
#if defined(__GNUC__)
  int highest;
  int vendor[4];
  vendor[3] = 0;

  print_cpuinfo();

  {
    long long begin;
    long long end;
    set_cycles(begin);
    {
      struct timespec req;
      req.tv_sec = 0;
      req.tv_nsec = 100000000; /* 100 million ns = 0.1s s */
      nanosleep( &req, NULL );
    }
    set_cycles(end);
    /* Ticks polled over 0.1s; print ticks / 100 M */
    printf("CPU speed: %1.2f GHz (%s)\n",
           (float)(end - begin)*10*1e-9, CPU_TYPE);
  }

# if defined(__i386__)
#  define cpuid(func,ax,bx,cx,dx) \
        __asm__ __volatile__ ("cpuid": \
                "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "0" (func));
  cpuid(0,highest,vendor[0],vendor[2],vendor[1]);
  printf(" (CPU vendor: %s)\n", (char *)vendor);
# endif
#endif
}

void TOPC_OPT_finalize( int num_tasks, int num_updates, int num_redos ) {
  if( TOPC_OPT_stats ) {
    struct rusage buf;

    /* getrusage is BSD */
    if ( -1 == getrusage( RUSAGE_SELF, &buf ) )
      PERROR("getrusage");

    printf("\nMASTER:\n");
    printf("Maximum resident set size: %ld\n", buf.ru_maxrss);
    printf("Page faults: %ld\n", buf.ru_majflt);
    printf("Swaps: %ld\n", buf.ru_nswap);
    printf("Voluntary context switches: %ld\n", buf.ru_nvcsw);
    printf("Involuntary context switches: %ld\n", buf.ru_nivcsw);
    print_cpu_speed();
    printf("CPU time (user s): %ld\n",
           (long)buf.ru_utime.tv_sec /* + buf.ru_utime.tv_usec/1000000 */);

    if  ( 0 ) { /* Wait until TOPC_gather() available to collect stats */
      /* getrusage is BSD */
      if ( -1 == getrusage( RUSAGE_CHILDREN, &buf ) )
        PERROR("getrusage");

      printf("\nSLAVE:\n");
      printf("Maximum resident set size: %ld\n", buf.ru_maxrss);
      printf("Page faults: %ld\n", buf.ru_majflt);
      printf("Swaps: %ld\n", buf.ru_nswap);
      printf("Voluntary context switches: %ld\n", buf.ru_nvcsw);
      printf("Involuntary context switches: %ld\n", buf.ru_nivcsw);
      printf("CPU time (user s): %ld\n",
             (long)buf.ru_utime.tv_sec /* + buf.ru_utime.tv_usec/1000000 */);
    }

    printf("\nParallel computation finished\n");
    system("date");
    system("uptime");
    printf("number of tasks:  %d, number of UPDATE's:  %d,"
           " number of REDO's:  %d\n", num_tasks, num_redos, num_updates);
    printf("Elapsed (wall-clock) time (s):  %d\n",
           (int)(time(NULL) - start_time) );
    printf("\n");

    /* We should also call TOPC_master_slave() with do_task() to collect
     * slave statistics.  Do we need to declare do_task() volatile?
     */
  }
}

char *TOPC_OPT_getpwd( char *buf, size_t size ) {
  char *p;
  struct stat dotstat, pwdstat;

  if ( (p = getenv ("PWD")) != 0
       && *p == '/'
       && stat (p, &pwdstat) == 0
       && stat (".", &dotstat) == 0
       && dotstat.st_ino == pwdstat.st_ino
       && dotstat.st_dev == pwdstat.st_dev )
    return strncpy(buf, p, size);
  else
    return getcwd(buf, size);
}


#ifdef TEST
int main( int argc, char **argv ) {
  TOPC_OPT_pre_init( &argc, &argv );
  TOPC_OPT_pre_init( &argc, &argv );
  exit(0);
}
#endif
