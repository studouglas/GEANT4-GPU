 /**********************************************************************
  * TOP-C (Task Oriented Parallel C)                                   *
  * Copyright (c) 2000 Gene Cooperman <gene@ccs.neu.edu>               *
  *                    and Victor Grinberg <victor@ccs.neu.edu>        *
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

#include <stdio.h>
#include <stdarg.h>
#include <errno.h>

/***********************************************************************
 * This is included in topc.c and comm-xxx.c and page.c .  It defines the
 * interface between the two.
 */

#ifndef _COMM_H
#define _COMM_H

/* #define HAVE_PTHREAD as default, but not if HAVE_PTHREAD == 0 */
#ifdef HAVE_PTHREAD
# if HAVE_PTHREAD == 0
#  undef HAVE_PTHREAD
# endif
#else
# define HAVE_PTHREAD
#endif

/* #define HAVE_SEMAPHORE as default, but not if HAVE_SEMAPHORE == 0 */
#ifdef HAVE_SEMAPHORE
# if HAVE_SEMAPHORE == 0
#  undef HAVE_SEMAPHORE
# endif
#else
# define HAVE_SEMAPHORE
#endif

/* If HAVE_THR_SETCONCURRENCY, undefine it and patch pthread_setconcurrency */
#ifdef HAVE_THR_SETCONCURRENCY
# ifndef HAVE_PTHREAD_SETCONCURRENCY
#  include <thread.h>
#  define pthread_getconcurrency thr_getconcurrency
#  define pthread_setconcurrency thr_setconcurrency
#  define HAVE_PTHREAD_SETCONCURRENCY
# endif
# undef HAVE_THR_SETCONCURRENCY
#endif

#include <stdio.h>  // Needed to define size_t

//Compile-time parameters (complete set in topc.c)
#define TOPC_MAX_SLAVES            1000       /* Max. number of slaves */
#define NUM_MSG_CACHE_PTR      100  /* Num pending msg's that can be cached */

extern char *COMM_mem_model;

/***********************************************************************
 * Interface to command line options (options.c)
 */
void TOPC_OPT_pre_init( int *argc, char ***argv );
void TOPC_OPT_post_init( int *argc, char ***argv );
void TOPC_OPT_finalize( int num_tasks, int num_redos, int num_updates );
char *TOPC_OPT_getpwd( char *buf, size_t size );
#define UNINITIALIZED		   -1

/***********************************************************************
 * TOP-C command line options for application (see options.c for defs)
 */
// These are also in topc.h
extern int TOPC_OPT_num_slaves;
extern int TOPC_OPT_aggregated_tasks;
extern int TOPC_OPT_slave_wait;
extern int TOPC_OPT_slave_timeout;
extern int TOPC_OPT_trace;
extern int TOPC_OPT_help;
extern int TOPC_OPT_verbose;
extern int TOPC_OPT_stats;
extern char * TOPC_OPT_procgroup;
extern char * TOPC_OPT_topc_log;
extern int TOPC_OPT_safety;

/***********************************************************************
 * Levels for TOPC_OPT_safety
 * These values are subject to change as more safety levels are added.
 */
#define SAFETY_NONE			 0
#define SAFETY_NO_MSG_PTR		 4
// Still not sure if memory manager has bug in shared memory in unusual case
#define SAFETY_NO_MEM_MGR		 6
#define SAFETY_NO_AGGREG		 8
#define SAFETY_NO_ABORT_TASKS		12
#define SAFETY_NO_RCV_THREAD		14
#define SAFETY_DFLT_ATOMIC_READ_WRITE	16
#define SAFETY_TWO_SLAVES		19
#define SAFETY_ONE_SLAVE		20
#define SAFETY_NO_TIMEOUT		21

/***********************************************************************
 * TOP-C data types (shared with topc.h)
 */
#ifndef TOPC_TYPES
#define TOPC_TYPES
typedef int TOPC_BOOL;
typedef struct {
  void  *data;
  size_t data_size;
} TOPC_BUF;
typedef void TOPC_FUNCTION();
struct TOPC_OPT {
       char *name;
       union { int integer; TOPC_FUNCTION *fnc; } value;
};
#endif

/***********************************************************************
 * TOP-C data types (not in topc.h)
 */
enum TAG {
  NO_TAG,
  TASK_INPUT_TAG,
  REDO_INPUT_TAG, /* Currently, TASK_INPUT_TAG used insted during REDO */
  CONTINUATION_INPUT_TAG,
  UPDATE_INPUT_TAG,
  UPDATE_OUTPUT_TAG,
  SLAVE_REPLY_TAG,
  END_MASTER_SLAVE_TAG,
  PING_TAG,
  CHDIR_TAG,
  SLAVE_WAIT_TAG,
  SOFT_ABORT_TAG,
  MEM_TAG, /* placeholder by MEM system; replaced by other tag later */
  PAGE_ACCESS_TAG,
  MAX_TAG /* not used, marks last valid tag */
};

extern TOPC_BUF (*COMM_do_task)(void *input); // for comm-seq.c
extern TOPC_BUF TOPC_MSG(void *input, size_t input_size); // for comm-seq.c
extern void (*COMM_slave_loop)(void);         // for comm-pthreads.c

/***********************************************************************
 * COMM layer functions (interface between comm-xxx.c and topc.c)
 */
void COMM_init(int *argc, char ***argv);
void COMM_finalize(void);
TOPC_BOOL COMM_initialized(void);
int COMM_is_shared_memory(void);
int COMM_node_count(void);
int COMM_rank(void);
TOPC_BOOL COMM_probe_msg(void);
TOPC_BOOL COMM_probe_msg_blocking(int);
TOPC_BOOL COMM_send_msg(void *msg, size_t msg_size, int dst, enum TAG tag);
TOPC_BOOL COMM_receive_msg(void **msg, size_t *msg_size, int *src, enum TAG *tag);
void COMM_free_receive_buf(void *msg);
TOPC_BOOL COMM_is_on_stack(void *buf, size_t size);
extern void *COMM_stack_bottom;
void COMM_soft_abort(void); // called by master
extern volatile int COMM_is_abort_pending; // read on slave

/***********************************************************************
 * SMP functions
 */
void COMM_begin_atomic_read(void);
void COMM_end_atomic_read(void);
void COMM_begin_atomic_write(void);
void COMM_end_atomic_write(void);
TOPC_BOOL COMM_is_in_atomic_section(void);

// end TOPC_H
#endif

/***********************************************************************
 * Functions from memory.c
 */
enum hdr_type {
  UNUSED_HDR, IS_MALLOC, IS_PRE_MSG, IS_PRE_MSG_PTR,
  IS_MSG, IS_MSG_PTR };
void *MEM_malloc(size_t size, int source, enum TAG tag, enum hdr_type type);
void MEM_register_buf(void *buf);
void MEM_free(void *buf);

/***********************************************************************
 ***********************************************************************
 ** Printout functions and debugging
 ***********************************************************************
 ***********************************************************************/
static void WARNING(char *format,...) {
  va_list ap;
  if (TOPC_OPT_verbose >= 0) {
    va_start(ap, format);
    fprintf(stderr, "*** TOP-C WARNING:  ");
    vfprintf(stderr, format, ap);
    fprintf(stderr, "\n");
    va_end(ap);
  }
}

//This function should perhaps take steps to perform proper shutdown
static void ERROR(char *format,...) {
  va_list ap;
  va_start(ap, format);
  fprintf(stderr, "*** TOP-C:  ");
  vfprintf(stderr, format, ap);
  fprintf(stderr, "\n");
  fflush(stderr);
  va_end(ap);
  exit(1);
}

//This function should perhaps take steps to perform proper shutdown
//Could allow arguments, as in ERROR()
static void PERROR(char *str) {
  fprintf(stderr, "*** TOP-C(%s:%d):  ", __FILE__, __LINE__);
  perror(str);
  exit(1);
}

/***********************************************************************
 * For debugging
 *     syscall(...) =>
 * CHECK(
 *     syscall(...)
 * )
 */
#define CHECK(x) {int val=x if (val == -1) {perror("TOP-C"); exit(1);}}
