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

/***********************************************************************
 * This is included in topc.c and the application.  It defines the
 * interface between the two.
 */

#ifndef _TOPC_H
#define _TOPC_H

/* Needed to define size_t and stderr */
#include <stdio.h>

#ifdef  __cplusplus
extern "C" {
#endif

/* major and minor version numbers for TOP-C 2.5.2 */
#define __TOPC_STANDARD__ 2
#define __TOPC_MAJOR__ 5
#define __TOPC_MINOR__ 2

/***********************************************************************
 * TOP-C data types (shared with comm.h)
 */
/* These are copied from comm.h */
#ifndef TOPC_TYPES
#define TOPC_TYPES
typedef int TOPC_BOOL;
typedef struct {
  void  *data;
  size_t data_size;
  /* size_t dummy1; [ For SGI Irix 6.5 and gcc 2.7.2 ] */
  /* size_t dummy2; [ For SGI Irix 6.5 and gcc 2.7.2 ] */
} TOPC_BUF;
/* This define is warning for obsolete TOP-C applications. */
#define TOPC_MSG_BUF Please_replace_TOPC_MSG_BUF_by_TOPC_MSG
typedef void TOPC_FUNCTION();
#endif

/* Utilities:  (not required for applications) */
TOPC_BOOL TOPC_is_master(void);
int TOPC_rank(void);
int TOPC_node_count(void); /* number of nodes, _including_ master */
int TOPC_num_slaves(void);
int TOPC_num_idle_slaves(void);

/***********************************************************************
 ***********************************************************************
 * TOPC_master_slave() interface
 ***********************************************************************
 ***********************************************************************/


/***********************************************************************
 * TOP-C task input type for return from set_task_output().  The syntax
 * is return TOPC_MSG(input, input_size);.  TOPC_MSG used to be a macro
 * that stored the length argument globally and returned the pointer.
 * In this version it's a function that packs them both into a struct
 */
extern TOPC_BUF NOTASK;

extern TOPC_BUF TOPC_MSG(void *input, size_t input_size);
extern TOPC_BUF TOPC_MSG_PTR(void *input, size_t input_size);


/***********************************************************************
 * TOP-C action type for return from check_task_result(), which should
 * actually return either NO_ACTION, REDO, UPDATE, or
 * CONTINUATION(TOPC_MSG(input, input_size)).  This simulates the syntax
 * required by the macros used in a previous version of TOP-C
 */
typedef struct {
  int type;
  char *name;
  void *continuation;
  size_t continuation_size;
} TOPC_ACTION;

extern TOPC_ACTION NO_ACTION, REDO, UPDATE;

extern TOPC_ACTION CONTINUATION(TOPC_BUF msg);

/***********************************************************************
 ***********************************************************************
 * TOPC options and utilities
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * TOP-C command line options for application (see options.c for defs)
 */

/* These are copied from comm.h */
extern int TOPC_OPT_num_slaves;
extern int TOPC_OPT_slave_timeout;
extern int TOPC_OPT_aggregated_tasks;
/* 0: no trace; 1: trace; 2: user functions, (*TOPC_OPT_trace_input)(), etc. */
extern int TOPC_OPT_trace;
extern int TOPC_OPT_help;
extern int TOPC_OPT_verbose;
extern char * TOPC_OPT_procgroup;
extern char * TOPC_OPT_topc_log;
extern int TOPC_OPT_safety;

#if defined(__cplusplus)
extern void (*TOPC_OPT_trace_input)(void *input);
extern void (*TOPC_OPT_trace_result)(void *input, void *output);
typedef void (*TOPC_trace_input_ptr)(void *);
typedef void (*TOPC_trace_result_ptr)(void *, void *);
#else
extern void (*TOPC_OPT_trace_input)();  /* args: (void *input) */
extern void (*TOPC_OPT_trace_result)(); /* args: (void *input, void *output) */
#endif

/***********************************************************************
 * TOP-C main functions (interface between topc.c and application)
 */

/* Declare beginning and end of TOP-C program. */
void TOPC_init(int *argc, char ***argv);
void TOPC_finalize(void);

/* Main interface to TOP-C */
#if defined(__cplusplus)
void TOPC_master_slave(
  TOPC_BUF (*generate_task_input_)(void),
  TOPC_BUF (*do_task_)(void *input),
  TOPC_ACTION (*check_task_result_)(void *input, void *output),
  void (*update_shared_data_)(void *input, void *output)
);

/* Lower level interface to TOP-C, for legacy and highly nested applications */
void TOPC_raw_begin_master_slave(
  TOPC_BUF (*do_task_)(void *input),
  TOPC_ACTION (*check_task_result_)(void *input, void *output),
  void (*update_shared_data_)(void *input, void *output)
);

// Thanks to Jason Ansel for suggesting this template trick:
typedef TOPC_BUF (*TOPC_genTaskInputPtr)();
typedef TOPC_BUF (*TOPC_doTaskPtr)(void *);
typedef TOPC_ACTION (*TOPC_checkTaskResultPtr)(void *, void *);
typedef void (*TOPC_updateSharedDataPtr)(void *, void *);

} // end of extern "C"

template <typename Tinput, typename Toutput>
  void TOPC_master_slave_cpp( TOPC_BUF (*genTaskInput)(),
		  	     TOPC_BUF (*doTask)(Tinput*),
		  	     TOPC_ACTION (*checkTaskResult)(Tinput*, Toutput*),
		  	     void (*updateSharedData)(Tinput*, Toutput*) ) {
    TOPC_master_slave( (TOPC_genTaskInputPtr)genTaskInput,
		       (TOPC_doTaskPtr)doTask,
		       (TOPC_checkTaskResultPtr)checkTaskResult,
		       (TOPC_updateSharedDataPtr)updateSharedData );
  }
template <typename Tinput, typename Toutput>
  void TOPC_master_slave_cpp( TOPC_BUF (*genTaskInput)(),
		  	     TOPC_BUF (*doTask)(Tinput*),
		  	     TOPC_ACTION (*checkTaskResult)(Tinput*, Toutput*),
		  	     void * updateSharedData ) {
    if ( updateSharedData != NULL ) {
      fprintf(stderr, "TOP-C:  Bad call to TOPC_master_slave.\n"
		      "    Fourth function (updateSharedData) was not NULL,"
		      "    and had args incompatible with checkTaskResult.\n");
      exit(1);
    }
    TOPC_master_slave( (TOPC_genTaskInputPtr)genTaskInput,
		       (TOPC_doTaskPtr)doTask,
		       (TOPC_checkTaskResultPtr)checkTaskResult,
		       NULL );
  }

template <typename Tinput, typename Toutput>
  void TOPC_raw_begin_master_slave_cpp( TOPC_BUF (*doTask)(Tinput*),
		  	     TOPC_ACTION (*checkTaskResult)(Tinput*, Toutput*),
		  	     void (*updateSharedData)(Tinput*, Toutput*) ) {
    TOPC_raw_begin_master_slave( (TOPC_doTaskPtr)doTask,
		       		 (TOPC_checkTaskResultPtr)checkTaskResult,
		       		 (TOPC_updateSharedDataPtr)updateSharedData );
}
template <typename Tinput, typename Toutput>
  void TOPC_raw_begin_master_slave_cpp( TOPC_BUF (*doTask)(Tinput*),
		  	     TOPC_ACTION (*checkTaskResult)(Tinput*, Toutput*),
		  	     void * updateSharedData ) {
    if ( updateSharedData != NULL ) {
      fprintf(stderr, "TOP-C:  Bad call to TOPC_raw_begin_master_slave.\n"
		      "    Third function (updateSharedData) was not NULL,"
		      "    and had args incompatible with checkTaskResult.\n");
      exit(1);
    }
    TOPC_raw_begin_master_slave( (TOPC_doTaskPtr)doTask,
		       		 (TOPC_checkTaskResultPtr)checkTaskResult,
		       		 NULL );
}

extern "C" {
#define TOPC_master_slave TOPC_master_slave_cpp
#define TOPC_raw_begin_master_slave TOPC_raw_begin_master_slave_cpp

#else
/* Else ANSI C and not C++ */
/* How does one declare callbacks as taking one or two pointer args in ANSI C?*/
void TOPC_master_slave(
  TOPC_BUF (*generate_task_input_)(void),
  TOPC_BUF (*do_task_)(),              /* args: (void *input) */
  TOPC_ACTION (*check_task_result_)(), /* args: (void *input, void *output) */
  void (*update_shared_data_)()        /* args: (void *input, void *output) */
);

/* Lower level interface to TOP-C, for legacy and highly nested applications */
void TOPC_raw_begin_master_slave(
  TOPC_BUF (*do_task_)(),              /* args: (void *input) */
  TOPC_ACTION (*check_task_result_)(), /* args: (void *input, void *output) */
  void (*update_shared_data_)()        /* args: (void *input, void *output) */
);
#endif
void TOPC_raw_end_master_slave(void);
void TOPC_raw_submit_task_input(TOPC_BUF input);
TOPC_BOOL TOPC_raw_wait_for_task_result(void);

/***********************************************************************
 * TOP-C Utility functions (interface between topc.c and application)
 */
TOPC_BOOL TOPC_is_up_to_date(void);
void TOPC_abort_tasks();
TOPC_BOOL TOPC_is_abort_pending(void);
TOPC_BOOL TOPC_is_REDO(void);
TOPC_BOOL TOPC_is_CONTINUATION(void);

/***********************************************************************
 ***********************************************************************
 * SMP-related functions (e.g.:  private (non-shared) variables)
 ***********************************************************************
 ***********************************************************************/


/***********************************************************************
 * TOPC_thread_private
 *
 * Usage:
 * typedef struct {int val1; float val2;} TOPC_thread_private_t;
 * void foo() {
 *   TOPC_thread_private.val1 = 42;
 *   TOPC_thread_private.val2 = 3.14;
 * }
 * void bar() {
 *   foo();
 *   if (TOPC_thread_private.val1 != 42) printf("ERROR");
 *   if (TOPC_thread_private.val2 != 3.14) printf("ERROR");
 * }
 */

/* Move expl. below to doc/ after final testing.     - Gene */
/*
doc:
    A TOP-C variable has either global scope or local scope.
    It is also shared or private.
    A variable is global or local according to the standard C specification.
    In the distributed memory (e.g.: --mpi), a variable is always private.
      Even a variable whose value is in shared data is private, but
      calls to UpdateSharedData() will update the private value with
      the latest value.  Thus, we say that such variables have lazy updates.
    In the sequential memory (e.g.: --seq), a variable is always shared.
      Thus UpdateSharedMemory executes only once, in the sequential process.
    In shared memory (e.g.: --pthread), a variable is shared by default.
      Further UpdateSharedMemory executes only once, in the master thread.
      However, certain strategies for high concurrency require the use
      of private variables to privately save temporary values and use them
      again during a REDO action.
      To accomodate this, one can declare a single private variable.
      The private variable has global scope, but each thread has a separate
      copy that can be read or written only by that thread.
      One is allowd only a single private variable, but its type can
      be declared arbitrarily.  Hence, in the example below, we declare
      TOPC_thread_private to be a struct, and then use
      TOPC_thread_private.val1 and TOPC_thread_private.val2 separately.
*/

/* comm layer: will malloc a TOPC_thread_private_t first time, then re-uses it*/
void *COMM_thread_private(size_t);
/* If TOPC_thread_private used, app must typedef ... TOPC_thread_private_t; */
#define TOPC_thread_private \
  ( *(TOPC_thread_private_t *) \
      COMM_thread_private( sizeof(TOPC_thread_private_t) ) )

/* EXPERIMENTAL:
 *   Ideally, we would want to define THREAD_PRIVATE(mytype, myvar);
 * to act as if:  THREAD_PRIVATE mytype myvar;
 * and to expand to:
 * #define myvar TOPC_thread_private_var
 * typedef mytype TOPC_thread_private_var_t
 * TOPC_thread_private_var_t myvar_debug{ return myvar; } // for GDB
 * as if:  TOPC_thread_private_var_t TOPC_thread_private_var; were defined.
 * OPTIONAL:  mytype TOPC_thread_private_var_init_value = XXX;
 */

/***********************************************************************
 *  Synchronization (atomic I/O) and pages of shared data
 *
 * TOPC_ATOMIC_READ(x) and TOPC_ATOMIC_WRITE(x)
 * In non-shared memory, this has no effect.
 * In shared memory, it provides a read lock or a write lock around code.
 * Usage:
 *   TOPC_ATOMIC_WRITE(x) { hash[i] = 17; }
 * Currently, x is not used, but eventually it will be a "page number".
 * TOPC_is_up_to_date() will check if each page is up_to_date.
 * This eliminates the problem of false sharing.
 * Maybe, page 0 will have special meaning:  reading page 0 means reading
 * all pages and writing page 0 means writing to all pages.
 */

/* This is copied from comm.h */
TOPC_BOOL COMM_is_in_atomic_section(void);

extern void TOPC_BEGIN_ATOMIC_READ(int pagenum);
extern void TOPC_END_ATOMIC_READ(int pagenum);
extern void TOPC_BEGIN_ATOMIC_WRITE(int pagenum);
extern void TOPC_END_ATOMIC_WRITE(int pagenum);

#define TOPC_ATOMIC_READ(pagenum) \
  for ( TOPC_BEGIN_ATOMIC_READ(pagenum); \
        COMM_is_in_atomic_section(); \
        TOPC_END_ATOMIC_READ(pagenum) ) /* application body appears here */

#define TOPC_ATOMIC_WRITE(pagenum) \
  for ( TOPC_BEGIN_ATOMIC_WRITE(pagenum); \
        COMM_is_in_atomic_section(); \
        TOPC_END_ATOMIC_WRITE(pagenum) ) /* application body appears here */

TOPC_BOOL TOPC_is_up_to_date(void);


#ifdef  __cplusplus
}
#endif

/* end TOPC_H */
#endif
