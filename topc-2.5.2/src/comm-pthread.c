 /**********************************************************************
  * TOP-C (Task Oriented Parallel C)                                   *
  * Copyright (c) 2000, 2004 Gene Cooperman <gene@ccs.neu.edu>         *
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
#include <unistd.h>
#include <stdlib.h> // malloc()/free(), abs()
#include <string.h> // strncmp()
#include "comm.h"    // comm.h also defines HAVE_PTHREAD, etc., by default

#ifdef HAVE_PTHREAD_SETCONCURRENCY
// Define __USE_UNIX98 this for sake of pthread_setconcurrency()                
// Consider no longer using old pthread_setconcurrency()                        
# define __USE_UNIX98
#endif
#include <pthread.h>
#ifdef HAVE_SEMAPHORE
# include <semaphore.h> // Solaris:  needs -lposix4
#else
# include "sem-pthread.h"
#endif

//Compile-time parameters (see comm.h and topc.c for more params)
#define PTHREAD_STACKSIZE (1 << 22) /* Request 4 MB stack */

// volatile:
// NOTE:  In waiting on condition var (as below), test_var must be volatile
//        or compiler may keep previous value of test_var in register.
// while(test_var) pthread_cond_wait(&cond_var, &mutex);

//Error checking, assuming return value of x == 0 means no error:
#define ERROR_CHECK(x) { if (x) PERROR(#x); }

//Alternative implementation to explore:
//  Use semaphore instead of msg_cond for master waiting for slave message.
//  Semaphore represents number of messages waiting via message[].next_msg

char *COMM_mem_model = "shared (pthread)";

/***********************************************************************
 ***********************************************************************
 ** Communications and MPI functions for pthread SMP layer
 ***********************************************************************
 ***********************************************************************/

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199506L
#endif

static sem_t master_semaphore; // master blocks until slave thread ready.
static pthread_key_t thread_private_key;

static size_t stacksize = 0;

/***********************************************************************
 ***********************************************************************
 * Synchronization for queue of messages from slaves to master,
 * for synchronization of slave waiting for msg from master
 *   (implemented by message[slave].sem),
 * and for single writer/multiple readers to synchronize
 *   do_task() (readers) and update_shared_data() (writers)
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 ***********************************************************************
 * STILL NEED TO THINK OUT free_receive_buffer(), lock(), semaphore()
 * Also, has slave released all locks, semaphores before exiting?
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Optional tracing of all synchronization (define DEBUG_SYNC to enable)
 */
/* #define DEBUG_SYNC */
#ifdef DEBUG_SYNC
#define DEBUG_SYNC_MAX 1000000
#define BAR 'a' /* begin atomic read */
#define EAR 'b' /* end atomic read */
#define BAW 'c' /* begin atomic write */
#define EAW 'd' /* end atomic write */
#define SSM 'e' /* slave sending message */
#define MRM 'f' /* master receiving message */
#define MSM 'g' /* master sending message */
#define SRM 'h' /* slave receiving message */
#define DEBUG_SYNC_TRACE(type) \
  if (sync_trace_idx<DEBUG_SYNC_MAX) sync_trace[sync_trace_idx++]=type
static int sync_trace_idx = 0;
static char sync_trace[DEBUG_SYNC_MAX];
void print_sync_trace() {
  int i;
  printf("\nDEBUG TRACE:\n  src/comm-pthread.c:  #define DEBUG_SYNC 1\n");
  for (i=0; i<sync_trace_idx;) {
    switch(sync_trace[i++]) {
    case BAR:
      printf("begin read (%c readers)\n", sync_trace[i++]); break;
    case EAR:
      printf("end read (%c readers)\n", sync_trace[i++]); break;
    case BAW:
      printf("begin write\n"); break;
    case EAW:
      printf("end write\n"); break;
    // Trace of next two may not be properly sync'ed with above
    case SSM:
      printf("slave %c sending msg\n", sync_trace[i++]); break;
    case MRM:
      printf("master receiving msg from slave %c\n", sync_trace[i++]); break;
    case MSM:
      printf("master sending msg to %c\n", sync_trace[i++]); break;
    case SRM:
      printf("slave %c receiving msg from master\n", sync_trace[i++]); break;
    }
  }
}
#endif

/***********************************************************************
 * Synchronization of messages between master and slaves
 */
static int is_initialized = 0;
static void free_key_value(void *value) { if (value) free(value); }
static int node_count;
static int last_rank;
static pthread_attr_t slave_attr;
static pthread_key_t stack_bottom_key;
static pthread_key_t rank_key;
static pthread_key_t is_atomic_section_key;
struct MSG {
  void *buf;
  enum TAG tag;
  volatile int in_use; // true if msg sent (by master or slave), not yet rec'd
  int next_msg; // in linked list of all msg's from slaves to master
  pthread_mutex_t mutex; // not currently used
  sem_t sem; // master posts, slave waits at message and at initialization
  sem_t sem_inuse; // master waits, slave posts, initial value = 1
};
static struct MSG message[TOPC_MAX_SLAVES];

// for FIFO queue of messages from slaves to master
#define NO_MSG -1
static volatile int rank_of_first_msg = NO_MSG;
static int rank_of_last_msg = NO_MSG;
static int wake_for_msg = 0;
static pthread_mutex_t msg_queue = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t msg_cond = PTHREAD_COND_INITIALIZER;
#define is_pending_msg() (rank_of_first_msg != NO_MSG)

// called by slave
static void add_next_msg(int myrank) {
  int do_wakeup;
  ERROR_CHECK(pthread_mutex_lock(&msg_queue));
  message[myrank].next_msg = NO_MSG;
  if (!is_pending_msg()) rank_of_first_msg = myrank;
  if (rank_of_last_msg != NO_MSG) // if prev thread on queue, tell it about me
    message[rank_of_last_msg].next_msg = myrank;
  rank_of_last_msg = myrank;
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(SSM);
  DEBUG_SYNC_TRACE('0'+myrank);
#endif
  do_wakeup = wake_for_msg;
  ERROR_CHECK(pthread_mutex_unlock(&msg_queue));
  if (do_wakeup) pthread_cond_signal(&msg_cond); // is_pending_msg() => 1, now
}
// called by master
static int pop_first_msg() {
  int first_rank;
  ERROR_CHECK(pthread_mutex_lock(&msg_queue));
  wake_for_msg = 1;
  while (!is_pending_msg()) pthread_cond_wait(&msg_cond, &msg_queue);
  wake_for_msg = 0;
  first_rank = rank_of_first_msg;
  // next_msg might be NO_MSG
  rank_of_first_msg = message[rank_of_first_msg].next_msg;
  // CAN THIS EVER HAPPEN?  ABOVE WE WAITED UNTIL is_pending_msg()
  if (!is_pending_msg()) rank_of_last_msg = NO_MSG;
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(MRM);
  DEBUG_SYNC_TRACE('0'+first_rank);
#endif
  ERROR_CHECK(pthread_mutex_unlock(&msg_queue));
  return first_rank;
}

/***********************************************************************
 * Synchronization among do_task() (readers) and update_shared_data() (writer)
 * Implements reader-writer synchronization with writer priority.
 */
// One writer:        COMM_update_shared_data()
// Multiple readers:  COMM_do_task()
// Code also works for multiple writers and readers if user wants to use it so.
static pthread_mutex_t global_lock1 = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t global_lock2 = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   global_cond = PTHREAD_COND_INITIALIZER;
static volatile int readers = 0;
static int wake_writer = 0;

static TOPC_BUF (*orig_do_task)(void *input) = NULL;
static void (*orig_update_shared_data)(void *input, void *output) = NULL;

void COMM_begin_atomic_read() {
  (*(int *)pthread_getspecific(is_atomic_section_key))++;
  ERROR_CHECK(pthread_mutex_lock( &global_lock1 ));
  ERROR_CHECK(pthread_mutex_lock( &global_lock2 ));
  readers++;
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(BAR);
  DEBUG_SYNC_TRACE('0'+readers);
#endif
  ERROR_CHECK(pthread_mutex_unlock( &global_lock2 ));
  ERROR_CHECK(pthread_mutex_unlock( &global_lock1 ));
}
void COMM_end_atomic_read() {
  int is_last_reader, val;
  ERROR_CHECK(pthread_mutex_lock( &global_lock2 ));
  readers--;
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(EAR);
  DEBUG_SYNC_TRACE('0'+readers);
#endif
  is_last_reader = (readers == 0);
  ERROR_CHECK(pthread_mutex_unlock( &global_lock2 ));
  val = (*(int *)pthread_getspecific(is_atomic_section_key))--;
  if (wake_writer && is_last_reader) pthread_cond_signal(&global_cond);
  if ( val < 0 )
    ERROR("more TOPC_END_ATOMIC_READ() than TOPC_BEGIN_ATOMIC_READ/WRITE()");
}

void COMM_begin_atomic_write() {
  (*(int *)pthread_getspecific(is_atomic_section_key))++;
  // Block new readers from starting
  ERROR_CHECK(pthread_mutex_lock( &global_lock1 ));
  ERROR_CHECK(pthread_mutex_lock( &global_lock2 ));
  wake_writer = 1; // Tell reader to wake me when he finishes reading
  // While writer is in cond_wait(), global_lock2 is unlocked (released).
  // When reader executes pthread_cond_signal(), writer wakes up
  while (readers>0) pthread_cond_wait(&global_cond, &global_lock2);
  wake_writer = 0; // No more readers; cancel wakeup call
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(BAW);
#endif
  ERROR_CHECK(pthread_mutex_unlock( &global_lock2 ));
}
void COMM_end_atomic_write() {
  int val;
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(EAW);
#endif
  // Allow new readers to enter
  ERROR_CHECK(pthread_mutex_unlock( &global_lock1 ));
  val = (*(int *)pthread_getspecific(is_atomic_section_key))--;
  if ( val < 0 )
    ERROR("more TOPC_END_ATOMIC_WRITE() than TOPC_BEGIN_ATOMIC_READ/WRITE()");
}
#if 0
/* BUG:  If application nests TOPC_ATOMIC_READ/WRITE{} inside
 *       TOPC_ATOMIC_BEGIN/END_ATOMIC_READ/WRITE,
 *       this code gets mixed up about whether it's in atomic section.
 */
TOPC_BOOL COMM_is_in_atomic_section() {
  /* The int value is initialized to 0, and toggles between 1 and 0 */
  return *(int *)pthread_getspecific(is_atomic_section_key) =
           1 - *(int *)pthread_getspecific(is_atomic_section_key);
}
#endif
TOPC_BOOL COMM_is_in_atomic_section() {
  int val;
  val = *(int *)pthread_getspecific(is_atomic_section_key);
  if ( val < 0 ) {
    printf("TOP-C:  more TOPC_END_ATOMIC_xxx() than TOPC_BEGIN_ATOMIC_xxx()");
    exit(1);
    /* NOTREACHED */
  }
  else return ( val > 0 ? 1 : 0 );
}

/***********************************************************************
 ***********************************************************************
 * GET RID OF CODE IN THIS SECTION WHEN SURE IT'S NOT NEEDED
 ***********************************************************************
 ***********************************************************************/
// set by topc.c
extern TOPC_BUF (*COMM_do_task)(void *input);
extern void (*COMM_update_shared_data)(void *input, void *output);


static TOPC_BUF COMM_do_task_wrapper(void *input) {
  TOPC_BUF buf;
  buf = orig_do_task(input);
  return buf;
}
// Note COMM_generate_task_input() also runs on master thread,
//   and so it cannot run while COMM_update_shared_data() runs.
static void COMM_update_shared_data_wrapper(void *input, void *output) {
  orig_update_shared_data(input, output);
}
// add wrapper if is_wrap != 0, and remove wrapper if is_wrap == 0
// Not used currently
static void add_wrapper_to_callbacks(int is_wrap) {
  if (is_wrap && orig_do_task == NULL) {
    // Make TOPC_master_slave() call our new wrapper functions
    orig_do_task = COMM_do_task;
    orig_update_shared_data = COMM_update_shared_data;
    COMM_do_task = COMM_do_task_wrapper;
    COMM_update_shared_data = COMM_update_shared_data_wrapper;
  } else if (!is_wrap && orig_do_task != NULL) {
    COMM_do_task = orig_do_task;
    COMM_update_shared_data = orig_update_shared_data;
    orig_do_task = NULL;
    orig_update_shared_data = NULL;
  }
}

/***********************************************************************
 ***********************************************************************
 * MPI-style Communications library on top of synchronization constructs
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Initialize underlying communications library
 */
void COMM_init(int *dummy_argc, char ***dummy_argv) {
  static int slave_attr_initialized = 0;
  static int main_rank = 0;
  static int main_is_atomic_section = 0;
  int i;

  if ( TOPC_OPT_num_slaves != UNINITIALIZED )
    node_count = TOPC_OPT_num_slaves + 1;
  if ( TOPC_OPT_num_slaves == UNINITIALIZED ) {
#ifdef _SC_NPROCESSORS_ONLN
    node_count = sysconf( _SC_NPROCESSORS_ONLN ); // num processors online
#else
    node_count = sysconf( 15 ); // On Solaris, _SC_NPROCESSORS_ONLN == 15
#endif
    if (node_count < 3) node_count = 3; // sysconf error => -1
  }
  if ( node_count > TOPC_MAX_SLAVES )
    ERROR("Configured for at most %d threads, %d requested.\n",
          TOPC_MAX_SLAVES, node_count );
  // This is "hint" to operating system -- not binding; not all O/S have it.
#ifdef HAVE_PTHREAD_SETCONCURRENCY
  /* threads:  slaves + 1  (no thread should need to context switch) */
  pthread_setconcurrency( node_count );
#endif
  last_rank = 0;

  if ( ! slave_attr_initialized ) {
    pthread_attr_init(&slave_attr);
// bound = permanently attach thread to execution vehicle
//   (if not bound, can have fewer execution vehicles in kernel,
//    which attach to different threads at different times, based
//    on also doing user space scheduling)
// scheduling in system scope (bound threads, scheduled in kernel) often
//    needs privilegs
#ifdef HAVE_PTHREAD_SCOPE_BOUND_NP
    pthread_attr_setscope(&slave_attr, PTHREAD_SCOPE_BOUND_NP);
#else
    pthread_attr_setscope(&slave_attr, PTHREAD_SCOPE_PROCESS);
#endif
    // pthread_attr_setscope(&slave_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_attr_setdetachstate(&slave_attr, PTHREAD_CREATE_DETACHED);

    pthread_key_create( &thread_private_key, free_key_value );
    pthread_key_create( &is_atomic_section_key, NULL );
    pthread_setspecific( is_atomic_section_key, &main_is_atomic_section );
    *(int *)pthread_getspecific( is_atomic_section_key ) = 0;
    pthread_key_create( &stack_bottom_key, NULL ); // for COMM_is_on_stack()
    pthread_key_create( &rank_key, NULL );
    pthread_setspecific( rank_key, &main_rank ); // set value of main() thread
  }
  slave_attr_initialized = 1;
  i = sem_init( &master_semaphore, 0, 0 );
  if (i == -1) PERROR("sem_init");
  is_initialized = 1;
}
//called by new slave
//slave_thread_create() calls pthread_create() which calls this as init routine
static void *slave_thread_init(void *rank_ptr) {
  int rank;                  // Is on stack for life of thread
  int is_atomic_section = 0; // Is on stack for life of thread
  int tid = pthread_self();  // Used only for debugging, e.g. gdb
  rank = *(int *)rank_ptr;
  pthread_setspecific( stack_bottom_key, &rank ); // for COMM_is_on_stack()
  pthread_setspecific( rank_key, &rank );
  pthread_setspecific( is_atomic_section_key, &is_atomic_section );
  sem_post( &master_semaphore ); // tell master last_rank now copied by slave
  // Is there a POSIX way for a slave thread (not process) to sleep?
  //     if (TOPC_OPT_slave_wait>0)
  //       sleep(TOPC_OPT_slave_wait);
  // master:COMM_finalize() will set COMM_do_task = pthread_exit()
  //   and next TASK_INPUT_TAG message for slave will "poison it".
  while (1)
    (*COMM_slave_loop)();
  return NULL;
}
//called by master
static void slave_thread_create() {
  pthread_t tid;
  int val;

  last_rank++; // pass last_rank to new slave
  pthread_mutex_init(&message[last_rank].mutex, NULL);
  val = sem_init( &(message[last_rank].sem), 0, 0 );
  if (val == -1) PERROR("sem_init");
  // Initialize to 1 (resource = 1 msg buf; master can put it in use once)
  sem_init( &(message[last_rank].sem_inuse), 0, 1 );
  if (val == -1) PERROR("sem_init");
  message[last_rank].in_use = 0;
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
  pthread_attr_setstacksize(&slave_attr, PTHREAD_STACKSIZE);
#endif
  val = pthread_create( &tid, &slave_attr, slave_thread_init, &last_rank );
  if (val != 0) ERROR("%s\n", strerror(val) );
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
  { int val = pthread_attr_getstacksize(&slave_attr, &stacksize);
    if (val) ERROR("%s\n", strerror(val) );
  }
#endif
  sem_wait( &master_semaphore ); // wait until slave copies last_rank
  if (val == -1) PERROR("sem_wait");
}

/***********************************************************************
 * Shutdown underlying communications library
 */
static TOPC_BUF do_task_exit(void *dummy_input) {
  extern TOPC_BUF NOTASK;
  pthread_exit(NULL);
  /* NOTREACHED */
  return NOTASK;
}
void COMM_finalize() {
  int i;

  // Slave shouldn't be arriving here, but just in case:
  if (COMM_rank() != 0) // if we're a slave, then exit
    pthread_exit(NULL);
  COMM_do_task = do_task_exit; // next task executed by slave will poison it
  if ( last_rank > 0 ) {  // If TOPC_master_slave was called, and slaves exist
    for (i=1; i < node_count; i++)  // num_slaves == node_count - 1
      COMM_send_msg(NULL, 0, i, TASK_INPUT_TAG);
    is_initialized = 0;
    last_rank = 0;
  }
#ifdef DEBUG_SYNC
  print_sync_trace();
#endif
}

/***********************************************************************
 * Test if underlying communications library is running
 */
TOPC_BOOL COMM_initialized() {
  return is_initialized;
}

/***********************************************************************
 * Get number of nodes
 */
int COMM_node_count() {
  return node_count;
}

/***********************************************************************
 * Get rank of this node
 */
int COMM_rank() {
  if ( is_initialized )
    return *(int *)pthread_getspecific( rank_key );
  else return 0;
}

/***********************************************************************
 * Return boolean, true if master and slave share memory
 */
int COMM_is_shared_memory() {
  return 1;  // true
}

/***********************************************************************
 * Return boolean, true if master and slave share memory
 */
void *COMM_stack_bottom = NULL; // set in topc.c

TOPC_BOOL COMM_is_on_stack(void *buf, size_t size) {
  /* argument size not currently used, but available for future */
  int stack_top[8];  /* This won't be put in register */
  void *stack_bottom;

  stack_bottom = pthread_getspecific( stack_bottom_key );
  if (! stack_bottom) // then this must be master thread
    stack_bottom = COMM_stack_bottom;
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
  else
    if ( (size_t)abs((char *)stack_bottom-(char *)stack_top)
	 > stacksize - (1 << 10))
      ERROR("TOPC_MSG() buffer (%d bytes) too large for\n"
            "    slave thread stack (size %d bytes)\n"
            "  Either re-compile %s with larger STACKSIZE value\n"
            "    or allocate buffer outside of stack.\n",
            size, stacksize, __FILE__);
#endif
  if ( (buf >= (void *)stack_top && buf <= stack_bottom)
       || (buf <= (void *)stack_top && buf >= stack_bottom) )
    return 1; // true
  else
    return 0; // false
}

/***********************************************************************
 * Set via SOFT_ABORT_TAG
 */
// read on slave by TOPC_is_abort_pending()
volatile int COMM_is_abort_pending = 0;
// called on master by TOPC_abort()
void COMM_soft_abort() {
  COMM_is_abort_pending = 0;
  }

/***********************************************************************
 * Check if a message from a slave is instantly available (non-blocking)
 */
TOPC_BOOL COMM_probe_msg() {

  return is_pending_msg();

  //Blocking version that works if called on slave:
  // struct MSG *msg_ptr;
  // msg_ptr = &message[COMM_rank()];
  // sem_wait( &msg_ptr->sem );
  // sem_post( &msg_ptr->sem ); // since available message was not consumed
  // return 1;
}

/***********************************************************************
 * Check for a message from a slave (blocking)
 */
TOPC_BOOL COMM_probe_msg_blocking(int rank) {
  // master calls, waits for message from slave <rank>
  if ( ! message[rank].in_use ) {
  ERROR_CHECK(pthread_mutex_lock(&msg_queue));
  wake_for_msg = 1;
  while ( ! message[rank].in_use ) pthread_cond_wait(&msg_cond, &msg_queue);
  wake_for_msg = 0;
  ERROR_CHECK(pthread_mutex_unlock(&msg_queue));
  }
  return 1;
}

/***********************************************************************
 * Send message
 */
TOPC_BOOL COMM_send_msg(void *msg, size_t dummy_msg_size, int dst,
			enum TAG tag) {
  struct MSG *msg_ptr;
  int myrank, i;

  if ( dst != 0 ) { // if we're master (guaranteed not busy)
    if ( last_rank == 0 ) {  // Only main() thread exists so far
      for (i=1; i < node_count; i++)  // num_slaves == node_count - 1
        slave_thread_create();
    }
    msg_ptr = &message[dst];
    sem_wait( &msg_ptr->sem_inuse );  // Wait if slave didn't yet read last msg
    msg_ptr->tag = tag;
    msg_ptr->buf = msg;
    msg_ptr->in_use = 1;
#ifdef DEBUG_SYNC
  DEBUG_SYNC_TRACE(MSM);
  DEBUG_SYNC_TRACE('0'+dst);
#endif
    sem_post( &msg_ptr->sem );
  }
  else { // else we're slave
    myrank = COMM_rank();
    msg_ptr = &message[myrank];
    if ( msg_ptr->in_use )
      ERROR("Internal error:  slave found msg buf already in use");
    msg_ptr->tag = tag;
    msg_ptr->buf = msg;
    msg_ptr->in_use = 1;
    add_next_msg(myrank);
    // pthread_mutex_unlock( &(msg_ptr->mutex) ); // IS NEEDED? write barr.?
  }
  return 1;
}

/***********************************************************************
 * Receive message into previously malloc'ed memory (blocking call)
 * Receive from anybody, and set each of four param's if non-NULL
 * (msg_size not set by current code)
 */
TOPC_BOOL COMM_receive_msg(void **msg, size_t *dummy_msg_size, int *src,
			   enum TAG *tag) {
  struct MSG *msg_ptr;
  int myrank, actual_src;

  myrank = COMM_rank();

  if ( myrank == 0 ) { // if we're master
    actual_src = pop_first_msg(); // master blocks here if no msg
    msg_ptr = &message[actual_src];
    if ( ! msg_ptr->in_use )
      ERROR("Internal error:  master waiting on single slave");
    if (src) *src = actual_src;
    if (tag) *tag = msg_ptr->tag;
    if (msg) *msg = msg_ptr->buf;
    msg_ptr->in_use = 0;
  }
  else { // else we're slave
    msg_ptr = &message[myrank];
    sem_wait( &msg_ptr->sem );
    if ( ! msg_ptr->in_use )
      ERROR("Internal error:  slave found msg buf with no message");
    // pthread_mutex_lock( &(msg_ptr->mutex) ); // IS THIS NEEDED? write barr.?
    // slave call to pthread_mutex_unlock() served as write barrier
    if (src) *src = 0;
    if (tag) *tag = msg_ptr->tag;
    if (msg) *msg = msg_ptr->buf;
#ifdef DEBUG_SYNC
    DEBUG_SYNC_TRACE(SRM);
    DEBUG_SYNC_TRACE('0'+myrank);
#endif
    msg_ptr->in_use = 0;
    sem_post( &msg_ptr->sem_inuse );  // Let master use msg_ptr again
  }
  return 1;
}

// balances MEM_malloc() in COMM_receive_msg(), no malloc() for pthread
void COMM_free_receive_buf(void *dummy_msg) {
  return;
}


/***********************************************************************
 ***********************************************************************
 * SMP-related functions (e.g.:  private (non-shared) variables)
 ***********************************************************************
 ***********************************************************************/
static pthread_mutex_t key_mutex = PTHREAD_MUTEX_INITIALIZER;

void *COMM_thread_private(size_t size) {
  void *thread_private_val;
  static int thread_private_initialized = 0;
  static size_t old_size = 0;

  /* Occurs if application called us before TOPC_init() */
  if (! thread_private_initialized) {
    pthread_mutex_lock(&key_mutex); /* Try again in critical section */
    if (! thread_private_initialized) {
      pthread_key_create( &thread_private_key, free_key_value );
      old_size = size;
      thread_private_initialized = 1;
    }
    pthread_mutex_unlock(&key_mutex);
  }
  if (size != old_size)
    ERROR("TOPC_thread_private_t redeclared as a type of a different size.");
  thread_private_val = pthread_getspecific(thread_private_key);
  if (thread_private_val) return thread_private_val;
  // else it was never initialized
  thread_private_val = malloc(size);
  //In case it's a pointer, initialize to NULL pointer for user
  if (size >= sizeof(void *))
    *(void **)thread_private_val = NULL;
  pthread_setspecific(thread_private_key, thread_private_val);

  return thread_private_val;
}

