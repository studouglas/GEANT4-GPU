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

/**********************************************************************
 * See ./configure --with-mpi-cc=<PROG>, etc.  (./configure --help)
 * for using another MPI with a program, PROG, to compile MPI files.
 * See the manual for further details of using another MPI
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h> // malloc(), rand()
#include <unistd.h> // alarm()
#include <signal.h> // signal(), pthread_kill()
#include <string.h> // strerror()
#include <assert.h>
#ifdef HAVE_MPI_H
# include <mpi.h>
#else
# include <mpinu.h>  // MPINU implementation of MPI
#endif
#include "comm.h"    // comm.h also defines HAVE_PTHREAD, etc., by default

#ifdef HAVE_PTHREAD
# include <pthread.h>
# ifdef HAVE_SEMAPHORE
#  include <semaphore.h> // Solaris:  needs -lposix4
# else
#  include "sem-pthread.h"
# endif
static pthread_t cache_thread;
#endif

//Compile-time parameters (set in comm.h)
// #define NUM_MSG_CACHE_PTR 1000  // Number of pending msg's that can be cached

// NOTE:  MPI returns MPI_SUCCESS or application-dependent indication
//        of failure.  We may MPI_SUCCESS to TRUE and others to FALSE.
#ifndef FALSE
enum {FALSE, TRUE};
#endif

char *COMM_mem_model = "distributed (mpi)";


/***********************************************************************
 ***********************************************************************
 ** Communications functions, implemented over MPI
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Initialize underlying communications library
 */
static void alarm_handler(int dummy_sig) {
  ERROR("%d seconds elapsed without reply by master:  slave exiting\n"
        "        Use `./a.out --TOPC-slave-timeout=XXX' to change interval\n",
        TOPC_OPT_slave_timeout);
}
static int use_alarm = 1;
void COMM_init(int *argc, char ***argv) {
  int flag;
  if (MPI_SUCCESS != MPI_Initialized(&flag))
    ERROR("COMM_init:  MPI_Initialized() failed");
  if (!flag) {
    if (TOPC_OPT_safety >= SAFETY_NO_TIMEOUT || TOPC_OPT_slave_timeout == 0)
      use_alarm = 0;
// SLAVES_DIE should be set to max number of seconds before they die.
#ifdef SLAVES_DIE
    // Kill slaves with odd-numbered rank after a while; for testing
    use_alarm = 0;
#endif
    if (use_alarm) {
      signal( SIGALRM, alarm_handler );
      alarm(TOPC_OPT_slave_timeout);
    }
    if (MPI_SUCCESS != MPI_Init(argc, argv))
      ERROR("TOPC_init:  Couldn't initialize.  (missing procgroup file?)"
            "\n\t    Try:  ./a.out --TOPC-procgroup=PROCGROUP_FILE");
    if (use_alarm) { /* Again, in case MPI_Init() called alarm() */
      signal( SIGALRM, alarm_handler );
      alarm(TOPC_OPT_slave_timeout);
    }
    COMM_rank();  // pre-compute rank as static local var.
    if (use_alarm && COMM_rank() == 0) { // no alarm on master
      alarm(0);
      use_alarm = 0;
    }
#ifdef HAVE_PTHREAD
    if (TOPC_OPT_safety < SAFETY_NO_RCV_THREAD)
      cache_thread = pthread_self();
#endif

#ifdef SLAVES_DIE
    if ( COMM_rank() % 2 == 1 )
      alarm( (int)(rand()/(float)RAND_MAX * SLAVES_DIE) );
#endif
  }
}

/***********************************************************************
 * Shutdown underlying communications library
 */
void COMM_finalize() {
  int flag;
  MPI_Initialized(&flag);
  if (flag) {
    int rank = COMM_rank();
#ifdef HAVE_PTHREAD
    if (TOPC_OPT_safety < SAFETY_NO_RCV_THREAD)
      // cache_thread is still self if COMM_receive() never called
      if (rank != 0 && ! pthread_equal(pthread_self(), cache_thread) ) {
        // MPI_Finalize() will close sockets while cache_thread could
        //   try to read from them, kill cache_thread now
        pthread_kill( cache_thread, SIGKILL );
        pthread_join( cache_thread, NULL );
      }
#endif
    if (MPI_SUCCESS != MPI_Finalize())
      WARNING("COMM_finalize:  MPI_Finalize() failed");
    if (rank != 0) exit(0); /* If slave, exit */
  }
}

/***********************************************************************
 * Test if underlying communications library is running
 */
TOPC_BOOL COMM_initialized() {
  int flag;
  MPI_Initialized(&flag);
  // flag is true (1) if initialized.  This agrees with TOPC_BOOL of TRUE
  return flag;
}

/***********************************************************************
 * Get number of nodes
 */
int COMM_node_count() {
  int count;
  MPI_Comm_size(MPI_COMM_WORLD, &count);
  return count;
}

/***********************************************************************
 * Get rank of this node
 */
int COMM_rank() {
  static int rank = -1; // -1 means don't know; static so we remember rank
  if (rank == -1 && COMM_initialized())
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Else in mpinu, absence of "-p4amslave" in argv[] indicates it's master
  // In our case, we can't be sure
  if ( rank == -1 )
    ERROR("TOPC_rank()/TOPC_is_master() called"
       " before TOPC_init() in dist. memory model\n"
       "  Can't determine if this is master or slave in this version of MPI.\n"
       "Try placing TOPC_init() earlier"
       " ( before call to TOPC_rank()/TOPC_is_master() )\n");
  return rank;
}

/***********************************************************************
 * Return boolean, true if master and slave share memory
 */
TOPC_BOOL COMM_is_shared_memory() {
  return FALSE;
}

/***********************************************************************
 * Return boolean, true if master and slave share memory
 */
void *COMM_stack_bottom = NULL; // set in topc.c

TOPC_BOOL COMM_is_on_stack(void *buf, size_t dummy_size) {
  int stack_top[8];  /* This won't be put in register */
  /* alt:  stack_top = alloca(8); autoconf:AC_FUNC_ALLOCA */

  if ( (buf >= (void *)stack_top && buf <= COMM_stack_bottom)
       || (buf <= (void *)stack_top && buf >= COMM_stack_bottom) )
    return TRUE;
  else
    return FALSE;
}

/***********************************************************************
 * Set via SOFT_ABORT_TAG
 */
// read on slave by TOPC_is_abort_pending()
volatile int COMM_is_abort_pending = 0;
// called on master by TOPC_abort()
void COMM_soft_abort() {
  int slave, num_slaves = COMM_node_count()-1;
#ifdef HAVE_PTHREAD
  int have_pthread = 1;
#else
  int have_pthread = 0;
#endif
  if ( ! have_pthread )
    WARNING("TOP-C compiled without threads; abort message will be delayed");
  else if ( TOPC_OPT_safety >= SAFETY_NO_RCV_THREAD )
    WARNING("--TOPC-safety >= %d (no receive threads); abort msg delayed",
	    SAFETY_NO_RCV_THREAD);
  for (slave = 1; slave <= num_slaves; ++slave)
    COMM_send_msg(NULL, 0, slave, SOFT_ABORT_TAG);
  }

/***********************************************************************
 * Check if a message from a slave is instantly available (non-blocking)
 */
TOPC_BOOL COMM_probe_msg() {
  MPI_Status status;
  int flag;

  if (MPI_SUCCESS !=
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status))
    ERROR("COMM_probe_msg");
  flag = ( flag == MPI_SUCCESS ? TRUE : FALSE );
  return flag;
}

/***********************************************************************
 * Check for a message from a slave (blocking)
 */
TOPC_BOOL COMM_probe_msg_blocking(int rank) {
  MPI_Status status;

  // Allow half hour (default) for master to reply and for our task, or we die.
  if (use_alarm) alarm(TOPC_OPT_slave_timeout);
  if (MPI_SUCCESS !=
      MPI_Probe(rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status))
    ERROR("COMM_probe_msg_blocking");
  return TRUE;
}

/***********************************************************************
 * Send message
 */
TOPC_BOOL COMM_send_msg(void *msg, size_t msg_size, int dst, enum TAG tag) {
  if(MPI_SUCCESS != MPI_Send(msg, (int)msg_size, MPI_BYTE, dst, tag, MPI_COMM_WORLD))
     return FALSE;
     else return TRUE;
}

/***********************************************************************
 * Receive message into malloc'ed memory (blocking call)
 * receive from anybody, and set each of four param's if non-NULL
 */
//The fields of struct msg_ptr correspond to the args of COMM_receive_msg()
struct msg_ptr {
  void *msg;
  size_t msg_size;
  int src;
  enum TAG tag;
  int inuse;
} msg_cache_ptr[NUM_MSG_CACHE_PTR];
static volatile struct msg_ptr *cache_ptr_head = msg_cache_ptr,
                               *cache_ptr_tail = msg_cache_ptr;
#ifdef HAVE_PTHREAD
static sem_t cache_semaphore;
static int sem_val = 0; // used by DEBUG
#endif

static TOPC_BOOL cache_msg() {
  struct msg_ptr msg_info;
  MPI_Status status;
  int size;

  TRYAGAIN: ;
  do {
    if (MPI_SUCCESS !=
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status)){
        if(COMM_rank() != 0)
           exit(1);
        else return FALSE;
    }
    MPI_Get_count(&status, MPI_BYTE, &size);
    //if size_t 64 bits, cast 32 bit int back to size_t
    msg_info.msg_size = size;

    if (status.MPI_TAG == SOFT_ABORT_TAG) {
      COMM_is_abort_pending = 1;
      assert(size == 0);
      if (MPI_SUCCESS !=
          MPI_Recv( NULL, 0, MPI_BYTE,
                    status.MPI_SOURCE, SOFT_ABORT_TAG, MPI_COMM_WORLD, &status))
        ERROR("cache_msg (receiving SOFT_ABORT_TAG)");
    }
  } while (status.MPI_TAG == SOFT_ABORT_TAG);

  // Add hook here if want options for caller to pass in buffer
  msg_info.msg= MEM_malloc(size, status.MPI_SOURCE, status.MPI_TAG, IS_PRE_MSG);
  MEM_register_buf(msg_info.msg);
  if (MPI_SUCCESS !=
      MPI_Recv( msg_info.msg, size, MPI_BYTE,
      status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status )){
    goto TRYAGAIN;
  }
  msg_info.src            = status.MPI_SOURCE;
  msg_info.tag            = (enum TAG)status.MPI_TAG;
  assert( (msg_info.inuse = 1) > 0 );

#ifndef HAVE_PTHREAD
  assert(cache_ptr_tail == cache_ptr_head);
#endif
  *(cache_ptr_tail++) = msg_info;
  if (cache_ptr_tail == msg_cache_ptr + NUM_MSG_CACHE_PTR)
    cache_ptr_tail = msg_cache_ptr;
#ifdef HAVE_PTHREAD
  if (TOPC_OPT_safety < SAFETY_NO_RCV_THREAD)
    if (COMM_rank() != 0) {
#     ifdef DEBUG
      printf("rank: %d, thread: %d posting (tag: %d); in queue: %d\n",
        COMM_rank(), pthread_self(), msg_info.tag, ++sem_val);
#     endif
      if (cache_ptr_tail == cache_ptr_head)
        ERROR("TOP-C:  Overflow:  More than %d pending messages cached.\n"
              "        Re-compile src/comm-mpi.c with larger value"
                       " for NUM_MSG_CACHE_PTR in src/comm.h\n",
              NUM_MSG_CACHE_PTR);
      sem_post(&cache_semaphore);
    }
#endif
  return TRUE;
}

#ifdef HAVE_PTHREAD
static void *cache_messages(void *dummy) {
  while (1)
    cache_msg();
  /* NOTREACHED */
  return NULL;
}
static void cache_messages_init() {
  static pthread_attr_t slave_attr;
  int val;

  val = sem_init( &cache_semaphore, 0, 0 );
  if (val == -1) PERROR("sem_init");
  pthread_attr_init(&slave_attr);
  pthread_attr_setdetachstate(&slave_attr, PTHREAD_CREATE_JOINABLE);
#ifdef DEBUG
  printf("rank: %d, thread created\n", COMM_rank());
#endif
  val = pthread_create( &cache_thread, &slave_attr, cache_messages, NULL );
  if (val != 0) ERROR("%s\n", strerror(val) );
}
#endif

TOPC_BOOL COMM_receive_msg(void **msg, size_t *msg_size, int *src, enum TAG *tag) {
  struct msg_ptr msg_info;
  static int cache_messages_inited = 0;

  // On slave, allow half hour (default) for master to reply, or we die.
  if (use_alarm) alarm(TOPC_OPT_slave_timeout);
#ifdef HAVE_PTHREAD
  if (TOPC_OPT_safety < SAFETY_NO_RCV_THREAD && COMM_rank() != 0) {
    /* The actual receive is done asynchronously by a separate thread */
    if (! cache_messages_inited) {
      cache_messages_init();
      cache_messages_inited = 1;
    }
#   ifdef DEBUG
    printf("rank: %d, thread: %d waiting\n", COMM_rank(), pthread_self());
#   endif
    sem_wait(&cache_semaphore);
#   ifdef DEBUG
    printf("rank: %d, thread: %d done waiting\n", COMM_rank(), pthread_self());
#   endif
  }
  else
    // Get and cache a message ourselves, if no thread
    if(cache_msg() == FALSE) return FALSE ;
#else
  /* The actual receive is done by cache_msg() */
  // Get and cache a message ourselves, if no thread
  if(cache_msg() == FALSE) return FALSE ;
#endif

  msg_info = *(cache_ptr_head++);
  if (cache_ptr_head == msg_cache_ptr + NUM_MSG_CACHE_PTR)
    cache_ptr_head = msg_cache_ptr;

  if (msg) *msg            = msg_info.msg;
  if (msg_size) *msg_size  = msg_info.msg_size;
  if (src) *src            = msg_info.src;
  if (tag) *tag            = msg_info.tag;
#ifdef HAVE_PTHREAD
# ifdef DEBUG
  printf("rank: %d, thread: %d   received tag: %d; in queue: %d\n",
    COMM_rank(), pthread_self(), msg_info.tag, --sem_val);
# endif
#endif
  assert( --(msg_info.inuse) == 0 );
  return TRUE;
}

// balances MEM_malloc(), etc. in receive thread for COMM_receive_msg()
void COMM_free_receive_buf(void *msg) {
  MEM_free(msg);
}

/***********************************************************************
 ***********************************************************************
 * SMP-related functions (e.g.:  private (non-shared) variables)
 ***********************************************************************
 ***********************************************************************/

void *COMM_thread_private(size_t size) { // trivial for non-shared memory
  static void *thread_private = NULL;
  if ( thread_private == NULL ) {
    thread_private = malloc(size);
    if (size >= sizeof(void *))
      *(void **)thread_private = NULL;
  }
  return thread_private;
}
void COMM_begin_atomic_read() {}
void COMM_end_atomic_read() {}
void COMM_begin_atomic_write() {}
void COMM_end_atomic_write() {}
TOPC_BOOL COMM_is_in_atomic_section() {
  static int is_in = 0;
  return (is_in = 1 - is_in);
}
