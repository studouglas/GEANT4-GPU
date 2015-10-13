 /**********************************************************************
  * TOP-C (Task Oriented Parallel C)                                   *
  * Copyright (c) 2000-2004 Gene Cooperman <gene@ccs.neu.edu>          *
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
#include <stdlib.h> // malloc(), atoi()
#include <string.h> // memcpy(), strlen()
#include <unistd.h> // getcwd(), chdir(), sleep()
#include <time.h> // time()
#include <assert.h>
#include <errno.h>
#include "topc.h"   // data structures shared with application
#ifdef  __cplusplus
# undef TOPC_master_slave
# undef TOPC_raw_begin_master_slave
#endif
#include "comm.h"   // not visible to applic.; shared with communication layer
		    // comm.h defines HAVE_PTHREAD, etc.

//Compile-time parameters
// Also:  comm.h defines TOPC_MAX_SLAVES
// #define TOPC_MAX_SLAVES            1000       /* Max. number of slaves */
#define UPDATE_QUEUE_SIZE          200        //Size of update ueue
#define NUM_REUSABLE_IDS           200000000  //Should be >> number of slaves
#define MAX_PATH_LEN               256        //for TOPC_OPT_getpwd()
#define MSG_CACHE_SIZE             5000       //Larger updates are queued
                                              //Models socket buf size of O/S
                                              // (SO_SNDBUF / SO_RCVBUF)
//Only allocated if TOPC_OPT_aggregated_tasks > 1
// 1<<20 (1 MB) chosen so transfer time larger than gigabit Ethernet latency
// For Fast Ethernet, could set this to only 100 KB
#define AGGREG_TASKS_SIZE	   1<<20

//For ease of further porting
#ifndef FALSE
enum {FALSE, TRUE};
#endif

//Global variables
static int num_slaves;                     //Total number of nodes less 1
static int num_idle_slaves;                //Slaves currently not doing a task
static int num_dead_slaves;                //Slaves w/ broken socket, too slow
static int num_tasks, num_tasks_on_slave, num_redos, num_updates;  //Statistics
static int last_task_id;                   //Last task sent to a slave
static int last_update_id;                 //Task that caused last update
static int slave_being_checked;            //Slave whose reply is being checked
static TOPC_BOOL aggregated_inputs_pending;//used: TOPC_OPT_aggregated_tasks>1

//Slave array, allocated on master
struct SLAVE_RECORD {
  TOPC_BOOL   busy;         //Is it doing a task?
  TOPC_BOOL   dead;         //Is slave dead?
  int    msg_cache_free;    //Free bytes in slave's msg_cache
  int    last_update_seen;  //ID of last seen queued update
  int    task_id;           //Task being done
  int    update_id;         //Last update task has seen
  int slow_on_task_id;      //Which task the slave is slow to do
  int timer;                //Elapsed time that the slave is slow on the task
  TOPC_BUF input;           //Task input being processed
  TOPC_BUF aggreg_inputs;   //Permanent buffer for aggregated task inputs
                            //submit_task_input() changes this
};
static struct SLAVE_RECORD *slave_array = NULL;
static size_t slave_array_size;            //Number of elements in slave array

//Globals for shared memory
static TOPC_BOOL is_shared_memory = 0;     //Set in TOPC_init()
struct THR_PRIV_DATA {
  enum TAG tag;
};
static struct THR_PRIV_DATA *per_slave_thr_priv = NULL, per_slave;
#define THR_PRIV \
  (is_shared_memory ? per_slave_thr_priv[COMM_rank()] : per_slave)

/***********************************************************************
 ***********************************************************************
 ** MPI tags:  (enum TAG) now defined in comm.h
 ***********************************************************************
 ***********************************************************************/


/***********************************************************************
 ***********************************************************************
 ** Printout functions:  now defined in comm.h
 ***********************************************************************
 ***********************************************************************/


/***********************************************************************
 ***********************************************************************
 ** TOP-C task and action interfaces (see topc.h)
 ***********************************************************************
 ***********************************************************************/

//User functions return TOPC_ACTION structs by value, so these ID
//codes are used to identify special return values
enum ACTION {
  NO_ACTION_ID, REDO_ID, UPDATE_ID, CONTINUE_ID
};

TOPC_ACTION NO_ACTION = {NO_ACTION_ID, "NO_ACTION", NULL, 0};
TOPC_ACTION REDO      = {REDO_ID, "REDO", NULL, 0};
TOPC_ACTION UPDATE    = {UPDATE_ID, "UPDATE", NULL, 0};
//Function to form continuation TOPC_ACTION
TOPC_ACTION CONTINUATION(TOPC_BUF msg) {
  TOPC_ACTION action;
  action.type              = CONTINUE_ID;
  action.name              = "CONTINUATION";
  action.continuation      = msg.data;
  action.continuation_size = msg.data_size;
  return action;
}

//Special return values
// NOTE:  dummy != NULL, so user TOPC_MSG(NULL,0) not confused with NOTASK
static char dummy[] = "NOTASK";
TOPC_BUF NOTASK   = {dummy, 0};

/***********************************************************************
 ***********************************************************************
 ** Task and updates are identified by reusable IDs, like PIDs on UNIX
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Compare IDs with respect to rollover.  For ease of initialization,
 * a negative value of first argument compares less then anything
 */
static int idcmp(int id1, int id2) {
  if (id1 == id2) return 0;
  if (id1 < id2 || id1 - id2 > NUM_REUSABLE_IDS / 2) return -1;
  return 1;
}

/***********************************************************************
 * Generate new task ID
 */
static int next_task_id() {
  if (++last_task_id == NUM_REUSABLE_IDS) last_task_id = 0;
  return last_task_id;
}


/***********************************************************************
 ***********************************************************************
 ** Synchronization (atomic I/O) and recording shared data page accesse
 ***********************************************************************
 ***********************************************************************/

static volatile TOPC_BOOL user_handles_own_synchronization = FALSE;

// PAGE_is_up_to_date()

void TOPC_BEGIN_ATOMIC_READ(int dummy_pagenum) {
  if (TOPC_OPT_safety < SAFETY_DFLT_ATOMIC_READ_WRITE)
    user_handles_own_synchronization = TRUE;
  // PAGE_atomic_read(dummy_pagenum, TOPC_rank());
  COMM_begin_atomic_read();
}
void TOPC_END_ATOMIC_READ(int dummy_pagenum) {
  COMM_end_atomic_read();
}
// TOPC_ATOMIC_READ(dummy_pagenum)   defined in topc.h

void TOPC_BEGIN_ATOMIC_WRITE(int dummy_pagenum) {
  // PAGE_atomic_write(dummy_pagenum, TOPC_rank(), last_update_id);
  COMM_begin_atomic_write();
}
void TOPC_END_ATOMIC_WRITE(int dummy_pagenum) {
  COMM_end_atomic_write();
}
// TOPC_ATOMIC_WRITE(dummy_pagenum)   defined in topc.h

/***********************************************************************
 ***********************************************************************
 ** Aggregated tasks
 **   Aggregated buffer:  BUF1_SIZE | BUF1 | BUF2_SIZE | BUF2 | ...
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Data structure utilities to marshal an aggregated tasks buffer
 * When buffer is complete, the last task must be NOTASK
 */

//These functions for use by debugger.  #define used for efficiency
static size_t AGGREG_TASK_SIZE(size_t size) {
  return size + sizeof(size_t); }
static size_t AGGREG_MSG_SIZE(void *input_buf) {
  return *(size_t *)(input_buf); }
static void * NEXT_AGGREG_MSG(char * aggreg_buf) {
  return aggreg_buf + sizeof(size_t) + AGGREG_MSG_SIZE(aggreg_buf); }
static TOPC_BOOL IS_EMPTY(void *input_buf) {
  return *(size_t *)(input_buf) == 0; }

// AGGREG_TASK_SIZE refers to size of task plus header holding size of task
#define AGGREG_TASK_SIZE(size) (size + sizeof(size_t))
#define IS_EMPTY(input_buf) ( *(size_t *)(input_buf) == 0 )
#define AGGREG_MSG_SIZE(input_buf) ( *(size_t *)(input_buf) )
#define AGGREG_MSG_DATA(input_buf) ( input_buf + sizeof(size_t) )
#define NEXT_AGGREG_MSG(aggreg_buf) \
  ( aggreg_buf + AGGREG_TASK_SIZE( AGGREG_MSG_SIZE(aggreg_buf) ) )

static void init_aggregate_buffer(TOPC_BUF * buf) {
  if (buf->data == NULL) // If buffer not yet created
    buf->data = malloc(4); // NOTE: if we malloc(0), some O/S return NULL
  buf->data_size = 0;
}
// NOTASK.data_size is 0
static void add_to_aggregate_buffer(TOPC_BUF task, TOPC_BUF * buf) {
  char * buf_data = buf->data;
  // copy header with size of next task
  memcpy(buf_data + buf->data_size, &task.data_size, sizeof(size_t));
  // copy data of next task
  if (task.data_size > 0)
    memcpy(buf_data + buf->data_size + sizeof(size_t),
	   task.data, task.data_size);
  buf->data_size = buf->data_size + sizeof(size_t) + task.data_size;
}
 
/***********************************************************************
 ***********************************************************************
 ** Define wrappers for callbacks:
 **   Potential uses:  protected stack call, task aggregation
 ** (much of this is only in experimental version)
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Define wrappers for callback functions
 *   do_task_wrapper/COMM_do_task() as wrapper that takes
 *   task input and repeatedly calls
 *   application-defined do_task()
 *   to produce for output.data
 */
static TOPC_BUF (*generate_task_input_orig)(void);
static TOPC_BUF (*do_task_orig)(void *input);
static TOPC_ACTION (*check_task_result_orig)(void *input, void* output);
static void (*update_shared_data_orig)(void *input, void *output);

static TOPC_BUF generate_task_input_wrapper() {
  return (*generate_task_input_orig)();
}
static TOPC_BUF do_task_wrapper(void *input) {
  static TOPC_BUF output = {NULL, 0};
  static int output_realloc_size = 0;
  TOPC_BUF tmp;

  if (TOPC_OPT_aggregated_tasks > 1) {
    assert( ! IS_EMPTY(input) );
    init_aggregate_buffer(&output);
    while ( AGGREG_MSG_SIZE(input) > 0 ) {
      // Process current task input
      ++num_tasks_on_slave;
      tmp = (*do_task_orig)((void *)AGGREG_MSG_DATA((char *)input));
      // output.data must hold current messages and tmp and NOTASK at end
      if (output_realloc_size
	  < output.data_size + AGGREG_TASK_SIZE(tmp.data_size)
	    + AGGREG_TASK_SIZE(NOTASK.data_size)) {
	output_realloc_size =
	  output.data_size + AGGREG_TASK_SIZE(tmp.data_size)
	  + AGGREG_TASK_SIZE(NOTASK.data_size);
	output.data = realloc(output.data, output_realloc_size);
      }
      add_to_aggregate_buffer(tmp, &output);
      MEM_free(tmp.data);
      input = NEXT_AGGREG_MSG((char *)input);
    }
    add_to_aggregate_buffer(NOTASK, &output);
    return output;
  } else
    return (*do_task_orig)(input);
}
static TOPC_ACTION check_task_result_wrapper(void *input, void *output) {
  return (*check_task_result_orig)(input, output);
}
static void update_shared_data_wrapper(void *input, void *output) {
  (*update_shared_data_orig)(input, output);
}

//Application-defined callbacks
//The communication layer may redefine them
//  (e.g.:  COMM_init() may add another wrapper)
TOPC_BUF (*COMM_generate_task_input)(void) = generate_task_input_wrapper;
TOPC_BUF (*COMM_do_task)(void *input) = do_task_wrapper;
TOPC_ACTION (*COMM_check_task_result)(void *input, void *output)
	= check_task_result_wrapper;
void (*COMM_update_shared_data)(void *input, void *output)
	= update_shared_data_wrapper;
//Can be called by communication layer
// Assigned to slave_loop() in TOPC_master_slave().
void (*COMM_slave_loop)(void) = NULL;


/***********************************************************************
 ***********************************************************************
 ** Dead Slaves (used on master)
 ***********************************************************************
 ***********************************************************************/

// Used only for debugging.
static int debug_num_dead_slaves() {
  int i;
  int dead_slaves = 0;
  for(i = 1; i <= num_slaves; i++)
    if (slave_array[i].dead == TRUE)
      dead_slaves ++;
  return dead_slaves;
}

/***********************************************************************
 * Do bookkeeping to declare slave as dead
 */
static void slave_is_dead(int rank) {
  slave_array[rank].dead = TRUE;
  num_dead_slaves++;
  if (! slave_array[rank].busy) {
    num_idle_slaves--;  // A dead idle slave is not really idle
  }
}

/***********************************************************************
 * Apply criteria for declaring if a (possibly slow) slave is dead
 * Dead if busy on same task for at least TOPC_OPT_slave_timeout seconds
 */
static TOPC_BOOL is_dead_slave(int rank) {
  time_t t;

  if (slave_array[rank].dead)
    return TRUE;
  // If slave doing task during three rounds of tasks for other slaves
  if ((slave_array[rank].task_id < last_task_id - 3 * num_slaves)
      || ((slave_array[rank].task_id > last_task_id) &&
	  (slave_array[rank].task_id
	   < NUM_REUSABLE_IDS + last_task_id - 3 * num_slaves))) {
    // And if this task was previously found to be slow
    if (slave_array[rank].task_id == slave_array[rank].slow_on_task_id) {
      // And if TOPC_OPT_slave_timeout seconds have elapsed
      if ( time(&t) - slave_array[rank].timer > TOPC_OPT_slave_timeout )
	return TRUE; // declare slave dead
      else return FALSE; // timeout not yet elapsed
    } else {
      // Else slave_array was slow on previous task, but not on current task
      // Note current task and reset timer
      slave_array[rank].slow_on_task_id = slave_array[rank].task_id;
      slave_array[rank].timer = time(&t);
    }
  }
  return FALSE;
}

/***********************************************************************
 * Check for new dead slave.  A dead slave is newly dead if the busy
 *   field is still TRUE.  Make it dead, change its busy field to FALSE,
 *   and return it.  It is now an old dead slave.
 * When this is called, if it finds a new dead slave, that slave
 *   has its tasks resubmitted elsewhere.
 */
static int check_for_dead_slave() {
  int rank;

  for (rank = 1; rank <= num_slaves; rank++) {
    if (slave_array[rank].busy == TRUE) {
      if (slave_array[rank].dead == FALSE) {
	// If a message to slave fails, socket connection must have died
	// DO WE WANT TO PING SO OFTEN?  SHOULD CHANGE THIS.
	if (COMM_send_msg(NULL, 0, rank, PING_TAG) == FALSE
	    || is_dead_slave(rank)) {
	  slave_is_dead(rank);
	  slave_array[rank].busy = FALSE;
	  return rank;
	}
      }
      // Other parts of code declare a slave dead when we can't send a message.
      // We recognize those "new dead slaves" because busy is TRUE.
      // Catch such "new" dead slaves here, and report them.
      if (slave_array[rank].dead == TRUE) {
	slave_array[rank].busy = FALSE;
	return rank;
      }
    }
  }
  return 0;
}

/***********************************************************************
 * Check for new dead slave, but only once per round of tasks
 *  (only once out of every `num_slaves' calls to this routine).
 */
static int new_dead_slave() {
  static int counter = -1;
  if (counter < 0)
    counter = num_slaves;
  counter --;

  if (counter < 0) {
    int id = check_for_dead_slave();
    assert( num_dead_slaves == debug_num_dead_slaves() );
    if (id) {
      fprintf(stderr, "***********************************************\n");
      fprintf(stderr, "************** SLAVE %d HAS DIED **************\n", id);
      fprintf(stderr, "***********************************************\n");
      fflush(stderr); // stderr should be unbuffered anyway, but why not
      if (num_dead_slaves == num_slaves - 1)
	fprintf(stderr, "*** ONE SLAVE LEFT.  NO WARNING GIVEN IF IT DIES!\n");
      return id;
    }
  }
  return 0;
}


/***********************************************************************
 ***********************************************************************
 ** Update queue on master
 ***********************************************************************
 ***********************************************************************/

//Assume update messages are either all short or all long.
//If no update messages pending, all messages are short,
//  and msg_cache for each slave has space, then
//  immediately broadcast them, using the network as our buffer.
//Otherwise, queue the messages, and then update_slave() sends
//  all pending updates to a slave just before sending the
//  next task input (whether due to REDO, CONTINUATION, or new task)
//If more slaves return while the master was doing its own update,
//  then those slaves will see all updates when the master recognizes them.
//Note that an urgent message (e.g.: SOFT_ABORT), can always be
//  probed by slave, since all updates fit in slave msg_cache.

//Update queue
struct UPDATE_RECORD {
  int    id;              //ID of task that resulted in the update
  void  *input;           //Update information
  size_t input_size;
  void  *output;
  size_t output_size;
  int    slaves_updated;  //Number of slaves this update has been sent to
};
static struct UPDATE_RECORD update_queue[UPDATE_QUEUE_SIZE],
                            *update_queue_tail, *update_queue_head;

static void free_input_data(void *ptr) { MEM_free(ptr); }
// More efficient version for now:
#define free_input_data(x) (MEM_free(x))

/***********************************************************************
 * Send update message to a given slave
 */
static void send_update(int slave, void *input, size_t input_size,
                        void *output, size_t output_size) {
  if (COMM_send_msg(input, input_size, slave, UPDATE_INPUT_TAG) == FALSE)
    slave_is_dead(slave);
  else if (COMM_send_msg(output, output_size, slave, UPDATE_OUTPUT_TAG)
	   == FALSE)
    slave_is_dead(slave);
}

/***********************************************************************
 * Broadcast update message
 */
static void broadcast_update(void *input, size_t input_size,
                             void *output, size_t output_size) {
  int slave;
  for (slave = 1; slave <= num_slaves; ++slave) {
    if (slave_array[slave].dead == FALSE) {
      send_update(slave, input, input_size, output, output_size);
      slave_array[slave].msg_cache_free -=
	(input_size + output_size + 2*sizeof(enum TAG) + 2*sizeof(input_size));
    }
  }
}

// COULD USE getsockopt(,,SO_RCVBUF,result,&result_size);
//  TO GET OR CHANGE DEFAULT SOCKET BUFFER SIZE FOR MORE ACCURACY.
// IF SIZE IS MORE THAN NETWORK BUFFER SIZE, THEN CONSIDER INCREASING
//  SO_SNDBUF and SO_RCVBUF.
// THIS COULD ALSO BE USEFUL FOR aggregation FOR DIST. MEMORY.

static TOPC_BOOL broadcast_will_not_block(int input_size, int output_size) {
  int slave;
  for (slave = 1; slave <= num_slaves; ++slave)
    if (input_size + output_size + 2*sizeof(enum TAG) + 2*sizeof(input_size)
          > slave_array[slave].msg_cache_free
        && slave_array[slave].busy)
      return FALSE;
  return TRUE;
}

/***********************************************************************
 * Find first update that the slave hasn't seen.  If there are no
 * unseen updates, return value points past the end of the queue
 */
static struct UPDATE_RECORD *find_first_unseen_update(int slave) {
  struct UPDATE_RECORD *l, *m, *r, *mm;
  int mm_id, last_update_seen = slave_array[slave].last_update_seen;

  l = update_queue_head;
  r = update_queue_tail;
  if (r < l) r += UPDATE_QUEUE_SIZE;
  last_update_seen
    = last_update_seen + NUM_REUSABLE_IDS / 2 < update_queue_head->id
      ? last_update_seen + NUM_REUSABLE_IDS : last_update_seen;

  while (r > l) {
    m = l + (r - l) / 2;
    mm = m < update_queue + UPDATE_QUEUE_SIZE ? m : m - UPDATE_QUEUE_SIZE;
    mm_id = mm->id + NUM_REUSABLE_IDS / 2 < update_queue_head->id
            ? mm->id + NUM_REUSABLE_IDS : mm->id;
    if (idcmp(last_update_seen, mm_id) < 0)
      r = m;
    else
      l = m + 1;
  }
  assert(l == r);
  return r < update_queue + UPDATE_QUEUE_SIZE ? r : r - UPDATE_QUEUE_SIZE;
}

/***********************************************************************
 * Send all necessary updates to a slave.  Free memory occupied by
 * updates that have been sent to all slaves
 */
static void update_slave(int slave) {
  struct UPDATE_RECORD *update;
  int slaves_updated;

  //Send all unseen updates to the slave.  Increment updated slave
  //count for each update.  When an update has been sent to all
  //slaves, remove it from queue.  The removal should only happen at
  //the head of the queue, because updates are sent in order
  update = find_first_unseen_update(slave);
  while (update != update_queue_tail) {
    send_update(slave, update->input, update->input_size, update->output,
                update->output_size);
    slave_array[slave].last_update_seen = update->id;

    slaves_updated = ++update->slaves_updated;
    assert(slaves_updated < num_slaves || update == update_queue_head);

    if (++update == update_queue + UPDATE_QUEUE_SIZE)
      update = update_queue;
    if (slaves_updated == num_slaves) {
      free_input_data(update_queue_head->input);
      COMM_free_receive_buf(update_queue_head->output);
      update_queue_head = update;
    }
  }
  //update_slave() always called when slave finishes task
  slave_array[slave].msg_cache_free = MSG_CACHE_SIZE;
}

/***********************************************************************
 * Queue an update for being sent to all slaves eventually.  The
 * update queue must have space on it
 */
static void queue_update(void *input, size_t input_size,
                         void *output, size_t output_size) {
  int slave;
  static int id = 0;

  update_queue_tail->id             = id++;
  if (id == NUM_REUSABLE_IDS)
    id = 0;
  update_queue_tail->input          = input;
  update_queue_tail->input_size     = input_size;
  update_queue_tail->output         = output;
  update_queue_tail->output_size    = output_size;
  update_queue_tail->slaves_updated = 0;
  if (++update_queue_tail == update_queue + UPDATE_QUEUE_SIZE)
    update_queue_tail = update_queue;
  if (update_queue_head == update_queue_tail)
    ERROR("Update queue overflow;\n"
          "      Recompile TOP-C with larger topc.c:UPDATE_QUEUE_SIZE");

  for (slave = 1; slave <= num_slaves; ++slave)
    if (! slave_array[slave].busy)
      update_slave(slave);
}

/***********************************************************************
 * Schedule global update.  Long updates are queued.  Short updates
 * are only queued if there is something on the queue already, to make
 * sure updates are not reordered.  Unless update is queued, frees
 * memory
 */
static void schedule_update(int id, void *input_data, size_t input_size,
                            void *output_data, size_t output_size) {
  if (broadcast_will_not_block(input_size, output_size)
      && update_queue_head == update_queue_tail) {
    broadcast_update(input_data, input_size, output_data, output_size);
    free_input_data(input_data);
    COMM_free_receive_buf(output_data);
  }
  else
    queue_update(input_data, input_size, output_data, output_size);
  last_update_id = id;
}

/***********************************************************************
 ***********************************************************************
 ** Tracing
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Default:  trace msg; value of NULL: print nothing
 * User can redefine initial binding to print buffer-specific tracing msg
 */

static void do_nothing() {}
void (*TOPC_OPT_trace_input)() = do_nothing;  /* args: (void *input) */
void (*TOPC_OPT_trace_result)() = do_nothing;
  /* args: (void *input, void *output) */

static void trace_input(void *input, int slave) {
  if (TOPC_OPT_trace == FALSE) return;
  else if (input == NOTASK.data) { printf("master: NOTASK\n"); return; }
  else { printf("master -> %d: ", slave);
         if (TOPC_OPT_trace == 2 && *TOPC_OPT_trace_input)
           (*TOPC_OPT_trace_input)(input);
         printf("\n");
         fflush(stdout);
       }
}
static void trace_result(int slave, void *input, void *output) {
  if (TOPC_OPT_trace == FALSE) return;
  else { printf("%d -> master: ", slave);
         if (TOPC_OPT_trace == 2 && TOPC_OPT_trace_result)
           (*TOPC_OPT_trace_result)(input, output);
         printf("\n");
         fflush(stdout);
       }
}
static void trace_action(TOPC_ACTION action, void *data, void *out, int slave) {
  if (TOPC_OPT_trace == FALSE) return;
  if (action.type == REDO_ID) { printf("REDO: "); trace_input(data, slave); }
  else if (action.type == UPDATE_ID)
    { printf("  UPDATE: ");
      if (TOPC_OPT_trace == 2)
        (*TOPC_OPT_trace_result)(data, out);
      printf("\n");
    }
  else if (action.type == CONTINUE_ID)
    { printf("  CONTINUATION: "); trace_input(data, slave); }
}

/***********************************************************************
 ***********************************************************************
 ** Master-slave processing
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Slave processing loop (slave)
 */
static void slave_loop() {
  enum TAG tag;
  void *input = NULL, *old_input, *output;
  size_t input_size, output_size;
  TOPC_BUF result = NOTASK;

  if (!is_shared_memory) /* is_shared_memory inited to 0, reset on master */
    num_tasks = num_tasks_on_slave = num_updates = 0;
  for (;;) {
    old_input = input;
    input = NULL; /* Tell COMM layer to provide buffer */
    COMM_receive_msg(&input, &input_size, NULL, &tag);
    // result.data could point into old_input.  Now safe to free old_input.
    if (old_input) COMM_free_receive_buf(old_input);
    /* safe to free old result upon seeing new input from master */
    /* but if TOPC_OPT_aggregated_tasks > 1, result was static buffer */
    if (result.data != NOTASK.data && TOPC_OPT_aggregated_tasks <= 1) {
      MEM_free(result.data);
      result = NOTASK;
    }
    /* Some compilers don't like this:  THR_PRIV.tag = tag; */
    if (is_shared_memory) per_slave_thr_priv[COMM_rank()].tag = tag;
    else per_slave.tag = tag;
    switch (tag) {
    case UPDATE_INPUT_TAG:
      output = NULL; /* Tell COMM layer to provide buffer */
      COMM_receive_msg(&output, &output_size, NULL, &tag);
      if (tag != UPDATE_OUTPUT_TAG)
        ERROR("Received %d instead of UPDATE_OUTPUT_TAG");

      if (!is_shared_memory)
        ++num_updates;
      (*COMM_update_shared_data)(input, output);
      // malloc(0) can return NULL instead of a 0-length buffer
      if (output) COMM_free_receive_buf(output);
      break;

    case TASK_INPUT_TAG:
    case REDO_INPUT_TAG:
    case CONTINUATION_INPUT_TAG:
      if (!is_shared_memory)
        ++num_tasks;
      COMM_is_abort_pending = 0;//Turn off aborts before new task, REDO, or CONT
      // NOTE:  result.data can be ptr into buffer for input
      result = (*COMM_do_task)(input);
      if (TOPC_OPT_aggregated_tasks <= 1)
	MEM_register_buf(result.data);
      COMM_send_msg(result.data, result.data_size, 0, SLAVE_REPLY_TAG);
      break;

    case END_MASTER_SLAVE_TAG:
      /* Last chance to free up result from last TASK_INPUT_TAG */
      COMM_free_receive_buf(input);
      /* Flush user prints on slave, or master may exit before they're seen */
      fflush(stdout); fflush(stderr);
      return;

    case PING_TAG:
      break;

    case CHDIR_TAG:
      if (0 != chdir((char *)input))
        WARNING("SLAVE %d:  Couldn't change directory to:\n  %s",
                TOPC_rank(), (char *)input);
      break;

    /* SLAVE_WAIT_TAG now obsolete;  Remove in next version. */
    case SLAVE_WAIT_TAG:
      sleep(*(int *)input);
      break;

    default:
      ERROR("Received invalid tag %d", tag);
    }
  }
}

/***********************************************************************
 * Find available slave.  Should only be called when it's known that
 * there is one (master)
 */
static int find_available_slave() {
  static int last_slave = 0;   //Slave to which a task was sent last
  int i;

  //Loop starts where we left off last time, to spread the load evenly
  //across all slaves
  for (i = 0; i < num_slaves; ++i) {
    if (++last_slave > num_slaves) last_slave = 1;
    if ( !slave_array[last_slave].busy && !slave_array[last_slave].dead )
      break;
  }
  assert(!slave_array[last_slave].busy);
  return last_slave;
}

/***********************************************************************
 * Write barrier:  Wait for all slaves to finish tasks (master)
 */
static void wait_for_all_slaves() {
  int rank = 1;
  for (rank = 1; rank <= num_slaves; rank++) {
    if (slave_array[rank].busy)
      COMM_probe_msg_blocking(rank);
  }
}

/***********************************************************************
 * Send task input, REDO, or CONTINUATION to slave (master)
 */
static void send_task_input(int slave, TOPC_BUF input, enum TAG tag) {
  if ( TOPC_OPT_aggregated_tasks > 1 )
    assert( input.data_size == 0 || AGGREG_MSG_SIZE(input.data) > 0 );
  //Send all pending updates to the slave
  update_slave(slave);

  //Record task information and send the task
  slave_array[slave].busy       = TRUE;
  //gdc - incrementing task_id here means REDO and CONTINUATION also get
  //      new task_id's.  Is this the desired behavior?
  slave_array[slave].task_id    = next_task_id();
  slave_array[slave].update_id  = last_update_id;
  if (COMM_send_msg(input.data, input.data_size, slave, tag) == FALSE)
    slave_is_dead(slave);
}

/***********************************************************************
 * Receive and process output of a slave computation (master)
 */
static void receive_task_output() {
  TOPC_ACTION action;
  TOPC_BUF msg_buf;
  void *output = NULL; // ptr to data of a message
  size_t output_size;
  int slave;
  enum TAG tag;
  // THESE NEXT USED ONLY if TOPC_OPT_aggregated_tasks > 1
  char *input_buf;  // ptr to next message of input
  char *output_buf; // ptr to next message of output
  void *output_orig; // ptr to all messages received by COMM_receive_msg

  // Get a message:
  // Ignore incomplete messages (COMM_receive_msg() == FALSE)
  // If received from slave that is dead, ignore it (MAYBE FREE output?)
  do {
    while (COMM_receive_msg(&output, &output_size, &slave, &tag) == FALSE)
      /* no body */;
  } while ( slave_array[slave].dead == TRUE );
  if ( tag != SLAVE_REPLY_TAG ) {
    printf("TOP-C:  internal error:  tag != SLAVE_REPLY_TAG\n");fflush(stdout);
    exit(1);
  }

  if (TOPC_OPT_aggregated_tasks > 1) {
    input_buf = slave_array[slave].aggreg_inputs.data;
    assert( ! IS_EMPTY(input_buf ) );
    if ( IS_EMPTY(input_buf) ) {
      ++num_idle_slaves;
      return;
    }
    output_buf = output;
    output_orig = output;
  }

 NEXT_AGGREG_TASK:
  // Now call check_task_result:
  slave_being_checked = slave;
  if (TOPC_OPT_aggregated_tasks > 1) {
    slave_array[slave].input.data_size = AGGREG_MSG_SIZE(input_buf);
    slave_array[slave].input.data = AGGREG_MSG_DATA(input_buf);
    input_buf = NEXT_AGGREG_MSG(input_buf);
    output = AGGREG_MSG_DATA(output_buf);
    output_buf = NEXT_AGGREG_MSG(output_buf);
  }

  trace_result(slave, slave_array[slave].input.data, output);
  action = (*COMM_check_task_result)(slave_array[slave].input.data, output);

  if (TOPC_OPT_aggregated_tasks > 1
      && action.type != NO_ACTION_ID) {
    printf("TOPC_OPT_aggregated_tasks > 1; task action wasn't NO_ACTION\n");
    printf("TOP-C doesn't yet implement this case for aggregated tasks.\n");
    fflush(stdout);
    exit(1);
  }
  slave_being_checked = 0;

  //Take required action
  switch (action.type) {
  case NO_ACTION_ID:
    if (TOPC_OPT_aggregated_tasks <= 1) {
      // If TOPC_OPT_aggregated_tasks > 1, do this just before returning
      slave_array[slave].busy = FALSE;
      ++num_idle_slaves;
      COMM_free_receive_buf(output);
      free_input_data(slave_array[slave].input.data);
    }
    break;

  case REDO_ID:
    trace_action(action, slave_array[slave].input.data, output, slave);
    COMM_free_receive_buf(output);
    msg_buf.data = slave_array[slave].input.data;
    msg_buf.data_size = slave_array[slave].input.data_size;
    send_task_input(slave, msg_buf, REDO_INPUT_TAG);
    ++num_redos;
    break;

  case CONTINUE_ID:
    trace_action(action, action.continuation, output, slave);
    COMM_free_receive_buf(output);
    msg_buf.data = action.continuation;
    msg_buf.data_size = action.continuation_size;
    send_task_input(slave, msg_buf, CONTINUATION_INPUT_TAG);
    break;

  case UPDATE_ID:
    trace_action(action, slave_array[slave].input.data, output, slave);
    slave_array[slave].busy = FALSE;
    ++num_idle_slaves;
    // If shared memory, default is that UPDATE acts as write barrier
    if ( is_shared_memory && ! user_handles_own_synchronization )
      wait_for_all_slaves();
    // Call application callback on master
    (*COMM_update_shared_data)(slave_array[slave].input.data, output);
    if ( is_shared_memory ) free_input_data(slave_array[slave].input.data);
    else schedule_update( slave_array[slave].task_id,
                          slave_array[slave].input.data,
                          slave_array[slave].input.data_size,
		          output, output_size);
    ++num_updates;
    break;

  default:
    ERROR("check_task_result() returned invalid action %d", action.type);
  }

  if (TOPC_OPT_aggregated_tasks > 1) {
    if (! IS_EMPTY(input_buf))
      goto NEXT_AGGREG_TASK;
    else {
      assert( IS_EMPTY(output_buf) );
      COMM_free_receive_buf(output_orig);
      slave_array[slave].busy = FALSE;
      ++num_idle_slaves;
    }
  }
}

/***********************************************************************
 * Submit task for dispatching to a slave.  Returns FALSE only when no
 * slaves are busy, so there's no chance of getting more tasks from
 * application code (master)
 */
static void wait_until_an_idle_slave(void);
static TOPC_BOOL submit_task_input(TOPC_BUF input) {
  static int num_inputs = 0;
  static TOPC_BUF aggreg_tasks = {NULL, 0};
  static int slave = 1;
  int size_for_new_input  // leaving space for final NOTASK)
    = AGGREG_TASK_SIZE(input.data_size)
      + AGGREG_TASK_SIZE(NOTASK.data_size);

  if (TOPC_OPT_aggregated_tasks > 1 && aggreg_tasks.data == NULL)
    aggreg_tasks.data = malloc(AGGREG_TASKS_SIZE);

  // Don't try to send messages overflowing AGGREG_TASKS_SIZE
  if (TOPC_OPT_aggregated_tasks > 1
      && aggreg_tasks.data_size + size_for_new_input >= AGGREG_TASKS_SIZE) {
    if (aggreg_tasks.data_size > 0 && input.data != NOTASK.data) {
      submit_task_input(NOTASK);  // submit previous tasks
      assert(aggreg_tasks.data_size == 0);
      wait_until_an_idle_slave(); // and then continue with current task
    }
    // But if only one message, and it's more than AGGREG_TASKS_SIZE, realloc
    if ( size_for_new_input >= AGGREG_TASKS_SIZE )
      aggreg_tasks.data = realloc(aggreg_tasks.data, size_for_new_input);
  }

  // MAYBE SHOULD MOVE new_dead_slave() LOGIC FROM wait_until_an_idle_slave
  //  TO HERE.  OR AT LEAST wait_until_an_idle_slave SHOULD CHECK
  //   EVEN WHEN MANY IDLE SLAVES.  ALSO CHECK BEFORE TERMINATION OF PAR.
  assert( num_idle_slaves > 0 );
  slave = find_available_slave();
  // trace_input must come after receive_task_output()
  if (TOPC_OPT_aggregated_tasks <= 1)
    trace_input(input.data, slave);

  if (TOPC_OPT_aggregated_tasks > 1) {
    trace_input(input.data, -1);
    add_to_aggregate_buffer(input, &aggreg_tasks);
    if (input.data != NOTASK.data) {
      num_inputs++;
      free_input_data(input.data);
    }
    if (num_inputs >= TOPC_OPT_aggregated_tasks)
      add_to_aggregate_buffer(NOTASK, &aggreg_tasks);
    // If NOTASK or if have our quota of aggregated tasks, then submit
    // If neither, then return FALSE now
    if (input.data != NOTASK.data
	&& num_inputs < TOPC_OPT_aggregated_tasks) {
      ++num_tasks;
      aggregated_inputs_pending = TRUE;
      return FALSE;
    }
    // Okay, we've got all the tasks, and we'll be sending them out.
    if (TOPC_OPT_trace != FALSE) {
      printf("master -> %d\n", slave);
      fflush(stdout);
    }
    //Copy from application space to TOP-C space
    {
      void * tmp = slave_array[slave].aggreg_inputs.data;
      slave_array[slave].aggreg_inputs.data = aggreg_tasks.data;
      slave_array[slave].aggreg_inputs.data_size = aggreg_tasks.data_size;
      aggreg_tasks.data = tmp;
      if ( aggreg_tasks.data == NULL )
        aggreg_tasks.data = malloc(AGGREG_TASKS_SIZE);
      slave_array[slave].input = slave_array[slave].aggreg_inputs;
      // Reset aggreg_tasks for next call
      num_inputs = 0;
      init_aggregate_buffer(&aggreg_tasks);
    }
  } else {
    //Copy from application space to TOP-C space
    slave_array[slave].input.data = input.data;
    slave_array[slave].input.data_size = input.data_size;
  }

  send_task_input(slave, slave_array[slave].input, TASK_INPUT_TAG);
  --num_idle_slaves;
  ++num_tasks;
  aggregated_inputs_pending = FALSE;
  return TRUE;
}

/***********************************************************************
 * Receive and process result = (input,output) of a slave computation (master)
 */
static void wait_until_an_idle_slave() {
  //NOTE:  receive_task_output() blocking due to call to COMM_receive_msg().
  while (num_idle_slaves == 0) {
    int id;
    receive_task_output();
    // If idle slave available, move task of dead slave to idle slave
    if (! is_shared_memory && num_idle_slaves > 0)
      if ( (id = new_dead_slave()) != 0 ) {
	submit_task_input(slave_array[id].input);
	if (TOPC_OPT_aggregated_tasks > 1)
	  submit_task_input(NOTASK);  // Force this to be sent now.
      }
  }
  assert( num_idle_slaves > 0 );
}

/***********************************************************************
 * Initialize master (master)
 */
static void master_init() {
  int slave;

  //Initialize statistics and run parameters
  num_tasks = num_tasks_on_slave = num_redos = num_updates = 0;
  num_idle_slaves = num_slaves = COMM_node_count() - 1;
  num_dead_slaves = 0;  //All slaves initially assumed to be alive.
  last_task_id   = -1;  //No last task ID
  last_update_id = -1;  //No updates seen so far
  slave_being_checked = 0;  //Not inside check_task_result()

  //Allocate and initialize slave array
  if (slave_array && (signed int)slave_array_size != num_slaves + 1) {
    free(slave_array);
    slave_array = NULL;
    if (is_shared_memory && per_slave_thr_priv) {
      free(per_slave_thr_priv);
      per_slave_thr_priv = NULL;
    }
  }
  if (!slave_array) {
    slave_array_size = num_slaves + 1;
    slave_array = calloc(slave_array_size, sizeof(struct SLAVE_RECORD));
    if (slave_array == NULL) ERROR("No memory for process table");
  }
  for (slave = 1; slave <= num_slaves; ++slave) {
    slave_array[slave].last_update_seen = -1;  //Seen none yet
    slave_array[slave].busy = FALSE;
    slave_array[slave].dead = FALSE;
    slave_array[slave].task_id = -1;
    slave_array[slave].slow_on_task_id = -1;
    slave_array[slave].timer = 0;
    slave_array[slave].msg_cache_free = MSG_CACHE_SIZE;
  }

  //Thread private values for do_task() on slave in shared memory
  if (is_shared_memory && !per_slave_thr_priv) {
    per_slave_thr_priv = calloc(num_slaves+1, sizeof(struct THR_PRIV_DATA));
    if (per_slave_thr_priv == NULL) ERROR("No memory for THR_PRIV_DATA");
  }

  //Initialize update queue
  update_queue_head = update_queue_tail = update_queue;
}

/***********************************************************************
 * Force slaves to change to same directory as master (ignore if no such dir)
 */
static void master_slave_chdir() {
  if ( ! COMM_initialized() )
    ERROR("master_slave_chdir() called before TOPC_init()");
  // If not shared memory, ask slave to change to same directory as master
  if ( ! COMM_is_shared_memory() ) {
    if (TOPC_is_master()) {
      int slave;
      char cwd[MAX_PATH_LEN];

      TOPC_OPT_getpwd(cwd, MAX_PATH_LEN);
      cwd[MAX_PATH_LEN-1] = '\0';
      // Emulate master side of master_slave loop
      for (slave = 1; slave <= num_slaves; ++slave) {
	if (slave_array[slave].dead == FALSE) {
	  COMM_send_msg(cwd, 1+strlen(cwd), slave, CHDIR_TAG);
	  COMM_send_msg(NULL, 0, slave, END_MASTER_SLAVE_TAG);
	}
      }
    }
    // each slave sees CHDIR_TAG, END_MASTER_SLAVE_TAG (then exits slave loop)
    else slave_loop();
  }
}

/***********************************************************************
 ***********************************************************************
 ** Interface (public) functions
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Get node count (pass to comm layer)
 */
int TOPC_node_count() {
  return COMM_node_count();
}

/***********************************************************************
 * Get number of slaves
 */
int TOPC_num_slaves() {
  return COMM_node_count() - 1;
}

/***********************************************************************
 * Get number of slaves
 */
int TOPC_num_idle_slaves() {
  if (COMM_rank() == 0)
    return num_idle_slaves;
  else {
    ERROR("TOPC_num_idle_slaves:  can only be called on master");
    return -1; // eliminate compiler warning
  }
}

/***********************************************************************
 * Get rank of this node (pass to comm layer)
 */
int TOPC_rank() {
  return COMM_rank();
}

/***********************************************************************
 * Figure out who is the master
 */
TOPC_BOOL TOPC_is_master() {
  return COMM_rank() == 0;
}

/***********************************************************************
 * Initialize TOP-C and COMM (MPI, etc.) (both master and slave)
 */

void TOPC_init(int *argc, char ***argv) {
  void *tmp;

  errno = 0;
  tmp = malloc(1000);
  if (tmp == NULL && errno != 0) {
    perror("TOP-C initialization:");
    if (errno == ENOMEM)
      printf("This can happen if your application dynamically OR statically\n"
             "  allocates a lot of memory.  For example,\n"
             "    char *x[100000000];\n"
             "  might cause this error, depending on your swap size.\n");
    exit(1);
  }
  else
    free(tmp);

  // Get --TOPC options; In distributed memory, this is also called on slaves.
  TOPC_OPT_pre_init( argc, argv );

  // Spawn slaves
  COMM_init(argc, argv);

  is_shared_memory = COMM_is_shared_memory();
  if (TOPC_is_master()) master_init();
  if (!TOPC_is_master() && TOPC_OPT_slave_wait > 0 )
    sleep(TOPC_OPT_slave_wait);
  master_slave_chdir();  // Place slaves in same directory as master

  // This strips away TOPC options, so they are invisible to application
  TOPC_OPT_post_init( argc, argv );
}

/***********************************************************************
 * Shutdown TOP-C and MPI (both master and slave)
 */
void TOPC_finalize() {
  if ( TOPC_is_master() )
    TOPC_OPT_finalize(num_tasks, num_redos, num_updates);
  COMM_finalize();
}

/***********************************************************************
 * Was the shared memory up to date?  Should only be called from within
 * check_task_result()
 */
TOPC_BOOL TOPC_is_up_to_date() {
  int cmp;

  if (slave_being_checked == 0)
    ERROR("TOPC_is_up_to_date() called from outside of check_task_result()");

  cmp = idcmp(slave_array[slave_being_checked].update_id, last_update_id);
  assert(cmp <= 0);
  return cmp == 0;
}

/***********************************************************************
 * Is this a REDO or CONTINUATION?  Should only be called from within
 * do_task()
 */
TOPC_BOOL TOPC_is_REDO(void) {
  if (TOPC_is_master())
    ERROR("TOPC_is_REDO() called from outside of do_task()");
  return THR_PRIV.tag == REDO_INPUT_TAG;
}
TOPC_BOOL TOPC_is_CONTINUATION(void) {
  if (TOPC_is_master())
    ERROR("TOPC_is_CONTINUATION() called from outside of do_task()");
  return THR_PRIV.tag == CONTINUATION_INPUT_TAG;
}

/***********************************************************************
 * Parallel execution.  This function is called on both master and
 * slave, and executes the parallel code supplied in the form of four
 * callback routines (both master and slave)
 */
void TOPC_master_slave(
  TOPC_BUF (*generate_task_input_)(void),
  TOPC_BUF (*do_task_)(),              /* args: (void *input) */
  TOPC_ACTION (*check_task_result_)(), /* args: (void *input, void *output) */
  void (*update_shared_data_)()        /* args: (void *input, void *output) */
) {
  TOPC_BUF input;
  int slave;

  //Used in calls to COMM_is_on_stack()
  //Must mark a point on stack before calls to do_task, generate_task_input
  //Beware of segmented memory models, where pointer comparisons in different
  //  segments may not work as expected
  COMM_stack_bottom = &input;

  if (!COMM_initialized())
    ERROR("TOPC_master_slave called before TOPC_init()");

  //Create buffers for aggregated tasks
  if (TOPC_OPT_aggregated_tasks > 1)
    for (slave = 1; slave <= num_slaves; ++slave) {
      slave_array[slave].aggreg_inputs.data_size = 0; // no aggreg tasks yet
      slave_array[slave].aggreg_inputs.data = malloc(AGGREG_TASKS_SIZE);
    }

  //Store callbacks into global variables.  If we want to support
  //  recursion these need to go on a stack
  // xxx_orig() is called by COMM_xxx for xxx a callback function.
  generate_task_input_orig  = generate_task_input_;
  do_task_orig              = do_task_;
  check_task_result_orig    = check_task_result_;
  update_shared_data_orig   = update_shared_data_;
  COMM_slave_loop           = slave_loop; // needed for pthreads/SMP COMM layer

  //Master loop --- heart of the TOP-C algorithm
  if ( TOPC_is_master() ) {
    while (1) {
      wait_until_an_idle_slave();      // And process any pending slave replies
      input = (*COMM_generate_task_input)();
      if (input.data != NOTASK.data) {
        MEM_register_buf(input.data);
	submit_task_input(input);            // lower num_idle_slaves
      } else if (aggregated_inputs_pending) {
	submit_task_input(NOTASK);           // even by sending pending inputs
      } else {                             // or else insure progress condition
	if (num_idle_slaves + num_dead_slaves < num_slaves)
	  receive_task_output();             // by blocking until slave reply
	else break;
      }
    } // termination condition:  _after_ all slaves idle, next input was NOTASK
    assert( (input.data == NOTASK.data)
	    && (num_idle_slaves + num_dead_slaves == num_slaves));

    //Check if user has not supplied any tasks for this run
    if (num_tasks == 0)
      WARNING("GenerateTaskInput() returned null input on first invocation");

    for (slave = 1; slave <= num_slaves; slave++) {
      update_slave(slave); // Needed in case we call TOPC_master_slave again.
      if (TOPC_OPT_aggregated_tasks > 1) {
	slave_array[slave].aggreg_inputs.data_size = 0;
	free(slave_array[slave].aggreg_inputs.data);
	slave_array[slave].aggreg_inputs.data = NULL;
      }
      if (COMM_send_msg(NULL, 0, slave, END_MASTER_SLAVE_TAG) == FALSE)
	slave_is_dead(slave);
    }
  }

  //Slave loop
  else slave_loop();
}

/***********************************************************************
 ***********************************************************************
 ** Soft abort
 ***********************************************************************
 ***********************************************************************/
void TOPC_abort_tasks() {
  if (!TOPC_is_master())
    ERROR("TOPC_abort_tasks called on a slave");
  if (TOPC_OPT_safety < SAFETY_NO_ABORT_TASKS)
    COMM_soft_abort();
}

/* What about in UpdateSharedData() on master? */
/* Should we restrict it to is_in_do_task, and add static version in topc.c? */
TOPC_BOOL TOPC_is_abort_pending() {
  if (TOPC_is_master())
    WARNING("TOPC_is_abort_pending:  called on master; result not useful");
  return COMM_is_abort_pending;
}

/***********************************************************************
 ***********************************************************************
 ** Alternate TOPC_raw master-slave interface:
 **
 **   The idea is that a traditional generate_task_input() is a kind of
 **   iterator that needs to be provided by application code.  However,
 **   sometimes the original sequential code produces task inputs
 **   inside of complicated nested loops.  In such cases, it is
 **   difficult to create a corresponding iterator (and co-routines or
 **   threads would in fact be the ideal language construct).
 **
 **   To get around this, we replace a single call to master_slave() by:
 **     TOPC_raw_begin_master(do_task, check_task_result, update_shared_data)
 **     TOPC_raw_submit_task_input(input)
 **     TOPC_raw_wait_for_task_result()
 **     TOPC_raw_end_master()
 **
 **   The application can then call submit_task_input() repeatedly with
 **   the new task inputs, before completing the computation by a call
 **   to TOPC_end_master_slave().
 **
 **   Slave only calls TOPC_raw_begin_master_slave() and
 **   TOPC_raw_end_master_slave(), whereas slave loop is in the former,
 **   and the latter does nothing
 ***********************************************************************
 ***********************************************************************/

/***********************************************************************
 * Start parallel execution on master
 */
void TOPC_raw_begin_master_slave(
  TOPC_BUF (*do_task_)(),              /* args: (void *input) */
  TOPC_ACTION (*check_task_result_)(), /* args: (void *input, void *output) */
  void (*update_shared_data_)()        /* args: (void *input, void *output) */
) {
  int stack_marker;
  int slave;

  //Used in calls to COMM_is_on_stack()
  //Must mark a point on stack before calls to do_task
  //Beware of segmented memory models, where pointer comparisons in different
  //  segments may not work as expected
  COMM_stack_bottom = &stack_marker;

  if (!COMM_initialized())
    ERROR("TOPC_raw_begin_master_slave called before TOPC_init()");

  //Create buffers for aggregated tasks
  if (TOPC_OPT_aggregated_tasks > 1)
    for (slave = 1; slave <= num_slaves; ++slave) {
      slave_array[slave].aggreg_inputs.data_size = 0; // no aggreg tasks yet
      slave_array[slave].aggreg_inputs.data = malloc(AGGREG_TASKS_SIZE);
    }

  // xxx_orig() is called by COMM_xxx for xxx a callback function.
  do_task_orig              = do_task_;
  check_task_result_orig    = check_task_result_;
  update_shared_data_orig   = update_shared_data_;
  COMM_slave_loop           = slave_loop; // needed for pthreads/SMP COMM layer

  if (!TOPC_is_master())
    slave_loop();
}

/***********************************************************************
 * End parallel execution on master
 */
void TOPC_raw_end_master_slave() {
  int slave;

  if (TOPC_is_master()) {
    if (num_tasks == 0)
     WARNING("TOPC_raw_submit_task_input() never called\n"
             "     called (or returned NOTASK).");
    if (aggregated_inputs_pending) {
      if (num_idle_slaves == 0)
	receive_task_output();
      assert(num_idle_slaves > 0);
      submit_task_input(NOTASK);
    }
    while (num_idle_slaves + num_dead_slaves < num_slaves)
      receive_task_output();

    for (slave = 1; slave <= num_slaves; ++slave) {
      if (slave_array[slave].dead == FALSE) {
	update_slave(slave);  //Needed in case we call TOPC_master_slave again.
	if (COMM_send_msg(NULL, 0, slave, END_MASTER_SLAVE_TAG) == FALSE)
	  slave_is_dead(slave);
      }
      if (TOPC_OPT_aggregated_tasks > 1) {
        slave_array[slave].aggreg_inputs.data_size = 0;
        free(slave_array[slave].aggreg_inputs.data);
        slave_array[slave].aggreg_inputs.data = NULL;
      }
    }
  }
}

/***********************************************************************
 * Submit task input to TOPC for parallel execution
 */
void TOPC_raw_submit_task_input(TOPC_BUF input) {
  static int is_warned = 0;
  if (TOPC_is_master() && input.data != NOTASK.data) {
    /* argument, input, is result of TOPC_MSG/TOPC_MSG_PTR */
    MEM_register_buf(input.data);
    wait_until_an_idle_slave();
    submit_task_input(input);
  } else if (!TOPC_is_master() && !is_warned) {
    is_warned = 1;
    WARNING("TOPC_raw_submit_task_input() called from slave.\n"
            "       Sample code showing correct usage:\n"
            "***  TOPC__raw_begin_master_slave(...);\n"
            "***  if ( TOPC_is_master() ) {\n"
            "***    ... TOPC_raw_submit_task_input(...) ...\n"
            "***  }\n"
            "***  TOPC_raw_end_master_slave()\n");
  }
}

/***********************************************************************
 * If pending task exists, wait for it, call CheckTaskResult() and return true
 * If no pending task exists, return false
 */
TOPC_BOOL TOPC_raw_wait_for_task_result() {
  if (num_idle_slaves < num_slaves) {
    do {
      receive_task_output();
    } while (COMM_probe_msg() || num_idle_slaves == 0);
    // NOTE:  COMM_probe_msg() is non-blocking.
    return TRUE;
  }
  else return FALSE;
}
