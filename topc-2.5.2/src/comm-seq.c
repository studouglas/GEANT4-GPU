/**********************************************************************
  * TOP-C (Task Oriented Parallel C)                                   *
  * Copyright (c) 2000 Gene Cooperman <gene@ccs.neu.edu>               *
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
#include <stdlib.h> // malloc()
#include "comm.h"    // comm.h also defines HAVE_PTHREAD, etc., by default

// NOTE:  MPI returns MPI_SUCCESS or application-dependent indication
//        of failure.  We may MPI_SUCCESS to TRUE and others to FALSE.
#ifndef FALSE
enum {FALSE, TRUE};
#endif

char *COMM_mem_model = "sequential";


/***********************************************************************
 ***********************************************************************
 ** Communications and MPI functions for sequential layer
 ** (it assumes 1 slave, and we're always excuting on master)
 ***********************************************************************
 ***********************************************************************/


/***********************************************************************
 * private COMM variables to emulate MPI, messages
 * (could emulate several slaves by using arrays here)
 */
static int is_initialized = 0;
static int is_output_msg_available = 0;
static void *output_msg = NULL;
static size_t output_msg_size = 999999;

/***********************************************************************
 * Initialize underlying communications library
 */
void COMM_init(int *adummy_rgc, char ***dummy_argv) {
  is_initialized = 1;
}

/***********************************************************************
 * Shutdown underlying communications library
 */
void COMM_finalize() {
  is_initialized = 0;
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
  return 2;  // master and slave
}

/***********************************************************************
 * Get rank of this node
 */
static int is_in_do_task = 0;

int COMM_rank() {
  if (is_in_do_task)
    return 1;
  else
    return 0;
}

/***********************************************************************
 * Return boolean, true if master and slave share memory
 */
int COMM_is_shared_memory() {
  return 0;  // false
}

/***********************************************************************
 * Return boolean, true if master and slave share memory
 */
void *COMM_stack_bottom = NULL; // set in topc.c

TOPC_BOOL COMM_is_on_stack(void *buf, size_t dummy_size) {
  int stack_top[8];  /* This won't be put in register */

  if ( (buf >= (void *)stack_top && buf <= COMM_stack_bottom)
       || (buf <= (void *)stack_top && buf >= COMM_stack_bottom) )
    return 1; // true
  else
    return 0; // false
}

/***********************************************************************
 * Set via SOFT_ABORT_TAG
 */
//read on slave by TOPC_is_abort_pending()
volatile int COMM_is_abort_pending = 0;
// called on master by TOPC_abort()
void COMM_soft_abort() {
  COMM_is_abort_pending = 0;
  }

/***********************************************************************
 * Check if a message from a slave is instantly available (non-blocking)
 */
TOPC_BOOL COMM_probe_msg() {
  return is_output_msg_available;
}

/***********************************************************************
 * Check for a message from a slave (blocking)
 */
TOPC_BOOL COMM_probe_msg_blocking(int dummy_rank) {
  if ( ! is_output_msg_available )
    ERROR("COMM_probe_msg_blocking:  Deadlock."
          "  Blocking and no msg available from slave.\n");
  return 1;
}

/***********************************************************************
 * Send message
 */
TOPC_BOOL COMM_send_msg(void *msg, size_t dummy_msg_size, int dummy_dst,
			enum TAG tag) {
  TOPC_BUF buf;

  // always on master;  always to slave
  switch( tag ) {
    case TASK_INPUT_TAG:
    case REDO_INPUT_TAG:
    case CONTINUATION_INPUT_TAG:
      is_in_do_task = 1;
      buf = (*COMM_do_task)(msg);
      if (TOPC_OPT_aggregated_tasks <= 1)
	MEM_register_buf(buf.data);
       is_in_do_task = 0;
      is_output_msg_available = 1;
      // flow through to next case
    case PING_TAG:
      output_msg      = buf.data;
      output_msg_size = buf.data_size; // Shouldn't be used by master_slave()
    default:  // for other cases, ignore the message
      ;
  }
  return 1;
}

/***********************************************************************
 * Receive message into malloc'ed memory (blocking call)
 * receive from anybody, and set each of four param's if non-NULL
 */
TOPC_BOOL COMM_receive_msg(void **msg, size_t *msg_size, int *src,
			   enum TAG *tag) {
  if ( ! is_output_msg_available )
    ERROR("COMM_receive_msg:  Deadlock."
           "  Blocking and no msg available from slave.\n");
  // always on master;  always from slave
  if (msg) *msg           = output_msg;
  if (msg_size) *msg_size = output_msg_size;
  if (src) *src           = 1;
  if (tag) *tag           = SLAVE_REPLY_TAG;
  is_output_msg_available = 0;
  return TRUE;
}

// balances MEM_malloc() in COMM_receive_msg(), no malloc() for seq
void COMM_free_receive_buf(void *dummy_msg) {
  return;
}

/***********************************************************************
 ***********************************************************************
 * SMP-related functions (e.g.:  private (non-shared) variables)
 ***********************************************************************
 ***********************************************************************/

void *COMM_thread_private(size_t size) { // trivial for non-shared memory
  static void *master_thr_priv = NULL, *slave_thr_priv = NULL;
  if ( slave_thr_priv == NULL ) {
    master_thr_priv = malloc(size);
    slave_thr_priv = malloc(size);
    if (size >= sizeof(void *)) {
      *(void **)master_thr_priv = NULL;
      *(void **)slave_thr_priv = NULL;
    }
  }
  if (is_in_do_task)
    return slave_thr_priv;
  else
    return master_thr_priv;
}
void COMM_begin_atomic_read() {}
void COMM_end_atomic_read() {}
void COMM_begin_atomic_write() {}
void COMM_end_atomic_write() {}
TOPC_BOOL COMM_is_in_atomic_section() {
  static int is_in = 0;
  return (is_in = 1 - is_in);
}
