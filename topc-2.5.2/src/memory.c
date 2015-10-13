 /**********************************************************************
  * TOP-C (Task Oriented Parallel C)                                   *
  * Copyright (c) 2000-2001 Gene Cooperman <gene@ccs.neu.edu>          *
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

/* EXPORTED FUNCTIONS:  TOPC_MSG, TOPC_MSG_PTR, MEM_malloc, MEM_register_buf,
 *                                                          MEM_free
 * This code does malloc/free while recording special header info.
 *   After GenerateTaskInput/DoTask is called, MEM_register_buf()
 *   verifies that TOPC_MSG()/TOPC_MSG_PTR() was called at least once.
 *   If called more than once, extra buffers are silently freed.
 *                                                          MEM_free
 * The code must deal with two kinds of buffers:
 *   (1) buffers dynamically allocated by MEM_malloc() via a call to malloc()
 *   (2) buffers located in application space (including NULL pointers)
 *       for which TOP-C creates a header.
 * Buffers of type (2) are created by the application and arise either through
 *   calls to TOPC_MSG_PTR() or in cases where TOP-C can prove it doesn't
 *   need to copy the messages into TOP-C space.
 *
 * Typical sequence:  TOPC_MSG/MEM_malloc, MEM_register_buf, MEM_free
 *
 * A buffer of type (1) or (2) has an associated struct mem_hdr, whose
 *   type field indicated its current state.
 * State transitions:  IS_MALLOC->IS_PRE_MSG->IS_MSG->UNUSED_HDR
 *                or:  IS_PRE_MSG_PTR->IS_MSG_PTR->UNUSED_HDR
 * Function called resulting in new state:
 * MEM_malloc()->IS_MALLOC->TOPC_MSG()->IS_PRE_MSG
 *             ->MEM_register_buf()->IS_MSG
 *             ->MEM_free()->UNUSED_HDR (and frees it)
 *
 * mem_hdr's implemented as doubly linked list
 *     (eventually, want to turn it into circular buffer):
 *   mem_hdr_start==mem_hdr_head      mem_hdr_register     mem_hdr_tail
 *                                      <--prev              next-->
 *     MEM_malloc enqueues new mem_hdr as mem_hdr_tail->next,
 *          and moves mem_hdr_tail toward it.
 *          Call to malloc() done within mutex crit. section.
 *     MEM_register_buf moves toward `next', and examines each mem_hdr.
 *     mem_hdr_head, mem_hdr_register, mem_hdr_tail always move toward `next'
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "comm.h"    // comm.h also defines HAVE_PTHREAD, etc., by default
#ifdef HAVE_PTHREAD
# include <pthread.h>
#endif

/***********************************************************************
 * Define reentrant version of mutex_lock/mutex_unlock
 *   In distributed memory, could remove mutexes (and for master in
 *   shared memory), but these are cheap.
 */
#ifdef HAVE_PTHREAD
static pthread_mutex_t mem_mgr = PTHREAD_MUTEX_INITIALIZER;
# ifdef DEBUG
#  define DEBUG_CHECK debug_check();
# else
#  define DEBUG_CHECK
# endif
/* MUTEX_LOCK/MUTEX_UNLOCK let thread recursively call lock w/o blocking */
  static volatile int recurse_init = 0;
  static pthread_key_t recurse_key;
# define MUTEX_LOCK(addr_mutex) \
{ int *recurse = NULL; \
  if (recurse_init && (recurse = pthread_getspecific(recurse_key)) != NULL \
      && *recurse > 0) \
    *recurse++; \
  else { \
    pthread_mutex_lock(addr_mutex); \
    if ( recurse == NULL ) { \
      if (! recurse_init) { \
        recurse_init = 1; \
        pthread_key_create( &recurse_key, NULL ); \
      } \
      pthread_setspecific(recurse_key, recurse = malloc(sizeof(int))); \
      if ( ! recurse ) PERROR("malloc"); \
      else *recurse = 0; \
    } \
    (*recurse)++; \
    DEBUG_CHECK; \
  } \
}
# define MUTEX_UNLOCK(addr_mutex) \
  if ( --(*(int *)pthread_getspecific(recurse_key)) <= 0 ) \
  { \
    DEBUG_CHECK; \
    pthread_mutex_unlock(addr_mutex); \
  }
#else
# define MUTEX_LOCK(addr_mutex)
# define MUTEX_UNLOCK(addr_mutex)
# undef volatile
# define volatile
#endif

/***********************************************************************
 * Access functions for a message
 */

/* msg = (hdr, buf); Want buf correctly aligned after
 * msg=malloc(sizeof_msg(size)); buf=((struct mem_hdr *)msg)+1;
 */
#define ROUNDUP(x) ( (((x)+sizeof(void*)-1) / sizeof(void*)) * sizeof(void*) )
#define sizeof_msg_header ( ROUNDUP(sizeof(struct mem_hdr)) )
#define sizeof_msg(size) ( sizeof_msg_header + ROUNDUP(size) )

/* (actual declaration in comm.h) */
// enum hdr_type {
//   UNUSED_HDR, IS_MALLOC, IS_PRE_MSG, IS_PRE_MSG_PTR,
//  IS_MSG, IS_MSG_PTR };

/* mem_hdr is not in use if buf == NULL */
static struct mem_hdr {
  struct mem_hdr *next;
  struct mem_hdr *prev;
  void *buf;
  enum hdr_type type;
  enum TAG tag;
  int source;
#ifdef HAVE_PTHREAD
  pthread_t owner;
#endif
  int size;
  int id;
} mem_hdr_start
#ifdef HAVE_PTHREAD
      = {NULL, NULL, NULL, UNUSED_HDR, NO_TAG, -1, (pthread_t)-1, -1, -1};
#else
      = {NULL, NULL, NULL, UNUSED_HDR, NO_TAG, -1,     -1, -1};
#endif

/* mem_hdr_head always points to empty mem_hdr_start; acts only as marker */
/* mem_hdr_register points to last hdr that has been checked */
static volatile struct mem_hdr *mem_hdr_head = &mem_hdr_start,
                               *mem_hdr_tail = &mem_hdr_start,
                               *mem_hdr_register = &mem_hdr_start;
static struct mem_hdr * MEM_buf_hdr(void *buf) /* used for debugging */
  { return (((struct mem_hdr *)(buf)) - 1); }
#define MEM_buf_hdr(buf) (((struct mem_hdr *)(buf)) - 1)
#define buf_of_hdr(hdr) ((void *)(((struct mem_hdr *)(hdr)) + 1))
#define MEM_buf_type(buf) (MEM_buf_hdr(buf)->type)
#define MEM_buf_tag(buf) (MEM_buf_hdr(buf)->tag)
#define MEM_buf_source(buf) (MEM_buf_hdr(buf)->source)
/* Note this is size of buffer, not including header.
 * For including pointer to header in size, use sizeof_msg(MEM_buf_size(buf)).
 */
#define MEM_buf_size(buf) (MEM_buf_hdr(buf)->size)

static void debug_check() {
  struct mem_hdr *hdr;
  for (hdr = (struct mem_hdr *)mem_hdr_head; hdr != NULL; hdr = hdr->next)
    if ( ((long)(hdr->next) > 0 && (long)(hdr->next) < 1000)
         || ((long)(hdr->prev) > 0 && (long)(hdr->prev) < 1000) )
      ERROR("Bad hdr->next");
  for (hdr = (struct mem_hdr *)mem_hdr_register; hdr != NULL ; hdr = hdr->prev)
    if (hdr->type == IS_PRE_MSG_PTR || hdr->type == IS_PRE_MSG)
      ERROR("debug_check:  mem_hdr_register overrun");
}

/***********************************************************************
 * malloc a buffer + header
 */
void *MEM_malloc(size_t size, int source, enum TAG tag, enum hdr_type type) {
  static int id = 0;
  struct mem_hdr *hdr;
  if (TOPC_OPT_safety >= SAFETY_NO_MEM_MGR) {
    void *ptr;
    if (size == 0)
      return NULL;
    MUTEX_LOCK(&mem_mgr);
    ptr = malloc(size);
    MUTEX_UNLOCK(&mem_mgr);
    if (ptr == NULL && size > 0)
      PERROR("malloc");
    else
      return ptr;
  }
  assert(tag != NO_TAG);
  MUTEX_LOCK(&mem_mgr);

  /* enqueues(malloc((sizeof_msg(size)); */
  assert(mem_hdr_tail->next == NULL);
  mem_hdr_tail->next = malloc(sizeof_msg(size));
  if (mem_hdr_tail->next == NULL)
    ERROR("No memory to malloc a new buffer.\n");
                             /* cast to discard volatile */
  mem_hdr_tail->next->prev = (struct mem_hdr *)mem_hdr_tail;
  mem_hdr_tail = mem_hdr_tail->next;
  mem_hdr_tail->next = NULL;
  hdr = (struct mem_hdr *)mem_hdr_tail; /* cast to discard volatile */
  hdr->id = id; /* used for debugging */
  if (id++ >= 1000000000) id = 0;
  assert(hdr->prev->next == hdr);
  assert(hdr->next == NULL);

  hdr->tag = tag;
  hdr->source = source;
  hdr->size = size;
  hdr->type = type;
#ifdef HAVE_PTHREAD
  hdr->owner = pthread_self();
#endif
  hdr->buf = buf_of_hdr(hdr);
  MUTEX_UNLOCK(&mem_mgr); /* MEM_register_buf needs above fields stable */
  assert(MEM_buf_hdr(hdr->buf) == hdr);
# ifdef DEBUG
printf("rank: %d, mem_hdr_head: %x, mem_hdr_head->next: %x, mem_hdr_tail: %x\n",
COMM_rank(), mem_hdr_head, mem_hdr_head->next, mem_hdr_tail);
  printf("rank: %d; buf_of_hdr(hdr): %x, size: %d, source: %d, tag: %d,"
         " type: %d, id: %d\n",
         COMM_rank(), buf_of_hdr(hdr), size, source, tag, type, hdr->id);
# endif
  return hdr->buf;
}
void *MEM_malloc_ptr(void *buf, size_t size, int source, enum TAG tag,
                     enum hdr_type type) {
  struct mem_hdr *hdr;
  if (TOPC_OPT_safety >= SAFETY_NO_MEM_MGR)
    return buf;
  hdr = MEM_buf_hdr(MEM_malloc(0, source, tag, type));
  hdr->buf = buf;
  hdr->size = size;
  return buf;
}

/***********************************************************************
 * Free any buffer created by MEM_malloc
 *  (only routine to change type to UNUSED_HDR)
 */
void MEM_free(void *buf) {
  struct mem_hdr *hdr = MEM_buf_hdr(buf);

  if (TOPC_OPT_safety >= SAFETY_NO_MEM_MGR) {
    if (buf != NULL)
      free(buf);
    return;
  }
  MUTEX_LOCK(&mem_mgr);
# ifdef DEBUG
printf("rank: %d, mem_hdr_head: %x, mem_hdr_head->next: %x, mem_hdr_tail: %x\n",
COMM_rank(), mem_hdr_head, mem_hdr_head->next, mem_hdr_tail);
  printf("rank: %d; owner: %d, free msg: %x, size: %d, source: %d, tag: %d"
         ", id: %d\n",
         COMM_rank(),
#ifdef HAVE_PTHREAD
	 MEM_buf_hdr(buf)->owner,
#endif
	 buf, MEM_buf_size(buf),
         MEM_buf_source(buf), MEM_buf_tag(buf), MEM_buf_hdr(buf)->id);
# endif
  if (hdr != mem_hdr_tail || hdr->buf != buf) {
    /* This logic for IS_MSG_PTR;  Probably efficient enough */
    hdr = (struct mem_hdr *)mem_hdr_tail; /* cast to discard volatile */
    while (hdr->buf != buf && hdr != mem_hdr_head)
      hdr = hdr->prev;
    assert(hdr != mem_hdr_head);
  }
  assert(hdr->buf == buf);

  /* dequeue(hdr) */
  assert(hdr != mem_hdr_head); /* mem_hdr_head only marker for head of queue */
  assert(hdr->type != UNUSED_HDR);
  assert(hdr->prev->next == hdr);
  assert(hdr->next == NULL || hdr->next->prev == hdr);
  hdr->type = UNUSED_HDR;
  hdr->prev->next = hdr->next;
  if (mem_hdr_register == hdr)
    mem_hdr_register = hdr->prev;
  if (hdr->next == NULL) {
    assert(hdr == mem_hdr_tail);
    mem_hdr_tail = hdr->prev;
    mem_hdr_tail->next = NULL;
  }
  else
    hdr->next->prev = hdr->prev;
  free(hdr);
  assert(mem_hdr_register == mem_hdr_tail || mem_hdr_register->next != NULL);
  MUTEX_UNLOCK(&mem_mgr);
}

/***********************************************************************
 * Called after TOPC_MSG/TOPC_MSG_PTR
 * Since TOPC_MSG() can MEM_malloc a new buffer that TOP-C frees,
 *   it's important that application not trick us by returning
 *   an old buffer.  So, after callback function calls TOPC_MSG()
 *   TOP-C always calls MEM_register_buf().
 * (MEM_register_buf and MEM_free only routines to modify mem_hdr_register)
 * (MEM_register_buf calls MEM_free, only routines
 *   to change type to UNUSED_HDR)
 */
static void msg_not_found() {
  ERROR("MEM_register_buf:  Return value of %s at rank %d does\n"
        "      not appear to be a new invocation of TOPC_MSG/TOPC_MSG_PTR.\n"
        "      Must return new invocation or NOTASK\n",
        (COMM_rank() == 0 ? "GenerateTaskInput()" : "DoTask()"), COMM_rank() );
}
void MEM_register_buf(void *buf) {
  int found = 0;
#ifdef HAVE_PTHREAD
  pthread_t tid = pthread_self();
#endif
  struct mem_hdr *hdr_register, *next_hdr_register;
  extern TOPC_BUF NOTASK;

  if (TOPC_OPT_safety >= SAFETY_NO_MEM_MGR) {
    return;
  }
  MUTEX_LOCK(&mem_mgr);
  next_hdr_register = NULL;
  for (hdr_register = (struct mem_hdr *)mem_hdr_register->next;
       hdr_register != NULL;
       hdr_register = (struct mem_hdr *)hdr_register->next) { /*cast, disc. vol*/
#ifdef HAVE_PTHREAD
    if ( pthread_equal(hdr_register->owner, tid) ) {
#else
    if (1) {
#endif
      switch (hdr_register->type) {
      case IS_PRE_MSG:
      case IS_PRE_MSG_PTR:
        if (hdr_register->buf == buf) { /* if it's this buf */
          hdr_register->type =
            (hdr_register->type==IS_PRE_MSG ? IS_MSG : IS_MSG_PTR);
          found = 1;
        } else {
          assert(hdr_register->prev->next == hdr_register);
          hdr_register = (struct mem_hdr *)hdr_register->prev; /*cast, disc. vol*/
	  fprintf(stderr, "*** TOP-C: MEM_register_buf:\n"
	                  "      WARNING: application callback function"
		          " called TOPC_MSG() twice.\n");
          MEM_free(hdr_register->next->buf); /* applic. created extra one */
        }
        break;
      case IS_MSG_PTR: /* MEM_register_buf seeing these a second time */
      case IS_MSG:
        assert(hdr_register->type == IS_MSG_PTR || hdr_register->buf != buf);
        break;
      default:
        assert(hdr_register == mem_hdr_head);
      }
    } else if (next_hdr_register == NULL
               && hdr_register->type != IS_MSG
               && hdr_register->type != IS_MSG_PTR) {
      /* else somebody else's header, mem_hdr_register will stop here */
      assert(hdr_register->type == IS_PRE_MSG
             || hdr_register->type == IS_PRE_MSG_PTR);
      assert(hdr_register->buf != buf); /* If we don't own it, not our buf */
#     ifdef DEBUG

#      ifdef HAVE_PTHREAD
      printf("thread id %d: hdr_register found IS_MSG_PTR or IS_MSG"
	     " from owner %d\n", tid, hdr_register->owner);
#      endif
#     endif
      next_hdr_register = hdr_register->prev;
    }
  }
  mem_hdr_register = (next_hdr_register==NULL ? mem_hdr_tail : next_hdr_register);
  assert(hdr_register == NULL);
  assert(mem_hdr_register == mem_hdr_tail || mem_hdr_register->next != NULL);
  MUTEX_UNLOCK(&mem_mgr);

  if (! found && buf != NOTASK.data)
    msg_not_found();
  return;
}

/***********************************************************************
 * Creates a TOPC_BUF; copies data from user space to TOP-C space.
 */
TOPC_BUF TOPC_MSG(void *data, size_t data_size) {
  if (data == NULL && data_size > 0)
    ERROR("TOPC_MSG() called with null pointer and positive size");
  // Always copy to TOP-C space on master; It's called by GenerateTask()
  //   for input task buffer, and saved for reuse by CheckTaskResult()
  if ( TOPC_OPT_safety < SAFETY_NO_MSG_PTR && COMM_rank() != 0
       && ! COMM_is_shared_memory() && ! COMM_is_on_stack(data, data_size) ) {
    /* for size 0, malloc a header only; equiv of TOPC_MSG_PTR() */
    MEM_malloc_ptr( data, data_size, COMM_rank(), MEM_TAG, IS_PRE_MSG_PTR );
    { TOPC_BUF tmp;
      tmp.data = data; tmp.data_size = data_size;
      return tmp;
    }
  } else {
    data = memcpy( MEM_malloc( data_size, COMM_rank(), MEM_TAG, IS_PRE_MSG ),
                   data, data_size );
    { TOPC_BUF tmp;
      tmp.data = data; tmp.data_size = data_size;
      return tmp;
    }
  }
}

/***********************************************************************
 * Creates a TOPC_BUF with pointer to buffer in user space.
 * TOPC_MSG() is safer;  Use this only for optimizations.
 */
TOPC_BUF TOPC_MSG_PTR(void *data, size_t data_size) {
  if (TOPC_OPT_safety >= SAFETY_NO_MSG_PTR
      || ( TOPC_OPT_aggregated_tasks > 1
	   && ! COMM_is_on_stack(data, data_size) ) )
    return TOPC_MSG(data, data_size);
  if (data == NULL && data_size > 0)
    ERROR("TOPC_MSG_PTR() called with null pointer and positive size");
  /* size 0, malloc a header only */
  MEM_malloc_ptr( data, data_size, COMM_rank(), MEM_TAG, IS_PRE_MSG_PTR );
  if (COMM_is_on_stack(data, data_size))
    WARNING("TOPC_MSG_PTR() was passed a pointer into stack on %s.\n%s",
            ( COMM_rank() == 0 ? "master" : "slave" ),
            "  Data buffer should be either static or global.");
  { TOPC_BUF tmp;
    tmp.data = data; tmp.data_size = data_size;
    return tmp;
  }
}
