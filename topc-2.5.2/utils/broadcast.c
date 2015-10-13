// Implements
//   void TOPC_broadcast(void (*)(void *) update_data, void *data, int size);
// This code assumes that TOP-C accesses idle slaves in round robin fashion

/* IMPORTANT:  For safety, should have check that master and slave are
 * both using broadcast in the current TOPC_master_slave
 * and notify user if not true.  Probably, exchange some tags.
 * */

#include <assert.h>
#include "topc.h"

static void *bcast_data;
static int bcast_data_size;
static void (*bcast_do_it)(void *data);

static TOPC_BUF bcast_gen_input() {
  static int i = 0;

  ++i;
  if (i == 1 + 1)
    bcast_data = NULL; /* then we've finished bcast_data request */
  else if (i > 1 + 1 && bcast_data != NULL) /* if new bcast_data request */
    i = 1;

  if (i > 1)
    return NOTASK;
  else
    return TOPC_MSG_PTR(bcast_data, bcast_data_size);
}

static TOPC_BUF bcast_do_task(void *data) {
  return TOPC_MSG(NULL, 0);
}

static TOPC_ACTION bcast_check_result(void *input, void *output) {
  return UPDATE;
}

static void bcast_update_data(void *data, void *output) {
  (*bcast_do_it)(data);
}

// TOPC_broadcast executes:  (*update_data)(data);
//  on all processes (including master).
void TOPC_broadcast(void (*update_data)(void *), void *data, int size) {
  int save_trace = TOPC_OPT_trace;
  TOPC_OPT_trace= 0;
  bcast_do_it = update_data;
  bcast_data = data;
  bcast_data_size = size;
  TOPC_master_slave(bcast_gen_input, bcast_do_task, bcast_check_result,
                    bcast_update_data);
  TOPC_OPT_trace = save_trace;
}

//*************************************************************************
// Example usage:
// static struct stats { int rank; struct rusage usage; } mystat;
// static void *get_stat()
//   { getrusage(RUSAGE_SELF, &mystat.usage); mystat.rank = TOPC_rank();
//     return &mystat; }
// struct stats *stat_array
//   = malloc(sizeof(struct stats) * (TOPC_num_slaves()+1));
// TOPC_gather( get_stats, stat_array, sizeof(struct stats) );
// Now, stat_array[5].usage indicates usage of slave stat_array[5].rank .
// Do we want:  TOPC_get_last_rank() ?

static int gather_is_done = 1;
static void * (*gather_do_it)(void);
static void *gather_array;
static int gather_array_elt_size;

// This code assumes TOP-C sends successive requests to successive slaves.
// Luckily, that is exactly what TOP-C does.
static TOPC_BUF gather_gen_input() {
  static int i = 0;
  static int num_slaves = -1;

  if (num_slaves == -1) num_slaves = TOPC_num_slaves();

  ++i;
  if (i == num_slaves + 1)
    gather_is_done = 1; /* then we've finished gather_array request */
  else if (i > num_slaves + 1 && ! gather_is_done) /* if new gather_array req */
    i = 1;

  if (i > num_slaves)
    return NOTASK;
  else
    return TOPC_MSG(NULL, 0);
}

static TOPC_BUF gather_do_task(void *input) {
  return TOPC_MSG( (*gather_do_it)(), gather_array_elt_size );
}

// gatherion_array has size at least:  size*(TOPC_num_slaves()+1)
#define gather_array_sub(i) (void *)((char *)gather_array + i*gather_array_elt_size)

static TOPC_ACTION gather_check_result(void *input, void *output) {
  static int slave = 0;
  ++slave;
  if (slave > TOPC_num_slaves()) slave = 1;
  memcpy(gather_array_sub(slave), output, gather_array_elt_size);
  return NO_ACTION;
}

void TOPC_gather(void * (*gather_data)(void),
                  void *gatherion_array, int elt_size) {
  int save_trace = TOPC_OPT_trace;
  TOPC_OPT_trace= 0;
  gather_is_done = 0;
  gather_do_it = gather_data;
  gather_array = gatherion_array;
  gather_array_elt_size = elt_size;
  memcpy(gather_array_sub(0), gather_data(), elt_size); // Do it on master
  // and do it from slaves
  TOPC_master_slave(gather_gen_input, gather_do_task, gather_check_result, NULL);
  TOPC_OPT_trace = save_trace;
}

//*************************************************************************

#include <stdlib.h>
// EXAMPLE:  sync. random state among multiple processes
//  Useful to do after TOPC_master_slave, when seeds may no longer sync.
//  If random() called from within do_task() and want reproducibility,
//    then set seed deterministically to next pseudo-random number
//    on master just before do_task, and then copy that seed to slave
//    just before slave executes do_task.
#if 0
static void set_random_state(void *state) {
  setstate(state);
}
#define SEED_SIZE sizeof(unsigned int)
static char *get_random_state() {
  char tmpstate[SEED_SIZE], *oldstate;
  oldstate = setstate(tmpstate);
  assert(oldstate != NULL);
  setstate(oldstate);
  return oldstate;
}
#else
static void set_random_state(void *state) {
  srandom(*(unsigned int *)state);
}
static unsigned int new_seed;
#define SEED_SIZE sizeof(new_seed)
static unsigned int *get_random_state() {
  new_seed = random();
  return &new_seed;
}
#endif

static void sync_random_state() {
  TOPC_broadcast(set_random_state, get_random_state(), SEED_SIZE);
}
