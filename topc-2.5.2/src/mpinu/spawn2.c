#include "mpi.h"
#include "mpiimpl.h"

int MPI_Spawn2() {
  int sd, ns;                /* socket descriptors            */
  struct init_msg msg1;
  int fromlen, rank;
  struct sockaddr_in new_sin;   /* used for each new connection   */

  //Leave the loader alone if no space in host table
  if (MPINU_num_slaves == PG_ARRAY_SIZE) return MPI_FAIL;

  //Start up new slave process
  MPINU_is_spawn2 = 1;
  MPINU_is_initialized = 0;
  do_unexec();
  MPINU_is_initialized = 1;
  MPINU_is_spawn2 = 0;
  if (!attach_new_slaves()) return MPI_FAIL;

  //Allocate new slave record. Easy
  rank = ++MPINU_num_slaves;

  //Accept connection from new slave, remember it
  fromlen = sizeof(new_sin);
  CALL_CHK( ns = accept, (MPINU_my_list_sd,
			  (struct sockaddr *)&new_sin, &fromlen) );
  MPINU_pg_array[rank].sd = ns;
  FD_SET(ns, &MPINU_fdset);
  if ( ns > MPINU_max_sd )
    MPINU_max_sd = ns;
#ifdef DEBUG
  printf("master:  accepted new slave connection\n");
  printf("master:  slave %d: new ns:  %d, current MPINU_max_sd: %d\n",
	 rank, ns, MPINU_max_sd);
#endif

  //Receive slave's listener_addr and store in field of MPINU_pg_array[rank]
  CALL_CHK( recv, (ns, (char *)&(MPINU_pg_array[rank].listener_addr),
            sizeof(struct sockaddr_in), 0) );

  //Send back number of slaves and the assigned rank
  msg1.len = htonl(sizeof(msg1));
  msg1.rank = htonl(rank);
  msg1.num_slaves = htonl(MPINU_num_slaves);
  send(ns, (char *)&msg1, sizeof(msg1), 0);

  //Send listener_addr fields to slave
  { char buf[PG_ARRAY_SIZE * sizeof(struct sockaddr_in)], *ptr;
    int i;

    ptr = buf;
    for ( i=0; i <= MPINU_num_slaves; i++ ) {
      memcpy( ptr, (char *)&(MPINU_pg_array[i].listener_addr),
	           sizeof(struct sockaddr_in) );
      ptr += sizeof(struct sockaddr_in);
    }
    send( MPINU_pg_array[rank].sd, buf,
          (1+MPINU_num_slaves)*sizeof(struct sockaddr_in), 0 );

    //Acknowledge that master and slave are synchronized
    *((INT *)buf) = htonl( sizeof(struct sockaddr_in) );
    send( MPINU_pg_array[rank].sd, buf, sizeof(INT), 0);
  }
}
