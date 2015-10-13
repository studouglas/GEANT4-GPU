# This file is used for TOP-C development, only -- not for making TOP-C
# Process this file with autoconf to produce a configure script.
# For debugging, invoke as:  env CFLAGS=-g ./configure
# For testing portability, invoke as:  env "CFLAGS=-Wall -W" ./configure
# For alternate C compiler, invoke as:
#                           env CC=cc ./configure --cache-file=/dev/null

AC_INIT(TOP-C,0.0.0,[gene@ccs.neu.edu])
AC_CONFIG_SRCDIR([src/topc.c])

# Set version number
TOPC_VERSION=2.5.2
TOPC_UPDATED="April, 2011"

# Checks for programs.
AC_PROG_CC
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_RANLIB

# Mostly for Solaris:
#  libxnet is a small subset.  If functions still missing, add other libs.
#  Note that last libraries added to $LIBS appears as first one.
AC_CHECK_FUNC(connect, , AC_CHECK_LIB(xnet, connect))
AC_CHECK_FUNC(socket, , AC_CHECK_LIB(socket, socket))
AC_CHECK_FUNC(gethostbyname, , AC_CHECK_LIB(nsl, gethostbyname))
LIBSFORSOCKETS="$LIBS"
AC_CHECK_LIB(efence, malloc, LIBEFENCE=-lefence, LIBEFENCE= )
LIBS=
AC_CHECK_FUNC(nanosleep, , AC_CHECK_LIB(rt, nanosleep))
LIBSFORSLEEP="$LIBS"
LIBS=

LIBSFORMPI="$LIBSFORSLEEP"
LIBSFORPTHREAD="$LIBSFORSLEEP"
LIBSFORSEQ="$LIBSFORSLEEP"

AC_ARG_ENABLE(cxx,
         [  --enable-cxx            provide topc++ for C++ [default=yes]],
         , enable_cxx=yes)
if test "x$enable_cxx" = "xyes"; then
  AC_PROG_CXX
fi

AC_ARG_ENABLE(posix-1c-threads,
         [  --enable-posix-1c-threads
                          turn on strict POSIX 1C [default=no]],
         , enable_posix_1c_threads=no)
if test "x$enable_posix_1c_threads" = "xyes"; then
  CFLAGS_FOR_POSIX_1C="-D_POSIX_C_SOURCE=199506L"
fi

AC_ARG_ENABLE(pthread,
         [  --enable-pthread        POSIX threads for shared memory [default=yes]],
         , enable_pthread=yes)

if test "x$enable_pthread" = "xyes"; then
  # AC_CHECK_HEADER(pthread.h)
  # This call adds -lpthread to LIBS
  AC_CHECK_LIB(pthread, pthread_create)
  # We temporarily add "-lpthread" because AC_CHECK_LIB fails to find it
  # on Alpha: OSF1-4.0 fails to find it.
  #   __pthread_create in:  /usr/shlib/libpthread.so
  #   and pthread.h redefines pthread_create to __pthread_create
  #   and declare return type as int.  This conflicts with char return
  #   type declared in AC_CHECK_LIB.
  AC_MSG_CHECKING(for working pthread library)
  ac_save_LIBS="$LIBS"
  LIBS="-lpthread  $LIBS"
  AC_TRY_RUN(#include <pthread.h>
    int main() {
    pthread_mutex_t mutex; if(0 != pthread_mutex_init(&mutex, NULL)) exit(1);
    return(0); }
	,
    HAVE_PTHREAD=yes, HAVE_PTHREAD=no, HAVE_PTHREAD=no)

  AC_MSG_RESULT($HAVE_PTHREAD)
  LIBS="$ac_save_LIBS"
  LIBPTHREAD="$LIBS"
  AC_CHECK_LIB(posix4, sem_init)
  AC_MSG_CHECKING(for working semaphore library)
  AC_TRY_RUN(#include <semaphore.h>
              int main() { sem_t sem; if(0 != sem_init(&sem, 0, 0)) exit(1);
              return(0); }
		,
              HAVE_SEMAPHORE=yes, HAVE_SEMAPHORE=no, HAVE_SEMAPHORE=no)
  AC_MSG_RESULT($HAVE_SEMAPHORE)
  LIBSFORPTHREAD="$LIBS $LIBSFORPTHREAD"
else
  HAVE_PTHREAD=no
fi

# LIBS="$LIBSFORSOCKETS $LIBSFORPTHREAD"

AC_ARG_ENABLE(strip-libs,
         [  --enable-strip-libs     Smaller image, but no symbolic debugging [default=no]],
         , enable_strip_libs=no)

if test "x$enable_strip_libs" = "xyes"; then
  STRIP_LIBS=yes
  STRIP=strip
else
  STRIP=:
fi

AC_ARG_WITH(mpi-cc,
[  --with-mpi-cc=PROG      Use PROG to compile mpi files [default=built-in MPI]
                          (Also consider: env CC=cc ./configure (use vendor cc);
                           see appendix of manual for more on non-default MPI)],
, with_mpi_cc=no)
AC_ARG_WITH(mpi-cxx,
[  --with-mpi-cxx=PROG     Use PROG to compile mpi files [default=built-in MPI]
                          (Also consider: env CXX=c++ ./configure (vendor C++))],
, with_mpi_cxx=no)
if test "x$with_mpi_cxx" = "xno"; then
  MPICXX='$'{CXX}
else
  MPICXX="$with_mpi_cxx"
fi
if test "x$with_mpi_cc" = "xno"; then
  libmpi="-lmpinu"
  MPICC='$'{CC}
else
  HAVE_MPI_H=1
#  AC_DEFINE(HAVE_MPI_H)  [Do it below, after resetting DEFS)
  MPICC="$with_mpi_cc"
fi
LIBSFORMPI="$libmpi $LIBSFORSOCKETS $LIBSFORMPI"
# topc-config.in:  --mpi)
#  Note that with_mpi_cc != no means libmpi="".
#  This is the right thing if using ${MPICC}
#  Maybe we should have undocumented --mpi-internal returning this,
#    which is called from topcc.in and topc++.in,
#    while --mpi  should indicate problem and:  exit 1
# # Do all of this for MPICXX also

# Checks for header files.
AC_HEADER_STDC
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([fcntl.h limits.h netdb.h netinet/in.h stdlib.h string.h sys/socket.h sys/time.h unistd.h])
AC_CHECK_HEADERS(pthread.h semaphore.h)

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_HEADER_TIME

# Checks for library functions.
# This defines HAVE_MALLOC if malloc(0) returns valid ptr instead of NULL.
AC_FUNC_MALLOC
# AC_FUNC_MMAP
AC_TYPE_SIGNAL
AC_FUNC_STAT
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([alarm getcwd gethostbyname gethostname memset select socket strchr strerror strtol])

# Erase old @DEFS@ and add to it
rm -rf conftest* confdefs.h
# AIX cpp loses on an empty file, so make sure it contains at least a newline.
echo > confdefs.h
if test "$HAVE_MPI_H" != ""; then
  AC_DEFINE(HAVE_MPI_H)
fi;

# AIX-cc doesn't allow "// ..." style commenting by default.
# AIX-xlc is IBM ANSI C, while cc is IBM extended C and xlC is C++/C
# OSF1: libpthread in weird place, not found by this script
# In linuxthreads-2.0.6 and also in some later versions using "old semaphore",
#   make parfactor-pthread occasionally fails.
#   Enable ASSERT in linuxthreads:queue.h for the 'smoking gun'
# We disable all of linuxthreads semaphores until we can test linuxthreads-2.0.7
target=`uname -s`
case $target-$CC in
  IRIX64-*) AC_DEFINE(HAVE_PROCESS_SCOPE_BOUND_NP) ;;
  AIX-xlc) CFLAGS="$CFLAGS -qcpluscmt" ;;
  AIX-xlC) CFLAGS="$CFLAGS -qcpluscmt" ;;
  AIX-cc) CFLAGS="$CFLAGS -qcpluscmt" ;;
  Linux-*) linux_pthread_sem_bug=yes ;;
  OSF1-*) if test x$HAVE_PTHREAD = xyes; then
            LIBSFORPTHREAD="-lpthread $LIBSFORPTHREAD"
          fi
          ;;
esac

# Substitute shell variables
AC_SUBST(TOPC_VERSION)
AC_SUBST(TOPC_UPDATED)
AC_SUBST(PWD)
AC_SUBST(HOSTNAME)
AC_SUBST(LIBSFORSOCKETS)
AC_SUBST(LIBSFORSLEEP)
AC_SUBST(LIBSFORMPI)
AC_SUBST(LIBPTHREAD)
AC_SUBST(HAVE_PTHREAD)
AC_SUBST(LIBSFORPTHREAD)
AC_SUBST(LIBSFORSEQ)
AC_SUBST(CFLAGS_FOR_POSIX_1C)
AC_SUBST(LIBEFENCE)
AC_SUBST(STRIP_LIBS)
AC_SUBST(STRIP)
AC_SUBST(MPICC)
AC_SUBST(MPICXX)

if test "x$HAVE_PTHREAD" = "xyes"; then
  # Older Solaris has only thr_setconcurrency()
  # AC_CHECK_FUNCS(FUNC)/AC_CHECK_HEADERS(file.h) (plural forms only)
  #   do AC_DEFINE(HAVE_FUNC), AC_DEFINE(HAVE_FILE_H)
  AC_CHECK_FUNCS(pthread_setconcurrency thr_setconcurrency)

  # Alternative, but would also need in conf.h.in:  #include <thread.h>
  # if test "$ac_cv_func_pthread_setconcurrency" != yes; then
  #   AC_CHECK_FUNC(thr_setconcurrency,
  #                 , AC_DEFINE(pthread_setconcurrency, thr_setconcurrency)
  #     	      AC_DEFINE(HAVE_PTHREAD_SETCONCURRENCY)
  #   )
  # fi
  if test "x$HAVE_SEMAPHORE" != "xyes"; then
    AC_DEFINE(HAVE_SEMAPHORE, 0)
  elif test "x$linux_pthread_sem_bug" = "xyes"; then
    AC_DEFINE(HAVE_SEMAPHORE,0)
  fi
else
  AC_DEFINE(HAVE_PTHREAD, 0)
fi

# if test "x$with_static_cache_size" = "xno"; then
#   AC_DEFINE(STATIC_CACHE_SIZE, 0)
# elif test "x$with_static_cache_size" != "xyes"; then
#   AC_DEFINE_UNQUOTED(STATIC_CACHE_SIZE, $with_static_cache_size)
# fi

AC_CONFIG_FILES([src/Makefile
          src/mpinu/Makefile src/mpinu/procgroup
          bin/Makefile bin/topcc bin/topc++ bin/topc-config
          doc/Makefile doc/topcc.1 doc/topc-version.texi
	  utils/ssh-killall Makefile README
         ])
AC_OUTPUT
chmod a+x utils/ssh-killall

if test "x$HAVE_PTHREAD" != xyes; then
  echo ""
  AC_MSG_WARN(*** Will not use pthreads in this configure)
elif test "x$HAVE_SEMAPHORE" != xyes; then
  echo ""
  AC_MSG_WARN(*** O/S semaphores not found.  Using TOP-C semaphores)
elif test "x$linux_pthread_sem_bug" = "xyes"; then
  echo ""
  AC_MSG_WARN(*** Using TOP-C semaphore due to possible bug in O/S sem)
  AC_MSG_WARN(*See http://www.ccs.neu.edu/home/gene/linuxthreads-bug.html,)
  AC_MSG_WARN(*  if you are curious about a bug in earlier Linux threads)
fi

chmod a+x bin/topcc bin/topc++ bin/topc-config

# --enable-batch not currently among our configure options
if test "x$enable_batch" != xyes; then
    echo ""
    echo "Now please type \`${TB}make${TN}' to compile. Good luck."
    echo ""
else
    make
fi
