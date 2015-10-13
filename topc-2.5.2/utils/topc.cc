// C++ interface for TOP-C
//   Gene Cooperman
// SEARCH ON parAPP1 and parAPP2 for example applications of this code.
//
// THIS DOES NOT COMPILE.  IT IS JUST AN ARCHITECTURE.
//
// The plan is to define a TOPC class with all the TOP-C callbacks.
// For each callback, e.g. doTask(), define a member function,
//   and also a static TOPC class function, s_doTask().
// The member function, doTask(), is declared virtual, so that the
//   user can form a derived class in which he or she writes a new doTask().
// In addition, define a static TOPC class function, master_slave().
//   master_slave() sets the TOPC class member:  TOPC * s_currTOPC = this;
// Meanwhile, the static class function, s_doTask(), is effectively
//   defined, using casts, as something like:
//     TOPC_BUF TOPC::s_doTask(void* in)
//       { (TOPC_BUF)(s_currTOPC->doTask((Tin)in)) };
// To make this all work, the class TOPC is modified to be
//   a template class:  TOPC<typename Tin,typename Tout>
// This context allows us to easily arrange defaults:
//   TOPC<typename Tinput = TOPC_BUF, typename Toutput = TOPC_BUF>
//   virtual doTask = 0;  // pure virtual
//   etc.
// In operation, the end user now typically does something like:
//   class Input { ... }
//   class Output { ... }
//   class parAPP1 : TOPC<Input,Output>, APPLICATION { ... };
//   class parAPP2 : TOPC<Input,Output>, APPLICATION { ... };
//   #include <topcpp> // which whould include topc.h
//   int main () { parAPP1 x; x.master_slave(); parAPP2 y; y.master_slave(); }
// TOP-C then looks for a procgroup file in the current directory to
//   specify where to run the slaves, and begins executing in parallel.

using namespace std;

#include <iostream>
#include <stdio.h>
#include "topc.h"

// Arbitrary hacks to set constants similar to TOP-C
#ifndef _TOPC_H
  typedef void * TOPC_BUF;
  inline int TOPC_num_slaves() { return 3; }
  inline int TOPC_is_up_to_date() { return true; }
  enum TOPC_ACTION {NO_ACTION, REDO, UPDATE, CONTINUATION};
#endif
static int dummy;

template <typename Tinput = void*, typename Toutput = void*>
class TOPC {
  private:
    static TOPC *s_currTOPC;
    static TOPC_BUF s_generateTaskInput();
    static TOPC_BUF s_doTask(void* input);
    static TOPC_ACTION s_checkTaskResult(TOPC_BUF input, TOPC_BUF output);
    static void s_updateSharedData(TOPC_BUF input, TOPC_BUF output);

    bool usingDefaultUpdateSharedData;

  public:
    static const Tinput NO_INPUT;

    inline bool TOPC<Tinput,Toutput>::is_up_to_data() {
      return (bool)TOPC_is_up_to_date();
    }
    virtual Toutput generateTaskInput();
    virtual Toutput doTask(Tinput) = 0;  // pure virtual
    virtual TOPC_ACTION checkTaskResult(Tinput input, Toutput);
    virtual void updateSharedData(Tinput input, Toutput output);
    void master_slave();
};

template <typename Tinput, typename Toutput>
TOPC<Tinput,Toutput> *TOPC<Tinput,Toutput>::s_currTOPC = NULL;

template <typename Tinput, typename Toutput>
// HACK for now:
Tinput const TOPC<Tinput,Toutput>::NO_INPUT = static_cast<Tinput>(&dummy);

template <typename Tinput, typename Toutput>
TOPC_BUF TOPC<Tinput,Toutput>::s_generateTaskInput() {
  // IF THIS DOESN'T RETURN TOPC_BUF, IT PROVES USER generateTaskInput()
  // DID NOT RETURN A POINTER;  USER ERROR;  CAN WE SIGNAL ERROR TO USER?
  return s_currTOPC->generateTaskInput();
}
template <typename Tinput, typename Toutput>
Tinput TOPC<Tinput,Toutput>::generateTaskInput() {
  cout << "default generateTaskInput" <<endl;
  static int slave = 0;
  static int last_slave = TOPC_num_slaves();
  if (slave >= last_slave) return static_cast<Tinput*>(NO_INPUT);
  slave++;
  return static_cast<Tinput*>&slave;
}

template <typename Tinput, typename Toutput>
TOPC_BUF TOPC<Tinput,Toutput>::s_doTask(void* input) {
  return s_currTOPC->doTask(static_cast<Tinput*>(input));
}

template <typename Tinput, typename Toutput>
TOPC_ACTION TOPC<Tinput,Toutput>::s_checkTaskResult
					(Tinput* input, Toutput* output) {
  return s_currTOPC->checkTaskResult(static_cast<Tinput*>(input),
				     static_cast<Toutput*>(output));
}
template <typename Tinput, typename Toutput>
TOPC_ACTION TOPC<Tinput,Toutput>::checkTaskResult(Tinput input, Toutput output)
{
  // IS THERE A WAY TO TEST IF updateSharedData HAS BEEN SHADOWED?
  // IF IT HAS NOT BEEN SHADOWED, THEN RETURN NO_ACTION.
  // AT THE VERY LEAST, WE COULD HAVE DEFAULT updateSharedData()
  // SET A SPECIAL VALUE IN `this', SO WE'LL KNOW IT WAS RUN AFTERWARDS
  cout << "default checkTaskResult" << endl;
  // if (usingDefaultUpdateSharedData) return NO_ACTION;
  if (! is_up_to_date()) return REDO;
  if (output == NULL) return NO_ACTION;
  else return UPDATE;
}

template <typename Tinput, typename Toutput>
void TOPC<Tinput,Toutput>::s_updateSharedData
					(TOPC_BUF input, TOPC_BUF output) {
  s_currTOPC->updateSharedData(static_cast<Tinput>(input),
		               static_cast<Toutput>(output));
}
template <typename Tinput, typename Toutput>
void TOPC<Tinput,Toutput>::updateSharedData(Tinput input, Toutput output)
{
  cout << "default updateSharedData" <<endl;
  // SHOULD WARN USER IF THIS IS CALLED.  USER SHOULD NEVER USE
  //   DEFAULT updateSharedData
  usingDefaultUpdateSharedData = true;
}

template <typename Tinput, typename Toutput>
void TOPC<Tinput,Toutput>::master_slave()
{
  TOPC::s_currTOPC = this;
  usingDefaultUpdateSharedData = false;

  cout << "master_slave:  \n";
  while (1) {
    TOPC_BUF input = s_generateTaskInput();
    if (input == NO_INPUT) break;
    TOPC_BUF output = s_doTask(input);
    cout << "input: " << *(int *)input << "; output: " << *(int *)output
	  << endl;
    char const *actions[] = {"NO_ACTION", "REDO", "UPDATE", "CONTINUATION"};
    cout << "TOPC_ACTION:  " << actions[s_checkTaskResult(input, output)]
	    << endl;
    s_updateSharedData(input, output);
    cout << endl;
  }

  TOPC::s_currTOPC = NULL;
  // Now we have static callbacks using type `TOPC_BUF' = `void *', and so
  //   we can call the C interface to implmment the TOP-C algorithm.
  // CALL:
  TOPC_master_slave(s_generateTaskInput,s_doTask,
                    s_checkTaskResult,s_UpdateSharedData);
}

// ======================================
// EXAMPLE 1:
// Trivial end-user interface:
// input and output buffers default to `void *'
// generateTaskInput() defaults to running once on each slave
//   with task inputs 1, 2, ..., num_slaves;  and then quitting.
// To run, create an instance, x, of parAPP1, and call:  x.master_slave();
// After compiling, ./a.out --TOPC-num_slaves=XXX can customize how many slaves

class parAPP1 : public TOPC<void*,void*> {
  inline TOPC_BUF doTask(void* input)
                    { cout << "new parAPP1::doTask" << endl; return input; }
};

// ======================================
// EXAMPLE 2:
// End-user interface:
// Assume input buffer and output buffer both of type `int *'
// The class parAPP2 uses multiple inheritance in this example, and
//   the end user calls APPLICATION:app_task in parAPP2::doTask().

class APPLICATION {
  private:
    int INCREMENT;
  public:
    inline APPLICATION() { INCREMENT = 100; }
    int app_task(int input) { return INCREMENT + input; }
};
class parAPP2 : public TOPC<int *,int *>, APPLICATION {
  int * generateTaskInput();
  int * doTask(int * input);
  TOPC_ACTION checkTaskResult(int * input, int *output);
};

// End-user implementation:
int * parAPP2::generateTaskInput()
{
  static bool done = false;
  if (done) return NO_INPUT;
  done = true;
  static int input;
  return &input;
}
int * parAPP2::doTask(int * input)
{
  cout << "new parAPP2::doTask:  " << *input << endl;
  // I think this is now safe in TOP-C.  Should check this:
  // It's probably now safe even without `static'
  static int output = app_task(*input);
  return &output;
}
TOPC_ACTION parAPP2::checkTaskResult(int * input, int *output)
{ return NO_ACTION; }

// ======================================

main ()
{
  int x = 3;
  parAPP1 par1;
  cout << "\n============================\nparAPP1: ";
  par1.master_slave();

  cout << "\n============================\nparAPP2: ";
  parAPP2 par2;
  par2.master_slave();
}
