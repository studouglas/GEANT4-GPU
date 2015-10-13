#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "mast.h"

//vietha 2003.05.09
#define SHADOWED_LABEL ("Shadowed")

int num_specs(struct mast *m) {
    int n = 0;
    struct specs *cursor;

    for(cursor = m->sp; cursor != NULL; cursor = cursor->next) {
        n++;
    }

    return n;
}

int num_fieldspecs(struct fieldspecs *f) {
    int n = 0;
    struct fieldspecs *cursor;

    for(cursor = f; cursor != NULL; cursor = cursor->next) {
        n++;
    }

    return n;
}

int num_formals(struct formals *fmls) {
    int n = 0;
    struct formals *cursor;

    for(cursor = fmls; cursor != NULL; cursor = cursor->next) {
        n++;
    }

    return n;
}

struct mast *Mast_init(char *m, struct specs *s) {
    struct mast *ret;
    ret = (struct mast*)malloc(sizeof(struct mast));

    ret->macros = m;
    ret->sp = s;

    return ret;
}

void Mast_free(struct mast *m) {
    if(m->sp != NULL)
        Specs_free(m->sp);
    free(m);
    m = NULL;
}

struct spec *Spec_init(char* template_decl, char *mt, struct formals *fmls, char* parent_classes, struct fieldspecs *f, char* constructor) {
    struct spec *ret;
    ret = (struct spec*)malloc(sizeof(struct spec));

    ret->mtype = mt;
    ret->fmls = fmls;
    ret->fsp = f;
    //vietha 2003.4.29
    ret->template_decl = template_decl; 	
    ret->parent_classes = parent_classes;
    ret->constructor_call = constructor;

    return ret;
}

void Spec_free(struct spec *s) {
    // fix mtype leak !

    if(s->fmls != NULL)
        Formals_free(s->fmls);
    if(s->fsp != NULL)
        Fieldspecs_free(s->fsp);

    free(s);
    s = NULL;
}

struct specs *Specs_init(struct spec *sp, struct specs *nt) {
    struct specs *ret;
    ret = (struct specs*)malloc(sizeof(struct specs));

    ret->sp = sp;
    ret->next= nt;

    return ret;
}

void Specs_free(struct specs *s) {
    if(s->sp != NULL)
        Spec_free(s->sp);
    if(s->next != NULL)
        Specs_free(s->next);
    free(s);
    s = NULL;
}

struct formals *Formals_init(char *type, char *name,
                                struct formals *next)
{
    struct formals *ret;
    ret = (struct formals*)malloc(sizeof(struct formals));

    ret->type = type;
    ret->name = name;
    ret->next = next;

    return ret;
}

void Formals_free(struct formals *f) {
    //Fix type and name leak!!

    if(f->next != NULL)
        Formals_free(f->next);

    free(f);
    f = NULL;
}

struct fieldspec *Fieldspec_init(char *t, char *nm, struct gss *g) { 
    struct fieldspec *ret;
    ret = (struct fieldspec*)malloc(sizeof(struct fieldspec));

    ret->type = t;
    ret->name = nm;
    ret->g = g;

    return ret;
}

void Fieldspec_free(struct fieldspec *fsp) {
    // Fix type and name leak!

    if(fsp->g != NULL)
        Gss_free(fsp->g);

    free(fsp);
    fsp = NULL;
}

struct gss *Gss_init(char *g, char *s, char *sz) {
    struct gss *ret;
    ret = (struct gss*)malloc(sizeof(struct gss));

    ret->getsp = g;
    ret->setsp = s;
    ret->sizesp = sz;

    return ret;
}

void Gss_free(struct gss *g) {
    // Fix get, set and size big leak!!

    free(g);
    g = NULL;
}


struct fieldspecs *Fieldspecs_init(struct fieldspec *f, struct fieldspecs *nt) {
    struct fieldspecs *ret;
    ret = (struct fieldspecs*)malloc(sizeof(struct fieldspecs));

    ret->fsp = f;
    ret->next= nt;

    return ret;
}

void Fieldspecs_free(struct fieldspecs *fsps) {
    if(fsps->next != NULL)
        Fieldspecs_free(fsps->next);
    if(fsps->fsp != NULL)
        Fieldspec_free(fsps->fsp);

    free(fsps);
    fsps = NULL;
}


void mast_print(struct mast *m) {
    //printf("macros: {\n%s\n}\n", m->macros);
    specs_print(m->sp);
}

void specs_print(struct specs *s) {
    if(s == NULL) {
        return;
    }
    spec_print(s->sp);
    specs_print(s->next);
}

void spec_print(struct spec *s) {
    //printf("mtype: %s\n", s->mtype);
    formals_print(s->fmls);
    fieldspecs_print(s->fsp);
}

void formals_print(struct formals *fmls) {
    //printf("formals: ( \n");
    //printf("%s %s", fmls->type, fmls->name);
    formalsrest_print(fmls->next);
}

void formalsrest_print(struct formals *fmlsrest) {
    if(fmlsrest != NULL) {
        //printf(", %s %s", fmlsrest->type, fmlsrest->name);
        formalsrest_print(fmlsrest->next);
    } else {
        //printf(" )\n");
    }
}

void fieldspecs_print(struct fieldspecs *f) {
     if(f == NULL) {
        return;
    }
    fieldspec_print(f->fsp);
    fieldspecs_print(f->next);
}

void fieldspec_print(struct fieldspec *f) {
    //printf("fieldspec: {\n");
    //printf("type: %s\n", f->type);
    //printf("name: %s\n", f->name);
    gss_print(f->g);
    //printf("}\n");
}

void gss_print(struct gss* g) {
    if(g == NULL) return;

    //printf("getspec: %s\n", g->getsp);
    //printf("setspec: %s\n", g->setsp);
    //printf("sizespec: %s\n", g->sizesp);
}

//concatenates variable number of strings, the first argument specifies
//the number of string to be concatenated. Returns the string concatenated
//caller should call free to free the storage allocated.
char *strconcat(int n, ...) {
    va_list ap;
    int i , total_len = 0;
    char **s;
    char *ret;

    va_start(ap, n);
    if(n <= 0) return NULL;

    s = (char **)malloc(n * sizeof(char *));

    for(i = 0; i < n; i++) {
        s[i] = va_arg(ap, char *);
        total_len += strlen(s[i]);
    }

    ret = (char *)malloc(total_len + 1);
    strcpy(ret, s[0]);
    for(i = 1; i < n; i++) {
        strcat(ret, s[i]);
    }
    free(s);
    va_end(ap);

    return ret;
}

/* Based on one spec (class ... marshal(..) {...})
   create a "Shadowed" class to access "protected" data fields of the main class
   vietha 2003.05.09 */
void generate_shadowClass(struct spec *s, FILE *fs) {
    char file_name[50], instance_var[51]; 
    char *mtype = s->mtype;
    char *type = s->fmls->type;
    char *name = s->fmls->name;
    int k, i;
    int numOffieldspecs = num_fieldspecs(s->fsp);

    struct fieldspec *fsp;
    char *ftype;  //field type
    char *fname;  //field name
    //vietha 2003.4.29
    char mtype_without_template[50];
    char* type_without_asterisk = strdup(type);
    char* ptr;

    // remove the "<...>" from the class type
    memset(mtype_without_template, 0, sizeof(mtype_without_template));
    typename_without_template(mtype,mtype_without_template);

    // remove the '*' from the type
    ptr = strrchr(type_without_asterisk,'*');
    if(ptr) *ptr='\0';

    fprintf(fs, "\n%s class %s;\n", s->template_decl, mtype_without_template);

    fprintf(fs, "\n%s class %s%s : public %s{\n", s->template_decl, SHADOWED_LABEL, mtype_without_template, type_without_asterisk);
    indent(fs,1);
    fprintf(fs, "friend class %s;\n", mtype);
    free(type_without_asterisk);

#if 0
    //class def
    fprintf(fs, "\n%s class %s%s %s{\n", s->template_decl, SHADOWED_LABEL, mtype_without_template, s->parent_classes);

    // Data field declarations
    //Declare a fields , these should be the same exact types and names
    //as the those specified by the user, I am making it public here for
    // accessing the private data fields of the main class
    fprintf(fs, "public:\n");
    for(k = 1; k <= numOffieldspecs; k++) {
       int i = 1;
       struct fieldspecs *cursor;

       if(s->fmls != NULL && s->fmls->next != NULL)
	   isMultipleMarshal = 1;

       //First go to the num fieldspec
       for(cursor = s->fsp; cursor != NULL; cursor = cursor->next) {
	   if(i == k)
	       break;
	   i++;
       }
       fsp = cursor->fsp;
      
       ftype = fsp->type;  //field type
       fname = fsp->name;  //field name
       //gss1 = fsp->g;

       indent(fs,1);
       fprintf(fs, "%s %s;\n", ftype, fname);
    }
#endif

    //Finish up
    fprintf(fs, "};\n\n");
}


//Based on one spec (class ... marshal(..) {...})
//creates files: one .h file
void generate_hfile(struct spec *s, FILE *fs) {
    //char file_name[50], instance_var[51]; 
    char *mtype = s->mtype;
    char *type = s->fmls->type;
    char *name = s->fmls->name;
    int len = 0, i;
    int numOffieldspecs = num_fieldspecs(s->fsp);

    //vietha 2003.4.29
    char mtype_without_template[50];
    memset(mtype_without_template, 0, sizeof(mtype_without_template));
    typename_without_template(mtype,mtype_without_template);

    //class def
    //vietha
    //fprintf(fs, "class %s : public %s {\n", mtype, "MarshaledObj");
    fprintf(fs, "%s class %s : public %s {\n", s->template_decl, mtype_without_template, "MarshaledObj");

    /* Data field declarations */
    //Declare a field for Foo *, this should be the same exact type
    //as the type specified by the user, I am making it public here only for
    //easiness for testing. (Although a friend class would be better)
    fprintf(fs, "public:\n");
    indent(fs,1);
    fprintf(fs, "%s %s;\n", type, name);
    //vietha 2003.05.09
    indent(fs,1);
    fprintf(fs, "%s%s* %s_%s;\n", SHADOWED_LABEL, mtype, SHADOWED_LABEL, name);

    fprintf(fs, "public:\n");
}

/* generate the implementation of functions of the Marshaled... Class
   Before, these were put in a .cpp file,
   but now I put them in the end of the .h file ,
   because the implementation of a class with template should be in the heaher files.
   vietha 2003/05/01
   */
void generate_cfile(struct spec *s, FILE *fs) {
    char file_name[50];
    char mtype_without_template[50];
    char *mtype = s->mtype; 
    char *type = s->fmls->type;
    char *name = s->fmls->name;
    int len = 0, i;
    int numOffieldspecs = num_fieldspecs(s->fsp);
    char type_no_star[128];  //type name without the *, used in "new type();"

    strcpy(type_no_star, type);
    type_no_star[strlen(type)-1] = '\0';
    
    //vietha 2003.4.29
    memset(mtype_without_template, 0, sizeof(mtype_without_template));
    typename_without_template(mtype,mtype_without_template);


    //Implement constructors
    //fprintf(fs, "%s %s::%s(%s %s) : MarshaledObj() {\n", s->template_decl, mtype, mtype_without_template, type, "objptr");
    fprintf(fs, "%s(%s %s) : MarshaledObj() {\n", mtype_without_template, type, "objptr");

    indent(fs,1);
    fprintf(fs, "msh_isUnmarshalDone = false;\n");
    indent(fs,1);
    fprintf(fs, "this->%s = objptr;\n", name);
    //vietha 2003.05.09
    indent(fs,1);
    fprintf(fs, "this->%s_%s = (%s%s*)this->%s;\n", SHADOWED_LABEL, name, SHADOWED_LABEL, mtype, name);


    //If objptr == NULL, then we stop here.
    indent(fs,1);
    fprintf(fs, "if (objptr == NULL)\n");
    indent(fs,2);
    fprintf(fs, "return;\n\n");

    for(i = 1; i <= numOffieldspecs; i++) {
    // Add shallow copy, vietha
    //for(i = 0; i <= numOffieldspecs; i++) {
        indent(fs,1);
        fprintf(fs, "marshal%d();\n", i);
    }
    fprintf(fs, "}\n\n");

    //fprintf(fs, "%s %s::%s(void *buf, char isUnmarshaling)\n", s->template_decl, mtype, mtype_without_template);
    fprintf(fs, "%s(void *buf, char isUnmarshaling = 'u')\n", mtype_without_template);
    fprintf(fs, ": MarshaledObj(buf, isUnmarshaling) {\n");
	indent(fs,1);
	fprintf(fs, "msh_isUnmarshalDone = false;\n");
    fprintf(fs, "}\n\n");

    //Destructor
    //fprintf(fs, "%s %s::~%s() {\n", s->template_decl, mtype, mtype_without_template);
    fprintf(fs, "~%s() {\n", mtype_without_template);
    indent(fs,1);
	fprintf(fs, "//if(msh_isUnmarshalDone && this->%s != NULL) {\n", name);
	indent(fs,2);
	fprintf(fs, "//delete this->%s;\n", name);
	indent(fs,1);
	fprintf(fs, "//}\n");
    fprintf(fs, "}\n\n");

    //unmarshal function unmarshals the whole object
    //fprintf(fs, "%s %s %s::unmarshal() {\n", s->template_decl, type, mtype);
    fprintf(fs, "%s unmarshal() {\n", type);
	indent(fs,1);
	fprintf(fs, "//We don't want to unmarshal the buffer is empty.\n");
	indent(fs,1);
	//fprintf(fs, "if(*(int *)this->msh_buffer == 0) {\n");
	fprintf(fs, "if(msh_size <= MSH_HEADER_SIZE) {\n");
	indent(fs,2);
	fprintf(fs, "//This is buggy, we can't always assume that\n");
	indent(fs,2);
	fprintf(fs, "//obj == NULL <==> List is empty.\n");
	indent(fs,2);
	fprintf(fs, "return NULL;\n");
	indent(fs,1);
	fprintf(fs, "} else {\n");
	indent(fs,2);

	// vietha 2004.08.29
	//fprintf(fs, "this->%s = new %s();\n", name, type_no_star);
	fprintf(fs, "%s\n", s->constructor_call);

	//vietha 2003.05.09
	indent(fs,2);
	fprintf(fs, "this->%s_%s = (%s%s*)this->%s;\n", SHADOWED_LABEL, name, SHADOWED_LABEL, mtype, name);

	indent(fs,2);
	fprintf(fs, "this->msh_isUnmarshalDone = true;\n");

    for(i = 1; i <= numOffieldspecs; i++) {
    // Add shallow copy, vietha
    //for(i = 0; i <= numOffieldspecs; i++) {
        indent(fs,2);
        fprintf(fs, "unmarshal%d();\n", i);
    }
	indent(fs,2);
	fprintf(fs, "return this->%s;\n", name);
	indent(fs,1);
	fprintf(fs, "}\n");
    fprintf(fs, "}\n\n");


    //vietha 2003.04.28
    //unmarshal function that copies to an already-existing object
    //fprintf(fs, "%s void %s::unmarshalTo(%s obj) {\n", s->template_decl, mtype, type);
    fprintf(fs, "void unmarshalTo(%s obj) {\n", type);
	indent(fs,1);
	fprintf(fs, "//We don't want to unmarshal the buffer is empty.\n");
	indent(fs,1);
	//fprintf(fs, "if(*(int *)this->msh_buffer == 0) {\n");
	fprintf(fs, "if(msh_size <= MSH_HEADER_SIZE) {\n");
	indent(fs,2);
	fprintf(fs, "//This is buggy, we can't always assume that\n");
	indent(fs,2);
	fprintf(fs, "//obj == NULL <==> List is empty.\n");
	indent(fs,2);
	fprintf(fs, "return;\n");
	indent(fs,1);
	fprintf(fs, "} else {\n");
	indent(fs,2);
	fprintf(fs, "this->%s = obj;\n", name);

	//vietha 2003.05.09
	indent(fs,2);
	fprintf(fs, "this->%s_%s = (%s%s*)this->%s;\n", SHADOWED_LABEL, name, SHADOWED_LABEL, mtype, name);

	indent(fs,2);
	fprintf(fs, "this->msh_isUnmarshalDone = true;\n");

    for(i = 1; i <= numOffieldspecs; i++) {
    // Add shallow copy, vietha
    //for(i = 0; i <= numOffieldspecs; i++) {
        indent(fs,2);
        fprintf(fs, "unmarshal%d();\n", i);
    }
	indent(fs,1);
	fprintf(fs, "}\n");
    fprintf(fs, "}\n\n");

    //Now define marshal##n and unmarshal##n for each field of obj
    for(i = 1; i <= numOffieldspecs; i++) {
    //for(i = 0; i <= numOffieldspecs; i++) {
        generate_marshalfunc(s, i, fs);
        generate_unmarshalfunc(s, i, fs);
    }

    //Done
}


//@param: sp is a spec. 
//@param: num is the number (index) of the current fieldspec
//@param: fs is the FILE ptr
//This function creates two functions: marshal##n and unmarshal##n
//vietha 2003/04/25
// num = 0 : marshaling the shallow-copy of the whole object

void generate_marshalfunc(struct spec *sp, int num, FILE *fs) {
    char *mtype = sp->mtype;
    struct fieldspec *fsp;

    char *ftype;  //field type
    char *fname;  //field name
    struct gss *gss1;
    char *getspec, *sizespec;

    char *msh_size = "msh_size";
    char *msh_buffer = "msh_buffer";
    char *msh_extent = "msh_extent";
    char *msh_cursor = "msh_cursor";
    /* vietha 2003.05.05 */
    char *msh_field_begin = "msh_field_begin";

    char *currentSize = "msh_currentSize";

    int isMultipleMarshal = 0;  // Is this spec for marshaling multiples?

    //vietha 2003/04/25
    char *type = sp->fmls->type;
    char *name = sp->fmls->name;
    char type_no_star[128];  //type name without the *, used in "new type();"
    strcpy(type_no_star, type);
    type_no_star[strlen(type)-1] = '\0';


    if(num == 0){ /* for the shallow-copy of the whole object*/
    }else{  /* for data fields */
      int i = 1;
      struct fieldspecs *cursor;

      if(sp->fmls != NULL && sp->fmls->next != NULL)
        isMultipleMarshal = 1;

      //First go to the num fieldspec
      for(cursor = sp->fsp; cursor != NULL; cursor = cursor->next) {
        if(i == num)
	  break;
        i++;
      }
      fsp = cursor->fsp;
      
      ftype = fsp->type;  //field type
      fname = fsp->name;  //field name
      gss1 = fsp->g;
    }

    //Start definition of marshal##n
    //fprintf(fs, "%s void %s::marshal%d() {\n", sp->template_decl, mtype, num);
    fprintf(fs, "void marshal%d() {\n", num);
    if(gss1 == NULL & (num!=0) ) {  //Default case, call library functions
	/* vietha 2003.05.05 */
	indent(fs, 1);
        fprintf(fs, "//Default case, call library function\n");
        indent(fs, 1);
        if(!isMultipleMarshal) {
            fprintf(fs, "marshalPrimitive((void*)&(%s->%s), sizeof(%s));\n",
                sp->fmls->name, fname, ftype);
        } else {
            fprintf(fs, "marshalPrimitive((void*)&%s, sizeof(%s));\n",fname, ftype);
        }
    } else {
      if(num!=0){
        sizespec = gss1->sizesp;
        getspec = gss1->getsp;
      }
        indent(fs, 1);
        fprintf(fs, "//declare field_size to be the size of this field\n");
        indent(fs, 1);
        fprintf(fs, "int %s = 0;\n", currentSize);

        indent(fs, 1);
        fprintf(fs, "if (isUnmarshaling())\n");
        indent(fs, 2);
        fprintf(fs, "throw \"Tried to marshal in obj marked isUnmarshaling == true\";\n\n");

	if(num == 0){  //for marshaling shallow-copy
	  indent(fs, 1);
	  fprintf(fs, "%s = sizeof(%s); \n", currentSize,type_no_star);
	}else{
	  indent(fs, 1);
	  fprintf(fs, "//Copy the sizespec into %s here:\n", currentSize);
	  indent(fs, 1);
	  fprintf(fs, "%s\n\n", sizespec);
	}

        indent(fs, 1);
        fprintf(fs, "//Increase the size of buffer if needed\n");
        indent(fs, 1);
        //fprintf(fs, "%s += %s + sizeof(int) + sizeof(int); // 4 bytes for the total size of field, 4 bytes for the number of elements in the array\n", msh_size, currentSize);
        //indent(fs, 1);
        //fprintf(fs, "EXTEND_BUFFER(%s);\n\n", msh_size);
        fprintf(fs, "EXTEND_BUFFER(%s + sizeof(int) + sizeof(int)); // 4 bytes for the total size of field, 4 bytes for the number of elements in the array (in the case of array marshaling)\n", currentSize);

        indent(fs, 1);
        //fprintf(fs, "//Put the size of the current field in the first sizeof(int) bytes\n");
	/* vietha 2003.05.05 */
        fprintf(fs, "//Mark the beginning position for this field, will write the total size of this field here later\n");
        indent(fs, 1);
        //fprintf(fs, "*(int*)%s = %s;\n\n", msh_cursor, currentSize);
	fprintf(fs, "%s = %s;\n\n", msh_field_begin, msh_cursor);

        indent(fs,1);
        fprintf(fs, "//Advance cursor of distance = sizeof(int)\n");
        indent(fs, 1);
        fprintf(fs, "%s += sizeof(int);\n\n", msh_cursor);

	if (num == 0){
	  indent(fs, 1);
          fprintf(fs, "memcpy(%s, %s, sizeof(%s));\n", msh_cursor, name, type_no_star);
	}else{
	  indent(fs, 1);
	  fprintf(fs, "//Now just copy \"get\" functions here\n");
          //the getspec is a (compound)stmt that copies the value of the current
          //field into msh_cursor, which is of type "char *", therefore, user
          //has essentially did the "marshaling" work for us, we just need to 
          //copy it
	  indent(fs, 1);
          fprintf(fs, "%s\n", getspec);
	}

        indent(fs,1);
        fprintf(fs, "//Now advance the cursor\n");
        indent(fs,1);
        fprintf(fs, "%s += %s;\n", msh_cursor, currentSize);

	/* vietha 2003.05.05 */
        indent(fs,1);
        fprintf(fs, "//Now set the size of this field\n");
	indent(fs,1);
	fprintf(fs,"int tmp; //use memcpy instead of *(int*)... =... to prevent bus error\n");
        indent(fs,1);
        fprintf(fs, "tmp = (%s-%s) - sizeof(int);\n", msh_cursor, msh_field_begin);
        indent(fs,1);
        fprintf(fs, "memcpy(%s, &tmp, sizeof(int));\n\n", msh_field_begin);

        indent(fs,1);
        fprintf(fs, "//Now set msh_size\n");
        indent(fs,1);
        fprintf(fs, "%s = %s - %s;\n", msh_size, msh_cursor, msh_buffer);
        indent(fs,1);
        /*fprintf(fs, "tmp = %s - sizeof(int);\n", msh_size);
        indent(fs,1);
        fprintf(fs, "memcpy(%s, &tmp, sizeof(int));\n\n", msh_buffer);*/
	// vietha 2004.08.29
        fprintf(fs, "MSH_SET_TOTALSIZE(%s);", msh_size);
    }

    indent(fs,1);
    //fprintf(fs, "*(int*)msh_header = msh_typechoice;\n");
    fprintf(fs, "MSH_SET_TYPECHOICE(msh_typechoice);\n");
    //Finish up
	  fprintf(fs, "}\n\n");
}

//@param: sp is a spec. 
//@param: num is the number (index) of the current fieldspec
//@param: fs is the FILE ptr
//This function creates two functions: marshal##n and unmarshal##n
//vietha 2003/04/25
// num = 0 : unmarshaling the shallow-copy of the whole object

void generate_unmarshalfunc(struct spec *sp, int num, FILE *fs) {
    char *mtype = sp->mtype;
    char *var_name = sp->fmls->name;
    char *type = sp->fmls->type;
    struct fieldspec *fsp;
    char *ftype;  //field type
    char *fname;  //field name
    struct gss *gss1;
    char *setspec, *sizespec;

    char *msh_size = "msh_size";
    char *msh_buffer = "msh_buffer";
    char *msh_extent = "msh_extent";
    char *msh_cursor = "msh_cursor";

    char *currentSize = "msh_currentSize";

    int isMultipleMarshal = 0;  // Is this spec for marshaling multiples?
    //vietha 2003/04/25
    char *name = sp->fmls->name;
    char type_no_star[128];  //type name without the *, used in "new type();"
    strcpy(type_no_star, type);
    type_no_star[strlen(type)-1] = '\0';

    if(num == 0){  //for the shallow-copy
    }else{

      int i = 1;
      struct fieldspecs *cursor;

      if(sp->fmls != NULL && sp->fmls->next != NULL)
        isMultipleMarshal = 1;

      //First go to the num fieldspec
      for(cursor = sp->fsp; cursor != NULL; cursor = cursor->next) {
	if(i == num)
	  break;
        i++;
      }
      fsp = cursor->fsp;

      ftype = fsp->type;  //field type
      fname = fsp->name;  //field name
      gss1 = fsp->g;
    }

    //Start definition of unmarshal##n
    //fprintf(fs, "%s void %s::unmarshal%d() {\n", sp->template_decl, mtype, num);
    fprintf(fs, "void unmarshal%d() {\n", num);
    if(gss1 == NULL & (num!=0)) {  //Default case, call library functions
	// If ftype is one of the "primitive" types,
	/*if(!strcmp(ftype, "int") || !strcmp(ftype, "double")) {
            indent(fs, 1);
            fprintf(fs, "//Default case, call library function\n");
            indent(fs, 1);
            if(!isMultipleMarshal) {
                fprintf(fs, "%s->%s = %sAs%s();\n",
                            sp->fmls->name, fname, "unmarshal", ftype);
            } else {
                fprintf(fs, "%s->%s = %sAs%s();\n",
                            "this", fname, "unmarshal", ftype);
            }
        }*/
	/* vietha 2003.05.05 */
	indent(fs, 1);
        fprintf(fs, "//Default case, call library function\n");
        indent(fs, 1);
        if(!isMultipleMarshal) {
            fprintf(fs, "unmarshalPrimitive((void*)&(%s->%s), sizeof(%s));\n",
                sp->fmls->name, fname, ftype);
        } else {
            fprintf(fs, "unmarshalPrimitive((void*)&%s, sizeof(%s));\n",fname, ftype);
        }
    } else {
      if(num!=0){
        sizespec = gss1->sizesp;
        setspec = gss1->setsp;
      }
        indent(fs, 1);
        fprintf(fs, "//declare currentSize to be the size of this field\n");
        indent(fs, 1);
        fprintf(fs, "int %s = 0;\n", currentSize);

        indent(fs, 1);
        fprintf(fs, "//copy the size of the current field into currentSize\n");
        indent(fs, 1);
        //fprintf(fs, "%s = *(int*)%s;\n", currentSize, msh_cursor);
	fprintf(fs, "memcpy(&%s, %s, sizeof(int));\n", currentSize, msh_cursor);

        indent(fs, 1);
        fprintf(fs, "%s += sizeof(int);\n", msh_cursor);

	if(num==0){  //unmarshaling the shallow-copy
	  indent(fs, 1);
          fprintf(fs, "memcpy(%s, %s, sizeof(%s));\n", name, msh_cursor, type_no_star);
	}else{
	  indent(fs, 1);
	  fprintf(fs, "//Now copy the setspec here\n");
	  indent(fs, 1);
	  fprintf(fs, "%s\n", setspec);
	}

        indent(fs, 1);
        fprintf(fs, "%s += %s;\n",
                        msh_cursor, currentSize);
        //indent(fs, 1);
        //fprintf(fs, "%s = %s - %s;\n", msh_size, msh_cursor, msh_buffer);

    }
    //Finish up
    fprintf(fs, "}\n\n");

}

void generate_files(struct mast *m, FILE* fs) {
    char *macros = m->macros;
    //char file_name[50];
    char *mtype = m->sp->sp->mtype;
    struct specs *cur1;
    //FILE *fs;
    //vietha 2003.4.29
    char mtype_without_template[50];
    memset(mtype_without_template, 0, sizeof(mtype_without_template));
    typename_without_template(mtype,mtype_without_template);
	
    //Create the .h file name
//    strcpy(file_name, mtype_without_template);
//    strcat(file_name, ".h");

//    fs = fopen(file_name, "w+");
//    if(fs == NULL) {
//        perror("Can't open file");
//        exit(-1);
//    }

    fprintf(fs, "// This file was generated automatically by marshalgen.\n"
            "\n");

    //macros
    fprintf(fs, "#ifndef %s_H\n", mtype_without_template);
    fprintf(fs, "#define %s_H\n\n", mtype_without_template);

    fprintf(fs, "%s\n", macros);

    //include .h file
    fprintf(fs, "#include <stdio.h>\n");
    fprintf(fs, "#include <string.h>\n");
    //include files
    fprintf(fs, "#include \"MarshaledObj.h\"\n");

    /* generate the "Shadowed" class to access private data fields
       and write it to the .h file */
    for(cur1 = m->sp; cur1 != NULL; cur1 = cur1->next) {
        //For a spec, if it has only one "type ID" pair
        if(cur1->sp->fmls != NULL && cur1->sp->fmls->next == NULL)
            generate_shadowClass(cur1->sp, fs);
    }

    // in case of multiple classes, go through each class
    for(cur1 = m->sp; cur1 != NULL; cur1 = cur1->next) {
	/* generate class decs for all specs
	   and write them to the .h file */
	generate_hfile(cur1->sp, fs);

	fprintf(fs, "\n\n// Function implementations\n\n");
	/* generate implemented function decs for all specs
	   and write them to the .h file, 
	   instead of writing to the ".cpp" file as before */
	generate_cfile(cur1->sp, fs);

	//Finish up
	fprintf(fs, "};\n");
    }


    fprintf(fs, "#endif\n\n");
    //Now free m
    Mast_free(m);
}


void indent(FILE *f, int n) {
    int i = 0;
    int tabstop = 4;

    for(i = 0; i < n * tabstop; i++) {
        fprintf(f, " ");
    }

}

//vietha 2003.04.29
/* remove the "<T>" from the class name "SomeThing<T>"  */
void typename_without_template(char* source, char* output){
  char *template_begin,*template_end;
  int len;

  template_begin= strchr(source,'<');
  if(!template_begin) {strcpy(output,source);return;}
  template_end = strrchr(template_begin,'>');
  if(!template_end) {strcpy(output,source);return;}
  template_end++; //points the part after '<T>'

  len = template_begin - source; //number of chars before '<'
  strncpy(output, source, len);
  strcat(output, template_end);
}
