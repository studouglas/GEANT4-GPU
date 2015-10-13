#include <stdio.h>
#include <stdlib.h>

struct mast {
	char *macros;
	struct specs *sp;
};

struct spec {
	char *mtype;
	struct formals *fmls;
	struct fieldspecs *fsp;
        //vietha 2003.4.29
	char* template_decl;
	char* parent_classes;
    //vietha 2004.08.29
    char* constructor_call;
};

struct formals {
    char *type;
    char *name;
    struct formals *next;
};

struct specs {
	struct spec *sp;
	struct specs *next;
};

struct fieldspec {
	char *type;
	char *name;
	struct gss *g;
};

struct gss {
	char *getsp;
	char *setsp;
	char *sizesp;
};

struct fieldspecs {
	struct fieldspec *fsp;
	struct fieldspecs *next;
};

int num_specs(struct mast *m);
int num_fieldspecs(struct fieldspecs *f);

char *strconcat(int n, ...);

struct mast *Mast_init(char *m, struct specs *s);
//struct spec *Spec_init(char *mt, struct formals *fmls, struct fieldspecs *f);
struct spec *Spec_init(char* template_decl, char *mt, struct formals *fmls, char* parent_classes, struct fieldspecs *f, char *constructor);
struct specs *Specs_init(struct spec *sp, struct specs *nt);
struct fieldspec *Fieldspec_init(char *t, char *nm, struct gss *g);
struct gss *Gss_init(char *g, char *s, char *sz);
struct fieldspecs *Fieldspecs_init(struct fieldspec *f, struct fieldspecs *nt);
struct formals *Formals_init(char *type, char *name,
                                struct formals *fmlsrest); 

void Mast_free(struct mast *m);
void Spec_free(struct spec *s);
void Specs_free(struct specs *s);
void Fieldspec_free(struct fieldspec *fsp);
void Fieldspecs_free(struct fieldspecs *fsps);
void Gss_free(struct gss *g);
void Formals_free(struct formals *f);

void mast_print(struct mast *m);
void specs_print(struct specs *s);
void spec_print(struct spec *s);
void formals_print(struct formals *fmls);
void formalsrest_print(struct formals *fmls);
void fieldspecs_print(struct fieldspecs *f);
void fieldspec_print(struct fieldspec *f);
void gss_print(struct gss* g);

//void generate_hfile_single(struct spec *s, FILE *fs);
//void generate_hfile_multiple(struct spec *s, FILE *fs);
//void generate_cfile_single(struct spec *s, FILE *fs);
//void generate_cfile_multiple(struct spec *s, FILE *fs);

void generate_hfile(struct spec *s, FILE *fs);
void generate_cfile(struct spec *s, FILE *fs);
void generate_marshalfunc(struct spec *sp, int num, FILE *fs);
void generate_unmarshalfunc(struct spec *sp, int num, FILE *fs);
void generate_files(struct mast *m, FILE *fs);
void indent(FILE *f, int n);

//vietha 2003.04.29
void typename_without_template(char* source, char* output);
//vietha 2003.05.09
void generate_shadowClass(struct spec *s, FILE *fs);
