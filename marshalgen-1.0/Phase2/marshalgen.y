/* For  us, semantic values are strings */
%{
#define YYSTYPE char *
#include "mast.h"
struct mast *tree;
%}

%token BEGIN_C END_C TOBECOPIED STMTS
%token CLASS MARSHALING UNMARSHALING CONSTRUCTOR
%token ID DOLLARS DOLLARSIZE
%token CONSTANT STRINGLITERAL BRACKETS ASTERISKS TEMPLATE
%token DOLLAR_THIS DOLLAR_ELEMENT DOLLAR_ELE_NUM DOLLAR_INDEX
%token DOLLAR_THIS_SHADOW TYPE_CHOICE
%token OTHER


%start program
%%

program
	: quotes specs {$$ = (char *)Mast_init($1, (struct specs*)$2);
						tree = (struct mast*)$$;}
	;

quotes
	:
	| BEGIN_C header END_C {$$ = $2;}
	;

header
	: OTHER			{$$ = $1;}
	| OTHER stmts		{$$ = (char *)strconcat(2, $1, $2);}
	;

specs
	: spec {$$ = (char*)Specs_init((struct spec*)$1, NULL);}
	| spec specs 
	{ $$ = (char *)Specs_init((struct spec*)$1, (struct specs*)$2);}
	;

spec
	: template_declare MARSHALING CLASS mtype '(' formals ')' parent_classes '{' fieldspecs  UNMARSHALING CONSTRUCTOR  constructor_call '}'
		{$$ = (char *)Spec_init($1,$4, (struct formals *)$6, $8,
                                    (struct fieldspecs*)$10, $13);}
	;

template_declare
        : TEMPLATE '<' class_option  template_name '>' {$$ = (char*)strconcat(7,$1," ",$2,$3," ",$4,$5);}
        | {$$ = " ";}
        ;

class_option : CLASS { $$ = $1;}
             | { $$ = " ";}
             ;

template_name:  ID { $$ = $1;}
             ;

formals
    : type ID formalsrest     { $$ = (char*)Formals_init($1, $2,
                                        (struct formals *)$3); }
    ;

formalsrest
    :                            { $$ = NULL; }
    | ',' type ID formalsrest    { $$ = (char*)Formals_init($2, $3,
                                            (struct formals *)$4) }
    ;

parent_classes:  ':' ID ID parent_list {$$ = (char*)strconcat(7, $1," ",$2," ",$3," ",$4);}
               | { $$ = "";}
               ;

parent_list : 
            ',' ID ID parent_list { $$ = (char*)strconcat(6, $1," ",$2," ",$3, $4);}
           | { $$ = "";}
           ;

fieldspecs
	:                   { $$ = NULL; }
	| fieldspec fieldspecs
			{$$ = (char*)Fieldspecs_init(
				(struct fieldspec*)$1, (struct fieldspecs*)$2);}
	;

fieldspec
	: type ID ';' gss
	{$$ = (char *)Fieldspec_init($1, $2, (struct gss*)$4);}
	;

/* gss - Get, Set, and Size */
gss
	:							{$$ = NULL;}
	| getspec setspec sizespec	{$$ = (char *)Gss_init($1, $2, $3);}
	;

getspec
	: '{' stmts '}'		{$$ = (char *)strconcat(3, "{", $2, "}");}
	;

setspec
	: '{' stmts '}'		{$$ = (char *)strconcat(3, "{", $2, "}");}
	;

sizespec
	: '{' stmts '}'		{$$ = (char *)strconcat(3, "{", $2, "}");}
	;

constructor_call
	: '{' stmts '}' 	{$$ = (char *)strconcat(3, "{", $2, "}");}
	;

stmts
	: OTHER			{$$ = $1;}
	| DOLLARS		{$$ = "msh_cursor";}
	| DOLLARSIZE		{$$ = "msh_currentSize"; }
        | DOLLAR_THIS		{$$ = "param"; }
        | DOLLAR_THIS_SHADOW	{$$ = "Shadowed_param"; }
	| DOLLAR_ELEMENT	{$$ = "anElement"; }
	| DOLLAR_ELE_NUM	{$$ = "elementNum"; }
        | DOLLAR_INDEX      	{$$ = "index"; }
        | TYPE_CHOICE      	{$$ = "msh_typechoice"; }

        | OTHER stmts		{$$ = (char *)strconcat(2, $1, $2);}
	| DOLLARS stmts		{$$ = (char *)strconcat(2, "msh_cursor", $2);}
	| DOLLARSIZE stmts	{$$ = (char *)strconcat(2, "msh_currentSize", $2);}
	| DOLLAR_THIS stmts	{$$ = (char *)strconcat(2, "param", $2);}
	| DOLLAR_THIS_SHADOW stmts {$$ = (char *)strconcat(2, "Shadowed_param", $2);}
	| DOLLAR_ELEMENT stmts	{$$ = (char *)strconcat(2, "anElement", $2);}
	| DOLLAR_ELE_NUM stmts	{$$ = (char *)strconcat(2, "elementNum", $2);}
	| DOLLAR_INDEX stmts	{$$ = (char *)strconcat(2, "index", $2);}
	| TYPE_CHOICE stmts	{$$ = (char *)strconcat(2, "msh_typechoice", $2);}
        ;

/* support templated class name */
mtype
        : ID {$$ = $1;} 
        | ID ASTERISKS {$$ = (char*)strconcat(2, $1, $2);}
        | ID '<' mtype '>' {$$ = (char*)strconcat(4, $1,$2,$3,$4);}
        | ID '<' mtype '>' ASTERISKS {$$ = (char*)strconcat(5, $1,$2,$3,$4,$5);}
        ;

/* type for declarations */
type
	: mtype				{$$ = $1;}
	| mtype BRACKETS		{$$ = (char*)strconcat(2, $1, $2);}
	| mtype '&'			{$$ = (char*)strconcat(2, $1, $2);}

	| CLASS mtype			{$$ = (char*)strconcat(2, $1, $2);}
	| CLASS mtype BRACKETS {$$ = (char*)strconcat(3, $1, $2, $3);}
	| CLASS mtype '&'		{$$ = (char*)strconcat(3, $1, $2, $3);}
	;


%%

#include <stdio.h>
#include <stdarg.h>

extern char yytext[];
extern int column;
int yy_line = 1;

yyerror(s)
char *s;
{
        fflush(stdout);
	printf("%s",yylval);
        printf("\n%*s\nLine %d: %*s\n", column, "^", yy_line, column, s);
}



main(int argc, char **argv)
{
    extern FILE *yyin;
    FILE *fout;
    if(argc < 3){
	printf("Usage: %s <input .msh file> <ouput filename>\n", argv[0]);
	exit(1);
    }
    yyin = fopen(argv[1],"r");

    fout = fopen(argv[2], "w+");
    if(fout == NULL) {
        printf("Can't create output file %s", argv[2]);
        exit(-1);
    }


    yyparse();
    mast_print(tree);
    /* printf("num_specs = %d, num_fieldspecs = %d\n", num_specs(tree),
	   num_fieldspecs(tree->sp->sp->fsp)); */
    generate_files(tree, fout);
    
    fclose(yyin);
    fclose(fout);
    exit(0);
}
