/*
 * %{ HEADER %}
 * marshaling class (...) { body }
 * body := decls; | decls; { GSS } { GSS } { GSS }
 * GSS means "Get, Set, or Size" (Get field for marshaling, Set field for
 *	                          unmarshaling, Size of field)
*/

D                       [0-9]
L                       [a-zA-Z_]
H                       [a-fA-F0-9]
E                       [Ee][+-]?{D}+
FS                      (f|F|l|L)
IS                      (u|U|l|L)*
COLON                   ":"

BEGIN_C					"%{"
END_C					"%}"
BRACKETS				"[]"

%{
#define YYSTYPE char *

#include <stdio.h>
#include <stdlib.h> /* for malloc() */
#include "marshalgen.tab.h"

static void count();
static int nested_braces = 0;
static void error(char *str) { printf(str); printf("/n"); abort(); }
extern int yy_line;
%}

%x HEADER
%x GSS
%x GSS_C_COMMENT
%x C_COMMENT


%%
[\n]		        { yy_line++; count(); }

"%{"			{ count(); BEGIN(HEADER); return(BEGIN_C); }
<HEADER>[^%]*		{ count(); return(OTHER); }
<HEADER>"%}"		{ count(); BEGIN(INITIAL); return(END_C); }

"{"			{ count(); if (++nested_braces == 2) BEGIN(GSS);
			  return('{'); }
"}"			{ count();
			  if (--nested_braces < 0) error("unbalanced braces");
			  return('}'); }

<GSS_C_COMMENT>[^*]*	{ count(); return(OTHER); }
<GSS_C_COMMENT>"*"+[^*/]* { count(); return(OTHER); }
<GSS_C_COMMENT>"*"+"/"	{ count(); BEGIN(GSS); return(OTHER); }

<C_COMMENT>[^*]*	{ count(); }
<C_COMMENT>"*"+[^*/]*	{ count(); }
<C_COMMENT>"*"+"/"	{ count(); BEGIN(INITIAL); }

<GSS>\"(\\.|[^\\"])*\"	{ count(); return(OTHER); /* STRING_LITERAL */ }
<GSS>"/*"		{ count(); BEGIN(GSS_C_COMMENT); return(OTHER); }
<GSS>"$"                { count(); return(OTHER); }
<GSS>"$$"               { count(); return(DOLLARS); }
<GSS>"$SIZE"		{count(); return(DOLLARSIZE); }
<GSS>"$THIS"	        {count(); return(DOLLAR_THIS); }
<GSS>"$THIS_SHADOW"	{count(); return(DOLLAR_THIS_SHADOW); }
<GSS>"$ELEMENT"		{count(); return(DOLLAR_ELEMENT); }

<GSS>"$ELE_NUM"	        {count(); return(DOLLAR_ELE_NUM); }
<GSS>"$ELE_COUNT"	{count(); return(DOLLAR_ELE_NUM); }

<GSS>"$INDEX"    	{count(); return(DOLLAR_INDEX); }
<GSS>"$ELE_INDEX"	{count(); return(DOLLAR_INDEX); }
<GSS>"$TYPE_CHOICE"	{count(); return(TYPE_CHOICE); }

<GSS>[^{}$]*             { count(); return(OTHER); }
<GSS>"{"                { count(); ++nested_braces; return(OTHER); }
<GSS>"}"                { count();
			  if (--nested_braces < 1) error("internal error");
                          if (nested_braces == 1)
                            { BEGIN(INITIAL); return('}'); }
			  else return(OTHER);
                        }

"//".*\n { yy_line++; count(); /* C++-style comments in .msh */ }

"("			{ count(); return('('); }
")"			{ count(); return(')'); }
","                     { count(); return(','); }
":"			{ count(); return(':'); }
";"			{ count(); return(';'); }
("*")+			{ count(); return(ASTERISKS); }
"&"			{ count(); return('&'); }
"[]"			{ count(); return(BRACKETS); }
"<"			{ count(); return('<'); }
">"			{ count(); return('>'); }

"template"              { count(); return(TEMPLATE); }
"class"                 { count(); return(CLASS); }
"struct"                 { count(); return(CLASS); }
"marshaling"            { count(); return(MARSHALING); }
"unmarshaling"          { count(); return(UNMARSHALING); }
"constructor"           { count(); return(CONSTRUCTOR); }
{D}+{IS}?               { count(); return(CONSTANT); }
{D}+{E}{FS}?            { count(); return(CONSTANT); }
{D}*"."{D}+({E})?{FS}?  { count(); return(CONSTANT); }
{D}+"."{D}*({E})?{FS}?  { count(); return(CONSTANT); }

{L}({L}|{D}|{COLON})*	{ count(); return(ID); }
[ \t\v\f]		{ count(); }
.			{ count(); return(OTHER); }

%%

yywrap()
{
        return(1);
}


int column = 0;

void count()
{
        int i;

        /* This was added to define the semantic value of the current yytext */
        yylval = memcpy( malloc(yyleng + 1), yytext, yyleng );
	yylval[yyleng] = '\0';

        for (i = 0; yytext[i] != '\0'; i++)
                if (yytext[i] == '\n')
                        column = 0;
                else if (yytext[i] == '\t')
                        column += 8 - (column % 8);
                else
                        column++;

        //ECHO;
}
