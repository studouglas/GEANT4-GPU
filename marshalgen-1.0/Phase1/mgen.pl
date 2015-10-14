#!/usr/bin/perl -w
use strict;
use 5;
use File::Basename;

#### Global variables

my %type_hash = ("" => 1, "primitive" => 1,"primitive_ptr" => 1,
	      "transient" => 1,"ptr_to_index" =>1 ,"ptr_shallow_copy" => 1,
	      "manual" => 1,"predefined" => 1, "predefined_ptr" => 1,
		 "ptr_as_array" => 1);
#	      "embedded" => 1,"embedded_ptr" => 1,"global" => 1,

my $shadowedClass_parm = "Shadowed_param";

my $line_counter = 0;

my $constructor_call;

####

if (@ARGV != 1){
    print STDERR "Usage: $0 source_file\n";
    exit -1;
}

my $in_file = $ARGV[0];
my $out_file;

#Construct the output filename:

if ($in_file =~ /(.*)\..*/){
  $out_file = "$1\.msh";
}
else {
  $out_file = "$in_file\.msh";
}

open(SRC,  $in_file ) or die "Unable to open input file: $in_file\n<$!>";
open(DEST, ">$out_file") or die "Unable to open output file: $out_file\n<$!>";
select((select(DEST), $| = 1)[0]); #autoflush

my @src;
chomp(@src = (<SRC>));
close SRC;

my $linebuf = "";
my $line;
my $begin = 0;
foreach $line (@src){
  $line_counter++;

  if ($line =~  /\/\/\s*MSH\_include\_end/){
    $begin or error_die("\\MSH_include_end comes before \\MSH_include_begin!", $line);
    $linebuf .= "$line";
    last;
  }
  if ($line =~  /\/\/\s*MSH\_include\_begin/){
    (!$begin) or error_die("Can't have nested \\MSH_include_begin annotations!", $line);
    $begin = 1;
  }
  if ($begin) {$linebuf .= "$line\n";}
}

# Output the following: 
#  %{
#    INCLUDE_MACROS [anything you want to include or defined]
#  %}

my $in_file_basename = basename($in_file);
print DEST "\%\{\n"; 
#print DEST "\#include \"$in_file\"\n"; 
print DEST "\#include \"$in_file_basename\"\n"; 
print DEST "$linebuf\n"; 
print DEST "\%\}\n\n";

print DEST "// This file is generated automatically from $in_file_basename .";
print DEST "  It is an\n"; 
print DEST "//   intermediate file useful for debugging,";
print DEST " but otherwise may be deleted.\n\n"; 

my $ann_begin = 0;
my $marshaledClass_parm;
my $parameter;

$marshaledClass_parm = $parameter = "param";
#$marshaledClass_parm = $parameter = "\$THIS";

my $i;
for ($i = 0; $i <= $#src; $i++){
  $line_counter = $i+1;

  $linebuf = "";
  $line = $src[$i];

  if($line =~ /(.*)\/\*\s*MSH(.*)\s*\:(.*)\*\//){ #single-line comment
      # convert "/* MSH... */" format to "// MSH..." format
      $linebuf = "$1//MSH$2:$3";
  }
  elsif ($line =~ /(.*)\/\*\s*MSH(.*)\s*\:(.*)/){ # multiple-line comment
      #Putting multi-line comments of /* MSH: ... */ into one line
    $linebuf = "$1//MSH$2:$3";
    while($i <= $#src){
      $line = $src[++$i];
      #if ($line =~ /(.*)\*\//) {$linebuf .= $1; last;}
      #else {$linebuf .= $line;}
      if ($line =~ /(.*)\*\//) {$linebuf .= "\n$1"; last;}  #remove the "*/" at the end
      else {$linebuf .= "\n$line";}
    }
  }
  else {$linebuf = $line;}
  my $lookahead_line;
  if ($i <$#src){ $lookahead_line = $src[$i+1];}

  #print "linebuf=$linebuf\n";

  if (!$ann_begin && $linebuf =~ /\/\/\s*MSH\_BEGIN/){
      $ann_begin = 1;
      my $buffer = "";
      my ($type, $className, $parent_classes, $template_declare) = getClassName($lookahead_line);

      #print "className=$className\n";

      $buffer .= $template_declare;
      if ($type == 0){$buffer .= " marshaling class Marshaled";}
      else {$buffer .= " marshaling struct Marshaled";}
      $buffer .= "$className \($className\* $parameter\) $parent_classes\{\n";
      print DEST $buffer;

      $constructor_call = "\t\$THIS = new $className();\n";
  }
  elsif ($ann_begin && $linebuf =~ /\/\/\s*MSH\_END/){
      $ann_begin = 0;

      ## generating the option containing the constructor call
      my $constructor_opt = "\tunmarshaling constructor {\n";
      $constructor_opt .= $constructor_call;
      $constructor_opt .= "\t}\n";
      print DEST $constructor_opt;

      print DEST "\}\n\n";
    }
  elsif ($linebuf =~ /\/\/\s*MSH_constructor\s*\:\s*([\s\S]*)/) {
      $constructor_call = "\t\$THIS = new $1;\n";
  }
  elsif ($linebuf =~ /\/\/\s*MSH_superclass\s*\:\s*([\s\S]*)/) 
  # || ($linebuf =~ /\/\*\s*MSH_superclass\:\s*([\s\S]*)\*\//))
  {
    my $marsh_buffer = marshal_baseclass($1);
    print DEST $marsh_buffer;
  }
  elsif ($linebuf =~ /\/\/\s*MSH_derivedclass\s*\:\s*([\s\S]*)/) 
   #|| ($linebuf =~ /\/\*\s*MSH_derivedclass\s*\:\s*([\s\S]*)\*\//))
  {
    my $marsh_buffer = marshal_derivedclass($1);
    print DEST $marsh_buffer;
  }
  elsif ($linebuf =~ /\/\/\s*MSH\s*\:.*/)
  # || ($linebuf =~ /\/\*\s*MSH\:.*/))
  {
    my $marsh_buffer = marshal_annot($linebuf);
    print DEST $marsh_buffer;
  }
}

close DEST;
exit 0;


#input: "template <class T> class G4THitsCollection : public G4HitsCollection \n"
#parent_classes: ": public G4HitsCollection \n"
#template_declare: "template <class T>"
#className: "G4THitsCollection<T>"



sub getClassName{
  my $line_buffer = shift;
  my $type = -1;
  my $parent_classes = "";
  my $className = "";
  my $template_declare = "";

  #if ($line_buffer =~ /template\s+\<class\s+(\S+)\>\s+class\s+(\S+)(.*)/){
  # extract until encounter '{'}
  if ($line_buffer =~ /template\s+\<class\s+(\S+)\>\s+class\s+(\S+)([^{]*)/){
    $className = "$2\<$1\>";
    $parent_classes = $3;
    $template_declare = "template \<class $1\>";
    $type = 0;
  }
  #elsif ($line_buffer =~ /template\s+\<struct\s+(\S+)\>\s+struct\s+\S+(.*)/){
  elsif ($line_buffer =~ /template\s+\<struct\s+(\S+)\>\s+struct\s+(\S+)([^{]*)/){
    $className = "$2\<$1\>";
    $parent_classes = $3;
    $template_declare = "template \<struct $1\>";
    $type = 1;
  }
  #elsif($line_buffer =~ /class\s+(\S+)/){
  elsif($line_buffer =~ /class\s+([a-zA-Z0-9_]+)/){
    $className = $1;
    $type = 0;
  }
  #elsif($line_buffer =~ /struct\s+(\S+)/){
  elsif($line_buffer =~ /struct\s+([a-zA-Z0-9_]+)/){
    $className = $1;
    $type = 1;
  }
  else{
    error_die("Could not find class name", $line_buffer);
  }

  #print "className=$className;template_decl=$template_declare\n";

return ($type, $className, $parent_classes, $template_declare);
}


sub marshal_annot{
  my $linebuf = shift;
  my $vartype;
  my $varname;
  my $anntype;
  my $marshalbuf;
  my $token;


  #if ($linebuf =~ /\/\/\s*MSH\_virtual\s*\:([\s\S]*)/){
  #     $linebuf = $1;
  #   }
  #elsif ($linebuf =~ /\/\*\s*MSH\_virtual\s*\:([\s\S]*)/){
  #  $linebuf = $1;
  #}
  #print "Linebuf=$linebuf\n";

  #if ($linebuf =~ /\s+([a-zA-Z0-9_]+)\:\:([a-zA-Z0-9_]+)/){
  if ($linebuf =~ /\s*(\S+\s*\**)\s+([a-zA-Z0-9_]+)\s*;/ ||
      $linebuf =~ /\s*(\S+\s+\**)\s*([a-zA-Z0-9_]+)\s*;/){
    $vartype = $1;
    $varname = $2;
  }
  else {error_die("Could not extract the name and the type of the data field", $linebuf);}
  #print "Vartype=$vartype; varname=$varname\n";

  if ($linebuf =~ /(.*\;)\s*\/\/\s*MSH\s*:\s*([a-zA-Z0-9_]*)\s*/){
    $anntype = $2;
    $token = $1;
  }
  #print "Token=$token; anntype=$anntype\n";

  error_die("Invalid annotation: $anntype",$linebuf) unless defined($type_hash{$anntype});

  unless ($anntype eq "" || $anntype eq "transient"){
    $marshalbuf = "\n$token\n";
    $marshalbuf .= annot_marshalling($linebuf, $anntype, $vartype, $varname);
    $marshalbuf .= annot_unmarshaling($linebuf, $anntype, $vartype, $varname);
    $marshalbuf .= annot_getSize($linebuf, $anntype, $vartype, $varname);
  }
  return $marshalbuf;

}

sub marshal_baseclass{
  my $baseclass = shift;
  #print "Baseclass=$baseclass;\n";

  my $annot;
  my $rand_num;

  $rand_num = int(rand(1000));

  # generate a dummy field
  $annot = "    int __dummy$rand_num; // marshaling code for MSH_superclass\n";
  ## marshaling code 
  $annot .= "    //FIELDMARSHAL:\n    {\n";
  $annot .= "\t\tMarshaled$baseclass marParent(\$THIS);\n";
  $annot .= "\t\tEXTEND_BUFFER(marParent.getBufferSize());\n";
  $annot .= "\t\tmemcpy(\$\$,marParent.getBuffer(), marParent.getBufferSize());\n";
  $annot .= "\t\t\$SIZE = marParent.getBufferSize();\n";
  $annot .= "\n    }\n";

  ## unmarshaling code 
  $annot .= "    //FIELD UNMARSHAL:\n    {\n";
  $annot .= "\t\tMarshaled$baseclass marObj(\$\$);\n";
  $annot .= "\t\tmarObj.unmarshalTo(\$THIS);\n";
  $annot .= "\n    }\n";

  ## size
  $annot .= "    //FIELD SIZE :\n    {\n";
  $annot .= "\t\t//code for size, just dummy code because the size will be set correctly at the end of marshaling code\n";
  #$annot .= "\t\t\$SIZE = 0;\n";
  $annot .= "\n    }\n";
  return $annot;
}

sub marshal_derivedclass{
  my $derivedclass = shift;
  #print "Derived class=$derivedclass;\n";
  $constructor_call = "";  # reset the code for constructor call

  my ($numOfTypes, $cref, $rref, $conref) = extractMultipleTypes($derivedclass);
  my @arrTypeChoices = @{$cref};
  my @arrTypeResults = @{$rref};
  my @arrConstructorArgs = @{$conref};
  my $isMultipleTypes = ($numOfTypes>=2);
  my $strElementType;

  ## generating calls to the correct constructors (constructor of the child class)
  if ($isMultipleTypes) {
      $constructor_call .= "\tif(0){}\n";
  }
  for (my $i=0;$i<$numOfTypes;$i++) {
      if (!$isMultipleTypes) {
	  #$strElementType = $derivedclass;
	  $strElementType = $arrTypeResults[$i];
      } else {
	  my $choice = $arrTypeChoices[$i];
	  #$constructor_call .= "\telse if\($choice\){\n";
	  $constructor_call .= "\telse if(\$TYPE_CHOICE == $i){\n";
	  $strElementType = $arrTypeResults[$i];
      }

      my $consArg = $arrConstructorArgs[$i];
      $constructor_call .= "\t\$THIS = new $strElementType($consArg);\n";
      if ($isMultipleTypes) {
	  $constructor_call .= "\t}\n";
      }
  }


  ## generating the marshaling/unmarshaling/size code
  my $annot;
  my $rand_num;

  $rand_num = int(rand(1000));

  # generate a dummy field
  $annot = "    int __dummy$rand_num; // marshaling code for MSH_derivedclass\n";

  ## marshaling code 
  $annot .= "    //FIELDMARSHAL:\n    {\n";
  if ($isMultipleTypes) {
      $annot .= "\tif(0){}\n";
  }
  for (my $i=0;$i<$numOfTypes;$i++) {
      if (!$isMultipleTypes) {
	  #$strElementType = $derivedclass;
	  $strElementType = $arrTypeResults[$i];
      } else {
	  my $choice = $arrTypeChoices[$i];
	  $annot .= "\telse if\($choice\){\n";
	  $strElementType = $arrTypeResults[$i];
      }

      $annot .= "\t\t$strElementType *aObj$rand_num = ($strElementType*)\$THIS;\n";
      $annot .= "\t\tMarshaled$strElementType marChild(aObj$rand_num);\n";
      $annot .= "\t\tEXTEND_BUFFER(marChild.getBufferSize());\n";
      $annot .= "\t\tmemcpy(\$\$,marChild.getBuffer(), marChild.getBufferSize());\n";
      $annot .= "\t\t\$SIZE = marChild.getBufferSize();\n";
      $annot .= "\t\t\$TYPE_CHOICE = $i;\n";
      if ($isMultipleTypes) {
	  $annot .= "\t}\n";
      }
  }

  $annot .= "\n    }\n";

  ## unmarshaling code 
  $annot .= "    //FIELD UNMARSHAL:\n    {\n";
  if ($isMultipleTypes) {
      $annot .= "\tif(0){}\n";
  }
  for (my $i=0;$i<$numOfTypes;$i++) {
      if (!$isMultipleTypes) {
	  #$strElementType = $derivedclass;
	  $strElementType = $arrTypeResults[$i];
      } else {
	  my $choice = $arrTypeChoices[$i];
	  #$annot .= "\telse if\($choice\){\n";
	  $annot .= "\telse if(\$TYPE_CHOICE == $i){\n";
	  $strElementType = $arrTypeResults[$i];
      }

      $annot .= "\t\tMarshaled$strElementType marObj(\$\$);\n";
      $annot .= "\t\tmarObj.unmarshalTo(($strElementType*)\$THIS);\n";
      if ($isMultipleTypes) {
	  $annot .= "\t}\n";
      }
  }
  $annot .= "\n    }\n";

  ## size
  $annot .= "    //FIELD SIZE :\n    {\n";
  $annot .= "\t\t//code for size, just dummy code because the size will be set correctly at the end of marshaling code\n";
  # $annot .= "\t\t\$SIZE = 0;\n";
  $annot .= "\n    }\n";
  return $annot;
}


sub annot_marshalling{
  my $abuf = shift;
  my $atype = shift;
  my $vartype = shift;
  my $varname = shift;
  my $annot = "    //FIELDMARSHAL:\n    {\n";

  my $strTypeOption = get_Option_Name( $abuf, "elementType:");

  if ($atype eq "primitive"){
      if(!$strTypeOption){ #if there is no customized "elementType"
	  my $strElementGet = get_Option_Code($abuf, "elementGet:");
	  if (!$strElementGet){
	      $annot .= "\tmemcpy\(\$\$\, \&$shadowedClass_parm\-\>$varname\, sizeof\($vartype\)\);"
	      }
	  else {
	      $annot .= "\t$vartype \$ELEMENT\;\n\t$strElementGet\n";
	      $annot .= "\tmemcpy(\$\$, \&\$ELEMENT, sizeof\($vartype\)\)\;"
	      }
      }else { #if there is customized "elementType" option
	  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
	  my @arrTypeChoices = @{$cref};
	  my @arrTypeResults = @{$rref};
	  my $isMultipleTypes = ($numOfTypes>=2);
	  my $strElementType;

      # I know I repeat here the same code as in function "annotate_ptr_as_array"
      # Should better unify them in one place, but no time yet. (vietha Aug 23,2004)
	  if ($isMultipleTypes) {
	      $annot .= "\tif(0){}\n";
	  }
	  for (my $i=0;$i<$numOfTypes;$i++) {
	          if (!$isMultipleTypes) {
		      # single type, strElemenType == strTypeVariable
		      $strElementType = $strTypeOption;
		  } else {
		      my $choice = $arrTypeChoices[$i];
		      $annot .= "\telse if\($choice\){\n";
		      $strElementType = $arrTypeResults[$i];
		  }

		  my $strElementGet = get_Option_Code($abuf, "elementGet:");
		  if (!$strElementGet){
		      $annot .= "\t\tmemcpy\(\$\$\, \&$shadowedClass_parm\-\>$varname\, sizeof\($strElementType\)\);\n";
		  }
		  else {
		      $annot .= "\t\t$strElementType \$ELEMENT\;\n\t$strElementGet\n";
		      $annot .= "\t\tmemcpy(\$\$, \&\$ELEMENT, sizeof\($strElementType\)\)\;\n"
		      }
		  if ($isMultipleTypes) {
		      $annot .= "\t}\n";
		  }
	  }
      }
  }
  elsif ($atype eq "primitive_ptr"){

    my ($strElementGet) = get_Option_Code($abuf, "elementGet:");
    if (!$strElementGet){
      my $vartypeDeref = chopOneAsterisk($vartype);
      $annot .= "\tmemcpy\(\$\$\, $shadowedClass_parm\-\>$varname\, sizeof\($vartypeDeref\)\);"
    }
    else {
      $annot .= "\t$vartype\* \$ELEMENT\;\n\t$strElementGet\n";
      $annot .= "\tmemcpy(\$\$\, \$ELEMENT, sizeof\($vartype\)\)\;"
    }
  }
  elsif ($atype eq "ptr_to_index"){
    my $array_size = get_array_info($abuf);
    $annot .= "      memcpy\( \$\$\, $marshaledClass_parm\-\>$varname";
    $annot .= ", sizeof\( $vartype \) \* $array_size \)\; ";
  }
  elsif ($atype eq "ptr_shallow_copy"){
    $annot .= "      memcpy\( \$\$\, \&\$varname\, sizeof\( int \)\)\;";
  }
  elsif ($atype eq "manual"){
    my ($marshal, $unmarshal, $getSize) = get_functions($abuf);
    #if( !$marshal){
    #  error_die("Marshaling code expected for \"manual\" annotation",$abuf);
    #}
    $annot .= $marshal;
  }
  elsif ($atype eq "predefined"){
      if(!$strTypeOption){ #if there is no customized "elementType"
	  my ($strElementGet) = get_Option_Code($abuf, "elementGet:");
	  if(!$strElementGet) { 
	      $annot.= "\tMarshaled$vartype var\(\&$shadowedClass_parm\-\>$varname\)\;\n";
	  }else{
	      $annot .= "\t$vartype \$ELEMENT\;\n";
	      $annot .= "\t$strElementGet\n";
	      $annot .= "\tMarshaled$vartype var\(\&\$ELEMENT\)\;\n";
	  }
	  $annot .= "\tEXTEND_BUFFER(var\.getBufferSize\(\)\)\;\n";
	  $annot .= "\t\$SIZE = var\.getBufferSize\(\)\;\n";
	  $annot .= "\tmemcpy\(\$\$\, var\.getBuffer\(\)\, var\.getBufferSize\(\)\)\;";
      }else { #if there is customized "elementType" option
	  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
	  my @arrTypeChoices = @{$cref};
	  my @arrTypeResults = @{$rref};
	  my $isMultipleTypes = ($numOfTypes>=2);
	  my $strElementType;
	  # I know I repeat here the same code as in function "annotate_ptr_as_array"
	  # Should better unify them in one place, but no time yet. (vietha Aug 23,2004)
	  if ($isMultipleTypes) {
	      $annot .= "\tif(0){}\n";
	  }
	  for (my $i=0;$i<$numOfTypes;$i++) {
	      if (!$isMultipleTypes) {
		  # single type, strElemenType == strTypeVariable
		  $strElementType = $strTypeOption;
	      } else {
		  my $choice = $arrTypeChoices[$i];
		  $annot .= "\telse if\($choice\){\n";
		  $strElementType = $arrTypeResults[$i];
	      }

	      # one of the type of choices might be primitive
	      my $isPrimitiveType=IsPrimitiveType($strElementType);

	      my $strElementGet = get_Option_Code($abuf, "elementGet:");
	      if (!$strElementGet){
		  if($isPrimitiveType){
		      $annot .= "\tEXTEND_BUFFER(sizeof\($strElementType\)\)\;\n";
		      $annot .= "\t\$SIZE = sizeof\($strElementType\);\n";
		      $annot .= "\t\tmemcpy\(\$\$\, \&$shadowedClass_parm\-\>$varname\, sizeof\($strElementType\)\);\n";
		  }else{
		      $annot.= "\t\tMarshaled$strElementType var\(\&$shadowedClass_parm\-\>$varname\)\;\n";
		      $annot .= "\tEXTEND_BUFFER(var\.getBufferSize\(\)\)\;\n";
		      $annot .= "\t\$SIZE = var\.getBufferSize\(\)\;\n";
		      $annot .= "\t\tmemcpy\(\$\$\, var\.getBuffer\(\)\, var\.getBufferSize\(\)\)\;\n";
		  }
	      }
	      else {
		  if($isPrimitiveType){
		      $annot .= "\t\t$strElementType \$ELEMENT\;\n\t$strElementGet\n";
		      $annot .= "\tEXTEND_BUFFER(sizeof\($strElementType\)\)\;\n";
		      $annot .= "\t\$SIZE = sizeof\($strElementType\);\n";
		      $annot .= "\t\tmemcpy(\$\$, \&\$ELEMENT, sizeof\($strElementType\)\)\;\n"

		  }else{
		      $annot .= "\t\t$strElementType \$ELEMENT\;\n";
		      $annot .= "\t\t$strElementGet\n";
		      $annot .= "\t\tMarshaled$strElementType var\(\&\$ELEMENT\)\;\n";
		      $annot .= "\tEXTEND_BUFFER(var\.getBufferSize\(\)\)\;\n";
		      $annot .= "\t\$SIZE = var\.getBufferSize\(\)\;\n";
		      $annot .= "\t\tmemcpy\(\$\$\, var\.getBuffer\(\)\, var\.getBufferSize\(\)\)\;\n";
		  }
	      }

	      if ($isMultipleTypes) {
		  $annot .= "\t}\n";
	      }
	  }
      }
  }
  elsif( $atype eq "predefined_ptr"){
    my ($strElementGet) = get_Option_Code($abuf, "elementGet:");
    my $vartypeDeref = chopOneAsterisk($vartype);
    if (!$strElementGet) { 
      	  $annot .= "\tMarshaled$vartypeDeref var\($shadowedClass_parm\-\>$varname\)\;\n";
        }else{
	  $annot .= "\t$vartype\* \$ELEMENT\;\n";
	  $annot .= "\t$strElementGet\n";
	  $annot .= "\tMarshaled$vartypeDeref var\(\$ELEMENT\)\;\n";
	}
        $annot .= "\tEXTEND_BUFFER(var\.getBufferSize\(\)\)\;\n";
        $annot .= "\t\$SIZE = var\.getBufferSize\(\)\;\n";
	$annot .= "\tmemcpy\(\$\$\, var\.getBuffer\(\)\, var\.getBufferSize\(\)\);";
  }
  elsif ($atype eq "ptr_as_array"){
    my $outstr = annotate_ptr_as_array($abuf);
    if (!$outstr){
      error_die("Failed to parse the annotation",$abuf);
    }
    else {$annot .= $outstr;}
  }
  #elsif ($atype eq "embedded" || $atype eq "embedded_ptr" || $atype eq "global"){
  #  die "The annotation type $atype does not exist!\n";
  #}
  $annot .= "\n    }\n";
  #$marshal = $annot;
  return $annot;
}


sub annotate_ptr_as_array {
  my $annot_buf = shift;
  my $annot = "";
  my $strTypeOption = get_Option_Name( $annot_buf, "elementType:");
  if ( !$strTypeOption ) {
    error_die("Unrecognizable options in ptr_as_array, \"elementType\" expected",$annot_buf);
  }

  # accept both keywords "elementCount" and "elementNum"
  my $strElementNum = get_Option_Code( $annot_buf, "elementCount:");
  if ( !$strElementNum ) {
      my $strElementNum = get_Option_Code( $annot_buf, "elementNum:");
      if ( !$strElementNum ) {
      error_die("Unrecognizable options in ptr_as_array, \"elementCount\" expected",$annot_buf);
  }
  }

  my $strElementGet = get_Option_Code( $annot_buf, "elementGet:");
  if (!$strElementGet ) {
    error_die("Unrecognizable options in ptr_as_array, \"elementGet\" expected",$annot_buf);
  }


  $annot .= "\tint copy_off \= 0\;\n";
  $annot .= "\tint \$ELE_COUNT\;\n";

  $annot .= "\t$strElementNum\n"; #!!!

  $annot .= "\tmemcpy\( \$\$\+copy_off\, \&\$ELE_COUNT\,sizeof\(int\)\)\;\n";
  $annot .= "\tcopy_off \+\= sizeof\(int\)\;\n";
  $annot.= "\tfor\(int \$ELE_INDEX\=0;\$ELE_INDEX\<\$ELE_COUNT\;\$ELE_INDEX\+\+\)\{\n";


  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
  my @arrTypeChoices = @{$cref};
  my @arrTypeResults = @{$rref};
  my $isMultipleTypes = ($numOfTypes>=2);
  my $strElementType;

  #print "numOfTypes=$numOfTypes\n";

  if ($isMultipleTypes) {
    $annot .="\t\tif(0){}\n";
  }
  my $asterisk;

  for (my $i=0;$i<$numOfTypes;$i++) {
    if (!$isMultipleTypes) {
      # single type, strElemenType == strTypeVariable
      $strElementType = $strTypeOption;
    } else {
      my $choice = $arrTypeChoices[$i];
      $annot .= "\t\telse if\($choice\){\n";
      $strElementType = $arrTypeResults[$i];
    }

    my $strElementTypeDeref =  $strElementType;
    # Since the name of type used with "Marshaled..." should not contain "*", we remove it
    if ($strElementTypeDeref =~ /\*/) {
      #$strElementTypeDeref =~ s/\*.*//; 
      $strElementTypeDeref = chopOneAsterisk($strElementTypeDeref);
      $asterisk = 1;
    }

    my $isPrimitiveType=IsPrimitiveType($strElementType);


    if ($isMultipleTypes) {
      if ($asterisk) {
	$annot .= "\t\t\tvoid\* \$ELEMENT\;\n";
      } else {
	my $result = $arrTypeResults[$i];
	$annot .= "\t\t\t$result \$ELEMENT \;\n";
      }
      $annot .= "\t\t\t$strElementGet\n";
      if($isPrimitiveType){
	  # nothing
      }elsif ($asterisk) {
	$annot .= "\t\t\tMarshaled$strElementTypeDeref marEle\(\($strElementType\)\$ELEMENT\)\;\n";
      } else {			#  if not, add "&" before passing to Marshaled....
	$annot .= "\t\t\tMarshaled$strElementTypeDeref marEle\(\($strElementType\*\)\&\$ELEMENT\)\;\n";
      }
    } else {
      $annot .= "\t\t\t$strElementType \$ELEMENT;\n\t\t\t$strElementGet\n";

      if($isPrimitiveType){
	  # nothing
      }elsif ($asterisk) {		# if the element is already a pointer 
	$annot .= "\t\t\tMarshaled$strElementTypeDeref marEle\(\$ELEMENT\)\;\n";
      } else {			#if not, add "&" before passing to Marshaled....
	$annot .= "\t\t\tMarshaled$strElementTypeDeref marEle\(\&\$ELEMENT);\n";
      }
    }

    if($isPrimitiveType){
	$annot.= "\t\t\tEXTEND_BUFFER\(sizeof\($strElementType\)\);\n";	
	$annot.= "\t\t\tmemcpy\(\$\$\+copy_off\, &\$ELEMENT, sizeof\($strElementType\)\);\n";
	$annot.= "\t\t\tcopy_off \+\= sizeof\($strElementType\);\n";
    }else{
	$annot.= "\t\t\tEXTEND_BUFFER\(marEle\.getBufferSize\(\)\);\n";
	$annot.= "\t\t\tmemcpy\(\$\$\+copy_off\, marEle\.getBuffer\(\)\, marEle\.getBufferSize\(\)\);\n";
	$annot.= "\t\t\tcopy_off \+\= marEle\.getBufferSize\(\);\n";
    }
    $annot.= "\t\t\}\n";

  }
  if ($isMultipleTypes) {
      $annot .= "\t}\n";
  }
  $annot.= "\t\$SIZE \= copy_off;\n";

  return $annot;
}


sub annot_unmarshaling{
  my $annot_buf = shift;
  my $annot_type = shift;
  my $vartype = shift;
  my $varname = shift;
  my $annot = "    //FIELDUNMARSHAL:\n    {\n";

  my $strTypeOption = get_Option_Name( $annot_buf, "elementType:");

  if ($annot_type eq "" || $annot_type eq "transient"){} #do nothing
  elsif ($annot_type eq "primitive"){
      if(!$strTypeOption){ #if there is no customized "elementType"
	  my  $strElementSet = get_Option_Code( $annot_buf, "elementSet:");
	  if(!$strElementSet) { #default
	      $annot .= "\tmemcpy(\&$shadowedClass_parm\-\>$varname\, \$\$\, sizeof\($vartype\)\);\n";}
	  else{
	      $annot .= "\t$vartype \$ELEMENT;\n";
	      $annot .= "\tmemcpy(\&\$ELEMENT, \$\$, sizeof\($vartype\)\);\n";
	      $annot .= "\t$strElementSet\n";
	  }
      }else { #if there is customized "elementType" option
	  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
	  my @arrTypeChoices = @{$cref};
	  my @arrTypeResults = @{$rref};
	  my $isMultipleTypes = ($numOfTypes>=2);
	  my $strElementType;
	  if ($isMultipleTypes) {
	      $annot .= "\tif(0){}\n";
	  }
	  for (my $i=0;$i<$numOfTypes;$i++) {
	          if (!$isMultipleTypes) {
		      # single type, strElemenType == strTypeVariable
		      $strElementType = $strTypeOption;
		  } else {
		      my $choice = $arrTypeChoices[$i];
		      $annot .= "\telse if\($choice\){\n";
		      $strElementType = $arrTypeResults[$i];
		  }

		  my $strElementSet = get_Option_Code($annot_buf, "elementSet:");
		  if (!$strElementSet){
		      $annot .= "\t\tmemcpy(\&$shadowedClass_parm\-\>$varname\, \$\$\, sizeof\($strElementType\)\);\n";
		  }
		  else {
		      $annot .= "\t$strElementType \$ELEMENT;\n";
		      $annot .= "\tmemcpy(\&\$ELEMENT, \$\$, sizeof\($strElementType\)\);\n";
		      $annot .= "\t$strElementSet\n";
		  }
		  if ($isMultipleTypes) {
		      $annot .= "\t}\n";
		  }
	  }
      }
  }
  elsif ($annot_type eq "primitive_ptr"){ #eg int* i;
    my $strElementSet = get_Option_Code($annot_buf, "elementSet:");
    my $vartypeDeref = chopOneAsterisk($vartype);
    $annot .= "\t$shadowedClass_parm\-\>$varname \= ($vartype)malloc\(sizeof\($vartypeDeref\)\);\n";
    if(!$strElementSet) { #default
      $annot .= "\tmemcpy($shadowedClass_parm\-\>$varname\, \$\$\, sizeof\($vartypeDeref\)\);\n";}
    else{
      $annot .= "\t$vartype\* \$ELEMENT;\n";
      $annot .= "\tmemcpy\(\$ELEMENT\, \$\$\, sizeof\($vartype\)\);\n";
      $annot .= "\t$strElementSet\n";
    }
  }
  elsif ($annot_type eq "ptr_to_index"){ #eg. double d_array[12]
    my $array_size = get_array_info($annot_buf);
    $annot .= "      memcpy\( $marshaledClass_parm\-\>$varname";
    $annot .= ", \$\$, sizeof\( $vartype \) \* $array_size \)\; ";
  }
  elsif ($annot_type eq "ptr_shallow_copy"){
    $annot .= "      memcpy( \&\$varname\, \$\$\, sizeof\( int \)\)\;";
  }
  elsif ($annot_type eq "manual"){# manual {marshal()} {unmarshal()} {getSize()}
    # get marshaling function from { marshal() }
    my ($marshal, $unmarshal, $getSize) = get_functions($annot_buf);
    $annot .= $unmarshal;
  }
  elsif ($annot_type eq "predefined"){
      if(!$strTypeOption){ #if there is no customized "elementType"
	  my $strElementSet = get_Option_Code($annot_buf, "elementSet:");
	  $annot .= "\tMarshaled$vartype var\(\$\$\, \'u\'\)\;\n";
	  if(!$strElementSet) {
	      $annot .="\tvar\.unmarshalTo\(\&$shadowedClass_parm\-\>$varname\)\;\n";
	  }
	  else{
	      $annot .= "\t$vartype \$ELEMENT;\n";
	      $annot .= "\tvar\.unmarshalTo\(\&\$ELEMENT\)\;\n";
	      $annot .= "\t$strElementSet\n";
	  }
      }else { #if there is customized "elementType" option
	  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
	  my @arrTypeChoices = @{$cref};
	  my @arrTypeResults = @{$rref};
	  my $isMultipleTypes = ($numOfTypes>=2);
	  my $strElementType;
	  if ($isMultipleTypes) {
	      $annot .= "\tif(0){}\n";
	  }
	  for (my $i=0;$i<$numOfTypes;$i++) {
	      if (!$isMultipleTypes) {
		  # single type, strElemenType == strTypeVariable
		  $strElementType = $strTypeOption;
	      } else {
		  my $choice = $arrTypeChoices[$i];
		  $annot .= "\telse if\($choice\){\n";
		  $strElementType = $arrTypeResults[$i];
	      }

	      # one of the type of choices might be primitive
	      my $isPrimitiveType=IsPrimitiveType($strElementType);

	      my $strElementSet = get_Option_Code($annot_buf, "elementSet:");
	      if (!$strElementSet){
		  if($isPrimitiveType){
		      $annot .= "\t\tmemcpy(\&$shadowedClass_parm\-\>$varname\, \$\$\, sizeof\($strElementType\)\);\n";
		  }
		  else{
		      $annot .= "\t\tMarshaled$strElementType var\(\$\$\, \'u\'\)\;\n";
		      $annot .="\t\tvar\.unmarshalTo\(\&$shadowedClass_parm\-\>$varname\)\;\n";
		  }
	      }
	      else {
		  if($isPrimitiveType){
		      $annot .= "\t\t$vartype \$ELEMENT;\n";
		      $annot .= "\t\tmemcpy(\&\$ELEMENT, \$\$, sizeof\($strElementType\)\);\n";
		      $annot .= "\t\t$strElementSet\n";
		  }else{
		      $annot .= "\t\tMarshaled$strElementType var\(\$\$\, \'u\'\)\;\n";
		      $annot .= "\t\t$strElementType \$ELEMENT;\n";
		      $annot .= "\t\tvar\.unmarshalTo\(\&\$ELEMENT\)\;\n";
		      $annot .= "\t\t$strElementSet\n";
		  }
	      }
	      if ($isMultipleTypes) {
		  $annot .= "\t}\n";
	      }
	  }
  }
  }
  elsif ($annot_type eq "predefined_ptr"){#eg: List *ls;//predefined_ptr
    my $strElementSet = get_Option_Code( $annot_buf, "elementSet:");
    my $vartypeDeref = chopOneAsterisk($vartype);
    $annot .= "\tMarshaled$vartypeDeref var\(\$\$\, \'u\'\);\n";
    if(!$strElementSet) {
      $annot .= "\t$shadowedClass_parm\-\>$varname \= var\.unmarshal\(\)\;\n";
    }
    else{
      $annot .= "\t$vartype\* \$ELEMENT\;\n";
      $annot .= "\t\$ELEMENT \= var\.unmarshal\(\)\;\n";
      $annot .= "\t$strElementSet\n";
    }
  }
  elsif ($annot_type eq "ptr_as_array"){
#      int copy_off = 0;
#	int $ELE_COUNT;
#	memcpy(&$ELE_COUNT, $$+copy_off, sizeof(int));
#	copy_off += sizeof(int);
#	for(int $ELE_INDEX=0;$ELE_INDEX<$ELE_COUNT;$ELE_INDEX++){
#	<!--elementType--> $ELEMENT;
# 	Marshaled<!--elementTypeDeref--> marEle($$+copy_off);
# 	$ELEMENT = marEle.unmarshal();
# 	copy_off += marEle.getBufferSize();
# 	<!--elementSet-->

    my ($strTypeOption, $strElementSet, $strElementTypeDeref);

    $strTypeOption = get_Option_Name( $annot_buf, "elementType:");
    if( !$strTypeOption) {
	error_die("Unrecognizable options in ptr_as_array, \"elementType\" expected",$annot_buf);
    }
    $strElementSet = get_Option_Code( $annot_buf, "elementSet:");
    if(!$strElementSet ) {
	error_die("Unrecognizable options in ptr_as_array, \"elementSet\" expected",$annot_buf);
    }
    # numOfTypes = 1: single type case
    # numOfTypes>= 2: multiple types, with CASE(...) syntax */
    # int numOfTypes = 0;
    # IF strTypeOption ==  CASE{T:DMXPmtHit,DMXScintHit}{DMXPmtHit*,DMXScintHit*}]
    #   THEN strTypeVariable = T
    #   arrTypeChoices = {DMXPmtHit,DMXScintHit}
    #   arrTypeResults = {DMXPmtHit*,DMXScintHit*}
    #   return (Number of types);
    #   ELSE return 1;

    my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
    my @arrTypeChoices = @{$cref};
    my @arrTypeResults = @{$rref};
    my $isMultipleTypes = ($numOfTypes>=2);

    if ($isMultipleTypes) {
      $annot .="\tif(0){}\n";
    }
    my $asterisk;

    for (my $i=0;$i<$numOfTypes;$i++) {
	my ($strElementType, $choice);
      if (!$isMultipleTypes) {
	# single type, strElemenType == strTypeVariable 
	$strElementType = $strTypeOption;
      } else {
	$choice = $arrTypeChoices[$i];
	$annot .= "\telse if\($choice\){\n";
	$strElementType = $arrTypeResults[$i];
      }

      my $strElementTypeDeref =  $strElementType;
      # Since the name of type used with "Marshaled..." should not contain "*", we remove it
      if ($strElementTypeDeref =~ /\*/) {
	#$strElementTypeDeref =~ s/\*.*//; 
	$strElementTypeDeref = chopOneAsterisk($strElementTypeDeref);
	$asterisk = 1;
      }

      my $isPrimitiveType=IsPrimitiveType($strElementType);

      $annot .= "\t\tint copy_off \= 0\;\n";
      $annot .= "\t\tint \$ELE_COUNT\;\n";
      $annot .= "\t\tmemcpy\(\&\$ELE_COUNT\, \$\$\+copy_off\, sizeof\(int\)\)\;\n";
      $annot .= "\t\tcopy_off \+\= sizeof\(int\)\;\n";
      $annot .= "\t\tfor\(int \$ELE_INDEX\=0;\$ELE_INDEX\<\$ELE_COUNT\;\$ELE_INDEX\+\+\)\{\n";
      if($isPrimitiveType){}	
      else{
	  $annot .= "\t\t\tMarshaled$strElementTypeDeref marEle\(\$\$\+copy_off\)\;\n";
      }
      #if ($isMultipleTypes) {
	#my $choice = $arrTypeChoices[$i];
	#my $result = $arrTypeResults[$i];
        if($isPrimitiveType){
	  $annot .= "\t\t\t$strElementType \$ELEMENT;\n";
	  $annot .= "\t\t\tmemcpy\(\&\$ELEMENT,\$\$\+copy_off, sizeof\($strElementType\)\);\n";
          $annot .= "\t\t\tcopy_off \+\= sizeof\($strElementType\);\n";
	}elsif ($asterisk) {
	  $annot .= "\t\t\t$strElementType \$ELEMENT \= \($strElementType\)marEle\.unmarshal\(\)\;\n";
	  $annot .= "\t\t\tcopy_off \+\= marEle\.getBufferSize\(\)\;\n";
	} else {
	  $annot .= "\t\t\t$strElementType *\$ELEMENT \= \($strElementType *\)marEle\.unmarshal\(\)\;\n";
	  $annot .= "\t\t\tcopy_off \+\= marEle\.getBufferSize\(\)\;\n";
	}
      #} else {
	#$annot .= "\t\t\t$strElementType \$ELEMENT \= \($strElementType\)marEle\.unmarshal\(\)\;\n";
      #}

      $annot .= "\t\t\t$strElementSet\n";
      $annot .= "\t\t\}\n";
      if ($isMultipleTypes) {
	  $annot .= "\t}\n";
      }
  }
}
  #elsif ($annot_type eq "embedded" || $annot_type eq "embedded_ptr" || $annot_type eq "global"){} #do nothing
  else {
      error_die("The annotation type $annot_type does not exist", $annot_buf);
  }
  $annot .= "\n    }\n";
  return $annot;
}


sub annot_getSize{

  my $annot_buf = shift;
  my $annot_type = shift;
  my $vartype = shift;
  my $varname = shift;
  my $annot = "";
  my $strMaxSize = "";
  $annot .="    //FIELDSIZE:\n    {\n";

  my $strTypeOption = get_Option_Name( $annot_buf, "elementType:");

  if ($annot_type eq "" || $annot_type eq "transient"){}#do nothing
  elsif ($annot_type eq "primitive"){
      if(!$strTypeOption){ #if there is no customized "elementType"
	  $annot .="\t\$SIZE = sizeof($vartype);\n";
      }else { #if there is customized "elementType" option
	  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
	  my @arrTypeChoices = @{$cref};
	  my @arrTypeResults = @{$rref};
	  my $isMultipleTypes = ($numOfTypes>=2);
	  my $strElementType;
	  if ($isMultipleTypes) {
	      $annot .= "\tif(0){}\n";
	  }
	  for (my $i=0;$i<$numOfTypes;$i++) {
	      if (!$isMultipleTypes) {
		  # single type, strElemenType == strTypeVariable
		  $strElementType = $strTypeOption;
	      } else {
		  my $choice = $arrTypeChoices[$i];
		  $annot .= "\telse if\($choice\){\n";
		  $strElementType = $arrTypeResults[$i];
	      }
	      $annot .="\t\t\$SIZE = sizeof($strElementType);\n";
	      if ($isMultipleTypes) {
		  $annot .= "\t}\n";
	      }
	  }
      }
  }
  elsif ($annot_type eq "primitive_ptr"){
    my $vartypeDeref = chopOneAsterisk($vartype);
    $annot .= "      \$SIZE = sizeof( $vartypeDeref\)\;";
  }
  elsif ($annot_type eq "ptr_shallow_copy"){
    $annot .= "      \$SIZE = sizeof( int );";
  }
  elsif ($annot_type eq "ptr_to_index"){
    my $size = get_array_info($annot_buf);
    $annot .= "      \$SIZE = sizeof( $vartype ) * $size\;";
  }
  elsif ($annot_type eq "manual"){
  #manual {marshal()} {unmarshal()} {getSize()}
  # get marshaling function from { marshal() }
    my ($marshal, $unmarshal, $getSize) = get_functions( $annot_buf);
    $annot .= $getSize;
    #if( !$annot){ die "wrong size function: \n$annot_buf\n";}
  }
  elsif ($annot_type eq "predefined"){
      $annot .= "\t// no need to declare size since \$SIZE is already assigned in the MARSHAL field\n";
      # option MaxSize is obsolete
#      my $strMaxSize = get_Option_Code( $annot_buf, "MaxSize:");
#      if($strMaxSize){
#	  $annot .= "\t$strMaxSize\n";
#      }els
#        if(!$strTypeOption){ #if there is no customized "elementType"
#  	  my $strElementGet = get_Option_Code($annot_buf, "elementGet:");
#  	  if (!$strElementGet){
#  	      $annot .= "\tMarshaled$vartype var(\&$shadowedClass_parm->$varname);\n";
#  	  }else{
#  	      $annot .= "\t$vartype \$ELEMENT\;\n\t$strElementGet\n";
#  	      $annot .= "\tMarshaled$vartype var(\&\$ELEMENT);\n";
#  	  }
#  	  $annot .= "\t\$SIZE = var.getBufferSize();\n";
#        }else { #if there is customized "elementType" option
#  	  my ($numOfTypes, $cref, $rref) = extractMultipleTypes($strTypeOption);
#  	  my @arrTypeChoices = @{$cref};
#  	  my @arrTypeResults = @{$rref};
#  	  my $isMultipleTypes = ($numOfTypes>=2);
#  	  my $strElementType;
#  	  if ($isMultipleTypes) {
#  	      $annot .= "\tif(0){}\n";
#  	  }
#  	  for (my $i=0;$i<$numOfTypes;$i++) {
#  	      if (!$isMultipleTypes) {
#  		  # single type, strElemenType == strTypeVariable
#  		  $strElementType = $strTypeOption;
#  	      } else {
#  		  my $choice = $arrTypeChoices[$i];
#  		  $annot .= "\telse if\($choice\){\n";
#  		  $strElementType = $arrTypeResults[$i];
#  	      }
#  	      # one of the type of choices might be primitive
#  	      my $isPrimitiveType=IsPrimitiveType($strElementType);
#  	      if($isPrimitiveType){
#  		  $annot .="\t\t\$SIZE = sizeof($strElementType);\n";
#  	      }else{
#  		  $annot .= "\t\tMarshaled$strElementType var(\&$shadowedClass_parm->$varname);\n";
#  		  $annot .= "\t\t\$SIZE = var.getBufferSize();\n";
#  	      }
#  	      if ($isMultipleTypes) {
#  		  $annot .= "\t}\n";
#  	      }
#  	  }
#        }
  }
  elsif ($annot_type eq "predefined_ptr"){ # eg: List *ls;//predefined_ptr
      $annot .= "\t// no need to declare size since \$SIZE is already assigned in the MARSHAL field\n";
#      my $vartypeDeref = chopOneAsterisk($vartype);
#      my ($strElementGet) = get_Option_Code($annot_buf, "elementGet:");
#      if (!$strElementGet) { 
#  	$annot .= "\t$vartype\* \$ELEMENT\;\n";
#  	$annot .= "\t$strElementGet\n";
#  	$annot .= "\tMarshaled$vartypeDeref var\(\$ELEMENT);\n";
#      }else{
#  	$annot .= "\tMarshaled$vartypeDeref var\($shadowedClass_parm->$varname);\n";
#      }
#      $annot .= "\t\$SIZE = var.getBufferSize();";
  }
  elsif ($annot_type eq "ptr_as_array"){
    #changed by imltv 10/13/03 ==>remove MAXSize annotation
  }
  #elsif ($annot_type eq "embedded" || $annot_type eq "embedded_ptr" || $annot_type eq "global"){} #do nothing
  else{
      error_die("The annotation type $annot_type does not exist", $annot_buf);
  }
  $annot .= "\n    }\n";
  return $annot;
}


#annot_buf is in the form of --- TYPE NAME [num]; //ptr_to_index
sub get_array_info {
  my $annot = shift;
  if (($annot =~ /\[(\d+)\]/) && $1){
    return $1;
  }
  else {
    error_die("Ill-formed array annotations", $annot);
  }
}

sub get_functions {
  # annot_buf is in the form of --- 
  # TYPE NAME; //manual {marshal()} {unmarshal()} {getSize()}
  #replaces get_marshal_func(), get_unmarshal_func(), and get_size_func()
  my $buf = shift;

  #if ($buf =~ /\{(.*)\}\s+\{(.*)\}\s+\{(.*)\}/){
  #print "buf=$buf\n";
  if ($buf =~ /\{([\s\S]*)\}\s+\{([\s\S]*)\}\s+\{([\s\S]*)\}/){
    my @rvalue = ($1, $2, $3);
    #print "Manual buffer: Marshal field = $1\n Unmarshal field = $2\n Marshal size = $3\n";
    return @rvalue;
  }
  else{ 
    error_die("Invalid format for \"manual\" annotation.\nMust be \{MARSHALING CODE\} \{UNMARSHALING CODE\} \{SIZE\}",$buf);
  }

}


sub get_Option_Name{
  my $annot_buf = shift;
  my $tag = shift;
  if ($annot_buf =~ /$tag\s*([^\]]*)/){return $1;}
  else {return;}
}

sub get_Option_Code{
  my $annot_buf = shift;
  my $tag = shift;
  if ($annot_buf =~ /$tag\s*\{([^\}]*)/){return $1;}
  else {return;}
}

#(MSH_IsSameClass<T,DMXPmtHit>::Is) => DMXPmtHit* | true => DMXScintHit*
#arrTypeChoices = {DMXPmtHit,DMXScintHit}
#arrTypeResults = {DMXPmtHit*,DMXScintHit*}
#return (Number of types);
#ELSE return 1;
#CONDITION_COMMAND ->  "(" C_expression ")" "==>" C_type 

sub extractMultipleTypes{
  my $strTypeOption = shift;

  #print "strTypeOption=$strTypeOption\n";

  my @conds = split(/\|/, $strTypeOption); #assumes the test does not contain the "|" token
  my $i = 0;
  my @arrTypeChoices = ();
  my @arrTypeResults = ();
  my @arrConstructorArgs = ();


  if ($#conds == 0){ #no conditionals, but one result
    if ($conds[0] =~ /\s*(.*)\((.*)\)/){  # Foo(1,2) (type with constructor)
      $arrTypeResults[0] = $1;
      $arrConstructorArgs[0] = $2;
    }elsif ($conds[0] =~ /\s*(.*)/){   # Foo  (only the type)
      $arrTypeResults[0] = $1;
      $arrConstructorArgs[0] = "";
    }
    else {
	error_die("Invalid type expression: $conds[0]",$strTypeOption);
    }

    return (1, \@arrTypeChoices, \@arrTypeResults,\@arrConstructorArgs);
  }

  for ($i =0; $i<= $#conds; $i++){
    #print "cond=$conds[$i]\n";  
    #if ($conds[$i] =~ /(.*)\s*\=\=\>\s*(.*)/){

    if ($conds[$i] =~ /(.*)\s*\=\>\s*(.*)\((.*)\)/){  # Foo(1,2) (type with constructor)
      $arrTypeChoices[$i] = $1;
      $arrTypeResults[$i] = $2;
      $arrConstructorArgs[$i] = $3;
    }elsif ($conds[$i] =~ /(.*)\s*\=\>\s*(.*)/){   # Foo  (only the type)
      $arrTypeChoices[$i] = $1;
      $arrTypeResults[$i] = $2;
      $arrConstructorArgs[$i] = "";
    }
    else {
	error_die("Invalid conditional expression: $conds[$i]",$strTypeOption);
    }
  }
  return ($#arrTypeChoices+1, \@arrTypeChoices, \@arrTypeResults, \@arrConstructorArgs);
}

sub chopOneAsterisk{
    my $type = shift;
    $type =~s/\*//;
    return $type;
}

sub IsPrimitiveType{
    my $type = shift;
    # trim the spaces
    $type =~ s/^\s+//;
    $type =~ s/\s+$//;
    #print "primitiveType=\'$type\'\n";
    if(($type eq 'bool')||($type eq 'char')||($type eq 'int')||($type eq 'long')
       ||($type eq 'float') || ($type eq 'double')){
	return 1;
    }
    return 0;
}

sub error_die {
    my $msg = shift;
    my $source = shift;
    die "Error: $msg\nAt line $line_counter: $source\n";
}


