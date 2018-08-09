// TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
// statements (perhaps through using lit config substitutions).
//
// RUN: %S/../../mlir-opt %s -o - -check-parser-errors

// Check different error cases.
// -----

extfunc @illegaltype(i) // expected-error {{expected type}}

// -----

extfunc @nestedtensor(tensor<tensor<i8>>) -> () // expected-error {{invalid tensor element type}}

// -----
// Test no map in memref type.
extfunc @memrefs(memref<2x4xi8, >) // expected-error {{expected list element}}

// -----
// Test non-existent map in memref type.
extfunc @memrefs(memref<2x4xi8, #map7>) // expected-error {{undefined affine map id 'map7'}}

// -----
// Test non hash identifier in memref type.
extfunc @memrefs(memref<2x4xi8, %map7>) // expected-error {{expected '(' at start of dimensional identifiers list}}

// -----
// Test non-existent map in map composition of memref type.
#map0 = (d0, d1) -> (d0, d1)

extfunc @memrefs(memref<2x4xi8, #map0, #map8>) // expected-error {{undefined affine map id 'map8'}}

// -----
// Test multiple memory space error.
#map0 = (d0, d1) -> (d0, d1)
extfunc @memrefs(memref<2x4xi8, #map0, 1, 2>) // expected-error {{multiple memory spaces specified in memref type}}

// -----
// Test affine map after memory space.
#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0, d1) -> (d0, d1)

extfunc @memrefs(memref<2x4xi8, #map0, 1, #map1>) // expected-error {{affine map after memory space in memref type}}

// -----

cfgfunc @foo()
cfgfunc @bar() // expected-error {{expected '{' in CFG function}}

// -----

extfunc missingsigil() -> (i1, affineint, f32) // expected-error {{expected a function identifier like}}


// -----

cfgfunc @bad_branch() {
bb42:
  br missing  // expected-error {{reference to an undefined basic block 'missing'}}
}

// -----

cfgfunc @block_redef() {
bb42:
  return
bb42:        // expected-error {{redefinition of block 'bb42'}}
  return
}

// -----

cfgfunc @no_terminator() {
bb40:
  return
bb41:
bb42:        // expected-error {{custom op 'bb42' is unknown}}
  return
}

// -----

cfgfunc @block_no_rparen() {
bb42 (%bb42 : i32: // expected-error {{expected ')' to end argument list}}
  return
}

// -----

cfgfunc @block_arg_no_ssaid() {
bb42 (i32): // expected-error {{expected SSA operand}}
  return
}

// -----

cfgfunc @block_arg_no_type() {
bb42 (%0): // expected-error {{expected ':' and type for SSA operand}}
  return
}

// -----

mlfunc @foo()
mlfunc @bar() // expected-error {{expected '{' before statement list}}

// -----

mlfunc @empty() {
// expected-error@-1 {{ML function must end with return statement}}
}

// -----

mlfunc @no_return() {
// expected-error@-1 {{ML function must end with return statement}}
  "foo"() : () -> ()
}

// -----

"       // expected-error {{expected}}
"

// -----

"       // expected-error {{expected}}

// -----

cfgfunc @bad_op_type() {
bb40:
  "foo"() : i32  // expected-error {{expected function type}}
  return
}
// -----

cfgfunc @no_terminator() {
bb40:
  "foo"() : ()->()
  ""() : ()->()  // expected-error {{empty operation name is invalid}}
  return
}

// -----

extfunc @illegaltype(i0) // expected-error {{invalid integer width}}

// -----

mlfunc @malformed_for_percent() {
  for i = 1 to 10 { // expected-error {{expected SSA identifier for the loop variable}}

// -----

mlfunc @malformed_for_equal() {
  for %i 1 to 10 { // expected-error {{expected '='}}

// -----

mlfunc @malformed_for_to() {
  for %i = 1 too 10 { // expected-error {{expected 'to' between bounds}}
  }
}

// -----

mlfunc @incomplete_for() {
  for %i = 1 to 10 step 2
}        // expected-error {{expected '{' before statement list}}

// -----

mlfunc @nonconstant_step(%1 : i32) {
  for %2 = 1 to 5 step %1 { // expected-error {{expected non-negative integer for now}}

// -----

mlfunc @non_statement() {
  asd   // expected-error {{custom op 'asd' is unknown}}
}

// -----

mlfunc @invalid_if_conditional1() {
  for %i = 1 to 10 {
    if () // expected-error {{expected '(' at start of dimensional identifiers list}}
  }
}

// -----

mlfunc @invalid_if_conditional2() {
  for %i = 1 to 10 {
    if ((i)[N] : (i >= ))  // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

mlfunc @invalid_if_conditional3() {
  for %i = 1 to 10 {
    if ((i)[N] : (i == 1)) // expected-error {{expected '0' after '=='}}
  }
}

// -----

mlfunc @invalid_if_conditional4() {
  for %i = 1 to 10 {
    if ((i)[N] : (i >= 2)) // expected-error {{expected '0' after '>='}}
  }
}

// -----

mlfunc @invalid_if_conditional5() {
  for %i = 1 to 10 {
    if ((i)[N] : (i <= 0 )) // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

mlfunc @invalid_if_conditional6() {
  for %i = 1 to 10 {
    if ((i) : (i)) // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----
// TODO (support if (1)?
mlfunc @invalid_if_conditional7() {
  for %i = 1 to 10 {
    if ((i) : (1)) // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

#map = (d0) -> (%  // expected-error {{invalid SSA name}}

// -----

cfgfunc @test() {
bb40:
  %1 = "foo"() : (i32)->i64 // expected-error {{expected 0 operand types but had 1}}
  return
}

// -----

cfgfunc @redef() {
bb42:
  %x = "xxx"(){index: 0} : ()->i32 // expected-error {{previously defined here}}
  %x = "xxx"(){index: 0} : ()->i32 // expected-error {{redefinition of SSA value '%x'}}
  return
}

// -----

cfgfunc @undef() {
bb42:
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undeclared SSA value}}
  return
}

// -----

mlfunc @missing_rbrace() {
  return
mlfunc @d() {return} // expected-error {{expected '}' after statement list}}

// -----

mlfunc @malformed_type(%a : intt) { // expected-error {{expected type}}
}

// -----

cfgfunc @resulterror() -> i32 {  // expected-error {{return has 0 operands, but enclosing function returns 1}}
bb42:
  return
}

// -----

mlfunc @mlfunc_resulterror() -> i32 {  // expected-error {{return has 0 operands, but enclosing function returns 1}}
  return
}

// -----

cfgfunc @argError() {
bb1(%a: i64):  // expected-error {{previously defined here}}
  br bb2
bb2(%a: i64):  // expected-error{{redefinition of SSA value '%a'}}
  return
}

// -----

cfgfunc @bbargMismatch(i32, f32) { // expected-error {{first block of cfgfunc must have 2 arguments to match function signature}}
bb42(%0: f32):
  return
}

// -----

cfgfunc @br_mismatch() {  // expected-error {{branch has 2 operands, but target block has 1}}
bb0:
  %0 = "foo"() : () -> (i1, i17)
  br bb1(%0#1, %0#0 : i17, i1)

bb1(%x: i17):
  return
}

// -----

// Test no nested vector.
extfunc @vectors(vector<1 x vector<1xi32>>, vector<2x4xf32>)
// expected-error@-1 {{invalid vector element type}}

// -----

// affineint is not allowed in a vector.
extfunc @vectors(vector<1 x affineint>) // expected-error {{invalid vector element type}}

// -----

cfgfunc @condbr_notbool() {
bb0:
  %a = "foo"() : () -> i32 // expected-error {{prior use here}}
  cond_br %a, bb0, bb0 // expected-error {{use of value '%a' expects different type than prior uses}}
// expected-error@-1 {{expected type was boolean (i1)}}
}

// -----

cfgfunc @condbr_badtype() {
bb0:
  %c = "foo"() : () -> i1
  %a = "foo"() : () -> i32
  cond_br %c, bb0(%a, %a : i32, bb0) // expected-error {{expected type}}
}

// -----

cfgfunc @condbr_a_bb_is_not_a_type() {
bb0:
  %c = "foo"() : () -> i1
  %a = "foo"() : () -> i32
  cond_br %c, bb0(%a, %a : i32, i32), i32 // expected-error {{expected basic block name}}
}

// -----

mlfunc @undef() {
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undefined SSA value %y}}
  return
}

// -----

mlfunc @duplicate_induction_var() {
  for %i = 1 to 10 {   // expected-error {{previously defined here}}
    for %i = 1 to 10 { // expected-error {{redefinition of SSA value '%i'}}
    }
  }
  return
}

// -----

mlfunc @duplicate_induction_var() {  // expected-error {{}}
  for %i = 1 to 10 {
  }
  "xxx"(%i) : (affineint)->()   // expected-error {{operand #0 does not dominate this use}}
  return
}

// -----

mlfunc @return_type_mismatch() -> i32 {
  // expected-error@-1 {{type of return operand 0 doesn't match function result type}}
  %0 = "foo"() : ()->f32
  return %0 : f32
}

// -----

mlfunc @return_inside_loop() -> i8 {
  for %i = 1 to 100 {
    %a = "foo"() : ()->i8
    return %a : i8
    // expected-error@-1 {{'return' op must be the last statement in the ML function}}
  }
}
