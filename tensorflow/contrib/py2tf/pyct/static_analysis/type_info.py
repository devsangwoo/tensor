# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Type resolution.

This analyzer uses known live values to further infer object types. This
may include for instance constructed objects and object member functions.

In addition, the analyzer will also process annotations for TF (staged) type
annotations.

Requires annotations generated by LiveValuesResolver.
"""

# TODO(mdan): This would be more robust with a CFG.
# Situations with multiple reaching modifications (e.g. modified inside and
# outside a control flow statement) should be more robustly detected and
# analyzed.

# TODO(mdan): Look into using Python AST's type annotation fields instead.
# It would be desirable to use that mechanism if we can.
# Some caveats to consider: We may need to annotate other nodes like
# Attribute. It may also not be feasible for us to faithfully to replicate
# PY3's type annotations where it isn't available. It would also require us
# to design rigorous type definitions that can accommodate Python types
# as well as TensorFLow dtypes and shapes.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import transformer
from tensorflow.python.util import tf_inspect


class Scope(object):
  """Tracks symbol value references.

  Attributes:
    values: A dict mapping string to gast.Node, containing the value that was
        most recently assigned to the symbol.
  """

  def __init__(self, parent):
    """Create a new scope.

    Args:
      parent: A Scope or None.
    """
    self.parent = parent
    self.values = {}

  def __repr__(self):
    return 'Scope[%s]' % self.values.keys()

  def copy(self):
    s = Scope(self.parent)
    s.values = self.values.copy()
    return s

  def setval(self, name, value):
    self.values[name] = value

  def hasval(self, name):
    return (name in self.values or
            (self.parent is not None and self.parent.hasval(name)))

  def getval(self, name):
    if name in self.values:
      return self.values[name]
    if self.parent is not None:
      return self.parent.getval(name)
    raise KeyError(name)


class TypeInfoResolver(transformer.Base):
  """Annotates symbols with type information where possible.

  Nodes currently annotated:
    * Call (helps detect class constructors)
    * Attribute (helps resolve object methods)
  """

  def __init__(self, context):
    super(TypeInfoResolver, self).__init__(context)
    self.scope = Scope(None)
    self.function_level = 0

  def visit_FunctionDef(self, node):
    self.scope = Scope(self.scope)
    self.function_level += 1
    self.generic_visit(node)
    self.function_level -= 1
    self.scope = self.scope.parent
    return node

  def _visit_block(self, block):
    self.scope = Scope(self.scope)
    for i, n in enumerate(block):
      block[i] = self.generic_visit(n)
    self.scope = self.scope.parent
    return block

  def visit_For(self, node):
    self.generic_visit(node.target)
    self.generic_visit(node.iter)
    node.body = self._visit_block(node.body)
    node.orelse = self._visit_block(node.orelse)
    return node

  def visit_While(self, node):
    self.generic_visit(node.test)
    node.body = self._visit_block(node.body)
    node.orelse = self._visit_block(node.orelse)
    return node

  def visit_If(self, node):
    self.generic_visit(node.test)
    node.body = self._visit_block(node.body)
    node.orelse = self._visit_block(node.orelse)
    return node

  def _process_function_arg(self, arg_name):
    str_name = str(arg_name)
    if self.function_level == 1 and str_name in self.context.arg_types:
      # Forge a node to hold the type information, so that method calls on
      # it can resolve the type.
      type_holder = arg_name.ast()
      type_string, type_obj = self.context.arg_types[str_name]
      anno.setanno(type_holder, 'type', type_obj)
      anno.setanno(type_holder, 'type_fqn', tuple(type_string.split('.')))
      self.scope.setval(arg_name, type_holder)

  def visit_arg(self, node):
    self._process_function_arg(anno.getanno(node.arg, anno.Basic.QN))
    return node

  def visit_Name(self, node):
    self.generic_visit(node)
    qn = anno.getanno(node, anno.Basic.QN)
    if isinstance(node.ctx, gast.Param):
      self._process_function_arg(qn)
    elif isinstance(node.ctx, gast.Load) and self.scope.hasval(qn):
      # E.g. if we had
      # a = b
      # then for future references to `a` we should have definition = `b`
      definition = self.scope.getval(qn)
      if anno.hasanno(definition, 'type'):
        anno.setanno(node, 'type', anno.getanno(definition, 'type'))
        anno.setanno(node, 'type_fqn', anno.getanno(definition, 'type_fqn'))
      if anno.hasanno(definition, 'element_type'):
        anno.setanno(node, 'element_type',
                     anno.getanno(definition, 'element_type'))
    return node

  def _process_tuple_assignment(self, source, t):
    for i, e in enumerate(t.elts):
      if isinstance(e, gast.Tuple):
        self._process_tuple_assignment(source, e)
      else:
        self.scope.setval(
            anno.getanno(e, anno.Basic.QN),
            gast.Subscript(source, gast.Index(i), ctx=gast.Store()))

  def _process_variable_assignment(self, source, targets):
    if isinstance(source, gast.Call):
      func = source.func
      if anno.hasanno(func, 'live_val'):
        func_obj = anno.getanno(func, 'live_val')
        if tf_inspect.isclass(func_obj):
          anno.setanno(source, 'is_constructor', True)
          anno.setanno(source, 'type', func_obj)
          anno.setanno(source, 'type_fqn', anno.getanno(func, 'fqn'))
          # TODO(mdan): Raise an error if constructor has side effects.
          # We can have a whitelist of no-side-effects constructors.
          # We can also step inside the constructor and further analyze.

    for t in targets:
      if isinstance(t, gast.Tuple):
        # need to recurse on the case of assigning nested tuples,
        # ex. a, (b, c) = f()
        self._process_tuple_assignment(source, t)
      elif isinstance(t, (gast.Name, gast.Attribute)):
        self.scope.setval(anno.getanno(t, anno.Basic.QN), source)
      else:
        raise ValueError('Dont know how to handle assignment to %s' % t)

  def visit_With(self, node):
    for wi in node.items:
      if wi.optional_vars is not None:
        self._process_variable_assignment(wi.context_expr, (wi.optional_vars,))
    self.generic_visit(node)
    return node

  def visit_Assign(self, node):
    self.generic_visit(node)
    self._process_variable_assignment(node.value, node.targets)
    return node

  def visit_Call(self, node):
    if anno.hasanno(node.func, 'live_val'):
      # Symbols targeted by the "set_type" marker function are assigned the data
      # type that it specified.
      if (anno.getanno(node.func, 'live_val') is
          self.context.type_annotation_func):
        # Expecting the actual type to be the second argument.
        if len(node.args) != 2:
          raise ValueError('"%s" must have exactly two parameters'
                           % self.context.type_annotation_func)
        if not anno.hasanno(node.args[0], anno.Basic.QN):
          raise ValueError('the first argument of "%s" must by a symbol'
                           % self.context.type_annotation_func)
        if not anno.hasanno(node.args[1], 'live_val'):
          raise ValueError(
              'the second argument of "%s" must be statically resolvable' %
              self.context.type_annotation_func)
        target_symbol = anno.getanno(node.args[0], anno.Basic.QN)
        element_type = anno.getanno(node.args[1], 'live_val')
        # Find the definition of this symbol and annotate it with the given
        # data type. That in turn will cause future uses of the symbol
        # to receive the same type annotation.
        definition = self.scope.getval(target_symbol)
        anno.setanno(node, 'element_type', element_type)
        anno.setanno(definition, 'element_type', element_type)
        # TODO(mdan): Should we update references between definition and here?
    return self.generic_visit(node)


def resolve(node, context):
  return TypeInfoResolver(context).visit(node)
