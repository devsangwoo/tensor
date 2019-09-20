#!/usr/bin/env python
"""A script to generate FileCheck statements for mlir unit tests.

This script is a utility to add FileCheck patterns to an mlir file.

NOTE: The input .mlir is expected to be the output from the parser, not a
stripped down variant.

Example usage:
$ generate-test-checks.py foo.mlir
$ mlir-opt foo.mlir -transformation | generate-test-checks.py

The script will heuristically insert CHECK/CHECK-LABEL commands for each line
within the file. By default this script will also try to insert string
substitution blocks for all SSA value names. The script is designed to make
adding checks to a test case fast, it is *not* designed to be authoritative
about what constitutes a good test!
"""

# Copyright 2019 The MLIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os  # Used to advertise this file's name ("autogenerated_note").
import re
import sys

ADVERT = '// NOTE: Assertions have been autogenerated by '

# Regex command to match an SSA identifier.
SSA_RE_STR = '[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*'
SSA_RE = re.compile(SSA_RE_STR)


# Class used to generate and manage string substitution blocks for SSA value
# names.
class SSAVariableNamer:

  def __init__(self):
    self.scopes = []
    self.name_counter = 0

  # Generate a subsitution name for the given ssa value name.
  def generate_name(self, ssa_name):
    variable = 'VAL_' + str(self.name_counter)
    self.name_counter += 1
    self.scopes[-1][ssa_name] = variable
    return variable

  # Push a new variable name scope.
  def push_name_scope(self):
    self.scopes.append({})

  # Pop the last variable name scope.
  def pop_name_scope(self):
    self.scopes.pop()


# Process a line of input that has been split at each SSA identifier '%'.
def process_line(line_chunks, variable_namer):
  output_line = ''

  # Process the rest that contained an SSA value name.
  for chunk in line_chunks:
    m = SSA_RE.match(chunk)
    ssa_name = m.group(0)

    # Check if an existing variable exists for this name.
    variable = None
    for scope in variable_namer.scopes:
      variable = scope.get(ssa_name)
      if variable is not None:
        break

    # If one exists, then output the existing name.
    if variable is not None:
      output_line += '[[' + variable + ']]'
    else:
      # Otherwise, generate a new variable.
      variable = variable_namer.generate_name(ssa_name)
      output_line += '[[' + variable + ':%.*]]'

    # Append the non named group.
    output_line += chunk[len(ssa_name):]

  return output_line + '\n'


# Pre-process a line of input to remove any character sequences that will be
# problematic with FileCheck.
def preprocess_line(line):
  # Replace any double brackets, '[[' with escaped replacements. '[['
  # corresponds to variable names in FileCheck.
  output_line = line.replace('[[', '{{\\[\\[}}')

  # Replace any single brackets that are followed by an SSA identifier, the
  # identifier will be replace by a variable; Creating the same situation as
  # above.
  output_line = output_line.replace('[%', '{{\\[}}%')

  return output_line


def main():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
      '--check-prefix', default='CHECK', help='Prefix to use from check file.')
  parser.add_argument(
      '-o',
      '--output',
      nargs='?',
      type=argparse.FileType('w'),
      default=sys.stdout)
  parser.add_argument(
      'input',
      nargs='?',
      type=argparse.FileType('r'),
      default=sys.stdin)
  args = parser.parse_args()

  # Open the given input file.
  input_lines = [l.rstrip() for l in args.input]
  args.input.close()

  output_lines = []

  # Generate a note used for the generated check file.
  script_name = os.path.basename(__file__)
  autogenerated_note = (ADVERT + 'utils/' + script_name)
  output_lines.append(autogenerated_note + '\n')

  # A map containing data used for naming SSA value names.
  variable_namer = SSAVariableNamer()
  for input_line in input_lines:
    if not input_line:
      continue
    lstripped_input_line = input_line.lstrip()

    # Lines with blocks begin with a ^. These lines have a trailing comment
    # that needs to be stripped.
    is_block = lstripped_input_line[0] == '^'
    if is_block:
      input_line = input_line.rsplit('//', 1)[0].rstrip()

    # Top-level operations are heuristically the operations at nesting level 1.
    is_toplevel_op = (not is_block and input_line.startswith('  ') and
                      input_line[2] != ' ' and input_line[2] != '}')

    # If the line starts with a '}', pop the last name scope.
    if lstripped_input_line[0] == '}':
      variable_namer.pop_name_scope()

    # If the line ends with a '{', push a new name scope.
    if input_line[-1] == '{':
      variable_namer.push_name_scope()

    # Preprocess the input to remove any sequences that may be problematic with
    # FileCheck.
    input_line = preprocess_line(input_line)

    # Split the line at the each SSA value name.
    ssa_split = input_line.split('%')

    # If this is a top-level operation use 'CHECK-LABEL', otherwise 'CHECK:'.
    if not is_toplevel_op or not ssa_split[0]:
      output_line = '// ' + args.check_prefix + ': '
      # Pad to align with the 'LABEL' statements.
      output_line += (' ' * len('-LABEL'))

      # Output the first line chunk that does not contain an SSA name.
      output_line += ssa_split[0]

      # Process the rest of the input line.
      output_line += process_line(ssa_split[1:], variable_namer)

    else:
      # Append a newline to the output to separate the logical blocks.
      output_lines.append('\n')
      output_line = '// ' + args.check_prefix + '-LABEL: '

      # Output the first line chunk that does not contain an SSA name for the
      # label.
      output_line += ssa_split[0] + '\n'

      # Process the rest of the input line on a separate check line.
      if len(ssa_split) > 1:
        output_line += '// ' + args.check_prefix + '-SAME:  '

        # Pad to align with the original position in the line.
        output_line += ' ' * len(ssa_split[0])

        # Process the rest of the line.
        output_line += process_line(ssa_split[1:], variable_namer)

    # Append the output line.
    output_lines.append(output_line)

  # Write the output.
  for output_line in output_lines:
    args.output.write(output_line)
  args.output.write('\n')
  args.output.close()


if __name__ == '__main__':
  main()
