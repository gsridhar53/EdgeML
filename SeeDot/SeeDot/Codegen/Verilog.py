# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

import IR.IR as IR
import IR.IRUtil as IRUtil

import Type as Type

from Codegen.CodegenBase import CodegenBase

from Util import *

class Verilog(CodegenBase):

	def __init__(self, writer, decls, expts, intvs, cnsts, expTables, VAR_IDF_INIT):
		self.out = writer
		self.decls = decls
		self.expts = expts
		self.intvs = intvs
		self.cnsts = cnsts
		self.expTables = expTables
		self.VAR_IDF_INIT = VAR_IDF_INIT

	# Print the compiled code (IR)
	def printAll(self, prog:IR.Prog, expr:IR.Expr):
		self._out_prefix()
		
		type = self.decls[expr.idf]
		if isinstance(type, Type.Tensor):
			shape_str = ''.join(['[' + str(n - 1) + ':0]' for n in type.shape])
		else:
			shape_str = ''

		#self.out.printf('%s%s, ', idf_str, shape_str, indent=True)
		self.out.printf("output logic [%d:0] %s%s;" % (Common.wordLength - 1, expr.idf, shape_str), indent=True)
		self.out.printf('\n')
		
		self.print(prog)
		
		self._out_suffix(expr)

	def _out_prefix(self):

		self.out.printf("`timescale 1ns/1ps\n\n")

		self.out.printf("module main(X, clk, rst);\n\n")

		self.out.increaseIndent()

		# declare vars
		self.printVarDecls()
		
		#self.out.printf('\n')

	def printVarDecls(self):

		self.out.printf("input clk, rst;\n", indent=True)

		for decl in self.decls:
			if decl in self.VAR_IDF_INIT:
				continue
			typ_str = IR.DataType.getIntStr()
			idf_str = decl
			type = self.decls[decl]
			if Type.isInt(type): shape_str = ''
			elif Type.isTensor(type): shape_str = ''.join(['[' + str(n - 1) + ':0]' for n in type.shape])
			self.out.printf('input [%d:0] %s%s;\n', Common.wordLength - 1, idf_str, shape_str, indent=True)
		self.out.printf('\n')

	def _out_suffix(self, expr:IR.Expr):
		self.out.printf('\n', indent=True)
		self.out.decreaseIndent()
		self.out.printf('endmodule\n', indent=True)

	def printFuncCall(self, ir):
		self.out.printf("%s %s(\n" % (ir.name, ir.name.lower()), indent = True)
		self.out.increaseIndent()
		self.out.printf(".wordLength(%d),\n" % Common.wordLength, indent = True)
		self.out.printf(".clk(clk),\n", indent = True)
		self.out.printf(".rst(rst),\n", indent = True)
		for arg, id in ir.argList.items():
			if isinstance(arg, IR.Var) and arg.idf in self.decls.keys():
				x = self.decls[arg.idf].dim
			else:
				x = 0
			self.out.printf('.', indent=True)
			self.out.printf('%s', id)
			self.out.printf('(')
			self.print(arg)
			self.out.printf('),\n')
		self.out.printf(");\n", indent=True)
		self.out.decreaseIndent()