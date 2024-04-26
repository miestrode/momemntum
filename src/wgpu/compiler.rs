use std::iter;

use serde::{Deserialize, Serialize};

use crate::{
    compiler::Compiler,
    graph::{ElemwiseOp, ExprBody, ExprId, Graph, Op},
    tensor::{Layout, Tensor},
};

use super::{
    expr::{WgpuExpr, WgpuOp},
    kernel,
};

#[derive(Serialize, Deserialize)]
pub(crate) enum WgpuStep {
    Allocate {
        id: ExprId,
        tensor: Tensor,
    },
    Deallocate(ExprId),
    Execute {
        output: ExprId,
        source: String,
        workgroups: [u32; 3],
        inputs: Vec<ExprId>,
        inputs_layout: Vec<(usize, bool)>,
    },
}

#[derive(Serialize, Deserialize)]
pub struct WgpuPlan {
    pub(crate) inputs: Vec<ExprId>,
    pub(crate) steps: Vec<WgpuStep>,
    pub(crate) outputs: Vec<ExprId>,
    pub(crate) output_layouts: Vec<Layout>,
}

pub struct WgpuCompiler {
    pub workgroup_size_x: u32,
}

impl Default for WgpuCompiler {
    fn default() -> Self {
        Self {
            workgroup_size_x: 256,
        }
    }
}

impl Compiler for WgpuCompiler {
    type CompileResult = WgpuPlan;

    fn compile(&self, graph: Graph) -> Self::CompileResult {
        let last_usages = graph.last_usages();

        let mut steps = Vec::with_capacity(graph.exprs.len());
        let mut layouts = Vec::with_capacity(graph.exprs.len());

        let mut aliases: Vec<ExprId> = Vec::with_capacity(graph.exprs.len());

        for (id, expr) in (0..).map(ExprId).zip(graph.exprs) {
            match expr.body {
                ExprBody::Op { op, children } => {
                    steps.push(WgpuStep::Execute {
                        output: id,
                        source: match op {
                            Op::Elemwise(op) => kernel::elemwise(
                                self.workgroup_size_x,
                                &expr.layout,
                                children.iter().map(|id| (*id, &layouts[id.0])).collect(),
                                WgpuExpr::new(
                                    match op {
                                        ElemwiseOp::Add => WgpuOp::Add,
                                        ElemwiseOp::Mul => WgpuOp::Mul,
                                        ElemwiseOp::Sin => WgpuOp::Sin,
                                    },
                                    children
                                        .iter()
                                        .map(|id| WgpuExpr::new_var(format!("elem_input_{}", id.0)))
                                        .collect(),
                                ),
                            ),
                            Op::Reduce { .. } => todo!(),
                            Op::Movement(_) => {
                                aliases.push(children[0]);
                                layouts.push(expr.layout);

                                continue;
                            }
                        },
                        workgroups: [
                            (expr.layout.elements() as u32).div_ceil(self.workgroup_size_x),
                            1,
                            1,
                        ],
                        inputs: iter::once(id).chain(children.iter().copied()).collect(),
                        inputs_layout: iter::once((expr.layout.size(), false))
                            .chain(children.iter().map(|id| (layouts[id.0].size(), true)))
                            .collect(),
                    });

                    for &child in children.iter().filter(|child| last_usages[child.0] == id) {
                        steps.push(WgpuStep::Deallocate(child));
                    }
                }
                ExprBody::Input(_) => {}
                ExprBody::Const(tensor) => steps.push(WgpuStep::Allocate { id, tensor }),
            }

            aliases.push(id);
            layouts.push(expr.layout);
        }

        WgpuPlan {
            inputs: graph.inputs,
            steps,
            output_layouts: graph
                .outputs
                .iter()
                .rev()
                .map(|id| layouts.remove(id.0))
                .collect(),
            outputs: graph.outputs.iter().map(|id| aliases[id.0]).collect(),
        }
    }
}
