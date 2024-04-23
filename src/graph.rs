use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use crate::tensor::{DimId, Layout, Shape, Tensor};

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExprId(pub(crate) usize);

impl Debug for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

#[derive(Copy, Clone)]
pub enum ElemwiseOp {
    Add,
    Mul,
    Sin,
}

impl Display for ElemwiseOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ElemwiseOp::Add => "add",
            ElemwiseOp::Mul => "mul",
            ElemwiseOp::Sin => "sin",
        })
    }
}

#[derive(Copy, Clone)]
pub enum ReduceOp {
    Sum,
    Max,
}

impl Display for ReduceOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
        })
    }
}

impl Debug for ReduceOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[derive(Clone)]
pub enum MovementOp {
    Reshape(Shape),
    Transpose,
    Squeeze,
}

impl Display for MovementOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MovementOp::Reshape(_) => "reshape",
            MovementOp::Transpose => "transpose",
            MovementOp::Squeeze => "squeeze",
        })
    }
}

#[derive(Clone)]
pub enum Op {
    Elemwise(ElemwiseOp),
    Reduce { op: ReduceOp, dims: Vec<DimId> },
    Movement(MovementOp),
}

impl Op {
    pub(crate) fn infer_layout(&self, children: &[&Layout]) -> Layout {
        match self {
            Op::Elemwise(_) => children[0].clone(),
            Op::Reduce {
                dims: reduce_dims, ..
            } => {
                let mut dims = children[0].dims().to_vec();

                for dim in reduce_dims {
                    dims[*dim] = 1;
                }

                Layout::from(dims)
            }
            Op::Movement(op) => match op {
                MovementOp::Reshape(shape) => children[0].reshape(shape.clone()),
                MovementOp::Transpose => {
                    let rank = children[0].shape().rank();

                    let mut dims = children[0].dims().to_vec();
                    dims.swap(rank - 2, rank - 1);

                    let mut strides = children[0].strides().to_vec();
                    strides.swap(rank - 2, rank - 1);

                    Layout {
                        shape: Shape {
                            dims: dims.into_boxed_slice(),
                            strides: strides.into_boxed_slice(),
                        },
                    }
                }
                MovementOp::Squeeze => {
                    let (dims, strides): (Vec<_>, Vec<_>) = children[0]
                        .dims()
                        .iter()
                        .copied()
                        .zip(children[0].strides().iter().copied())
                        .filter(|&(dim, _)| dim != 1)
                        .unzip();

                    Layout {
                        shape: Shape {
                            dims: dims.into_boxed_slice(),
                            strides: strides.into_boxed_slice(),
                        },
                    }
                }
            },
        }
    }
}

impl Op {
    fn parameters(&self) -> Vec<(&'static str, Box<dyn Debug + '_>)> {
        match self {
            Op::Reduce { dims, .. } => vec![("dims", Box::new(dims))],
            Op::Movement(MovementOp::Reshape(shape)) => vec![("shape", Box::new(shape))],
            _ => vec![],
        }
    }
}

impl Debug for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Op::Elemwise(op) => op.to_string(),
            Op::Reduce { op, .. } => op.to_string(),
            Op::Movement(op) => op.to_string(),
        })?;

        let parameters = self.parameters();

        if !parameters.is_empty() {
            write!(
                f,
                "({})",
                self.parameters()
                    .into_iter()
                    .map(|(parameter, value)| format!("{parameter}={value:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub(crate) enum ExprBody {
    Op { op: Op, children: Vec<ExprId> },
    Input(Layout),
    Const(Tensor),
}

impl Debug for ExprBody {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ExprBody::Op { op, children } => {
                let mut result = f.debug_tuple(&format!("{op:?}"));

                for child in children {
                    result.field(child);
                }

                result.finish()
            }
            ExprBody::Input(_) => f.write_str("?"),
            ExprBody::Const(_) => f.write_str(".."),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprInfo {
    pub(crate) body: ExprBody,
    pub(crate) layout: Layout,
    pub(crate) last_usage: ExprId,
}

#[derive(Default)]
pub struct Graph {
    pub(crate) inputs: Vec<ExprId>,
    pub(crate) exprs: Vec<ExprInfo>,
    pub(crate) outputs: Vec<ExprId>,
}

impl Index<ExprId> for Graph {
    type Output = ExprInfo;

    fn index(&self, index: ExprId) -> &Self::Output {
        &self.exprs[index.0]
    }
}

impl IndexMut<ExprId> for Graph {
    fn index_mut(&mut self, index: ExprId) -> &mut Self::Output {
        &mut self.exprs[index.0]
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn last_usages(&self) -> Vec<ExprId> {
        self.exprs.iter().map(|expr| expr.last_usage).collect()
    }

    fn add_expr(&mut self, expr: ExprBody) -> ExprId {
        let id = ExprId(self.exprs.len());

        let layout = match &expr {
            ExprBody::Op { op, children } => {
                for expr in children.iter().copied() {
                    self[expr].last_usage = id;
                }

                op.infer_layout(
                    &children
                        .iter()
                        .copied()
                        .map(|expr| &self[expr].layout)
                        .collect::<Vec<_>>(),
                )
            }
            ExprBody::Input(layout) => layout.clone(),
            ExprBody::Const(tensor) => tensor.layout.clone(),
        };

        self.exprs.push(ExprInfo {
            body: expr,
            layout,
            last_usage: id,
        });

        id
    }

    pub fn add_input(&mut self, layout: Layout) -> ExprId {
        let id = self.add_expr(ExprBody::Input(layout));

        self.inputs.push(id);

        id
    }

    pub fn add_const(&mut self, tensor: Tensor) -> ExprId {
        self.add_expr(ExprBody::Const(tensor))
    }

    pub fn add_op(&mut self, op: Op, children: &[ExprId]) -> ExprId {
        self.add_expr(ExprBody::Op {
            op,
            children: children.to_owned(),
        })
    }

    pub fn add_output(&mut self, expr: ExprId) {
        self.outputs.push(expr);
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}) -> ({}) {{",
            self.inputs
                .iter()
                .map(|input| format!("{input:?}: {}", self[*input].layout))
                .collect::<Vec<_>>()
                .join(", "),
            self.outputs
                .iter()
                .map(|output| format!("{output:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;

        if f.alternate() {
            writeln!(f)?;
        } else {
            write!(f, " ")?;
        }

        f.write_str(
            &self
                .exprs
                .iter()
                .zip((0..self.exprs.len()).map(ExprId))
                .filter(|(info, _)| !matches!(info.body, ExprBody::Input(..)))
                .map(|(node, id)| {
                    format!(
                        "{}{id:?}: {} = {:?};",
                        if f.alternate() { "    " } else { "" },
                        node.layout,
                        node.body
                    )
                })
                .collect::<Vec<_>>()
                .join(if f.alternate() { "\n" } else { " " }),
        )?;

        if f.alternate() {
            writeln!(f)?;
        } else {
            write!(f, " ")?;
        }

        write!(f, "}}")
    }
}
