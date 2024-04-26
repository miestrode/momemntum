use crate::graph::{ElemwiseOp, ExprId, Graph, Op};

pub struct Add {
    left: ExprId,
    right: ExprId,
}

impl Add {
    pub fn new(left: ExprId, right: ExprId) -> Self {
        Self { left, right }
    }

    pub fn build(&self, graph: &mut Graph) -> ExprId {
        graph.add_op(Op::Elemwise(ElemwiseOp::Add), &[self.left, self.right])
    }
}

pub struct Mul {
    left: ExprId,
    right: ExprId,
}

impl Mul {
    pub fn new(left: ExprId, right: ExprId) -> Self {
        Self { left, right }
    }

    pub fn build(&self, graph: &mut Graph) -> ExprId {
        graph.add_op(Op::Elemwise(ElemwiseOp::Mul), &[self.left, self.right])
    }
}
