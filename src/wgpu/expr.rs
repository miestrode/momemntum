use std::fmt::Display;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum WgpuOp {
    Add,
    Mul,
    Sin,
    Var(String),
}

impl Display for WgpuOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            WgpuOp::Add => "+",
            WgpuOp::Mul => "*",
            WgpuOp::Sin => "sin",
            WgpuOp::Var(variable) => variable.as_str(),
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct WgpuExpr {
    op: WgpuOp,
    children: Vec<WgpuExpr>,
}

impl WgpuExpr {
    pub fn new(op: WgpuOp, children: Vec<WgpuExpr>) -> Self {
        Self { op, children }
    }

    pub fn new_var(name: String) -> Self {
        Self {
            op: WgpuOp::Var(name),
            children: vec![],
        }
    }
}

impl Display for WgpuExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.op {
            WgpuOp::Add | WgpuOp::Mul => {
                write!(
                    f,
                    "({}) {} ({})",
                    &self.children[0], self.op, &self.children[1]
                )
            }
            WgpuOp::Sin => write!(
                f,
                "{}({})",
                self.op,
                self.children
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            WgpuOp::Var(variable) => f.write_str(variable),
        }
    }
}
