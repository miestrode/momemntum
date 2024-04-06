use crate::graph::{Graph, Tensor};

pub trait Runnable {
    fn run(self, inputs: Vec<Tensor>) -> Vec<Tensor>;
}

pub trait Compiler {
    type CompileResult: Runnable;

    fn compile(&self, graph: Graph) -> Self::CompileResult;
}
