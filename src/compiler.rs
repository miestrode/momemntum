use crate::{graph::Graph, tensor::Tensor};

pub trait Runner {
    type Compiler: Compiler;

    type Runnable;

    fn preprocess(&mut self, result: <Self::Compiler as Compiler>::CompileResult)
        -> Self::Runnable;

    fn run(&mut self, runnable: Self::Runnable, inputs: Vec<Tensor>) -> Vec<Tensor>;
}

pub trait Compiler {
    type CompileResult;

    fn compile(&self, graph: Graph) -> Self::CompileResult;
}
