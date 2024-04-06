use crate::{
    compiler::{Compiler, Runnable},
    graph::{Graph, Tensor},
};

pub struct Wgpu;

impl Runnable for () {
    fn run(self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        todo!()
    }
}

impl Compiler for Wgpu {
    type CompileResult = ();

    fn compile(&self, graph: Graph) -> Self::CompileResult {
        todo!()
    }
}
