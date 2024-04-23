use momentum::{
    compiler::{Compiler, Runner},
    graph::{ElemwiseOp, Graph, Op},
    tensor::{Layout, Tensor},
    wgpu::{compiler::WgpuCompiler, runner::WgpuRunner},
};

fn main() {
    let mut graph = Graph::new();

    let a = graph.add_input(Layout::from([1]));
    let b = graph.add_input(Layout::from([1]));
    let c = graph.add_input(Layout::from([1]));

    let result = graph.add_op(Op::Elemwise(ElemwiseOp::Mul), &[a, b]);
    let result = graph.add_op(Op::Elemwise(ElemwiseOp::Add), &[result, c]);

    graph.add_output(result);

    println!("{graph:#?}");

    let compiler = WgpuCompiler::default();
    let mut runner = WgpuRunner::new();

    let runnable = runner.preprocess(compiler.compile(graph));

    println!(
        "{:#?}",
        runner.run(
            runnable,
            vec![Tensor::from_parts(Box::new([2.0]), Layout::from([1]))]
        )
    );
}
