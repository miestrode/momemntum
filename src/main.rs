use momentum::{
    builder,
    compiler::{Compiler, Runner},
    graph::Graph,
    tensor::{Layout, Tensor},
    wgpu::{compiler::WgpuCompiler, runner::WgpuRunner},
};

fn main() {
    let mut graph = Graph::new();

    let a = graph.add_input(Layout::scalar());
    let b = graph.add_input(Layout::scalar());
    let c = graph.add_input(Layout::scalar());

    let result = builder::Add::new(builder::Mul::new(a, b).build(&mut graph), c).build(&mut graph);

    graph.add_output(result);

    println!("{graph:#?}");

    let compiler = WgpuCompiler::default();
    let mut runner = WgpuRunner::new();

    let runnable = runner.preprocess(compiler.compile(graph));

    println!(
        "{:#?}",
        runner.run(
            runnable,
            vec![
                Tensor::from_scalar(2.0),
                Tensor::from_scalar(2.0),
                Tensor::from_scalar(2.0)
            ]
        )
    );
}
