use momentum::{
    compiler::{Compiler, Runnable},
    graph::{Graph, Layout, Op},
    wgpu::compiler::Wgpu,
};

fn main() {
    let mut graph = Graph::new();

    let input = graph.add_input(Layout::from([1]));
    let result = graph.add_op(Op::Mul, &[input, input]);

    graph.add_output(result);

    println!("{:?}", Wgpu.compile(graph).run(vec![]));
}
