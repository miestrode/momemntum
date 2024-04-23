use std::sync::OnceLock;

use tera::{Context, Tera};

use crate::{
    graph::{ElemwiseOp, ReduceOp},
    tensor::{DimId, Layout},
};

const ELEMWISE: &str = "elemwise";
const REDUCE: &str = "reduce";

fn tera() -> &'static Tera {
    static TERA: OnceLock<Tera> = OnceLock::new();

    TERA.get_or_init(|| {
        let mut tera = Tera::default();

        tera.add_template_files([
            ("./src/wgpu/templates/elemwise.wgsl.tera", Some(ELEMWISE)),
            ("./src/wgpu/templates/reduce.wgsl.tera", Some(REDUCE)),
        ])
        .expect("could not create templates");

        tera
    })
}

pub(crate) fn unary_elemwise(workgroup_size_x: u32, op: ElemwiseOp, elements: usize) -> String {
    let mut context = Context::new();

    context.insert("workgroup_size_x", &workgroup_size_x);
    context.insert("op", &op.to_string());
    context.insert("elements", &elements);

    tera()
        .render(UNARY_ELEMWISE, &context)
        .expect("template execution failed")
}

pub(crate) fn binary_elemwise(
    workgroup_size_x: u32,
    op: BinaryElemwiseOp,
    left_input: &Layout,
    right_input: &Layout,
) -> String {
    let mut context = Context::new();

    context.insert("workgroup_size_x", &workgroup_size_x);
    context.insert("op", &op.to_string());
    context.insert("left_strides", &left_input.strides());
    context.insert("right_strides", &right_input.strides());
    context.insert("elements", &left_input.elements());

    tera()
        .render(BINARY_ELEMWISE, &context)
        .expect("template execution failed")
}

pub(crate) fn reduce(
    workgroup_size_x: u32,
    op: ReduceOp,
    input: &Layout,
    output: &Layout,
    dims: &[DimId],
) -> String {
    let mut context = Context::new();

    context.insert("workgroup_size_x", &workgroup_size_x);
    context.insert("strides", &input.strides());
    context.insert("new_strides", &output.strides());
    context.insert("reduce_dims", dims);
    context.insert("input_dims", input.dims());
    context.insert("op", &op.to_string());

    tera()
        .render(REDUCE, &context)
        .expect("template execution failed")
}
