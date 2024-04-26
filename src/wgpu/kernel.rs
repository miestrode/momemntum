use std::{collections::HashMap, iter, sync::OnceLock};

use serde::{Deserialize, Serialize};
use tera::{Context, Tera};

use crate::{graph::ExprId, tensor::Layout};

use super::expr::WgpuExpr;

const ELEMWISE: &str = "elemwise";
const REDUCE: &str = "reduce";

fn tera() -> &'static Tera {
    static TERA: OnceLock<Tera> = OnceLock::new();

    TERA.get_or_init(|| {
        let mut tera = Tera::default();

        tera.add_template_files([
            ("./src/wgpu/templates/common.wgsl.tera", Some("common")),
            ("./src/wgpu/templates/elemwise.wgsl.tera", Some(ELEMWISE)),
            ("./src/wgpu/templates/reduce.wgsl.tera", Some(REDUCE)),
        ])
        .expect("could not create templates");

        tera
    })
}

#[derive(Serialize, Deserialize)]
struct LayoutInfo {
    elements: usize,
    strides: Vec<usize>,
    dims: Vec<usize>,
}

impl LayoutInfo {
    fn new(layout: &Layout) -> Self {
        Self {
            elements: layout.elements(),
            strides: layout.strides().to_vec(),
            dims: layout.dims().to_vec(),
        }
    }
}

pub(crate) fn elemwise(
    workgroup_size_x: u32,
    output_layout: &Layout,
    layouts: HashMap<ExprId, &Layout>,
    expr: WgpuExpr,
) -> String {
    let mut context = Context::new();

    context.insert("workgroup_size_x", &workgroup_size_x);
    context.insert(
        "layouts",
        &layouts
            .iter()
            .map(|(id, layout)| (format!("input_{}", id.0), *layout))
            .chain(iter::once((String::from("output"), output_layout)))
            .map(|(id, layout)| (id, LayoutInfo::new(layout)))
            .collect::<HashMap<_, _>>(),
    );
    context.insert(
        "inputs",
        &layouts
            .keys()
            .map(|id| format!("input_{}", id.0))
            .collect::<Vec<_>>(),
    );
    context.insert("expr", &expr.to_string());

    tera()
        .render(ELEMWISE, &context)
        .expect("template execution failed")
}

// pub(crate) fn reduce(
//     workgroup_size_x: u32,
//     op: ReduceOp,
//     input: &Layout,
//     output: &Layout,
//     dims: &[DimId],
// ) -> String {
//     let mut context = Context::new();
//
//     context.insert("workgroup_size_x", &workgroup_size_x);
//     context.insert("strides", &input.strides());
//     context.insert("new_strides", &output.strides());
//     context.insert("reduce_dims", dims);
//     context.insert("input_dims", input.dims());
//     context.insert("op", &op.to_string());
//
//     tera()
//         .render(REDUCE, &context)
//         .expect("template execution failed")
// }
