use std::{borrow::Cow, collections::HashMap, num::NonZeroU64};

use pollster::FutureExt;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, ComputePipelineDescriptor,
    Device, Instance, InstanceDescriptor, Maintain, MapMode, PipelineLayoutDescriptor, Queue,
    RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    SubmissionIndex,
};

use crate::{
    compiler::Runner,
    graph::ExprId,
    tensor::{Layout, Tensor},
};

use super::compiler::{WgpuCompiler, WgpuPlan, WgpuStep};

#[derive(Debug)]
pub(crate) enum ConcreteWgpuStep {
    Allocate {
        id: ExprId,
        tensor: Tensor,
    },
    Deallocate(ExprId),
    Execute {
        output: ExprId,
        output_size: u64,
        compute_pipeline: ComputePipeline,
        bind_group_layout: BindGroupLayout,
        workgroups: [u32; 3],
        inputs: Vec<ExprId>,
    },
}

#[derive(Debug)]
pub struct ConcreteWgpuPlan {
    pub(crate) inputs: Vec<ExprId>,
    pub(crate) steps: Vec<ConcreteWgpuStep>,
    pub(crate) outputs: Vec<ExprId>,
    pub(crate) output_layouts: Vec<Layout>,
}

pub struct WgpuRunner {
    device: Device,
    queue: Queue,
    buffers: HashMap<ExprId, Buffer>,
}

impl Default for WgpuRunner {
    fn default() -> Self {
        let instance = Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .block_on()
            .expect("could not find adapter");

        Self::new_with_adapter(adapter).block_on()
    }
}

impl WgpuRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn new_with_adapter(adapter: Adapter) -> Self {
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .expect("could not get device");

        Self {
            device,
            queue,
            buffers: HashMap::new(),
        }
    }

    fn track(&mut self, id: ExprId, buffer: Buffer) {
        self.buffers.insert(id, buffer);
    }

    fn allocate(&mut self, id: ExprId, tensor: &Tensor) {
        self.track(
            id,
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&tensor.data),
                usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            }),
        );
    }

    fn deallocate(&mut self, id: ExprId) {
        self.buffers.remove(&id);
    }

    fn retrieve(&self, id: ExprId, layout: Layout) -> Tensor {
        let buffer = &self.buffers[&id];
        let staging_buffer = self.create_staging_buffer(&layout);

        let mut encoder = self.device.create_command_encoder(&Default::default());

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, layout.size() as u64);

        let copy_submission = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |_| {});

        self.device
            .poll(Maintain::WaitForSubmissionIndex(copy_submission));

        let data = buffer_slice.get_mapped_range();
        let tensor = Tensor {
            data: bytemuck::cast_slice(&data).to_vec().into_boxed_slice(),
            layout,
        };

        drop(data);
        staging_buffer.unmap();

        tensor
    }

    fn create_shader_module(&self, contents: &str) -> ShaderModule {
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(contents)),
        })
    }

    fn create_staging_buffer(&self, layout: &Layout) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: layout.size() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn create_output_buffer(&mut self, id: ExprId, size: u64) {
        self.track(
            id,
            self.device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        );
    }

    fn create_compute_pipeline(
        &self,
        module: &ShaderModule,
        entry_point: &str,
        bind_group_layout: &BindGroupLayout,
    ) -> ComputePipeline {
        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(
                    &self
                        .device
                        .create_pipeline_layout(&PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[bind_group_layout],
                            push_constant_ranges: &[],
                        }),
                ),
                module,
                entry_point,
            })
    }

    fn create_bind_group_layout(&self, inputs_layout: &[(usize, bool)]) -> BindGroupLayout {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &inputs_layout
                    .iter()
                    .enumerate()
                    .map(|(index, &(size, read_only))| BindGroupLayoutEntry {
                        binding: index as u32,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(size as u64).unwrap()),
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
    }

    fn create_bind_group(&self, layout: &BindGroupLayout, buffers: &[&Buffer]) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries: buffers
                .iter()
                .enumerate()
                .map(|(index, buffer)| BindGroupEntry {
                    binding: index as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>()
                .as_slice(),
        })
    }

    fn create_command_encoder(&self) -> CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    fn start_compute_pass(
        &self,
        mut encoder: CommandEncoder,
        compute_pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        workgroups: [u32; 3],
    ) -> SubmissionIndex {
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());

            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        self.queue.submit(Some(encoder.finish()))
    }

    fn execute_pipeline(
        &self,
        compute_pipeline: &ComputePipeline,
        workgroups: [u32; 3],
        bind_group_layout: &BindGroupLayout,
        buffers: &[ExprId],
    ) {
        let buffers = buffers
            .iter()
            .map(|id| &self.buffers[&id])
            .collect::<Vec<_>>();

        let bind_group = self.create_bind_group(bind_group_layout, &buffers);

        let encoder = self.create_command_encoder();

        self.start_compute_pass(encoder, compute_pipeline, &bind_group, workgroups);
    }
}

impl Runner for WgpuRunner {
    type Compiler = WgpuCompiler;

    type Runnable = ConcreteWgpuPlan;

    fn preprocess(&mut self, plan: WgpuPlan) -> ConcreteWgpuPlan {
        ConcreteWgpuPlan {
            inputs: plan.inputs,
            steps: plan
                .steps
                .into_iter()
                .map(|step| match step {
                    WgpuStep::Allocate { id, tensor } => ConcreteWgpuStep::Allocate { id, tensor },
                    WgpuStep::Deallocate(id) => ConcreteWgpuStep::Deallocate(id),
                    WgpuStep::Execute {
                        output,
                        source,
                        workgroups,
                        inputs,
                        inputs_layout,
                    } => {
                        let module = self.create_shader_module(&source);
                        let bind_group_layout = self.create_bind_group_layout(&inputs_layout);

                        ConcreteWgpuStep::Execute {
                            output,
                            output_size: inputs_layout[0].0 as u64,
                            compute_pipeline: self.create_compute_pipeline(
                                &module,
                                "main",
                                &bind_group_layout,
                            ),
                            bind_group_layout,
                            workgroups,
                            inputs,
                        }
                    }
                })
                .collect(),
            outputs: plan.outputs,
            output_layouts: plan.output_layouts,
        }
    }

    fn run(&mut self, plan: ConcreteWgpuPlan, inputs: Vec<Tensor>) -> Vec<Tensor> {
        for (index, input) in inputs.iter().enumerate() {
            self.allocate(plan.inputs[index], input);
        }

        for step in plan.steps {
            match step {
                ConcreteWgpuStep::Allocate { id, tensor } => {
                    self.allocate(id, &tensor);
                }
                ConcreteWgpuStep::Deallocate(id) => {
                    self.deallocate(id);
                }
                ConcreteWgpuStep::Execute {
                    output,
                    output_size,
                    compute_pipeline,
                    bind_group_layout,
                    workgroups,
                    inputs,
                } => {
                    self.create_output_buffer(output, output_size);

                    self.execute_pipeline(
                        &compute_pipeline,
                        workgroups,
                        &bind_group_layout,
                        &inputs,
                    );
                }
            }
        }

        plan.outputs
            .into_iter()
            .zip(plan.output_layouts)
            .map(|(id, layout)| self.retrieve(id, layout))
            .collect()
    }
}
