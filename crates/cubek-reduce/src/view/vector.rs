use std::cmp::max;

use cubecl::prelude::*;
use cubecl::std::tensor::layout::plain::{PlainLayout, PlainLayoutLaunch};
use cubecl::std::tensor::layout::{Coords1d, Coords3d, Layout, LayoutExpand};

#[derive(CubeType, CubeLaunch, Clone)]
/// A 3D tile (batch, group, vec)
pub struct GroupedVectorLayout {
    layout: PlainLayout,
    /// The number of vector in a group.
    stride_group: u32,
    /// The number of groups in a batch.
    stride_batch: u32,
}

impl<'a, R: Runtime> GroupedVectorLayoutLaunch<'a, R> {
    /// Creates group vector from the shape and strides and the vectorization axis .
    ///
    /// # Important
    ///
    /// The vectorization axis must respect the following properties:
    ///
    /// - `strides[axis] == 1`
    /// - `shape[axis] % line_size == 0`
    /// - `shape[0..axis]` don't mix with `shape[axis..rank-1]`.
    pub fn from_shape_strides(
        shape: &[usize],
        strides: &[usize],
        line_size: u8,
        vectorization_axis: u8,
    ) -> Self {
        let layout = PlainLayoutLaunch::from_shape(shape, line_size);
        let mut stride_batch = 1u32;
        let mut stride_group = 1u32;
        let mut done = false;

        for (axis, stride) in strides.iter().enumerate().rev() {
            if axis == vectorization_axis as usize {
                done = true;
                assert!(strides[axis] == 1);
                continue;
            }

            if !done {
                stride_group *= *stride as u32 / line_size as u32;
            } else {
                stride_batch *= *stride as u32 / line_size as u32;
            }
        }

        println!("strides_batch: {stride_batch} lines");
        println!("strides_group: {stride_group} lines");

        Self {
            layout,
            stride_batch: ScalarArg::new(stride_batch),
            stride_group: ScalarArg::new(stride_group),
            _phantom_runtime: std::marker::PhantomData,
            _phantom_a: std::marker::PhantomData,
        }
    }
}

#[cube]
impl Layout for GroupedVectorLayout {
    type Coordinates = Coords3d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> u32 {
        let (batch, group, vector) = pos;
        let pos = (batch * self.stride_batch) + (group * self.stride_group) + vector;
        self.layout.to_source_pos(pos)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (u32, bool) {
        let (batch, group, vector) = pos;
        let pos = batch * self.stride_batch + group * self.stride_group + vector;
        (
            self.layout.to_source_pos(pos),
            self.layout.is_in_bounds(pos),
        )
    }

    fn shape(&self) -> Self::Coordinates {
        (
            self.stride_batch,
            self.stride_group,
            self.layout.shape() / (self.stride_batch * self.stride_group),
        )
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (batch, group, vector) = pos;
        let pos = batch * self.stride_batch + group * self.stride_group + vector;
        self.layout.is_in_bounds(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::{
        TestRuntime,
        std::tensor::{TensorHandle, View, into_contiguous, launch::ViewArg},
    };
    use cubecl_common::bytes::Bytes;

    #[test]
    fn grouped_vector_axis2_origin() {
        TestCase {
            shape: vec![4, 4, 4],
            transposed: false,
            line_size: 4,
            expected: vec![0.0, 1.0, 2.0, 3.0],
            batch: 0,
            group: 0,
            pos: 0,
        }
        .run()
    }

    #[test]
    fn grouped_vector_axis2_middle() {
        TestCase {
            shape: vec![4, 4, 4],
            transposed: false,
            line_size: 4,
            expected: vec![40.0, 41.0, 42.0, 43.0],
            batch: 2,
            group: 2,
            pos: 0,
        }
        .run()
    }

    #[test]
    fn grouped_vector_axis1_origin_transposed() {
        TestCase {
            shape: vec![4, 4, 4],
            transposed: true,
            line_size: 4,
            expected: vec![0.0, 4.0, 8.0, 12.0],
            batch: 0,
            group: 0,
            pos: 0,
        }
        .run()
    }

    #[test]
    fn grouped_vector_axis1_origin_transposed_2() {
        TestCase {
            shape: vec![3, 4, 2],
            transposed: true,
            line_size: 2,
            expected: vec![1.0, 3.0],
            batch: 0,
            group: 1,
            pos: 0,
        }
        .run()
    }

    #[cube(launch_unchecked)]
    fn run_test(
        input: &View<Line<f32>, Coords3d>,
        output: &mut Tensor<Line<f32>>,
        batch: u32,
        group: u32,
        pos: u32,
    ) {
        let value = input[(batch, group, pos)];
        output[0] = value;
    }

    struct TestCase {
        shape: Vec<usize>,
        transposed: bool,
        line_size: u8,
        expected: Vec<f32>,
        batch: u32,
        group: u32,
        pos: u32,
    }

    impl TestCase {
        fn transpose(origin: &[usize]) -> Vec<usize> {
            let rank = origin.len();
            let mut output = origin.to_vec();
            let tmp = output[rank - 1];
            output[rank - 1] = output[rank - 2];
            output[rank - 2] = tmp;
            output
        }

        fn arange_input(
            client: &ComputeClient<TestRuntime>,
            shape: &[usize],
            transposed: bool,
        ) -> TensorHandle<TestRuntime> {
            let storage = f32::as_type_native_unchecked();
            let num_elems = shape.iter().product::<usize>();
            let elems: Vec<f32> = (0..num_elems).into_iter().map(|e| e as f32).collect();
            let bytes = Bytes::from_elems(elems);
            let alloc = client.create_tensor(bytes, &shape, storage.size());

            if transposed {
                // We only transpose the strides to generate not contiguous data.
                let strides = Self::transpose(&alloc.strides);
                let shape = Self::transpose(shape);
                let handle = TensorHandle::new(
                    alloc.handle,
                    shape.clone(),
                    strides.clone(),
                    f32::as_type_native_unchecked(),
                );
                let result = into_contiguous(&client, &handle.as_ref(), storage).unwrap();
                let strides = Self::transpose(&result.strides);
                TensorHandle::new(
                    result.handle,
                    shape,
                    strides,
                    f32::as_type_native_unchecked(),
                )
            } else {
                TensorHandle::new(alloc.handle, shape.to_vec(), alloc.strides, storage)
            }
        }

        fn run(self) {
            let client = TestRuntime::client(&Default::default());
            let input = Self::arange_input(&client, &self.shape, self.transposed);
            let axis = if self.transposed { 1 } else { 2 };
            let grouped_vector = GroupedVectorLayoutLaunch::from_shape_strides(
                &input.shape,
                &input.strides,
                self.line_size,
                axis,
            );

            let buffer: ArrayArg<'_, TestRuntime> = unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &input.handle,
                    input.shape.iter().product(),
                    self.line_size,
                )
            };

            let input = ViewArg::<'_, Coords3d, TestRuntime>::new::<GroupedVectorLayout>(
                buffer,
                grouped_vector,
            );

            let output = TensorHandle::empty(
                &client,
                vec![self.line_size as usize],
                f32::as_type_native_unchecked(),
            );

            unsafe {
                run_test::launch_unchecked(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    input,
                    output.as_arg(self.line_size),
                    ScalarArg::new(self.batch),
                    ScalarArg::new(self.group),
                    ScalarArg::new(self.pos),
                )
                .unwrap();
            }

            let output = client.read_one(output.handle);
            let elems = output.elems::<f32>().unwrap();

            assert_eq!(elems, &self.expected);
        }
    }
}
