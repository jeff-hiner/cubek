use cubecl::{
    benchmark::{Benchmark, TimingMethod},
    frontend, future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    random::random_uniform,
    reduce::instructions::{ReduceFn, ReduceFnConfig},
};
use std::marker::PhantomData;

#[allow(dead_code)]
struct ReduceBench<R: Runtime, E> {
    shape: Vec<usize>,
    device: R::Device,
    axis: usize,
    client: ComputeClient<R>,
    _e: PhantomData<E>,
}

impl<R: Runtime, E: Float> Benchmark for ReduceBench<R, E> {
    type Input = (TensorHandle<R>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let elem = E::as_type_native_unchecked();

        let input = TensorHandle::empty(&client, self.shape.clone(), elem);
        random_uniform(&client, 0., 1., input.as_ref(), elem).unwrap();
        let mut shape_out = self.shape.clone();
        shape_out[self.axis] = 1;
        let out = TensorHandle::empty(&client, shape_out, elem);

        (input, out)
    }

    fn execute(&self, (input, out): Self::Input) -> Result<(), String> {
        cubek::reduce::reduce::<R, ReduceFn>(
            &self.client,
            input.as_ref(),
            out.as_ref(),
            self.axis,
            None,
            ReduceFnConfig::Sum,
            cubek::reduce::ReduceDtypes {
                input: E::as_type_native_unchecked(),
                output: E::as_type_native_unchecked(),
                accumulation: f32::as_type_native_unchecked(),
            },
        )
        .map_err(|err| format!("{err}"))?;

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "reduce-axis({})-{}-{:?}",
            self.axis,
            E::as_type_native_unchecked(),
            self.shape
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: R::Device) {
    let client = R::client(&device);
    for axis in [0, 1, 2] {
        let bench = ReduceBench::<R, E> {
            shape: vec![32, 512, 2048],
            axis,
            client: client.clone(),
            device: device.clone(),
            _e: PhantomData,
        };
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::System).unwrap());
    }
}

fn main() {
    run::<cubecl::TestRuntime, f32>(Default::default());
}
