use tracel_xtask::prelude::*;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct CubeCLTestCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    args: CubeCLTestCmdArgs,
    _env: Environment,
    _context: Context,
) -> anyhow::Result<()> {
    let backends: &[&str] = if args.ci {
        &["cubecl/wgpu", "cubecl/cpu"]
    } else {
        &["cubecl/wgpu"]
    };
    for backend in backends {
        helpers::custom_crates_tests(
            vec![
                "cubek-matmul",
                "cubek-convolution",
                "cubek-attention",
                "cubek-random",
                "cubek-reduce",
            ],
            vec!["--features", backend],
            None,
            None,
            &format!("Test on backend {backend:?}"),
        )?;
    }
    Ok(())
}
